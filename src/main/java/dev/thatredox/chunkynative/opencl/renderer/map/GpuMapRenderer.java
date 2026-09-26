package dev.thatredox.chunkynative.opencl.renderer.map;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Tab;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.WritableImage;
import javafx.scene.image.WritablePixelFormat;
import org.jocl.*;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.map.MapBuffer;
import se.llbit.chunky.ui.ChunkMap;
import se.llbit.chunky.world.ChunkView;
import se.llbit.log.Log;

import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Collection;

import sun.misc.Unsafe;

/**
 * Overrides Chunky's CPU-based 2D map rendering with GPU-accelerated pixel scaling.
 *
 * <p>Since Chunky's plugin API does not expose the map rendering pipeline, this class
 * uses the main tab transformer hook to defer initialization until the JavaFX scene is
 * ready, then uses reflection to locate {@link ChunkyFxController} → {@link ChunkMap}
 * → {@link MapBuffer}.  The existing MapBuffer is wrapped by a {@link GpuMapBuffer}
 * that delegates all data operations to the original but overrides
 * {@code drawBuffered()} with an OpenCL nearest-neighbour upscaler.
 */
public class GpuMapRenderer {
    private static final WritablePixelFormat<IntBuffer> PIXEL_FORMAT =
            PixelFormat.getIntArgbInstance();

    private final Chunky chunky;

    // Reflection handles for MapBuffer's private fields
    private Field mbPixelsField;
    private Field mbWidthField;
    private Field mbHeightField;
    private Field mbViewField;
    private Field mbCachedField;
    private Field mbImageField;

    // GPU resources (lazily created, reused across frames). Both buffers are
    // allocated with CL_MEM_ALLOC_HOST_PTR so the driver places them in
    // pinned host memory; on integrated GPUs (shared RAM) map/unmap is then a
    // pointer hand-back instead of a full-frame copy in each direction.
    private cl_kernel scaleKernel;
    private cl_mem gpuSrcBuffer;
    private cl_mem gpuDstBuffer;
    private int lastSrcSize;
    private int lastDstSize;
    private WritableImage gpuImage;
    // One-time downgrade flags: if a driver refuses to map a pinned buffer,
    // fall back permanently (per context) to plain write/read transfers.
    private boolean srcMapUploadOk = true;
    private boolean dstMapReadOk = true;
    // Host-side destination scratch, only used on the dst read fallback path.
    private int[] dstScratch;

    // Source-content version last uploaded to gpuSrcBuffer. While the version
    // is unchanged (pure pan/zoom, hover repaints) the upload is skipped
    // entirely — the kernel re-reads the buffer already resident on the GPU.
    private long lastUploadedVersion = Long.MIN_VALUE;

    // Parameters of the frame currently held in gpuImage. When both the
    // content version and all scale parameters match, drawBuffered just
    // re-blits gpuImage without touching OpenCL at all.
    private boolean lastDrawnValid = false;
    private long lastDrawnVersion;
    private int lastDrawnDstW, lastDrawnDstH;
    private float lastDrawnScale;
    private int lastDrawnOffX, lastDrawnOffZ;

    // Dedicated command queue: this code runs on the JavaFX Application Thread,
    // so it must NOT share the render thread's queue (concurrent host access to
    // one OpenCL queue is undefined behavior). Independent queues on the same
    // context are safe.
    private cl_command_queue mapQueue;
    // The ContextManager the cached queue/kernel/buffers belong to. When it is
    // swapped (device switch / reload) the old resources are released and
    // rebuilt — reusing them would enqueue on a released context and leak.
    private ContextManager mapCtx;

    private volatile boolean installed = false;

    public GpuMapRenderer(Chunky chunky) {
        this.chunky = chunky;
    }

    /**
     * Install the GPU map renderer hook via the main tab transformer.
     * Must be called during {@code Plugin.attach()} before the UI launches.
     */
    public static GpuMapRenderer install(Chunky chunky) {
        GpuMapRenderer renderer = new GpuMapRenderer(chunky);

        se.llbit.chunky.plugin.TabTransformer prev = chunky.getMainTabTransformer();
        chunky.setMainTabTransformer(tabs -> {
            Collection<Tab> result = prev.apply(tabs);

            // Grab any tab so we can reach the Scene later
            Tab savedTab = null;
            for (Tab t : result) {
                savedTab = t;
                break;
            }
            final Tab tab = savedTab;

            // Double-deferred runLater to ensure the full scene graph is assembled
            Platform.runLater(() -> Platform.runLater(() -> {
                try {
                    if (tab == null) return;
                    javafx.scene.control.TabPane tabPane = tab.getTabPane();
                    if (tabPane == null) return;
                    Scene scene = tabPane.getScene();
                    if (scene == null) return;
                    renderer.doInstall(scene);
                } catch (Exception e) {
                    Log.warn("ChunkyCL: GPU map renderer install failed; the 2D "
                            + "map will fall back to CPU rendering. This usually "
                            + "means chunky's UI internals changed in a way the "
                            + "plugin's reflection-based hook can't follow.", e);
                }
            }));

            return result;
        });

        return renderer;
    }

    // ---- Installation via reflection ----

    private void doInstall(Scene scene) throws Exception {
        // 1. Locate mapOverlay Canvas (event handlers are registered here)
        Canvas mapOverlay = (Canvas) scene.lookup("#mapOverlay");
        if (mapOverlay == null) {
            Log.warn("ChunkyCL: Could not find #mapOverlay via scene lookup");
            return;
        }

        // 2. Extract ChunkMap from a method-reference event handler on mapOverlay.
        //    ChunkyFxController registers e.g. mapOverlay.setOnMousePressed(map::onMousePressed)
        //    The lambda captures the ChunkMap instance in a synthetic field.
        ChunkMap chunkMap = extractFromHandler(mapOverlay.getOnMousePressed(), ChunkMap.class);
        if (chunkMap == null)
            chunkMap = extractFromHandler(mapOverlay.getOnScroll(), ChunkMap.class);
        if (chunkMap == null)
            chunkMap = extractFromHandler(mapOverlay.getOnMouseDragged(), ChunkMap.class);
        if (chunkMap == null) {
            Log.warn("ChunkyCL: Could not extract ChunkMap from mapOverlay event handlers");
            return;
        }

        // 3. Get the original MapBuffer from ChunkMap
        Field mapBufferField = ChunkMap.class.getDeclaredField("mapBuffer");
        mapBufferField.setAccessible(true);
        MapBuffer originalBuffer = (MapBuffer) mapBufferField.get(chunkMap);
        if (originalBuffer == null) {
            Log.warn("ChunkyCL: ChunkMap.mapBuffer is null");
            return;
        }

        // 4. Prepare reflection handles for MapBuffer's private state. If chunky
        //    renames any of these fields, report exactly which one so the cause
        //    is obvious without poring through stack traces.
        mbPixelsField  = requireField(MapBuffer.class, "pixels");
        mbWidthField   = requireField(MapBuffer.class, "width");
        mbHeightField  = requireField(MapBuffer.class, "height");
        mbViewField    = requireField(MapBuffer.class, "view");
        mbCachedField  = requireField(MapBuffer.class, "cached");
        mbImageField   = requireField(MapBuffer.class, "image");

        // 5. Replace mapBuffer with our GpuMapBuffer wrapper.
        //    The field is 'protected final', so Field.set() may fail on some JVMs.
        //    Fall back to sun.misc.Unsafe if needed.
        GpuMapBuffer gpuBuffer = new GpuMapBuffer(originalBuffer, this);
        try {
            mapBufferField.set(chunkMap, gpuBuffer);
        } catch (IllegalAccessException e) {
            // Final field – use Unsafe to bypass
            Field unsafeField = Unsafe.class.getDeclaredField("theUnsafe");
            unsafeField.setAccessible(true);
            Unsafe unsafe = (Unsafe) unsafeField.get(null);
            long offset = unsafe.objectFieldOffset(mapBufferField);
            unsafe.putObject(chunkMap, offset, gpuBuffer);
        }

        installed = true;
        Log.info("ChunkyCL: GPU map renderer installed successfully");
    }

    private static Field requireField(Class<?> cls, String name) throws NoSuchFieldException {
        try {
            Field f = cls.getDeclaredField(name);
            f.setAccessible(true);
            return f;
        } catch (NoSuchFieldException e) {
            throw new NoSuchFieldException("ChunkyCL: required private field '"
                    + cls.getSimpleName() + "." + name + "' was not found. Chunky "
                    + "may have renamed it; the GPU map hook needs to be updated.");
        }
    }

    // ---- GPU-accelerated drawBuffered ----

    /**
     * GPU version of {@link MapBuffer#drawBuffered(GraphicsContext)}.
     * Reads the delegate's private pixel buffer via reflection, performs
     * nearest-neighbour upscaling on the GPU, and draws the result.
     *
     * @return true on success, false to signal the caller to fall back to CPU
     */
    boolean gpuDrawBuffered(GraphicsContext gc, GpuMapBuffer wrapper) {
        if (!installed) return false;

        MapBuffer delegate = wrapper.getDelegate();
        try {
            synchronized (delegate) {
                ChunkView view = (ChunkView) mbViewField.get(delegate);
                if (view.width <= 0 || view.height <= 0) return true; // nothing to draw

                int[] pixels = (int[]) mbPixelsField.get(delegate);
                int srcWidth  = mbWidthField.getInt(delegate);
                int srcHeight = mbHeightField.getInt(delegate);
                if (pixels == null || srcWidth <= 0 || srcHeight <= 0) return true;

                int dstWidth  = view.width;
                int dstHeight = view.height;

                // Compute source window (matches CPU MapBuffer logic)
                float scale = view.scale / (float) view.chunkScale;
                int srcOffsetX = (int) (0.5 + view.chunkScale * (view.x0 - view.px0));
                int srcOffsetZ = (int) (0.5 + view.chunkScale * (view.z0 - view.pz0));

                long version = wrapper.contentVersion();

                // Fast path: identical frame — re-blit without touching OpenCL.
                if (lastDrawnValid && gpuImage != null
                        && version == lastDrawnVersion
                        && dstWidth == lastDrawnDstW && dstHeight == lastDrawnDstH
                        && scale == lastDrawnScale
                        && srcOffsetX == lastDrawnOffX && srcOffsetZ == lastDrawnOffZ) {
                    gc.clearRect(0, 0, dstWidth, dstHeight);
                    gc.drawImage(gpuImage, 0, 0);
                    return true;
                }

                // Invalidate up front: if the scale below throws or falls back
                // to CPU, gpuImage no longer matches the recorded parameters.
                lastDrawnValid = false;

                if (!gpuScaleToImage(pixels, srcWidth, srcHeight,
                        dstWidth, dstHeight, scale, srcOffsetX, srcOffsetZ, version)) {
                    return false; // fall back to CPU
                }

                // Store back so MapBuffer considers itself cached
                mbImageField.set(delegate, gpuImage);
                mbCachedField.setBoolean(delegate, true);

                gc.clearRect(0, 0, dstWidth, dstHeight);
                gc.drawImage(gpuImage, 0, 0);

                lastDrawnVersion = version;
                lastDrawnDstW = dstWidth;
                lastDrawnDstH = dstHeight;
                lastDrawnScale = scale;
                lastDrawnOffX = srcOffsetX;
                lastDrawnOffZ = srcOffsetZ;
                lastDrawnValid = true;
                return true;
            }
        } catch (Exception e) {
            Log.warn("ChunkyCL: GPU map draw failed; falling back to CPU for this frame", e);
            return false;
        }
    }

    /**
     * Drops the delegate's cached-image flag. Called by {@link GpuMapBuffer} when it
     * skips a redundant {@code redrawView()} so a CPU fallback would still re-scale
     * at the current pan offset instead of blitting a stale cached image.
     */
    void invalidateDelegateImageCache(MapBuffer delegate) {
        if (!installed) return;
        try {
            mbCachedField.setBoolean(delegate, false);
        } catch (Exception ignored) {
            // Field access already worked during install; nothing sensible to do here.
        }
    }

    // ---- OpenCL nearest-neighbour scaling ----

    /**
     * Scales the source pixels on the GPU and writes the result into {@link #gpuImage}
     * (allocating/resizing it as needed).
     *
     * <p>The source upload is skipped whenever {@code contentVersion} matches the
     * version already resident in {@code gpuSrcBuffer} — during pans and zooms only
     * the kernel parameters change, so the map moves without any host→GPU transfer.
     * Transfers that do happen go through pinned-memory map/unmap, which on an
     * integrated GPU is a single memcpy (upload) and a zero-copy read (download)
     * instead of the double copies of write/read-buffer.
     *
     * @return true on success, false to signal the caller to fall back to CPU
     */
    private boolean gpuScaleToImage(int[] src, int srcWidth, int srcHeight,
                                    int dstWidth, int dstHeight, float scale,
                                    int srcOffsetX, int srcOffsetZ, long contentVersion) {
        ByteBuffer mappedDst = null;
        try {
            ContextManager ctx = ContextManager.get();
            // Rebuild on first use or after a device switch/reload: the cached
            // queue/kernel/buffers belong to a specific (now possibly released)
            // context. Reusing them across a context swap enqueues on a dead
            // context (CL_INVALID_CONTEXT) and leaks the old handles.
            if (mapCtx != ctx) {
                releaseGpuResources();
                mapCtx = ctx;
                mapQueue = ctx.context.createCommandQueue();
            }
            int srcSize = src.length;
            int dstSize = dstWidth * dstHeight;
            if (dstSize <= 0) return false;

            // (Re-)allocate GPU buffers when sizes change
            if (gpuSrcBuffer == null || srcSize != lastSrcSize) {
                if (gpuSrcBuffer != null) clReleaseMemObject(gpuSrcBuffer);
                gpuSrcBuffer = clCreateBuffer(ctx.context.context,
                        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                        (long) Sizeof.cl_int * srcSize, null, null);
                lastSrcSize = srcSize;
                lastUploadedVersion = Long.MIN_VALUE; // new buffer is empty
            }
            if (gpuDstBuffer == null || dstSize != lastDstSize) {
                if (gpuDstBuffer != null) clReleaseMemObject(gpuDstBuffer);
                gpuDstBuffer = clCreateBuffer(ctx.context.context,
                        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                        (long) Sizeof.cl_int * dstSize, null, null);
                lastDstSize = dstSize;
            }

            // Upload source pixels only when their content actually changed.
            if (contentVersion != lastUploadedVersion) {
                boolean uploaded = false;
                if (srcMapUploadOk) {
                    int[] err = new int[1];
                    ByteBuffer mappedSrc = null;
                    try {
                        mappedSrc = clEnqueueMapBuffer(mapQueue, gpuSrcBuffer, CL_TRUE,
                                CL_MAP_WRITE_INVALIDATE_REGION, 0,
                                (long) Sizeof.cl_int * srcSize, 0, null, null, err);
                    } catch (CLException clFailure) {
                        // JOCL has exceptions enabled globally (Device.<clinit>), so
                        // this — not the errcode-out param — is the normal failure
                        // path. Carry the REAL status through so a transient error
                        // isn't mistaken for "this device can't map".
                        err[0] = clFailure.getStatus();
                    } catch (Exception mapFailure) {
                        err[0] = CL_INVALID_OPERATION; // unknown cause: assume unsupported
                    }
                    if (mappedSrc != null && err[0] == CL_SUCCESS) {
                        try {
                            mappedSrc.order(ByteOrder.nativeOrder()).asIntBuffer().put(src);
                        } finally {
                            clEnqueueUnmapMemObject(mapQueue, gpuSrcBuffer, mappedSrc,
                                    0, null, null);
                        }
                        uploaded = true;
                    } else {
                        // A driver that hands back both a pointer and an error is
                        // out of spec, but if it does, the region is still mapped
                        // to the host — release it rather than leak it forever.
                        if (mappedSrc != null) {
                            try {
                                clEnqueueUnmapMemObject(mapQueue, gpuSrcBuffer, mappedSrc,
                                        0, null, null);
                            } catch (Exception ignored) { }
                        }
                        latchSrcMapUnsupported(err[0]);
                    }
                }
                if (!uploaded) {
                    clEnqueueWriteBuffer(mapQueue, gpuSrcBuffer, CL_TRUE, 0,
                            (long) Sizeof.cl_int * srcSize, Pointer.to(src), 0, null, null);
                }
                lastUploadedVersion = contentVersion;
            }

            // Create kernel on first use
            if (scaleKernel == null) {
                scaleKernel = clCreateKernel(ctx.renderer.kernel, "mapScale", null);
            }

            int ai = 0;
            clSetKernelArg(scaleKernel, ai++, Sizeof.cl_mem, Pointer.to(gpuSrcBuffer));
            clSetKernelArg(scaleKernel, ai++, Sizeof.cl_mem, Pointer.to(gpuDstBuffer));
            clSetKernelArg(scaleKernel, ai++, Sizeof.cl_int, Pointer.to(new int[]{srcWidth}));
            clSetKernelArg(scaleKernel, ai++, Sizeof.cl_int, Pointer.to(new int[]{srcHeight}));
            clSetKernelArg(scaleKernel, ai++, Sizeof.cl_int, Pointer.to(new int[]{dstWidth}));
            clSetKernelArg(scaleKernel, ai++, Sizeof.cl_int, Pointer.to(new int[]{dstHeight}));
            clSetKernelArg(scaleKernel, ai++, Sizeof.cl_float, Pointer.to(new float[]{scale}));
            clSetKernelArg(scaleKernel, ai++, Sizeof.cl_int, Pointer.to(new int[]{srcOffsetX}));
            clSetKernelArg(scaleKernel, ai++, Sizeof.cl_int, Pointer.to(new int[]{srcOffsetZ}));

            clEnqueueNDRangeKernel(mapQueue, scaleKernel, 1,
                    null, new long[]{dstSize}, null, 0, null, null);

            if (gpuImage == null
                    || (int) gpuImage.getWidth() != dstWidth
                    || (int) gpuImage.getHeight() != dstHeight) {
                gpuImage = new WritableImage(dstWidth, dstHeight);
            }

            if (dstMapReadOk) {
                // Blocking map waits for the kernel; on pinned memory this is a
                // pointer hand-back, and setPixels copies straight out of it.
                int[] err = new int[1];
                try {
                    mappedDst = clEnqueueMapBuffer(mapQueue, gpuDstBuffer, CL_TRUE,
                            CL_MAP_READ, 0, (long) Sizeof.cl_int * dstSize,
                            0, null, null, err);
                } catch (CLException clFailure) {
                    // See the upload path: with JOCL exceptions enabled this is the
                    // normal failure route, so preserve the real status code.
                    err[0] = clFailure.getStatus();
                } catch (Exception mapFailure) {
                    err[0] = CL_INVALID_OPERATION; // unknown cause: assume unsupported
                }
                if (mappedDst != null && err[0] != CL_SUCCESS) {
                    // Out-of-spec driver: pointer AND error. Unmap so the buffer
                    // doesn't stay host-owned for the rest of the context's life.
                    try {
                        clEnqueueUnmapMemObject(mapQueue, gpuDstBuffer, mappedDst,
                                0, null, null);
                    } catch (Exception ignored) { }
                    mappedDst = null;
                }
                if (mappedDst == null) {
                    latchDstMapUnsupported(err[0]);
                }
            }
            if (mappedDst != null) {
                gpuImage.getPixelWriter().setPixels(0, 0, dstWidth, dstHeight, PIXEL_FORMAT,
                        mappedDst.order(ByteOrder.nativeOrder()).asIntBuffer(), dstWidth);
            } else {
                // Fallback: blocking read into reusable scratch, then copy.
                if (dstScratch == null || dstScratch.length != dstSize) {
                    dstScratch = new int[dstSize];
                }
                clEnqueueReadBuffer(mapQueue, gpuDstBuffer, CL_TRUE, 0,
                        (long) Sizeof.cl_int * dstSize, Pointer.to(dstScratch), 0, null, null);
                gpuImage.getPixelWriter().setPixels(
                        0, 0, dstWidth, dstHeight, PIXEL_FORMAT, dstScratch, 0, dstWidth);
            }
            return true;

        } catch (Exception e) {
            Log.warn("ChunkyCL: GPU mapScale kernel failed; this frame will fall back to CPU", e);
            // The source buffer may be partially written; force a re-upload.
            lastUploadedVersion = Long.MIN_VALUE;
            return false;
        } finally {
            if (mappedDst != null) {
                try {
                    clEnqueueUnmapMemObject(mapQueue, gpuDstBuffer, mappedDst, 0, null, null);
                } catch (Exception ignored) {
                    // Queue may already be dead on a context swap; nothing to do.
                }
            }
        }
    }

    /**
     * Decide whether a failed buffer map means "this driver does not support
     * mapping" (latch the fallback permanently for this context) or was merely
     * transient (fall back for this frame only, and retry next frame).
     *
     * <p>The distinction matters because the map queue shares a device with the
     * render queue: a GPU watchdog reset triggered by a heavy render surfaces here
     * as CL_OUT_OF_RESOURCES, and latching on that would silently and permanently
     * downgrade the map path for a reason that has nothing to do with the map.
     */
    private static boolean isMapUnsupported(int err) {
        return err == CL_INVALID_OPERATION || err == CL_INVALID_VALUE
                || err == CL_MAP_FAILURE;
    }

    private void latchSrcMapUnsupported(int err) {
        if (isMapUnsupported(err)) {
            srcMapUploadOk = false;
            Log.info("ChunkyCL: this device does not support mapped uploads for the 2D "
                    + "map (error " + err + "); using plain buffer writes instead.");
        }
    }

    private void latchDstMapUnsupported(int err) {
        if (isMapUnsupported(err)) {
            dstMapReadOk = false;
            Log.info("ChunkyCL: this device does not support mapped readback for the 2D "
                    + "map (error " + err + "); using plain buffer reads instead.");
        }
    }

    /** Release all cached GPU resources (kernel, buffers, queue). */
    private void releaseGpuResources() {
        if (scaleKernel != null) { clReleaseKernel(scaleKernel); scaleKernel = null; }
        if (gpuSrcBuffer != null) { clReleaseMemObject(gpuSrcBuffer); gpuSrcBuffer = null; }
        if (gpuDstBuffer != null) { clReleaseMemObject(gpuDstBuffer); gpuDstBuffer = null; }
        if (mapQueue != null) { clReleaseCommandQueue(mapQueue); mapQueue = null; }
        dstScratch = null;
        lastSrcSize = 0;
        lastDstSize = 0;
        lastUploadedVersion = Long.MIN_VALUE;
        lastDrawnValid = false;
        // A new context may support mapping even if the old one didn't.
        srcMapUploadOk = true;
        dstMapReadOk = true;
    }

    /** Release GPU resources. */
    public void cleanup() {
        releaseGpuResources();
        mapCtx = null;
    }

    // ---- Reflection helpers ----

    /**
     * Extracts a captured instance of {@code targetClass} from a lambda / method-reference.
     * Java compiles {@code obj::method} into a synthetic class with a field holding {@code obj}.
     */
    @SuppressWarnings("unchecked")
    private static <T> T extractFromHandler(Object handler, Class<T> targetClass) {
        if (handler == null) return null;
        if (targetClass.isInstance(handler)) return (T) handler;
        for (Field f : handler.getClass().getDeclaredFields()) {
            try {
                f.setAccessible(true);
                Object val = f.get(handler);
                if (targetClass.isInstance(val)) return (T) val;
            } catch (Exception ignored) {}
        }
        return null;
    }
}
