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

    // GPU resources (lazily created, reused across frames)
    private cl_kernel scaleKernel;
    private cl_mem gpuSrcBuffer;
    private cl_mem gpuDstBuffer;
    private int lastSrcSize;
    private int lastDstSize;
    private WritableImage gpuImage;

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
                    Log.warn("ChunkyCL: Failed to install GPU map renderer", e);
                    System.err.println("[ChunkyCL] GPU map install failed: " + e);
                    e.printStackTrace();
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

        // 4. Prepare reflection handles for MapBuffer's private state
        mbPixelsField  = MapBuffer.class.getDeclaredField("pixels");
        mbWidthField   = MapBuffer.class.getDeclaredField("width");
        mbHeightField  = MapBuffer.class.getDeclaredField("height");
        mbViewField    = MapBuffer.class.getDeclaredField("view");
        mbCachedField  = MapBuffer.class.getDeclaredField("cached");
        mbImageField   = MapBuffer.class.getDeclaredField("image");
        mbPixelsField.setAccessible(true);
        mbWidthField.setAccessible(true);
        mbHeightField.setAccessible(true);
        mbViewField.setAccessible(true);
        mbCachedField.setAccessible(true);
        mbImageField.setAccessible(true);

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
        System.err.println("[ChunkyCL] GPU map renderer installed");
    }

    // ---- GPU-accelerated drawBuffered ----

    /**
     * GPU version of {@link MapBuffer#drawBuffered(GraphicsContext)}.
     * Reads the delegate's private pixel buffer via reflection, performs
     * nearest-neighbour upscaling on the GPU, and draws the result.
     *
     * @return true on success, false to signal the caller to fall back to CPU
     */
    boolean gpuDrawBuffered(GraphicsContext gc, MapBuffer delegate) {
        if (!installed) return false;

        try {
            synchronized (delegate) {
                ChunkView view = (ChunkView) mbViewField.get(delegate);
                if (view.width <= 0 || view.height <= 0) return true; // nothing to draw

                boolean cached = mbCachedField.getBoolean(delegate);
                if (cached) {
                    // The image is already up to date – just blit it
                    WritableImage img = (WritableImage) mbImageField.get(delegate);
                    if (img != null) {
                        gc.clearRect(0, 0, view.width, view.height);
                        gc.drawImage(img, 0, 0);
                        return true;
                    }
                }

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

                int[] scaled = gpuScale(pixels, srcWidth, srcHeight,
                        dstWidth, dstHeight, scale, srcOffsetX, srcOffsetZ);
                if (scaled == null) return false; // fall back to CPU

                // Write result to a WritableImage
                if (gpuImage == null
                        || (int) gpuImage.getWidth() != dstWidth
                        || (int) gpuImage.getHeight() != dstHeight) {
                    gpuImage = new WritableImage(dstWidth, dstHeight);
                }
                gpuImage.getPixelWriter().setPixels(
                        0, 0, dstWidth, dstHeight, PIXEL_FORMAT, scaled, 0, dstWidth);

                // Store back so MapBuffer considers itself cached
                mbImageField.set(delegate, gpuImage);
                mbCachedField.setBoolean(delegate, true);

                gc.clearRect(0, 0, dstWidth, dstHeight);
                gc.drawImage(gpuImage, 0, 0);
                return true;
            }
        } catch (Exception e) {
            System.err.println("[ChunkyCL] gpuDrawBuffered error: " + e.getMessage());
            return false;
        }
    }

    // ---- OpenCL nearest-neighbour scaling ----

    private int[] gpuScale(int[] src, int srcWidth, int srcHeight,
                           int dstWidth, int dstHeight, float scale,
                           int srcOffsetX, int srcOffsetZ) {
        try {
            ContextManager ctx = ContextManager.get();
            int srcSize = src.length;
            int dstSize = dstWidth * dstHeight;
            if (dstSize <= 0) return null;

            // (Re-)allocate GPU buffers when sizes change
            if (gpuSrcBuffer == null || srcSize != lastSrcSize) {
                if (gpuSrcBuffer != null) clReleaseMemObject(gpuSrcBuffer);
                gpuSrcBuffer = clCreateBuffer(ctx.context.context, CL_MEM_READ_ONLY,
                        (long) Sizeof.cl_int * srcSize, null, null);
                lastSrcSize = srcSize;
            }
            if (gpuDstBuffer == null || dstSize != lastDstSize) {
                if (gpuDstBuffer != null) clReleaseMemObject(gpuDstBuffer);
                gpuDstBuffer = clCreateBuffer(ctx.context.context, CL_MEM_WRITE_ONLY,
                        (long) Sizeof.cl_int * dstSize, null, null);
                lastDstSize = dstSize;
            }

            // Upload source pixels (blocking – Java arrays are non-direct buffers)
            clEnqueueWriteBuffer(ctx.context.queue, gpuSrcBuffer, CL_TRUE, 0,
                    (long) Sizeof.cl_int * srcSize, Pointer.to(src), 0, null, null);

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

            clEnqueueNDRangeKernel(ctx.context.queue, scaleKernel, 1,
                    null, new long[]{dstSize}, null, 0, null, null);

            // Blocking read – waits for kernel to finish
            int[] dst = new int[dstSize];
            clEnqueueReadBuffer(ctx.context.queue, gpuDstBuffer, CL_TRUE, 0,
                    (long) Sizeof.cl_int * dstSize, Pointer.to(dst), 0, null, null);
            return dst;

        } catch (Exception e) {
            System.err.println("[ChunkyCL] GPU mapScale failed: " + e.getMessage());
            return null;
        }
    }

    /** Release GPU resources. */
    public void cleanup() {
        if (scaleKernel != null) { clReleaseKernel(scaleKernel); scaleKernel = null; }
        if (gpuSrcBuffer != null) { clReleaseMemObject(gpuSrcBuffer); gpuSrcBuffer = null; }
        if (gpuDstBuffer != null) { clReleaseMemObject(gpuDstBuffer); gpuDstBuffer = null; }
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
