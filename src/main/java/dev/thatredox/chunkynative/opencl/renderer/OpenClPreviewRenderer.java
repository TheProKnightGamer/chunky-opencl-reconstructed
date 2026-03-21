package dev.thatredox.chunkynative.opencl.renderer;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.renderer.scene.*;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.*;
import se.llbit.chunky.renderer.DefaultRenderManager;
import se.llbit.chunky.renderer.Renderer;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Scene;

import java.util.Arrays;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

import static org.jocl.CL.*;

public class OpenClPreviewRenderer implements Renderer {
    private BooleanSupplier postRender = () -> true;

    // Lock to synchronize render() and sceneReset() so GPU buffers are not
    // freed/recreated while a kernel is still referencing them.
    private final ReentrantLock sceneLock = new ReentrantLock();

    // Cached kernel and canvas config to avoid per-frame allocation overhead.
    // Volatile for visibility across threads (sceneLock guards actual access).
    private volatile cl_kernel cachedKernel = null;
    private volatile ContextManager cachedContext = null;
    private volatile int[] prevCanvasConfig = null;
    private volatile ClIntBuffer cachedCanvasConfig = null;

    @Override
    public String getId() {
        return "ChunkyClPreviewRenderer";
    }

    @Override
    public String getName() {
        return "Chunky CL Preview Renderer";
    }

    @Override
    public String getDescription() {
        return "A work in progress OpenCL renderer.";
    }

    @Override
    public void setPostRender(BooleanSupplier callback) {
        postRender = callback;
    }

    private cl_kernel getPreviewKernel(ContextManager context) {
        if (cachedKernel == null || cachedContext != context) {
            if (cachedKernel != null) clReleaseKernel(cachedKernel);
            cachedKernel = clCreateKernel(context.renderer.kernel, "preview", null);
            cachedContext = context;
        }
        return cachedKernel;
    }

    private ClIntBuffer getCanvasConfig(Scene scene, ContextManager context) {
        int[] config = new int[] {
            scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight(),
            scene.canvasConfig.getCropWidth(), scene.canvasConfig.getCropHeight(),
            scene.canvasConfig.getCropX(), scene.canvasConfig.getCropY()
        };
        if (!Arrays.equals(config, prevCanvasConfig)) {
            if (cachedCanvasConfig != null) cachedCanvasConfig.close();
            cachedCanvasConfig = new ClIntBuffer(config, context.context);
            prevCanvasConfig = config;
        }
        return cachedCanvasConfig;
    }

    @Override
    public void render(DefaultRenderManager manager) throws InterruptedException {
        ContextManager context = ContextManager.get();
        ClSceneLoader sceneLoader = context.sceneLoader;

        cl_event[] renderEvent = new cl_event[1];
        Scene scene = manager.bufferedScene;
        int[] imageData = scene.getBackBuffer().data;

        sceneLock.lock();
        try {
            // Ensure the scene is loaded
            sceneLoader.ensureLoad(manager.bufferedScene);

            // Get cached or create kernel and canvas config
            cl_kernel kernel = getPreviewKernel(context);
            ClIntBuffer clCanvasConfig = getCanvasConfig(scene, context);

            ClCamera camera = new ClCamera(scene, context.context);
            ClMemory buffer = new ClMemory(clCreateBuffer(context.context.context, CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_int * imageData.length, null, null));

            try (ClCamera ignored1 = camera;
                 ClMemory ignored2 = buffer) {

                // Generate the camera rays
                camera.generate(null, false);

                renderEvent[0] = new cl_event();

                int argIndex = 0;
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.projectorType.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.cameraSettings.get()));

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getOctreeDepth().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getOctreeData().get()));

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterOctreeDepth().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterOctreeData().get()));

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBlockPalette().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getQuadPalette().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getAabbPalette().get()));

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWorldBvh().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getActorBvh().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getTrigPalette().get()));

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getTexturePalette().getAtlas()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getMaterialPalette().get()));

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSky().skyTexture.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSky().skyIntensity.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSun().get()));

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(clCanvasConfig.get()));
                // New: water config for preview
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterConfig().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getChunkBitmap().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[] { sceneLoader.getChunkBitmapSize() }));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeData().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[] { sceneLoader.getBiomeDataSize() }));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[] { sceneLoader.getBiomeYLevels() }));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(buffer.get()));

                try {
                    clEnqueueNDRangeKernel(context.context.queue, kernel, 1, null,
                            new long[]{imageData.length}, null, 0, null,
                            renderEvent[0]);

                    clEnqueueReadBuffer(context.context.queue, buffer.get(), CL_TRUE, 0,
                            (long) Sizeof.cl_int * imageData.length, Pointer.to(imageData),
                            1, renderEvent, null);
                } catch (CLException e) {
                    System.err.println("[ChunkyCL] Preview kernel error: " + e.getMessage());
                    clReleaseEvent(renderEvent[0]);
                    return;
                }

                manager.redrawScreen();
                postRender.getAsBoolean();
            }

            clReleaseEvent(renderEvent[0]);
        } finally {
            if (sceneLock.isHeldByCurrentThread()) {
                sceneLock.unlock();
            }
        }
    }

    @Override
    public boolean autoPostProcess() {
        return false;
    }

    @Override
    public void sceneReset(DefaultRenderManager manager, ResetReason reason, int resetCount) {
        sceneLock.lock();
        try {
            ContextManager.get().sceneLoader.load(resetCount, reason, manager.bufferedScene);
        } finally {
            sceneLock.unlock();
        }
    }
}
