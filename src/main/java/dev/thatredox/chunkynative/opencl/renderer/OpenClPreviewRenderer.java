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
    private volatile ClMemory cachedOutputBuffer = null;
    private volatile int cachedOutputSize = -1;

    // Reusable scratch for clSetKernelArg(int) calls. Avoids allocating a new
    // int[1] + Pointer per arg per render(). clSetKernelArg copies the value
    // synchronously so the same scratch can serve every int arg, written and
    // forgotten in sequence.
    private final int[] argIntScratch = new int[1];
    private final Pointer argIntScratchPtr = Pointer.to(argIntScratch);

    private void setIntKernelArg(cl_kernel kernel, int idx, int value) {
        argIntScratch[0] = value;
        clSetKernelArg(kernel, idx, Sizeof.cl_int, argIntScratchPtr);
    }

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
            // Context change invalidates all GPU resources
            if (cachedCanvasConfig != null) { cachedCanvasConfig.close(); cachedCanvasConfig = null; }
            prevCanvasConfig = null;
            if (cachedOutputBuffer != null) { cachedOutputBuffer.close(); cachedOutputBuffer = null; }
            cachedOutputSize = -1;
        }
        return cachedKernel;
    }

    private ClMemory getOutputBuffer(int size, ContextManager context) {
        if (cachedOutputSize != size) {
            if (cachedOutputBuffer != null) cachedOutputBuffer.close();
            cachedOutputBuffer = new ClMemory(clCreateBuffer(context.context.context, CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_int * size, null, null));
            cachedOutputSize = size;
        }
        return cachedOutputBuffer;
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

        Scene scene = manager.bufferedScene;
        int[] imageData = scene.getBackBuffer().data;

        sceneLock.lock();
        try {
            // Ensure the scene is loaded
            sceneLoader.ensureLoad(manager.bufferedScene);

            // Get cached or create kernel and canvas config
            cl_kernel kernel = getPreviewKernel(context);
            ClIntBuffer clCanvasConfig = getCanvasConfig(scene, context);

            ClCamera camera = new ClCamera(scene, context.context, false);
            ClMemory buffer = getOutputBuffer(imageData.length, context);

            try (ClCamera ignored1 = camera) {

                // Generate the camera rays
                camera.generate(null, false);

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
                setIntKernelArg(kernel, argIndex++, sceneLoader.getChunkBitmapSize());
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeData().get()));
                setIntKernelArg(kernel, argIndex++, sceneLoader.getBiomeDataSize());
                setIntKernelArg(kernel, argIndex++, sceneLoader.getBiomeYLevels());
                // Cloud config (renderConfig) + cloud bitmap so the preview can render clouds.
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getRenderConfig().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getCloudData().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(buffer.get()));

                try {
                    clEnqueueNDRangeKernel(context.context.queue, kernel, 1, null,
                            new long[]{imageData.length}, null, 0, null, null);

                    // No event needed: the queue is in-order, so this blocking
                    // read already waits for the kernel enqueued above.
                    clEnqueueReadBuffer(context.context.queue, buffer.get(), CL_TRUE, 0,
                            (long) Sizeof.cl_int * imageData.length, Pointer.to(imageData),
                            0, null, null);
                } catch (CLException e) {
                    se.llbit.log.Log.warn("ChunkyCL: Preview kernel error: " + e.getMessage());
                    return;
                }

                manager.redrawScreen();
                postRender.getAsBoolean();
            }
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
