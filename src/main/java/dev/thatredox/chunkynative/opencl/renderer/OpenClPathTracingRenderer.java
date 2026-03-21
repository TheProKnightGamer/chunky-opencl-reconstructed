package dev.thatredox.chunkynative.opencl.renderer;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.renderer.scene.*;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.*;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.util.TaskTracker;
import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

public class OpenClPathTracingRenderer implements Renderer {
    private BooleanSupplier postRender = () -> true;

    @Override
    public String getId() {
        return "ChunkyClRenderer";
    }

    @Override
    public String getName() {
        return "ChunkyClRenderer";
    }

    @Override
    public String getDescription() {
        return "ChunkyClRenderer";
    }

    @Override
    public void setPostRender(BooleanSupplier callback) {
        postRender = callback;
    }

    @Override
    public void render(DefaultRenderManager manager) throws InterruptedException {
        ContextManager context = ContextManager.get();
        ClSceneLoader sceneLoader = context.sceneLoader;
        ReentrantLock renderLock = new ReentrantLock();
        Scene scene = manager.bufferedScene;
        double[] sampleBuffer = scene.getSampleBuffer();
        int pixelCount = sampleBuffer.length / 3;
        float[] passBuffer = new float[pixelCount * 4];
        sceneLoader.ensureLoad(manager.bufferedScene);
        cl_kernel kernel = clCreateKernel(context.renderer.kernel, "render", null);
        ClCamera camera = new ClCamera(scene, context.context);
        ClMemory buffer = new ClMemory(clCreateBuffer(context.context.context, CL_MEM_WRITE_ONLY,
                (long) Sizeof.cl_float * passBuffer.length, null, null));
        ClMemory dynamicConfig = new ClMemory(
                clCreateBuffer(context.context.context, CL_MEM_READ_ONLY, Sizeof.cl_int * 5, null, null));
        ClMemory emitterIntensityMem = new ClMemory(
                clCreateBuffer(context.context.context, CL_MEM_READ_ONLY, Sizeof.cl_float, null, null));
        ClIntBuffer clCanvasConfig = new ClIntBuffer(new int[]{
                scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight(),
                scene.canvasConfig.getCropWidth(), scene.canvasConfig.getCropHeight(),
                scene.canvasConfig.getCropX(), scene.canvasConfig.getCropY()
        }, context.context);
        ClIntBuffer clRayDepth = new ClIntBuffer(scene.getRayDepth(), context.context);
        ClIntBuffer emptyGridPlaceholder = new ClIntBuffer(new int[]{0, 0, 0, 0, 0, 0, 0}, context.context);
        try (ClCamera camRes = camera;
             ClMemory bufRes = buffer;
             ClMemory dynamicCfgRes = dynamicConfig;
             ClMemory emitterIntensityRes = emitterIntensityMem;
             ClIntBuffer canvasCfgRes = clCanvasConfig;
             ClIntBuffer rayDepthRes = clRayDepth;
             ClIntBuffer emptyGridRes = emptyGridPlaceholder) {
            camera.generate(renderLock, true);
            int argIndex = 0;
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.projectorType.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.cameraSettings.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(
                    camera.apertureMaskBuffer != null ? camera.apertureMaskBuffer.get() : emptyGridPlaceholder.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[]{camera.apertureMaskWidth}));
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
            int matCacheWords = Math.min(2048, sceneLoader.getMaterialPalette().wordCount());
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[]{matCacheWords}));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_uint * Math.max(matCacheWords, 1), null);
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSky().skyTexture.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSky().skyIntensity.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSun().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(dynamicConfig.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(emitterIntensityMem.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(
                    sceneLoader.getEmitterPositions() != null ? sceneLoader.getEmitterPositions().get() : emptyGridPlaceholder.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(
                    sceneLoader.getPositionIndexes() != null ? sceneLoader.getPositionIndexes().get() : emptyGridPlaceholder.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(
                    sceneLoader.getConstructedGrid() != null ? sceneLoader.getConstructedGrid().get() : emptyGridPlaceholder.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(
                    sceneLoader.getGridConfig() != null ? sceneLoader.getGridConfig().get() : emptyGridPlaceholder.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(clCanvasConfig.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(clRayDepth.get()));
            int iterationsArgIndex = argIndex;
            int iterationsPerLaunch = 1;
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[]{iterationsPerLaunch}));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getFogData().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterConfig().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getRenderConfig().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getCloudData().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterNormalMap().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[]{sceneLoader.getWaterNormalMapWidth()}));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeData().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[]{sceneLoader.getBiomeDataSize()}));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getChunkBitmap().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[]{sceneLoader.getChunkBitmapSize()}));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(buffer.get()));
            int[] cfg = new int[5];
            float[] emitterIntensityArr = new float[1];
            float lastEmitterIntensity = Float.NaN;
            Random rand = new Random(0);

            // --- Calibration: measure GPU speed to auto-tune iterationsPerLaunch ---
            cfg[0] = rand.nextInt();
            cfg[1] = 0;
            cfg[2] = scene.getEmittersEnabled() ? 1 : 0;
            cfg[3] = scene.getEmitterSamplingStrategy().ordinal();
            cfg[4] = Math.max(1, scene.getCurrentBranchCount());
            clEnqueueWriteBuffer(context.context.queue, dynamicConfig.get(), CL_TRUE, 0,
                    Sizeof.cl_int * cfg.length, Pointer.to(cfg), 0, null, null);
            emitterIntensityArr[0] = (float) scene.getEmitterIntensity();
            lastEmitterIntensity = emitterIntensityArr[0];
            clEnqueueWriteBuffer(context.context.queue, emitterIntensityMem.get(), CL_TRUE, 0,
                    Sizeof.cl_float, Pointer.to(emitterIntensityArr), 0, null, null);

            int calSize = Math.min(pixelCount, Math.max(4096, pixelCount / 4));
            long calStart = System.nanoTime();
            clEnqueueNDRangeKernel(context.context.queue, kernel, 1,
                    null, new long[]{calSize}, null, 0, null, null);
            clFinish(context.context.queue);
            double calMs = (System.nanoTime() - calStart) / 1e6;
            double fullFrameMs = (calMs / calSize) * pixelCount;

            // Target ~800ms per dispatch (leaves headroom below 2s TDR)
            if (fullFrameMs > 0 && fullFrameMs < 800.0) {
                iterationsPerLaunch = Math.max(1, Math.min(100, (int) (800.0 / fullFrameMs)));
            }
            clSetKernelArg(kernel, iterationsArgIndex, Sizeof.cl_int,
                    Pointer.to(new int[]{iterationsPerLaunch}));
            System.err.printf("[ChunkyCL] Cal: %.1fms/%dpx, frame=%.0fms, ipl=%d%n",
                    calMs, calSize, fullFrameMs, iterationsPerLaunch);

            int sppSinceRedraw = 0;
            while (scene.spp < scene.getTargetSpp()) {
                renderLock.lock();
                try {
                    cfg[0] = rand.nextInt();
                    cfg[1] = 0;
                    cfg[2] = scene.getEmittersEnabled() ? 1 : 0;
                    cfg[3] = scene.getEmitterSamplingStrategy().ordinal();
                    cfg[4] = scene.getCurrentBranchCount();
                    // Blocking write: Java arrays are non-direct buffers
                    clEnqueueWriteBuffer(context.context.queue, dynamicConfig.get(), CL_TRUE, 0,
                            Sizeof.cl_int * cfg.length, Pointer.to(cfg), 0, null, null);
                    float curEmitterIntensity = (float) scene.getEmitterIntensity();
                    if (curEmitterIntensity != lastEmitterIntensity) {
                        lastEmitterIntensity = curEmitterIntensity;
                        emitterIntensityArr[0] = curEmitterIntensity;
                        clEnqueueWriteBuffer(context.context.queue, emitterIntensityMem.get(), CL_TRUE, 0,
                                Sizeof.cl_float, Pointer.to(emitterIntensityArr), 0, null, null);
                    }
                    // Single dispatch for all pixels (no batching overhead)
                    clEnqueueNDRangeKernel(context.context.queue, kernel, 1,
                            null, new long[]{pixelCount}, null, 0, null, null);
                } finally {
                    renderLock.unlock();
                }
                // Blocking read implicitly waits for kernel completion (no need for clFinish)
                clEnqueueReadBuffer(context.context.queue, buffer.get(), CL_TRUE, 0,
                        (long) Sizeof.cl_float * passBuffer.length, Pointer.to(passBuffer), 0, null, null);
                int sppBefore = scene.spp;
                int passSpp = iterationsPerLaunch * scene.getCurrentBranchCount();
                // Precompute blend weights: old * prevWeight + new * passWeight
                double prevWeight = (double) sppBefore / (sppBefore + passSpp);
                double passWeight = (double) passSpp / (sppBefore + passSpp);
                for (int p = 0; p < pixelCount; p++) {
                    int sampleIdx = p * 3;
                    int passIdx = p * 4;
                    sampleBuffer[sampleIdx] = sampleBuffer[sampleIdx] * prevWeight + passBuffer[passIdx] * passWeight;
                    sampleBuffer[sampleIdx + 1] = sampleBuffer[sampleIdx + 1] * prevWeight + passBuffer[passIdx + 1] * passWeight;
                    sampleBuffer[sampleIdx + 2] = sampleBuffer[sampleIdx + 2] * prevWeight + passBuffer[passIdx + 2] * passWeight;
                }
                scene.spp = sppBefore + passSpp;
                sppSinceRedraw += passSpp;
                // Refresh screen every ~10 SPP (matching CPU update frequency)
                if (sppSinceRedraw >= 10) {
                    scene.postProcessFrame(TaskTracker.Task.NONE);
                    manager.redrawScreen();
                    sppSinceRedraw = 0;
                }
                if (camera.needGenerate) {
                    renderLock.lock();
                    try {
                        camera.generate(renderLock, true);
                    } finally {
                        renderLock.unlock();
                    }
                }
                if (postRender.getAsBoolean()) {
                    break;
                }
            }
            // Flush final frame if there are undrawn samples
            if (sppSinceRedraw > 0) {
                scene.postProcessFrame(TaskTracker.Task.NONE);
                manager.redrawScreen();
            }
            if (scene.spp >= scene.getTargetSpp()) {
                scene.spp = scene.getTargetSpp() + 1;
            }
            postRender.getAsBoolean();
        } finally {
            clReleaseKernel(kernel);
        }
    }

    private boolean isSaveEvent(SnapshotControl control, Scene scene, int spp) {
        return control.saveSnapshot(scene, spp) || control.saveRenderDump(scene, spp);
    }

    @Override
    public boolean autoPostProcess() {
        return false;
    }

    @Override
    public void sceneReset(DefaultRenderManager manager, ResetReason reason, int resetCount) {
        ContextManager.get().sceneLoader.load(resetCount, reason, manager.bufferedScene);
    }
}