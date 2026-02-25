package dev.thatredox.chunkynative.opencl.renderer;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.renderer.scene.*;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.*;

import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.util.TaskTracker;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ForkJoinTask;
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
        cl_event[] renderEvent = new cl_event[1];
        Scene scene = manager.bufferedScene;

        double[] sampleBuffer = scene.getSampleBuffer();
        int pixelCount = sampleBuffer.length / 3;
        float[] passBuffer = new float[pixelCount * 4]; // RGBA from GPU kernel

        // Ensure the scene is loaded
        sceneLoader.ensureLoad(manager.bufferedScene);
        {
            // Load the kernel
            cl_kernel kernel = clCreateKernel(context.renderer.kernel, "render", null);

            // Generate the camera
            ClCamera camera = new ClCamera(scene, context.context);
            // Buffer will contain the per-launch averaged color for `iterations` samples.
            // Kernel writes a float3 per pixel (RGB average) into this buffer.
            ClMemory buffer = new ClMemory(clCreateBuffer(context.context.context, CL_MEM_WRITE_ONLY,
                (long) Sizeof.cl_float * passBuffer.length, null, null)); // 4 floats per pixel (RGBA)
            ClMemory dynamicConfig = new ClMemory(
                clCreateBuffer(context.context.context, CL_MEM_READ_ONLY, Sizeof.cl_int * 5, null, null));
            ClMemory emitterIntensityMem = new ClMemory(
                clCreateBuffer(context.context.context, CL_MEM_READ_ONLY, Sizeof.cl_float, null, null));
            ClIntBuffer clCanvasConfig = new ClIntBuffer(new int[] {
                    scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight(),
                    scene.canvasConfig.getCropWidth(), scene.canvasConfig.getCropHeight(),
                    scene.canvasConfig.getCropX(), scene.canvasConfig.getCropY()
            }, context.context);
            ClIntBuffer clRayDepth = new ClIntBuffer(scene.getRayDepth(), context.context);
            ClIntBuffer emptyGridPlaceholder = new ClIntBuffer(new int[] {0,0,0,0,0,0,0}, context.context);

            try (ClCamera camRes = camera;
                 ClMemory bufRes = buffer;
                 ClMemory dynamicCfgRes = dynamicConfig;
                 ClMemory emitterIntensityRes = emitterIntensityMem;
                 ClIntBuffer canvasCfgRes = clCanvasConfig;
                 ClIntBuffer rayDepthRes = clRayDepth;
                 ClIntBuffer emptyGridRes = emptyGridPlaceholder) {

                // Generate initial camera rays
                camera.generate(renderLock, true);

                // Set kernel args that do not change per-iteration
                int argIndex = 0;
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.projectorType.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.cameraSettings.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.apertureMaskBuffer != null ? camera.apertureMaskBuffer.get() : emptyGridPlaceholder.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[] { camera.apertureMaskWidth }));

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
                // Per-work-group material cache: cache the first N ints of the palette
                // into __local memory for faster access.  Cap to the actual palette
                // size so the cooperative copy never reads out-of-bounds.
                final int matCacheWords = Math.min(2048, sceneLoader.getMaterialPalette().wordCount());
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[] { matCacheWords }));
                // Allocate local memory for the cache (size = matCacheWords * sizeof(uint))
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_uint * Math.max(matCacheWords, 1), null);

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSky().skyTexture.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSky().skyIntensity.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getSun().get()));

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(dynamicConfig.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(emitterIntensityMem.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getEmitterPositions() != null ? sceneLoader.getEmitterPositions().get() : emptyGridPlaceholder.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getPositionIndexes() != null ? sceneLoader.getPositionIndexes().get() : emptyGridPlaceholder.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getConstructedGrid() != null ? sceneLoader.getConstructedGrid().get() : emptyGridPlaceholder.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getGridConfig() != null ? sceneLoader.getGridConfig().get() : emptyGridPlaceholder.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(clCanvasConfig.get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(clRayDepth.get()));
                // Iterations per kernel launch. Lower values reduce GPU execution time
                // per dispatch, avoiding Windows TDR (Timeout Detection and Recovery)
                // which kills kernels running longer than ~2 seconds. The host loop
                // compensates by dispatching more frequently.
                final int iterationsPerLaunch = 1;
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[] { iterationsPerLaunch }));

                // New buffers: fog, water, render config, cloud data, water normal map
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getFogData().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterConfig().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getRenderConfig().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getCloudData().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterNormalMap().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[] { sceneLoader.getWaterNormalMapWidth() }));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeData().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[] { sceneLoader.getBiomeDataSize() }));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getChunkBitmap().get()));
                clSetKernelArg(kernel, argIndex++, Sizeof.cl_int, Pointer.to(new int[] { sceneLoader.getChunkBitmapSize() }));

                clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(buffer.get()));

                // Warm-up dispatch: force the GPU driver to compile the kernel to
                // native ISA before the real render loop.  Many drivers (especially
                // NVIDIA / AMD on Windows) defer final code-generation until the
                // first clEnqueueNDRangeKernel.  For a complex kernel like this one,
                // lazy compilation can take several seconds.  If that compilation
                // happens on a large dispatch (millions of pixels), the combined
                // compile + execute time exceeds the Windows TDR timeout (~2 s) and
                // the driver kills the kernel with error -777.  A single-pixel
                // warm-up separates compilation time from execution time so
                // subsequent full-size dispatches complete well within TDR.
                {
                    int[] warmupCfg = new int[] { 0, 0, 0, 0, 1 }; // seed=0, spp=0, emitters=off, strategy=0, branches=1
                    clEnqueueWriteBuffer(context.context.queue, dynamicConfig.get(), CL_TRUE, 0,
                        Sizeof.cl_int * warmupCfg.length, Pointer.to(warmupCfg), 0, null, null);
                    clEnqueueWriteBuffer(context.context.queue, emitterIntensityMem.get(), CL_TRUE, 0,
                        Sizeof.cl_float, Pointer.to(new float[]{0.0f}), 0, null, null);
                    cl_event warmupEvent = new cl_event();
                    clEnqueueNDRangeKernel(context.context.queue, kernel, 1, null,
                        new long[]{1}, new long[]{1}, 0, null, warmupEvent);
                    clWaitForEvents(1, new cl_event[] { warmupEvent });
                    clReleaseEvent(warmupEvent);
                }

                int bufferSppReal = 0;
                int logicalSpp = scene.spp;
                final int[] sceneSpp = {scene.spp};
                long lastCallback = 0;

                Random rand = new Random(0);

                ForkJoinTask<?> cameraGenTask = Chunky.getCommonThreads().submit(() -> 0);
                ForkJoinTask<?> bufferMergeTask = Chunky.getCommonThreads().submit(() -> 0);

                // This is the main rendering loop. This deals with dispatching rendering tasks. The majority of time is spent
                // waiting for the OpenCL renderer to complete.
                while (logicalSpp < scene.getTargetSpp()) {
                    renderLock.lock();
                    renderEvent[0] = new cl_event();

                    int currentBranchCount = scene.getCurrentBranchCount();
                    int[] cfg = new int[] { rand.nextInt(), bufferSppReal, scene.getEmittersEnabled() ? 1 : 0, scene.getEmitterSamplingStrategy().ordinal(), currentBranchCount };
                    try {
                        clEnqueueWriteBuffer(context.context.queue, dynamicConfig.get(), CL_TRUE, 0,
                            Sizeof.cl_int * cfg.length, Pointer.to(cfg), 0, null, null);
                        clEnqueueWriteBuffer(context.context.queue, emitterIntensityMem.get(), CL_TRUE, 0, Sizeof.cl_float,
                            Pointer.to(new float[]{(float) scene.getEmitterIntensity()}), 0, null, null);
                        clEnqueueNDRangeKernel(context.context.queue, kernel, 1, null,
                            new long[]{pixelCount}, null, 0, null,
                            renderEvent[0]);
                        clWaitForEvents(1, renderEvent);
                        clReleaseEvent(renderEvent[0]);
                    } catch (org.jocl.CLException e) {
                        // Kernel execution failed — query detailed status and break out
                        try {
                            int[] execStatus = new int[1];
                            clGetEventInfo(renderEvent[0], CL_EVENT_COMMAND_EXECUTION_STATUS,
                                Sizeof.cl_int, Pointer.to(execStatus), null);
                            System.err.println("[ChunkyOCL] Kernel execution status code: " + execStatus[0]);
                        } catch (Exception ignored) {}
                        try { clReleaseEvent(renderEvent[0]); } catch (Exception ignored) {}
                        renderLock.unlock();
                        System.err.println("[ChunkyOCL] OpenCL kernel error: " + e.getMessage());
                        System.err.println("[ChunkyOCL] This may indicate a GPU timeout (TDR), driver bug, or out-of-bounds access.");
                        System.err.println("[ChunkyOCL] Try: (1) updating GPU drivers, (2) reducing scene complexity, (3) clearing the OpenCL cache at "
                            + context.context.getCacheDir());
                        e.printStackTrace();
                        break;
                    }
                    renderLock.unlock();
                    // Each kernel call contributed iterationsPerLaunch * branchCount samples per pixel
                    bufferSppReal += iterationsPerLaunch * currentBranchCount;

                    if (camera.needGenerate && cameraGenTask.isDone()) {
                        cameraGenTask = Chunky.getCommonThreads().submit(() -> camera.generate(renderLock, true));
                    }

                    boolean saveEvent = isSaveEvent(manager.getSnapshotControl(), scene, logicalSpp + bufferSppReal);

                    // Decide whether to merge and report progress this iteration.
                    // We merge when: (a) the previous merge is done and we have enough
                    // samples, (b) the canvas needs every frame finalized, or (c) a
                    // save/snapshot is due.
                    boolean readyToMerge = bufferMergeTask.isDone() || saveEvent;
                    if (!readyToMerge) continue;

                    // If the buffer is not being displayed and this isn't a save,
                    // throttle callbacks and require accumulated SPP before doing a
                    // merge (to amortize readback cost). Use an adaptive threshold:
                    // start low for instant visual feedback, ramp up as SPP grows.
                    if (!scene.shouldFinalizeBuffer() && !saveEvent) {
                        int sppPerLaunch = iterationsPerLaunch * currentBranchCount;
                        int mergeThreshold = Math.min(1024, Math.max(sppPerLaunch, sceneSpp[0]));
                        if (bufferSppReal < mergeThreshold) {
                            // Not enough samples for a merge yet; do a periodic
                            // progress-only callback so the UI timer stays updated.
                            long time = System.currentTimeMillis();
                            if (time - lastCallback > 100 && !manager.shouldFinalize()) {
                                lastCallback = time;
                                if (postRender.getAsBoolean()) break;
                            }
                            continue;
                        }
                    }

                    // Wait for the previous merge to finish before starting new work.
                    bufferMergeTask.join();

                    // For non-save events, issue a callback now (with old scene.spp)
                    // to let DefaultRenderManager accumulate render time and check
                    // for state changes. For save events, skip this callback and
                    // only issue one after the merge updates scene.spp, so the
                    // frameCompletionListener sees the correct SPP for the save check.
                    if (!saveEvent) {
                        if (postRender.getAsBoolean()) break;
                    }

                    clEnqueueReadBuffer(context.context.queue, buffer.get(), CL_TRUE, 0,
                            (long) Sizeof.cl_float * passBuffer.length, Pointer.to(passBuffer),
                            0, null, null);
                    int sampSpp = sceneSpp[0];
                    int passSpp = bufferSppReal;
                    double sinv = 1.0 / (sampSpp + passSpp);
                    bufferSppReal = 0;

                    bufferMergeTask = Chunky.getCommonThreads().submit(() -> {
                        // Merge GPU RGBA (4 floats/pixel) into CPU RGB (3 floats/pixel) sampleBuffer.
                        // Alpha is not accumulated into sampleBuffer (CPU computes alpha
                        // separately via AlphaBuffer at export time, matching CPU behavior).
                        int pc = sampleBuffer.length / 3;
                        Arrays.parallelSetAll(sampleBuffer, i -> {
                            int pixel = i / 3;
                            int channel = i % 3;
                            int gpuIdx = pixel * 4 + channel; // skip alpha (channel 3)
                            return (sampleBuffer[i] * sampSpp + passBuffer[gpuIdx] * passSpp) * sinv;
                        });
                        sceneSpp[0] += passSpp;
                        scene.spp = sceneSpp[0];
                        scene.postProcessFrame(TaskTracker.Task.NONE);
                        manager.redrawScreen();
                    });
                    logicalSpp += passSpp;

                    // For save events, wait for merge to complete (so scene.spp is
                    // updated) then issue the callback so the frameCompletion
                    // listener sees the correct SPP and can trigger the save.
                    if (saveEvent) {
                        bufferMergeTask.join();
                        if (postRender.getAsBoolean()) break;
                    }
                }

                cameraGenTask.join();
                bufferMergeTask.join();

                // Ensure the framework properly handles render completion.
                // The CPU renderer naturally exceeds the target SPP by one frame,
                // which triggers the renderCompletionListener (it checks spp > target).
                // The GPU renderer may hit the target exactly, so we bump spp by 1
                // to match CPU behavior, then issue a final postRender callback.
                // This ensures: (a) the final time slice is accumulated into renderTime,
                // (b) frameCompletionListener fires with the correct SPP for dumps/saves,
                // (c) renderCompletionListener fires and pauses the render properly.
                if (scene.spp >= scene.getTargetSpp()) {
                    scene.spp = scene.getTargetSpp() + 1;
                }
                postRender.getAsBoolean();
            }

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
