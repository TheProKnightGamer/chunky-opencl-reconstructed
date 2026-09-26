package dev.thatredox.chunkynative.opencl.renderer;

import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.context.Device;
import dev.thatredox.chunkynative.opencl.renderer.scene.*;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.jocl.*;
import se.llbit.chunky.renderer.*;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.log.Log;
import se.llbit.util.TaskTracker;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Random;
import java.util.function.BooleanSupplier;
import java.util.stream.IntStream;

public class OpenClPathTracingRenderer implements Renderer {
    private BooleanSupplier postRender = () -> true;

    // Cached GPU resources persisted across render() calls.
    // Avoids repeated GPU allocation/deallocation when rapidly switching scenes.
    private cl_kernel cachedKernel = null;
    private ContextManager cachedCtx = null;
    // Two output buffers for async double-buffered readback. While the CPU
    // blends pass N's data, the GPU is already producing pass N+1 in the
    // other buffer. Eliminates the GPU stall that the previous synchronous
    // path created.
    private final ClMemory[] cachedOutputBuffers = new ClMemory[2];
    private ClMemory cachedDynamicConfig = null;
    private ClMemory cachedEmitterIntensityMem = null;
    private ClIntBuffer cachedCanvasConfig = null;
    private ClIntBuffer cachedRayDepth = null;
    private ClIntBuffer cachedEmptyGrid = null;
    private int cachedPixelCount = -1;
    private int[] cachedCanvasData = null;
    private int cachedRayDepthVal = -1;
    // Two pass buffers, one per output buffer, so async reads write into
    // different host memory than the slot the CPU is currently blending.
    //
    // Pinned-memory pipeline:
    //   cachedOutputBuffers[i] is allocated with CL_MEM_ALLOC_HOST_PTR so
    //   the driver places it in pinned (page-locked) host memory accessible
    //   to the GPU via DMA. Each iteration we clEnqueueMapBuffer the buffer
    //   for read; on pinned memory this is a pointer hand-back, not a copy.
    //   These two arrays cache the most recent map result per ping-pong
    //   slot so the inner loop can blend without re-allocating wrappers.
    //   They are populated lazily inside the render loop because
    //   clEnqueueMapBuffer requires an active queue.
    private final ByteBuffer[] cachedPassByteBuffers = new ByteBuffer[2];
    private final FloatBuffer[] cachedPassFloatBuffers = new FloatBuffer[2];

    // GPU sample-accumulator path (Tier-2 #6): when the device supports
    // cl_khr_fp64 ContextManager.accumulator is non-null and we run an
    // additional kernel after the path tracer to blend each pass into a
    // GPU-resident fp64 buffer. The host only reads the accumulator at
    // redraw time (every ~2 s), eliminating the per-iteration 32 MB
    // transfer + parallel CPU multiply-add over millions of pixels.
    //
    // PARITY: bit-identical to the CPU blend (see accumulator.c for the
    // FP_CONTRACT / FMA reasoning).
    //
    // FALLBACK: when ContextManager.accumulator is null (compile failed,
    // typically because the device lacks fp64 support) we fall back to
    // the CPU-blend path that uses cachedOutputBuffers / cachedPassFloatBuffers
    // above. Both paths coexist; the renderer picks at render() entry.
    private cl_kernel cachedAccumulateKernel = null;
    private ClMemory cachedAccumulator = null;       // pixelCount * 3 doubles
    private ClMemory cachedAccPassBuffer = null;     // pixelCount * 4 floats (single, not pinned)

    // --- Launch-size control -------------------------------------------------
    // Every kernel launch must stay FAR below the OS GPU watchdog (~2 s on
    // Windows TDR, and on Linux for a display-attached GPU). When a launch
    // exceeds it the driver resets the device and every subsequent CL call
    // fails with CL_OUT_OF_RESOURCES / CL_INVALID_COMMAND_QUEUE.
    //
    // Two knobs bound the work in one launch, applied in a strict order so they
    // cannot fight each other:
    //   sliceCount — SAFETY knob. A frame is split into S dispatches. Slice s
    //     owns the G-sized pixel blocks q where q % S == s (G = the global work
    //     size), so the slices INTERLEAVE across the frame rather than cutting it
    //     into contiguous horizontal bands. That matters: a band of sky costs a
    //     fraction of a band of dense terrain, so with contiguous bands the
    //     per-launch budget below — which is an average — would hide a heavy band
    //     that alone trips the watchdog. Interleaved slices are cost-uniform, so
    //     the average IS the worst case. Slicing is bit-exact (the kernel keys the
    //     RNG seed, camera ray and output slot off the GLOBAL pixel index) and
    //     costs only a few extra launches per frame (~10-50 us each). This is what
    //     rescues a 4K frame too heavy to render in one launch — the case ipl
    //     alone cannot fix, because ipl bottoms out at 1, which is still a whole
    //     frame.
    //   iterationsPerLaunch (ipl) — THROUGHPUT knob. Batches several full-frame
    //     passes into one launch to amortise launch overhead. Only grown once a
    //     whole frame already fits comfortably in a single launch.
    //
    // Both are driven ONLY by measured launch times, never by extrapolation. The
    // old 32k-pixel calibration extrapolated from the frame's first pixel
    // indices — the TOP ROWS, which are mostly sky and far cheaper than a real
    // pixel — and that underestimate is what tripped the watchdog.
    private int cachedIpl = 1;
    private int cachedSliceCount = -1;
    private int calibratedForPixels = -1;
    private int lastSceneGeneration = Integer.MIN_VALUE;

    private static final double TARGET_LAUNCH_MS = 500.0;
    private static final double LAUNCH_HARD_CEIL_MS = 1100.0;
    private static final int MAX_IPL = 100;
    private static final int MAX_SLICES = 4096;
    private static final long PERSISTENT_THREAD_CAP = 262144L;
    /**
     * Work-items per compute unit on NVIDIA: two waves of the render kernel at its
     * 128-register cap (2 x 256-item work-groups per SM). Launching more than that
     * only queues work-groups behind each other; on an RTX 3070 two waves ran 4-6%
     * faster than the {@link #PERSISTENT_THREAD_CAP} default.
     */
    private static final long NV_THREADS_PER_COMPUTE_UNIT = 1024L;
    /** Upper bound on the work-items in one dispatch; see {@link #threadCapFor}. */
    private long persistentThreadCap = PERSISTENT_THREAD_CAP;
    /**
     * Work-group granularity. The global work size is always a multiple of this.
     * <p>Required for correctness of a sort: {@code local_work_size} is passed as
     * null, so the driver must choose a local size that evenly DIVIDES the global
     * size. An awkward global size (odd, or prime) forces some drivers all the way
     * down to local_work_size = 1, which would run the kernel's cooperative
     * matCache copy as up to 8192 serial global loads per work-item and collapse
     * occupancy — a 100x+ slowdown that could itself trip the watchdog this whole
     * mechanism exists to avoid.
     */
    private static final long WORKGROUP_GRANULARITY = 256L;
    /**
     * How many G-sized pixel blocks each slice should get, so that a slice samples
     * enough of the frame for its cost to be representative. Only a lower bound on
     * sampling quality — coverage is exact for any value.
     */
    private static final int BLOCKS_PER_SLICE = 16;
    /**
     * The warm-up renders {@link #WARMUP_BLOCKS} blocks of {@link #WARMUP_BLOCK_PIXELS}
     * pixels each, spread evenly across the WHOLE frame (top to bottom) rather than a
     * contiguous prefix.
     *
     * <p>This is the root-cause fix for the original watchdog bug. The old warm-up
     * measured pixels [0, 32768) — the frame's TOP ROWS, which for a typical outdoor
     * shot are almost pure sky. A sky ray misses everything and terminates instantly;
     * a terrain ray does rayDepth bounces of octree DDA plus shadow rays, and can cost
     * 100x+ more. Extrapolating a whole frame from that prefix underestimated it by
     * one to two orders of magnitude, and the resulting first launch ran for seconds.
     * No safety factor can paper over a sample that unrepresentative — so the sample
     * is now representative instead, and only a modest margin is applied on top.
     */
    private static final int WARMUP_BLOCKS = 16;
    private static final long WARMUP_BLOCK_PIXELS = 2048; // multiple of WORKGROUP_GRANULARITY
    private static final double WARMUP_SAFETY = 2.0;

    /**
     * Work-items per dispatch. Chosen so each slice covers ~{@link #BLOCKS_PER_SLICE}
     * interleaved blocks (enough of the frame to be cost-representative), capped at
     * the device's {@link #threadCapFor persistent-thread cap}, and always a
     * multiple of {@link #WORKGROUP_GRANULARITY}.
     */
    private long globalSizeFor(int pixelCount, int sliceCount) {
        long ideal = Math.max(1L,
                (long) pixelCount / Math.max(1L, (long) sliceCount * BLOCKS_PER_SLICE));
        long g = Math.min(ideal, persistentThreadCap);
        g = ((g + WORKGROUP_GRANULARITY - 1) / WORKGROUP_GRANULARITY) * WORKGROUP_GRANULARITY;
        return Math.max(WORKGROUP_GRANULARITY, Math.min(g, persistentThreadCap));
    }

    /** The persistent-thread cap for a device: two resident waves on NVIDIA. */
    private static long threadCapFor(Device device) {
        if (!device.supportsNvCompilerOptions()) return PERSISTENT_THREAD_CAP;
        long units = device.getInts(CL_DEVICE_MAX_COMPUTE_UNITS, 1)[0];
        long cap = units * NV_THREADS_PER_COMPUTE_UNIT;
        cap = (cap / WORKGROUP_GRANULARITY) * WORKGROUP_GRANULARITY;
        return Math.max(WORKGROUP_GRANULARITY, Math.min(cap, PERSISTENT_THREAD_CAP));
    }

    /**
     * Feedback controller for the launch-size knobs. Called once per pass with that
     * pass's measured wall time AND the exact configuration that pass was dispatched
     * with; updates {@link #cachedIpl} / {@link #cachedSliceCount} in place.
     *
     * <p>Every term is derived from the MEASURED configuration, never from the
     * current one. This matters because on the CPU-blend path the pass being timed
     * is the previous one, whose ipl/slice size may already differ from the current
     * values. Dividing an old pass's cost by the new (larger) batch size would
     * understate the per-unit cost and grow the batch again on the strength of a
     * measurement that never justified it — a self-sustaining doubling ramp that
     * walks straight into the watchdog. Keying off the measured config makes the
     * lag merely slow convergence instead of breaking it.
     *
     * <p>Growth is capped at x2 per step, since the measurement justifying it was
     * taken at the smaller size and cost is not perfectly linear in launch size
     * (cache effects, divergence). Between TARGET and HARD_CEIL nothing changes,
     * giving the controller a hysteresis band so it settles instead of oscillating.
     */
    private void adaptLaunchSize(double passMs, int measuredIpl, int measuredSliceCount,
                                 int pixelCount) {
        // Slices are interleaved and therefore cost-uniform, so the mean launch
        // time is also the worst-case launch time — which is what the watchdog
        // actually cares about. (With contiguous slices this average would be a
        // lie: a heavy terrain band could be several times the mean.)
        double perLaunchMs = passMs / Math.max(1, measuredSliceCount);

        if (perLaunchMs > LAUNCH_HARD_CEIL_MS) {
            // Too close to the watchdog. Shed iterations first (pure throughput);
            // only once a launch is a single pass do we cut the frame into more
            // slices, which is what actually rescues a frame too heavy for one
            // launch.
            if (measuredIpl > 1) {
                cachedIpl = Math.max(1, measuredIpl / 2);
            } else {
                cachedSliceCount = Math.min(MAX_SLICES, Math.max(2, measuredSliceCount * 2));
            }
            return;
        }

        if (perLaunchMs >= TARGET_LAUNCH_MS) {
            return; // inside the hysteresis band — leave both knobs alone
        }

        double headroom = TARGET_LAUNCH_MS / Math.max(perLaunchMs, 0.01);
        if (measuredSliceCount > 1) {
            // Coalesce slices back toward a whole frame before touching ipl.
            int shrunk = (int) Math.ceil(measuredSliceCount / Math.min(headroom, 2.0));
            cachedSliceCount = Math.max(1, Math.min(measuredSliceCount, shrunk));
        } else {
            // A whole frame already fits in one launch; batch passes for throughput.
            int ideal = Math.max(1, Math.min(MAX_IPL, (int) (measuredIpl * headroom)));
            cachedIpl = ideal > measuredIpl ? Math.min(measuredIpl * 2, ideal) : ideal;
        }
    }

    /**
     * Enqueues one full frame as {@code sliceCount} interleaved dispatches, each
     * small enough to stay under the GPU watchdog. Slice s walks the pixels
     * {@code s*G, s*G+1, … } stepping by {@code sliceCount*G}, so it owns the
     * G-sized blocks q where {@code q % sliceCount == s}. Every pixel in
     * [0, pixelCount) is therefore covered exactly once and the frame is
     * bit-identical to what a single dispatch would have produced.
     *
     * @return the number of dispatches used
     */
    private int dispatchFrame(cl_command_queue queue, cl_kernel kernel,
                              int pixelStartArgIndex, int pixelStrideArgIndex,
                              int pixelCountArgIndex, int pixelCount, int sliceCount,
                              long[] globalScratch) {
        long g = globalSizeFor(pixelCount, sliceCount);
        // s*g and sliceCount*g both stay well inside int range: g is derived from
        // pixelCount/(sliceCount*BLOCKS_PER_SLICE), so sliceCount*g <= pixelCount.
        int stride = (int) (sliceCount * g);
        setIntKernelArg(kernel, pixelStrideArgIndex, stride);
        setIntKernelArg(kernel, pixelCountArgIndex, pixelCount);
        globalScratch[0] = g;
        for (int s = 0; s < sliceCount; s++) {
            setIntKernelArg(kernel, pixelStartArgIndex, (int) (s * g));
            // OpenCL snapshots kernel arguments at enqueue time, so rewriting
            // pixelStart between back-to-back enqueues is safe and each dispatch
            // keeps its own range.
            clEnqueueNDRangeKernel(queue, kernel, 1, null, globalScratch, null, 0, null, null);
        }
        return sliceCount;
    }

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

    private void releaseAllCached() {
        for (int i = 0; i < cachedOutputBuffers.length; i++) {
            if (cachedOutputBuffers[i] != null) { cachedOutputBuffers[i].close(); cachedOutputBuffers[i] = null; }
        }
        if (cachedDynamicConfig != null) { cachedDynamicConfig.close(); cachedDynamicConfig = null; }
        if (cachedEmitterIntensityMem != null) { cachedEmitterIntensityMem.close(); cachedEmitterIntensityMem = null; }
        if (cachedCanvasConfig != null) { cachedCanvasConfig.close(); cachedCanvasConfig = null; }
        if (cachedRayDepth != null) { cachedRayDepth.close(); cachedRayDepth = null; }
        if (cachedEmptyGrid != null) { cachedEmptyGrid.close(); cachedEmptyGrid = null; }
        if (cachedAccumulator != null) { cachedAccumulator.close(); cachedAccumulator = null; }
        if (cachedAccPassBuffer != null) { cachedAccPassBuffer.close(); cachedAccPassBuffer = null; }
        if (cachedAccumulateKernel != null) { clReleaseKernel(cachedAccumulateKernel); cachedAccumulateKernel = null; }
        cachedPixelCount = -1;
        cachedCanvasData = null;
        cachedRayDepthVal = -1;
        for (int i = 0; i < cachedPassByteBuffers.length; i++) {
            cachedPassByteBuffers[i] = null;
            cachedPassFloatBuffers[i] = null;
        }
        calibratedForPixels = -1;
    }

    @Override
    public void render(DefaultRenderManager manager) throws InterruptedException {
        ContextManager context = ContextManager.get();
        ClSceneLoader sceneLoader = context.sceneLoader;
        Scene scene = manager.bufferedScene;
        double[] sampleBuffer = scene.getSampleBuffer();
        int pixelCount = sampleBuffer.length / 3;
        sceneLoader.ensureLoad(manager.bufferedScene);

        // A scene (re)load invalidates the launch-size tuning: a stale
        // iterations-per-launch carried over from a lighter scene can push a
        // single launch past the GPU watchdog on the heavier one.
        int sceneGeneration = sceneLoader.getResetGeneration();
        if (sceneGeneration != lastSceneGeneration) {
            lastSceneGeneration = sceneGeneration;
            cachedIpl = 1;
            cachedSliceCount = -1;
            calibratedForPixels = -1;
        }

        // GPU fp64 accumulator path is enabled iff the device supports
        // cl_khr_fp64 AND its kernel compiled. ContextManager handles the
        // try/catch and leaves accumulator==null on failure. We pick the
        // path once per render() call so the inner loop has no extra
        // branching.
        final boolean useGpuAcc = context.accumulator != null;

        // --- Cached resource management ---
        // Kernel (recreate only on context change)
        if (cachedKernel == null || cachedCtx != context) {
            if (cachedKernel != null) clReleaseKernel(cachedKernel);
            cachedKernel = clCreateKernel(context.renderer.kernel, "render", null);
            cachedCtx = context;
            persistentThreadCap = threadCapFor(context.device);
            releaseAllCached();
        }
        if (useGpuAcc && cachedAccumulateKernel == null) {
            cachedAccumulateKernel = clCreateKernel(context.accumulator.kernel, "accumulate", null);
        }

        if (useGpuAcc) {
            // GPU-accumulator path needs:
            //   - a single fp32 pass buffer the path tracer writes into
            //   - a persistent fp64 accumulator (3 doubles per pixel)
            // No pinning needed because the host never reads pass buffer
            // and only reads accumulator at redraw time (every ~2 s).
            if (cachedPixelCount != pixelCount) {
                int floats = pixelCount * 4;
                long accBytes = (long) Sizeof.cl_double * pixelCount * 3;
                if (cachedAccPassBuffer != null) cachedAccPassBuffer.close();
                cachedAccPassBuffer = new ClMemory(clCreateBuffer(context.context.context,
                        CL_MEM_READ_WRITE,
                        (long) Sizeof.cl_float * floats, null, null));
                if (cachedAccumulator != null) cachedAccumulator.close();
                cachedAccumulator = new ClMemory(clCreateBuffer(context.context.context,
                        CL_MEM_READ_WRITE,
                        accBytes, null, null));
                // Free the unused CPU-blend buffers — they'd waste memory
                // for a path we won't take this run.
                for (int i = 0; i < cachedOutputBuffers.length; i++) {
                    if (cachedOutputBuffers[i] != null) {
                        cachedOutputBuffers[i].close();
                        cachedOutputBuffers[i] = null;
                    }
                    cachedPassByteBuffers[i] = null;
                    cachedPassFloatBuffers[i] = null;
                }
                cachedPixelCount = pixelCount;
                calibratedForPixels = -1;
            }
        } else {
            // CPU-blend path: original two-buffer pinned async pipeline.
            //
            // CL_MEM_ALLOC_HOST_PTR asks the driver to allocate the cl_mem in
            // pinned (page-locked) host memory accessible to the GPU via DMA.
            // On discrete GPUs this avoids the staging copy that a regular
            // CL_MEM_WRITE_ONLY buffer pays on every clEnqueueReadBuffer; on
            // integrated GPUs it's typically the same memory the GPU writes to,
            // so the "read" becomes a near-free map.
            if (cachedPixelCount != pixelCount) {
                int floats = pixelCount * 4;
                for (int i = 0; i < cachedOutputBuffers.length; i++) {
                    if (cachedOutputBuffers[i] != null) cachedOutputBuffers[i].close();
                    cachedOutputBuffers[i] = new ClMemory(clCreateBuffer(context.context.context,
                            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                            (long) Sizeof.cl_float * floats, null, null));
                    cachedPassByteBuffers[i] = null;
                    cachedPassFloatBuffers[i] = null;
                }
                cachedPixelCount = pixelCount;
                calibratedForPixels = -1;
            }
        }

        // DynamicConfig (always 5 ints, content written via clEnqueueWriteBuffer)
        if (cachedDynamicConfig == null) {
            cachedDynamicConfig = new ClMemory(
                    clCreateBuffer(context.context.context, CL_MEM_READ_ONLY, Sizeof.cl_int * 5, null, null));
        }

        // EmitterIntensity (always 1 float)
        if (cachedEmitterIntensityMem == null) {
            cachedEmitterIntensityMem = new ClMemory(
                    clCreateBuffer(context.context.context, CL_MEM_READ_ONLY, Sizeof.cl_float, null, null));
        }

        // CanvasConfig (6 ints, recreate on change)
        int[] canvasData = new int[]{
                scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight(),
                scene.canvasConfig.getCropWidth(), scene.canvasConfig.getCropHeight(),
                scene.canvasConfig.getCropX(), scene.canvasConfig.getCropY()
        };
        if (!Arrays.equals(canvasData, cachedCanvasData)) {
            if (cachedCanvasConfig != null) cachedCanvasConfig.close();
            cachedCanvasConfig = new ClIntBuffer(canvasData, context.context);
            cachedCanvasData = canvasData;
        }

        // RayDepth (1 int, recreate on change)
        int rayDepth = scene.getRayDepth();
        if (cachedRayDepthVal != rayDepth) {
            if (cachedRayDepth != null) cachedRayDepth.close();
            cachedRayDepth = new ClIntBuffer(rayDepth, context.context);
            cachedRayDepthVal = rayDepth;
        }

        // EmptyGrid placeholder (constant)
        if (cachedEmptyGrid == null) {
            cachedEmptyGrid = new ClIntBuffer(new int[]{0, 0, 0, 0, 0, 0, 0}, context.context);
        }

        cl_kernel kernel = cachedKernel;
        ClMemory dynamicConfig = cachedDynamicConfig;
        ClMemory emitterIntensityMem = cachedEmitterIntensityMem;
        ClIntBuffer clCanvasConfig = cachedCanvasConfig;
        ClIntBuffer clRayDepth = cachedRayDepth;
        ClIntBuffer emptyGridPlaceholder = cachedEmptyGrid;

        // Camera must be per-render (changes with each scene)
        ClCamera camera = new ClCamera(scene, context.context);
        try {
            // render() runs entirely on this thread, so no lock is passed to ClCamera.generate.
            camera.generate(null, true);
            int argIndex = 0;
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.projectorType.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(camera.cameraSettings.get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(
                    camera.apertureMaskBuffer != null ? camera.apertureMaskBuffer.get() : emptyGridPlaceholder.get()));
            setIntKernelArg(kernel, argIndex++, camera.apertureMaskWidth);
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
            // LDS material-palette cache. Cap chosen to fit comfortably in
            // typical workgroup-shared memory (modern GPUs have >=64 KB per
            // workgroup; 8192 ints = 32 KB leaves room for other __local
            // allocations and lets material-rich modded scenes (>2048 word
            // palettes) fully cache instead of falling through to global
            // for the tail.
            // Off on NVIDIA: there the palette reads are L1 hits anyway, and the up to
            // 32 KB of shared memory per work-group comes out of the same on-chip
            // storage that holds the kernel's register spills (3% faster without it
            // on an RTX 3070).
            int matCacheWords = context.device.supportsNvCompilerOptions()
                    ? 0 : Math.min(8192, sceneLoader.getMaterialPalette().wordCount());
            setIntKernelArg(kernel, argIndex++, matCacheWords);
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
            int iterationsPerLaunch = cachedIpl;
            setIntKernelArg(kernel, argIndex++, iterationsPerLaunch);
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getFogData().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterConfig().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getRenderConfig().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getCloudData().get()));
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getWaterNormalMap().get()));
            setIntKernelArg(kernel, argIndex++, sceneLoader.getWaterNormalMapWidth());
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getBiomeData().get()));
            setIntKernelArg(kernel, argIndex++, sceneLoader.getBiomeDataSize());
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(sceneLoader.getChunkBitmap().get()));
            setIntKernelArg(kernel, argIndex++, sceneLoader.getChunkBitmapSize());
            int outputArgIndex = argIndex;
            // Output buffer arg: GPU-accumulator path writes into a single
            // non-pinned buffer (consumed by the accumulate kernel, never
            // mapped to host); CPU-blend path uses ping-pong slot 0.
            ClMemory initialOutputBuffer = useGpuAcc ? cachedAccPassBuffer : cachedOutputBuffers[0];
            clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(initialOutputBuffer.get()));
            // Pixel walk (start, stride, count) consumed by the kernel's outer
            // loop. dispatchFrame() rewrites these per slice; the warm-up below
            // overrides them for its small subset.
            int pixelStartArgIndex = argIndex;
            setIntKernelArg(kernel, argIndex++, 0);
            int pixelStrideArgIndex = argIndex;
            setIntKernelArg(kernel, argIndex++, (int) persistentThreadCap);
            int pixelCountArgIndex = argIndex;
            setIntKernelArg(kernel, argIndex++, pixelCount);
            int[] cfg = new int[5];
            // Cached pointers/arrays for the inner loop. Recreating Pointer.to(...)
            // every iteration would otherwise allocate per spp pass.
            final Pointer cfgPtr = Pointer.to(cfg);
            float[] emitterIntensityArr = new float[1];
            final Pointer emitterIntensityPtr = Pointer.to(emitterIntensityArr);
            // Persistent-threads launch: instead of one work-item per pixel
            // (which forces every warp's lockstep to sync on the longest path
            // in the warp) we launch a smaller fixed number of work-items
            // and let each one process multiple pixels in a grid-stride loop.
            // When a pixel's path terminates fast the work-item picks up
            // the next pixel without waiting for its warp peers.
            //
            // Cap chosen as 256k threads (PERSISTENT_THREAD_CAP) — empirically
            // enough to fill modern GPUs (40-80 SMs × ~2-4k threads in flight)
            // while small enough that each thread sees ~8 pixels at 1080p,
            // giving the divergence-reduction benefit a chance to amortise.
            // For a slice smaller than the cap the launch falls back to one
            // thread per pixel. dispatchFrame() applies this per slice.
            final long[] dispatchGlobal = new long[1];
            // The accumulate kernel is a 1:1 elementwise fp64 blend guarded by
            // `if (gid >= pixelCount) return` — it has NO grid-stride loop, so it
            // must be launched with at least one work-item per pixel. Reusing
            // dispatchGlobal (the path tracer's persistent-thread count, capped at
            // 262144) left every pixel past index 262144 un-accumulated: those
            // pixels stayed at the accumulator's initial value (0 = black), and
            // the black region grew with resolution. Round up to a multiple of a
            // typical work-group size; the kernel's guard no-ops the surplus
            // work-items. (Trivial blend → no divergence, so 1 thread/pixel is
            // optimal here; persistent threads only help the divergent ray loop.)
            final long accLocalSize = 256L;
            final long[] accumulateGlobal = new long[]{
                    ((pixelCount + accLocalSize - 1) / accLocalSize) * accLocalSize};
            float lastEmitterIntensity = Float.NaN;
            Random rand = new Random(0);
            // Pre-bind output buffer arg pointers — two for the CPU-blend
            // double-buffer, one (and unused for slot 1) for the GPU-acc path.
            final Pointer[] outputArgPtrs = useGpuAcc
                    ? new Pointer[]{Pointer.to(cachedAccPassBuffer.get()), null}
                    : new Pointer[]{
                            Pointer.to(cachedOutputBuffers[0].get()),
                            Pointer.to(cachedOutputBuffers[1].get())
                    };

            // --- Calibration: reuse cached result or measure GPU speed ---
            cfg[0] = rand.nextInt();
            cfg[1] = 0;
            cfg[2] = scene.getEmittersEnabled() ? 1 : 0;
            cfg[3] = scene.getEmitterSamplingStrategy().ordinal();
            cfg[4] = Math.max(1, scene.getCurrentBranchCount());
            clEnqueueWriteBuffer(context.context.queue, dynamicConfig.get(), CL_TRUE, 0,
                    Sizeof.cl_int * cfg.length, cfgPtr, 0, null, null);
            emitterIntensityArr[0] = (float) scene.getEmitterIntensity();
            lastEmitterIntensity = emitterIntensityArr[0];
            clEnqueueWriteBuffer(context.context.queue, emitterIntensityMem.get(), CL_TRUE, 0,
                    Sizeof.cl_float, emitterIntensityPtr, 0, null, null);

            if (calibratedForPixels != pixelCount) {
                // Warm-up: render a REPRESENTATIVE sample of the frame and use it to
                // size the first pass. The kernel's (start, stride, count) walk makes
                // this a one-liner: with pixelStart = 0, a global size of G and a
                // stride of S, one dispatch renders G-pixel blocks at 0, S, 2S, …,
                // i.e. WARMUP_BLOCKS blocks spread evenly from the top of the frame
                // to the bottom. That samples sky AND terrain in roughly their true
                // proportions, unlike the old contiguous [0, 32768) prefix (see
                // WARMUP_BLOCKS).
                //
                // It also still materialises the lazily-allocated buffers on a cheap
                // launch, so a scene that genuinely exhausts device memory fails here
                // on a small, debuggable dispatch rather than mid-render.
                long warmG;
                int warmStride;
                if (pixelCount <= WARMUP_BLOCKS * WARMUP_BLOCK_PIXELS) {
                    // Frame is small enough to just render all of it: one work-item
                    // per pixel (rounded up to a whole work-group; the surplus items
                    // fall straight out of the kernel's loop), stride = the frame, so
                    // the "sample" is the entire image and the extrapolation is exact.
                    warmG = ((pixelCount + WORKGROUP_GRANULARITY - 1)
                            / WORKGROUP_GRANULARITY) * WORKGROUP_GRANULARITY;
                    warmStride = pixelCount;
                } else {
                    warmG = WARMUP_BLOCK_PIXELS;
                    warmStride = pixelCount / WARMUP_BLOCKS; // > warmG, so blocks never overlap
                }
                // Pixels the dispatch actually renders, counted exactly so the
                // extrapolation below is honest.
                long sampled = 0;
                for (long base = 0; base < pixelCount; base += warmStride) {
                    sampled += Math.min(warmG, pixelCount - base);
                }

                setIntKernelArg(kernel, iterationsArgIndex, 1);
                setIntKernelArg(kernel, pixelStartArgIndex, 0);
                setIntKernelArg(kernel, pixelStrideArgIndex, warmStride);
                setIntKernelArg(kernel, pixelCountArgIndex, pixelCount);
                dispatchGlobal[0] = warmG;
                long calStart = System.nanoTime();
                clEnqueueNDRangeKernel(context.context.queue, kernel, 1,
                        null, dispatchGlobal, null, 0, null, null);
                clFinish(context.context.queue);
                double calMs = (System.nanoTime() - calStart) / 1e6;

                // Extrapolate to a full frame from the representative sample, keep a
                // modest safety margin, and pick enough slices to land near TARGET.
                double frameMs = calMs * ((double) pixelCount / Math.max(1, sampled));
                int seedSlices = (int) Math.max(1, Math.min(MAX_SLICES,
                        Math.ceil(frameMs * WARMUP_SAFETY / TARGET_LAUNCH_MS)));
                iterationsPerLaunch = 1;
                cachedIpl = 1;
                cachedSliceCount = seedSlices;
                calibratedForPixels = pixelCount;
                setIntKernelArg(kernel, iterationsArgIndex, iterationsPerLaunch);
                Log.info(String.format(
                        "[ChunkyCL] warm-up: %.1f ms for %d sampled px -> est. frame %.0f ms; "
                                + "first pass in %d slice(s), ipl=1",
                        calMs, sampled, frameMs, seedSlices));
            }
            // Clamp: a cached value from a previous render of this same scene +
            // resolution is reused, but it must still be a legal slice count.
            if (cachedSliceCount <= 0) {
                cachedSliceCount = 1;
            }
            cachedSliceCount = Math.max(1, Math.min(cachedSliceCount, MAX_SLICES));

            // Double-buffered pipeline. Two output buffers + two host-side
            // staging arrays. Each iteration:
            //   1. Dispatch the kernel into ping-pong slot[curIdx].
            //   2. Issue a non-blocking read of slot[curIdx]'s contents.
            //   3. If a previous iteration's read is still pending, wait for
            //      it (it usually has finished by now because the GPU is busy
            //      with step 1) and blend that pass into sampleBuffer.
            // This way the GPU starts pass N+1 while the CPU is still digesting
            // pass N. Compared to the previous synchronous loop the steady-state
            // throughput is bounded by max(GPU-time, CPU-blend+readback) instead
            // of the sum.
            //
            // committedSpp is the running spp count actually baked into
            // sampleBuffer — it lags scene.spp by at most one in-flight pass.
            //
            // Redraw cadence is adaptive:
            //   - First commit always triggers a redraw so users see something
            //     within the first iteration (time-to-first-frame matters for
            //     UX on slow scenes where one pass can take many seconds).
            //   - After that, redraw when EITHER 10+ samples have accumulated
            //     OR 2 seconds have elapsed since the last redraw.
            //   - Hard floor of 100 ms between redraws prevents thrashing on
            //     very fast renders where every iteration produces 100+ spp;
            //     postProcessFrame and redrawScreen aren't free.
            int sppSinceRedraw = 0;
            int curIdx = 0;
            int committedSpp = scene.spp;
            long lastRedrawNanos = 0; // 0 forces redraw on first commit
            final long minRedrawIntervalNanos = 100_000_000L;  // 100 ms
            final long maxRedrawIntervalNanos = 2_000_000_000L; // 2 s

            if (useGpuAcc) {
                // === GPU fp64 accumulator path ===
                //
                // Per iteration:
                //   1. Update dynamicConfig (cfg[]) and emitterIntensity if changed.
                //   2. Dispatch path-trace kernel into cachedAccPassBuffer.
                //   3. Compute the blend weights (prevWeight/passWeight) for
                //      this pass and feed them to the accumulate kernel.
                //   4. Dispatch accumulate kernel: accumulator[i] =
                //         accumulator[i] * prevWeight + passBuffer[i] * passWeight
                //      bit-identically to the CPU loop.
                //   5. Adaptive redraw: if it's time, blocking-read the
                //      accumulator into sampleBuffer, postProcess + redrawScreen.
                //
                // The host never sees the pass buffer; the only host-side
                // transfer is the periodic accumulator → sampleBuffer copy
                // (typically every 100 ms - 2 s, vs. the old per-iteration
                // 32 MB readback at potentially 100+ Hz).
                //
                // Initial state: seed the accumulator. For a fresh render
                // (committedSpp == 0) the first accumulate pass runs with
                // prevWeight == 0.0, so the accumulator's prior contents are
                // irrelevant — a non-blocking GPU-side zero fill replaces the
                // blocking host upload (50 MB at 1080p, ~800 MB at 8K).
                // clEnqueueFillBuffer is core OpenCL 1.2, guaranteed present
                // because this path only runs when the accumulator program
                // built via the 1.2-only clCompileProgram/clLinkProgram.
                // Resumed renders (spp > 0) still upload sampleBuffer so they
                // pick up where they left off. The targetSpp > 0 guard keeps
                // the upload in the degenerate case where the loop below never
                // runs and the final readback would otherwise replace
                // sampleBuffer's contents with zeros instead of round-tripping
                // them.
                long accBytes = (long) Sizeof.cl_double * pixelCount * 3;
                if (committedSpp == 0 && scene.getTargetSpp() > 0) {
                    clEnqueueFillBuffer(context.context.queue, cachedAccumulator.get(),
                            Pointer.to(new double[]{0.0}), Sizeof.cl_double,
                            0, accBytes, 0, null, null);
                } else {
                    clEnqueueWriteBuffer(context.context.queue, cachedAccumulator.get(),
                            CL_TRUE, 0, accBytes, Pointer.to(sampleBuffer), 0, null, null);
                }

                // Bind the kernel-arg slots that don't change per iteration.
                // Slots 2/3 (weights) are set inside the loop.
                clSetKernelArg(cachedAccumulateKernel, 0, Sizeof.cl_mem, Pointer.to(cachedAccumulator.get()));
                clSetKernelArg(cachedAccumulateKernel, 1, Sizeof.cl_mem, Pointer.to(cachedAccPassBuffer.get()));
                setIntKernelArg(cachedAccumulateKernel, 4, pixelCount);

                // Reusable scratch arrays for kernel args; avoids per-iteration
                // boxing allocations.
                final double[] weightScratch = new double[1];

                while (scene.spp < scene.getTargetSpp()) {
                    long launchStartNanos = System.nanoTime();
                    cfg[0] = rand.nextInt();
                    cfg[1] = 0;
                    cfg[2] = scene.getEmittersEnabled() ? 1 : 0;
                    cfg[3] = scene.getEmitterSamplingStrategy().ordinal();
                    cfg[4] = Math.max(1, scene.getCurrentBranchCount());
                    clEnqueueWriteBuffer(context.context.queue, dynamicConfig.get(), CL_TRUE, 0,
                            Sizeof.cl_int * cfg.length, cfgPtr, 0, null, null);
                    float curEmitterIntensity = (float) scene.getEmitterIntensity();
                    if (curEmitterIntensity != lastEmitterIntensity) {
                        lastEmitterIntensity = curEmitterIntensity;
                        emitterIntensityArr[0] = curEmitterIntensity;
                        clEnqueueWriteBuffer(context.context.queue, emitterIntensityMem.get(), CL_TRUE, 0,
                                Sizeof.cl_float, emitterIntensityPtr, 0, null, null);
                    }
                    // Path-trace this pass into the single fp32 buffer, split
                    // across as many dispatches as the watchdog budget needs.
                    // Snapshot the config actually used so the controller below
                    // scores the measurement against the right configuration.
                    int usedSliceCount = cachedSliceCount;
                    int usedIpl = iterationsPerLaunch;
                    dispatchFrame(context.context.queue, kernel,
                            pixelStartArgIndex, pixelStrideArgIndex, pixelCountArgIndex,
                            pixelCount, usedSliceCount, dispatchGlobal);

                    int passSpp = usedIpl * Math.max(1, scene.getCurrentBranchCount());
                    // Weights computed in fp64 the same way as the CPU loop:
                    //   prevWeight = committedSpp / (committedSpp + passSpp)
                    //   passWeight = passSpp     / (committedSpp + passSpp)
                    long total = (long) committedSpp + (long) passSpp;
                    double prevWeight = (double) committedSpp / (double) total;
                    double passWeight = (double) passSpp     / (double) total;

                    weightScratch[0] = prevWeight;
                    clSetKernelArg(cachedAccumulateKernel, 2, Sizeof.cl_double, Pointer.to(weightScratch));
                    weightScratch[0] = passWeight;
                    clSetKernelArg(cachedAccumulateKernel, 3, Sizeof.cl_double, Pointer.to(weightScratch));

                    clEnqueueNDRangeKernel(context.context.queue, cachedAccumulateKernel, 1,
                            null, accumulateGlobal, null, 0, null, null);

                    // Wait for this pass and time it for the launch-size
                    // controller. The clFinish costs nothing extra here: the
                    // next iteration's blocking config write would drain the
                    // in-order queue anyway, and there is no per-iteration
                    // host work to overlap with on this path. Timing the whole
                    // pass (every slice) keeps the estimate representative of
                    // the full frame rather than of one cheap region.
                    clFinish(context.context.queue);
                    double passMs = (System.nanoTime() - launchStartNanos) / 1e6;
                    // clFinish above means passMs times exactly the pass just
                    // dispatched, so the measured config IS the current one.
                    adaptLaunchSize(passMs, usedIpl, usedSliceCount, pixelCount);
                    if (cachedIpl != iterationsPerLaunch) {
                        iterationsPerLaunch = cachedIpl;
                        setIntKernelArg(kernel, iterationsArgIndex, iterationsPerLaunch);
                    }

                    committedSpp = (int) Math.min((long) Integer.MAX_VALUE, total);
                    scene.spp = committedSpp;
                    sppSinceRedraw += passSpp;

                    long nowNanos = System.nanoTime();
                    long sinceLast = lastRedrawNanos == 0
                            ? Long.MAX_VALUE
                            : nowNanos - lastRedrawNanos;
                    boolean enoughSamples = sppSinceRedraw >= 10;
                    boolean tooLongSinceRedraw = sinceLast >= maxRedrawIntervalNanos;
                    boolean throttled = sinceLast < minRedrawIntervalNanos;
                    if ((enoughSamples || tooLongSinceRedraw) && !throttled) {
                        // Blocking read: postProcessFrame needs sampleBuffer
                        // populated before it runs. Fortunately the queue is
                        // in-order, so this also waits for the current
                        // accumulate kernel to finish.
                        clEnqueueReadBuffer(context.context.queue, cachedAccumulator.get(),
                                CL_TRUE, 0, accBytes, Pointer.to(sampleBuffer), 0, null, null);
                        scene.postProcessFrame(TaskTracker.Task.NONE);
                        manager.redrawScreen();
                        sppSinceRedraw = 0;
                        lastRedrawNanos = nowNanos;
                    }

                    if (camera.needGenerate) {
                        camera.generate(null, true);
                    }
                    if (postRender.getAsBoolean()) {
                        break;
                    }
                }
                // Final readback so sampleBuffer is consistent with scene.spp
                // when render() returns. postProcessFrame + redrawScreen too,
                // if there are committed samples that haven't been displayed.
                clEnqueueReadBuffer(context.context.queue, cachedAccumulator.get(),
                        CL_TRUE, 0, accBytes, Pointer.to(sampleBuffer), 0, null, null);
                if (sppSinceRedraw > 0) {
                    scene.postProcessFrame(TaskTracker.Task.NONE);
                    manager.redrawScreen();
                }
                // Do NOT inflate scene.spp past the target here. Chunky's
                // DefaultRenderManager completes on `spp >= targetSpp` (it used
                // to use `>`, which is why an overshoot hack lived here);
                // inflating now just reports one more sample than was rendered.
                postRender.getAsBoolean();
                return;
            }

            // === CPU-blend path (fallback when fp64 accumulator unavailable) ===
            // pendingRead is held in a single-element array so the inner
            // try/finally can null it out (Java doesn't allow assignment to
            // captured locals in lambdas, but we don't need a lambda — we
            // just want a stable reference for the finally block).
            cl_event pendingRead = null;
            int pendingPassSpp = 0;
            int pendingIdx = -1;
            // The launch config the in-flight pass was dispatched with. The timing
            // below completes one pass behind the one just enqueued, so the
            // controller must score it against THIS config, not the current one.
            int pendingIpl = 0;
            int pendingSliceCount = 0;
            try {
            while (scene.spp < scene.getTargetSpp()) {
                long launchStartNanos = System.nanoTime();
                cfg[0] = rand.nextInt();
                cfg[1] = 0;
                cfg[2] = scene.getEmittersEnabled() ? 1 : 0;
                cfg[3] = scene.getEmitterSamplingStrategy().ordinal();
                cfg[4] = Math.max(1, scene.getCurrentBranchCount());
                // Blocking writes — JOCL forbids non-blocking ops on
                // pointers backed by Java heap arrays (the GC could move
                // them before the GPU consumes the data). The cfg / emitter
                // writes are 20 bytes and 4 bytes respectively, so the
                // host-side stall is microseconds and not worth converting
                // to direct ByteBuffers. The big read below uses a direct
                // ByteBuffer specifically so it CAN be non-blocking.
                clEnqueueWriteBuffer(context.context.queue, dynamicConfig.get(), CL_TRUE, 0,
                        Sizeof.cl_int * cfg.length, cfgPtr, 0, null, null);
                float curEmitterIntensity = (float) scene.getEmitterIntensity();
                if (curEmitterIntensity != lastEmitterIntensity) {
                    lastEmitterIntensity = curEmitterIntensity;
                    emitterIntensityArr[0] = curEmitterIntensity;
                    clEnqueueWriteBuffer(context.context.queue, emitterIntensityMem.get(), CL_TRUE, 0,
                            Sizeof.cl_float, emitterIntensityPtr, 0, null, null);
                }
                clSetKernelArg(kernel, outputArgIndex, Sizeof.cl_mem, outputArgPtrs[curIdx]);
                int usedSliceCount = cachedSliceCount;
                int usedIpl = iterationsPerLaunch;
                dispatchFrame(context.context.queue, kernel,
                        pixelStartArgIndex, pixelStrideArgIndex, pixelCountArgIndex,
                        pixelCount, usedSliceCount, dispatchGlobal);
                int passSpp = usedIpl * Math.max(1, scene.getCurrentBranchCount());
                cl_event readEvent = new cl_event();
                // Map the pinned output buffer for read. With pinned memory
                // (CL_MEM_ALLOC_HOST_PTR) this is a pointer hand-back rather
                // than a 32 MB copy. The map is non-blocking; the event
                // signals when the GPU has finished writing and the host can
                // safely read.
                long sizeBytes = (long) Sizeof.cl_float * pixelCount * 4;
                int[] errcode = new int[1];
                ByteBuffer mapped = clEnqueueMapBuffer(context.context.queue,
                        cachedOutputBuffers[curIdx].get(),
                        CL_FALSE, CL_MAP_READ, 0, sizeBytes,
                        0, null, readEvent, errcode);
                if (mapped == null || errcode[0] != CL_SUCCESS) {
                    // readEvent was passed to clEnqueueMapBuffer; if the call
                    // populated it before failing it would otherwise leak
                    // here. Best-effort release before propagating.
                    try { clReleaseEvent(readEvent); } catch (Exception ignored) {}
                    throw new RuntimeException("clEnqueueMapBuffer failed (code "
                            + errcode[0] + ") on output slot " + curIdx);
                }
                // Native byte order so the float layout matches the GPU's
                // memory; this is required even on x86 to avoid a JOCL
                // big-endian default in some versions.
                mapped.order(ByteOrder.nativeOrder());
                cachedPassByteBuffers[curIdx] = mapped;
                cachedPassFloatBuffers[curIdx] = mapped.asFloatBuffer();

                // Snapshot the pass we're about to process and IMMEDIATELY
                // promote the freshly-enqueued event into pendingRead so the
                // outer finally is guaranteed to release it even if the
                // wait / blend / postProcessFrame below throws. The mapped
                // pointer is also captured so the unmap can run on whatever
                // we just blended.
                cl_event toProcess = pendingRead;
                int toProcessIdx = pendingIdx;
                int toProcessSpp = pendingPassSpp;
                int toProcessIpl = pendingIpl;
                int toProcessSlices = pendingSliceCount;
                pendingRead = readEvent;
                pendingPassSpp = passSpp;
                pendingIdx = curIdx;
                pendingIpl = usedIpl;
                pendingSliceCount = usedSliceCount;
                curIdx ^= 1;

                if (toProcess != null) {
                    clWaitForEvents(1, new cl_event[]{toProcess});
                    blendPass(cachedPassFloatBuffers[toProcessIdx], sampleBuffer, pixelCount,
                            committedSpp, toProcessSpp);
                    // Release the host-side mapping. In-order queue means the
                    // next dispatch into this slot (two iterations from now)
                    // will wait for the unmap to complete. Without this the
                    // OpenCL spec considers the cl_mem still owned by the host.
                    clEnqueueUnmapMemObject(context.context.queue,
                            cachedOutputBuffers[toProcessIdx].get(),
                            cachedPassByteBuffers[toProcessIdx],
                            0, null, null);
                    cachedPassByteBuffers[toProcessIdx] = null;
                    cachedPassFloatBuffers[toProcessIdx] = null;
                    clReleaseEvent(toProcess);
                    // The queue is in-order and the config write at the top of this
                    // iteration is blocking, so by the time we get here the pass
                    // being timed is the one ENQUEUED LAST ITERATION (toProcess),
                    // not the one enqueued just above. Score the measurement against
                    // THAT pass's config — using the current (possibly already
                    // grown) config would understate its per-unit cost and grow the
                    // batch again on a measurement that never justified it.
                    //
                    // The window also includes the CPU blend of the previous pass.
                    // When the blend dominates this OVER-estimates GPU time, which
                    // only biases the launch size DOWN — the safe direction for the
                    // watchdog.
                    double passMs = (System.nanoTime() - launchStartNanos) / 1e6;
                    adaptLaunchSize(passMs, toProcessIpl, toProcessSlices, pixelCount);
                    if (cachedIpl != iterationsPerLaunch) {
                        iterationsPerLaunch = cachedIpl;
                        setIntKernelArg(kernel, iterationsArgIndex, iterationsPerLaunch);
                    }
                    committedSpp += toProcessSpp;
                    scene.spp = committedSpp;
                    sppSinceRedraw += toProcessSpp;
                    long nowNanos = System.nanoTime();
                    long sinceLast = lastRedrawNanos == 0
                            ? Long.MAX_VALUE
                            : nowNanos - lastRedrawNanos;
                    boolean enoughSamples = sppSinceRedraw >= 10;
                    boolean tooLongSinceRedraw = sinceLast >= maxRedrawIntervalNanos;
                    boolean throttled = sinceLast < minRedrawIntervalNanos;
                    if ((enoughSamples || tooLongSinceRedraw) && !throttled) {
                        scene.postProcessFrame(TaskTracker.Task.NONE);
                        manager.redrawScreen();
                        sppSinceRedraw = 0;
                        lastRedrawNanos = nowNanos;
                    }
                }

                if (camera.needGenerate) {
                    camera.generate(null, true);
                }
                if (postRender.getAsBoolean()) {
                    break;
                }
            }
            // Drain the last in-flight pass so its samples aren't lost.
            if (pendingRead != null) {
                clWaitForEvents(1, new cl_event[]{pendingRead});
                blendPass(cachedPassFloatBuffers[pendingIdx], sampleBuffer, pixelCount,
                        committedSpp, pendingPassSpp);
                clEnqueueUnmapMemObject(context.context.queue,
                        cachedOutputBuffers[pendingIdx].get(),
                        cachedPassByteBuffers[pendingIdx],
                        0, null, null);
                cachedPassByteBuffers[pendingIdx] = null;
                cachedPassFloatBuffers[pendingIdx] = null;
                clReleaseEvent(pendingRead);
                pendingRead = null;
                committedSpp += pendingPassSpp;
                scene.spp = committedSpp;
                sppSinceRedraw += pendingPassSpp;
            }
            } finally {
                // Defensive cleanup: if anything in the loop or drain threw,
                // make sure the in-flight cl_event is released and any open
                // mapping is closed so we don't leak GPU-side handles or
                // pinned-memory ownership.
                if (pendingRead != null) {
                    try {
                        clReleaseEvent(pendingRead);
                    } catch (Exception ignored) { }
                }
                for (int i = 0; i < cachedPassByteBuffers.length; i++) {
                    if (cachedPassByteBuffers[i] != null && cachedOutputBuffers[i] != null) {
                        try {
                            clEnqueueUnmapMemObject(context.context.queue,
                                    cachedOutputBuffers[i].get(),
                                    cachedPassByteBuffers[i],
                                    0, null, null);
                        } catch (Exception ignored) { }
                        cachedPassByteBuffers[i] = null;
                        cachedPassFloatBuffers[i] = null;
                    }
                }
            }
            // Flush final frame if there are undrawn samples
            if (sppSinceRedraw > 0) {
                scene.postProcessFrame(TaskTracker.Task.NONE);
                manager.redrawScreen();
            }
            // Do NOT inflate scene.spp past the target here. Chunky's
            // DefaultRenderManager completes on `spp >= targetSpp` (it used
            // to use `>`, which is why an overshoot hack lived here);
            // inflating now just reports one more sample than was rendered.
            postRender.getAsBoolean();
        } catch (CLException e) {
            logClRenderFailure(e);
            throw e;
        } finally {
            camera.close();
        }
    }

    /**
     * Attaches a human-readable explanation to the OpenCL error codes users
     * actually hit, before the exception propagates to Chunky's render manager.
     */
    private static void logClRenderFailure(CLException e) {
        int status = e.getStatus();
        if (status == CL_OUT_OF_RESOURCES || status == CL_INVALID_COMMAND_QUEUE) {
            Log.error("[ChunkyCL] Render failed with " + stringFor_errorCode(status)
                    + ". On Windows this usually means the GPU watchdog (TDR, ~2 s) killed an"
                    + " over-long kernel launch and reset the device — NOT that the GPU ran out"
                    + " of memory. The launch batcher self-tunes upward from 1 pass per launch,"
                    + " so if this persists even a single pass exceeds the watchdog: lower the"
                    + " render resolution, or raise the TdrDelay registry value.");
        } else if (status == CL_MEM_OBJECT_ALLOCATION_FAILURE || status == CL_OUT_OF_HOST_MEMORY) {
            Log.error("[ChunkyCL] Render failed with " + stringFor_errorCode(status)
                    + ": the scene plus render buffers exceed available device/host memory."
                    + " Try fewer chunks or a lower render resolution.");
        }
    }

    /**
     * Parallel sample accumulator. Each pixel is independent so the blend is
     * safe to fan out across cores; the per-pixel arithmetic (a*w1 + b*w2) is
     * the same as the previous serial version, so the result is bit-exact
     * regardless of the worker count.
     *
     * Reads from a direct FloatBuffer (the off-heap target of the async
     * GPU readback). FloatBuffer.get(int) is concurrent-read-safe.
     */
    private static void blendPass(FloatBuffer passBuffer, double[] sampleBuffer,
                                  int pixelCount, int sppBefore, int passSpp) {
        double prevWeight = (double) sppBefore / (sppBefore + passSpp);
        double passWeight = (double) passSpp / (sppBefore + passSpp);
        // Chunk size tuned so each task is ~16k pixels; large enough to amortise
        // the fork/join overhead, small enough that a 4K image still fans out.
        final int chunk = 16384;
        int chunks = (pixelCount + chunk - 1) / chunk;
        IntStream.range(0, chunks).parallel().forEach(c -> {
            int start = c * chunk;
            int end = Math.min(start + chunk, pixelCount);
            for (int p = start; p < end; p++) {
                int sampleIdx = p * 3;
                int passIdx = p * 4;
                sampleBuffer[sampleIdx]     = sampleBuffer[sampleIdx]     * prevWeight + passBuffer.get(passIdx)     * passWeight;
                sampleBuffer[sampleIdx + 1] = sampleBuffer[sampleIdx + 1] * prevWeight + passBuffer.get(passIdx + 1) * passWeight;
                sampleBuffer[sampleIdx + 2] = sampleBuffer[sampleIdx + 2] * prevWeight + passBuffer.get(passIdx + 2) * passWeight;
            }
        });
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
