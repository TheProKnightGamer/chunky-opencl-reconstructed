// FRAGMENT, not a header: included INSIDE kernel bodies (render in
// rayTracer.c and the wavefront kernels), so it has no include guard and is
// meant to be included more than once.
//
// The render kernel's parameter list, shared by every kernel that renders
// paths (render, wfPrimary, wfShade, wfConnect, wfExtend, wfFinish) so the host
// binds the same arguments at the same indices on all of them.
    // Small uniform buffers in __constant memory: dedicated broadcast cache
    // on most GPUs, reduces L1 pressure on the big buffers below. Total
    // footprint of all __constant args here is well under 1 KB so it fits
    // comfortably in any device's CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    // (spec minimum is 64 KB on OpenCL 1.2+).
    __constant const int* projectorType,
    __global const float* cameraSettings,
    __global const int* apertureMask,
    int apertureMaskWidth,

    __constant const int* octreeDepth,
    __global const int* octreeData,

    __constant const int* waterOctreeDepth,
    __global const int* waterOctreeData,

    __global const int* bPalette,
    __global const int* quadModels,
    __global const int* aabbModels,

    __global const int* worldBvhData,
    __global const int* actorBvhData,
    __global const int* bvhTrigs,

    image2d_array_t textureAtlas,
    __global const int* matPalette,
    int matCacheWords,
    __local unsigned int* matCache,

    image2d_t skyTexture,

    __constant const float* skyIntensity,
    __constant const int* sunData,

    __constant const int* dynamicConfig,
    __constant const float* emitterIntensity,
    __global const int* emitterPositions,
    __global const int* positionIndexes,
    __global const int* constructedGrid,
    __constant const int* gridConfig,
    __constant const int* canvasConfig,
    __constant const int* rayDepth,
    int iterations,

    // New buffers for expanded features
    __constant const float* fogData,
    __constant const float* waterConfig,
    __constant const float* renderConfig,
    __global const int* cloudData,
    __global const float* waterNormalMap,
    int waterNormalMapW,
    __global const int* biomeData,
    int biomeDataSize,
    __global const int* chunkBitmap,
    int chunkBitmapSize,

    __global float* res,

    // Pixel walk for this dispatch: start at pixelStart and step by pixelStride
    // until pixelCount. We over-launch fewer work-items than pixels and let each
    // work-item iterate over multiple pixels via a grid-stride loop; this keeps
    // warps full when individual paths terminate at very different bounce depths
    // instead of stalling lockstep on the longest path in the warp.
    //
    // The host may SPLIT a frame across S dispatches so that no single kernel
    // launch runs long enough to trip the OS GPU watchdog (~2 s on Windows TDR,
    // and on Linux for a display-attached GPU), which resets the device and
    // surfaces as CL_OUT_OF_RESOURCES / CL_INVALID_COMMAND_QUEUE. With G =
    // get_global_size(0), dispatch s passes pixelStart = s*G and pixelStride =
    // S*G, so it owns the G-sized pixel blocks q where q % S == s.
    //
    // The slices INTERLEAVE rather than partitioning the frame into contiguous
    // bands, which matters for two reasons:
    //   - Every pixel is still covered exactly once: for pixel p, write p = q*G
    //     + r (r < G); it belongs to dispatch s = q % S, work-item r, on loop
    //     iteration k = q / S. That (s, r, k) is unique, so the slices are a
    //     partition, not an overlap.
    //   - Each slice samples the WHOLE image instead of one horizontal band. A
    //     band of sky costs a fraction of a band of dense terrain, so contiguous
    //     slices have wildly uneven cost and the host's per-launch time budget —
    //     which it can only measure as an average — would hide a heavy band that
    //     alone trips the watchdog. Interleaved slices are cost-uniform, so the
    //     average IS the worst case.
    // Coalescing is unaffected: within one iteration the G work-items still read
    // and write G consecutive pixels.
    //
    // gid below is the GLOBAL pixel index in every case, so the RNG seed, the
    // camera ray and the output slot do not depend on how the frame is sliced —
    // slicing is bit-exact, not an approximation.
    int pixelStart,
    int pixelStride,
    int pixelCount

