// FRAGMENT, not a header: included INSIDE kernel bodies (render in
// rayTracer.c and the wavefront kernels), so it has no include guard and is
// meant to be included more than once.
//
// Kernel prologue: reads the scene configuration out of the arguments into
// locals (scene, fogConfig, sun, the cached* values, camCache, ...). Declares wid.
    // The walk's step is pixelStride (= sliceCount * get_global_size(0)), passed
    // in by the host rather than read from get_global_size here, so that a single
    // dispatch covers only its own interleaved share of the frame.
    int wid = get_global_id(0);

    // Cooperative copy of material palette ints into per-work-group local cache.
    if (matCacheWords > 0) {
        int lid = get_local_id(0);
        int lsize = get_local_size(0);
        // matCacheWords is bounded (host caps it) and lsize is typically
        // 64-256, so this loop runs at most ~32-128 times per work-item.
        // The hint lets the compiler unroll a few iterations and issue
        // wider memory transactions where possible.
        __attribute__((opencl_unroll_hint(8)))
        for (int i = lid; i < matCacheWords; i += lsize) {
            matCache[i] = matPalette[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    SceneConfig scene;
    scene.materialPalette = MaterialPalette_new(matPalette, matCacheWords, matCache);
    scene.octree = Octree_create(octreeData, *octreeDepth);
    scene.waterOctree = Octree_create(waterOctreeData, *waterOctreeDepth);
    scene.worldBvh = Bvh_new(worldBvhData, bvhTrigs, &scene.materialPalette);
    scene.actorBvh = Bvh_new(actorBvhData, bvhTrigs, &scene.materialPalette);
    scene.blockPalette = BlockPalette_new(bPalette, quadModels, aabbModels, &scene.materialPalette);
    // Cap the octree DDA at the worst-case number of leaf-cell crossings for a
    // ray traversing the whole octree (the 3D-DDA diagonal bound, 3 * 2^depth),
    // not a flat 256. The old 256 cap made distant geometry along shallow angles
    // in large/wide scenes terminate early and dissolve into sky (CPU is
    // uncapped). Clamped to 8192 so a pathological miss-ray on a huge octree
    // can't stall a dispatch. Normal rays exit early (on hit or out-of-bounds),
    // so this only affects long miss-rays.
    scene.drawDepth = min(3 << max(scene.octree.depth, scene.waterOctree.depth), 8192);

    // Load water config: [enabled, height, chunkClip, octreeSize, shadingStrategy,
    //                      animationTime, visibility, r, g, b, useCustomColor, ior,
    //                      shaderIterations, shaderFrequency, shaderAmplitude, shaderSpeed,
    //                      waterOpacity]
    scene.waterPlaneEnabled = (waterConfig[0] > 0.5f);
    scene.waterPlaneHeight = waterConfig[1];
    scene.waterPlaneChunkClip = (waterConfig[2] > 0.5f);
    scene.octreeSize = waterConfig[3];
    scene.waterShadingStrategy = (int)waterConfig[4];
    scene.animationTime = waterConfig[5];
    scene.waterVisibility = waterConfig[6];
    scene.waterColor = (float3)(waterConfig[7], waterConfig[8], waterConfig[9]);
    scene.useCustomWaterColor = (waterConfig[10] > 0.5f);
    scene.waterIor = waterConfig[11];
    scene.waterShaderParams.iterations = (int)waterConfig[12];
    scene.waterShaderParams.baseFrequency = waterConfig[13];
    scene.waterShaderParams.baseAmplitude = waterConfig[14];
    scene.waterShaderParams.animationSpeed = waterConfig[15];
    scene.waterOpacity = waterConfig[16];
    scene.hasWaterBlocks = (waterConfig[17] > 0.5f);
    scene.waterBounds = AABB_new(waterConfig[18], waterConfig[21], waterConfig[19],
                                 waterConfig[22], waterConfig[20], waterConfig[23]);
    scene.waterNormalMap = waterNormalMap;
    scene.waterNormalMapW = waterNormalMapW;

    // Load render config: [sunSamplingStrategy, transparentSky, branchCount,
    //                       cloudsEnabled, cloudHeight, cloudSize,
    //                       preventNormalEmitterWithSampling, strictDirectLight,
    //                       fancierTranslucency, transmissivityCap, biomeColorsEnabled,
    //                       cloudOffsetX, cloudOffsetZ, yMin, yMax, biomeYLevels]
    int sunSamplingStrategy = (int)renderConfig[0];
    // Sun-strategy boolean flags. These mirror chunky's
    // SunSamplingStrategy.java truth table 1:1; if upstream changes the
    // table (or adds a strategy), update both sides together.
    //   Strategy        | doSampling | diffuseSun | sunLuminosity | importance
    //   OFF             |   false    |   true     |    true       |   false
    //   NON_LUMINOUS    |   false    |   false    |    false      |   false
    //   FAST            |   true     |   false    |    false      |   false
    //   IMPORTANCE      |   false    |   true     |    true       |   true
    //   HIGH_QUALITY    |   true     |   true     |    true       |   false
    bool sunDoSampling   = (sunSamplingStrategy == SUN_SAMPLING_FAST
                         || sunSamplingStrategy == SUN_SAMPLING_HIGH_QUALITY);
    bool sunDiffuseSun   = (sunSamplingStrategy == SUN_SAMPLING_OFF
                         || sunSamplingStrategy == SUN_SAMPLING_IMPORTANCE
                         || sunSamplingStrategy == SUN_SAMPLING_HIGH_QUALITY);
    bool sunIsLuminosity = (sunSamplingStrategy == SUN_SAMPLING_OFF
                         || sunSamplingStrategy == SUN_SAMPLING_IMPORTANCE
                         || sunSamplingStrategy == SUN_SAMPLING_HIGH_QUALITY);
    bool sunIsImportance = (sunSamplingStrategy == SUN_SAMPLING_IMPORTANCE);
    scene.transparentSky = (renderConfig[1] > 0.5f);
    // branchCount is passed via dynamicConfig[4] so it updates per-frame (ramp-up logic)
    int branchCount = dynamicConfig[4];
    if (branchCount < 1) branchCount = 1;
    scene.cloudsEnabled = (renderConfig[3] > 0.5f);
    scene.cloudHeight = renderConfig[4];
    scene.cloudSize = renderConfig[5];
    scene.cloudOffsetX = renderConfig[11];
    scene.cloudOffsetZ = renderConfig[12];
    scene.cloudData = cloudData;
    bool preventNormalEmitterWithSampling = (renderConfig[6] > 0.5f);
    bool strictDirectLight = (renderConfig[7] > 0.5f);
    bool fancierTranslucency = (renderConfig[8] > 0.5f);
    float transmissivityCap = renderConfig[9];
    scene.biomeColorsEnabled = (renderConfig[10] > 0.5f);
    scene.biomeData = biomeData;
    scene.biomeDataSize = biomeDataSize;
    scene.biomeYLevels = max(1, (int)renderConfig[15]);
    scene.yMin = renderConfig[13];
    scene.yMax = renderConfig[14];
    scene.chunkBitmap = chunkBitmap;
    scene.chunkBitmapSize = chunkBitmapSize;

    // Load fog config
    FogConfig fogConfig = Fog_load(fogData);

    Sun sun = Sun_new(sunData);

    // Cache frequently-read global memory values (constant during dispatch)
    unsigned int randSeed = dynamicConfig[0];
    bool emittersEnabled = dynamicConfig[2] != 0;
    int samplingStrategy = dynamicConfig[3];
    float cachedEmitterIntensity = *emitterIntensity;
    int cachedRayDepth = *rayDepth;
    float cachedSkyIntensity = *skyIntensity;
    // Sun pow(gamma) and 1/luminosity are precomputed on host (PackedSun
    // slots 14 and 15). Saves one pow() and one divide per work-item.
    float cachedSunPower = sun.intensityPowGamma;
    float cachedSunLumInv = sun.luminosityInv;
    // Light for water in-scattering, relative to the default day it was tuned in.
    float3 waterSunLight = Water_sunLight(sun.color.xyz, cachedSunPower);
    float waterSkyLight = Water_skyLight((float3)(waterConfig[24], waterConfig[25], waterConfig[26]));
    int gc_cellSize = 0, gc_offsetX = 0, gc_sizeX = 0;
    int gc_offsetY = 0, gc_sizeY = 0, gc_offsetZ = 0, gc_sizeZ = 0;
    if (emittersEnabled && samplingStrategy != 0 && gridConfig[2] > 0) {
        gc_cellSize = gridConfig[0];
        gc_offsetX = gridConfig[1];
        gc_sizeX = gridConfig[2];
        gc_offsetY = gridConfig[3];
        gc_sizeY = gridConfig[4];
        gc_offsetZ = gridConfig[5];
        gc_sizeZ = gridConfig[6];
    }

    // Cache camera constants once (avoids re-reading global memory each iteration).
    // This is gid-independent so we hoist it outside the grid-stride loop —
    // every pixel processed by this work-item shares the same camera setup.
    CameraCache camCache = CameraCache_load(projectorType, cameraSettings, canvasConfig);

