#include "../opencl.h"
#include "octree.h"
#include "rt.h"
#include "block.h"
#include "material.h"
#include "kernel.h"
#include "camera.h"
#include "bvh.h"
#include "sky.h"
#include "fog.h"
#include "water.h"

// Like isInWater(), but also returns the water-octree block at floor(origin)
// via *outBlock (0 if empty/out of bounds), so getDirectLightAttenuation can
// set shadow.material without a second octree descent.
bool isInWaterWithBlock(SceneConfig scene, float3 origin, int* outBlock) {
    int x = (int)floor(origin.x);
    int y = (int)floor(origin.y);
    int z = (int)floor(origin.z);
    int waterBlock = Octree_get(&scene.waterOctree, x, y, z);
    *outBlock = waterBlock;
    // Water plane check
    if (scene.waterPlaneEnabled && origin.y < scene.waterPlaneHeight) {
        if (scene.waterPlaneChunkClip && scene.chunkBitmapSize > 0) {
            int cx = (int)floor(origin.x) >> 4;
            int cz = (int)floor(origin.z) >> 4;
            if (cx >= 0 && cx < scene.chunkBitmapSize && cz >= 0 && cz < scene.chunkBitmapSize) {
                int bitIndex = cz * scene.chunkBitmapSize + cx;
                int word = scene.chunkBitmap[bitIndex >> 5];
                bool isLoaded = (word >> (bitIndex & 31)) & 1;
                if (!isLoaded) {
                    return true; // Below water plane in unloaded chunk
                }
            } else {
                return true; // Outside bitmap = unloaded = underwater
            }
        } else {
            return true; // Below water plane, no chunk clipping
        }
    }
    // Water octree check
    if (waterBlock == 0) return false;
    int modelType = scene.blockPalette.blockPalette[waterBlock + 0];
    if (modelType != 5) return false;
    // It's a water block — check if position is below the surface
    int waterData = scene.blockPalette.blockPalette[waterBlock + 2];
    bool isFull = (waterData >> WATER_FULL_BLOCK) & 1;
    if (isFull) return true;
    float fracY = origin.y - (float)y;
    return fracY < 0.875f; // 14/16 = standard water height
}

// Check if a ray origin is inside water, matching CPU Scene.isInWater().
// Checks water plane first, then the water octree.
bool isInWater(SceneConfig scene, float3 origin) {
    int unusedBlock;
    return isInWaterWithBlock(scene, origin, &unusedBlock);
}

// Sun sampling strategies
#define SUN_SAMPLING_OFF         0
#define SUN_SAMPLING_NON_LUMINOUS 1
#define SUN_SAMPLING_FAST        2
#define SUN_SAMPLING_IMPORTANCE  3
#define SUN_SAMPLING_HIGH_QUALITY 4

// Cached camera constants (populated once per work-item from global memory).
// pset0..pset3 are the four projector-settings floats at cameraSettings[14..17];
// they're scene-uniform and would otherwise be re-read from __global memory
// inside every Camera_* call inside the iteration loop. Caching them as
// scalars puts them in registers so the iteration loop touches __global
// cameraSettings only for the pre-gen ray fallback (projType == -1).
typedef struct {
    int projType;
    int width;
    float halfWidth, invHeight;
    int cropX, cropY;
    float shiftX, shiftY;
    float3 cameraPos, m1s, m2s, m3s;
    float pset0, pset1, pset2, pset3, pset4;
} CameraCache;

// projectorType (1 int) and canvasConfig (6 ints) are uniform → __constant.
// cameraSettings stays __global because pre-gen ray modes can hold one ray
// per pixel (up to ~12 MB at 1080p) which won't fit in constant memory.
CameraCache CameraCache_load(
        __constant const int* projectorType,
        __global const float* cameraSettings,
        __constant const int* canvasConfig
) {
    CameraCache cc;
    cc.projType = *projectorType;
    if (cc.projType != -1) {
        cc.cameraPos = vload3(0, cameraSettings);
        cc.m1s = vload3(1, cameraSettings);
        cc.m2s = vload3(2, cameraSettings);
        cc.m3s = vload3(3, cameraSettings);
        cc.width = canvasConfig[0];
        int fullWidth = canvasConfig[2];
        int fullHeight = canvasConfig[3];
        cc.cropX = canvasConfig[4];
        cc.cropY = canvasConfig[5];
        cc.halfWidth = fullWidth / (2.0f * fullHeight);
        cc.invHeight = 1.0f / fullHeight;
        cc.shiftX = cameraSettings[12];
        cc.shiftY = cameraSettings[13];
        // Pull projector-specific settings (aperture/fov/subject-distance/
        // shape, depending on projType) into register scalars so the
        // iteration-loop Camera_* calls don't re-read them from __global
        // memory each pass.
        cc.pset0 = cameraSettings[14];
        cc.pset1 = cameraSettings[15];
        cc.pset2 = cameraSettings[16];
        cc.pset3 = cameraSettings[17];
        cc.pset4 = cameraSettings[18];  // 5th slot (parallel DoF subjectDistance)
    }
    return cc;
}

Ray ray_to_camera(
        CameraCache cc,
        const __global float* cameraSettings,
        const __global int* apertureMask,
        int apertureMaskWidth,
        int gid,
        Random random
) {
    Ray ray;
    if (cc.projType != -1) {
        float x = -cc.halfWidth + ((gid % cc.width) + Random_nextFloat(random) + cc.cropX) * cc.invHeight;
        float y = -0.5f + ((gid / cc.width) + Random_nextFloat(random) + cc.cropY) * cc.invHeight;
        // Lens shift applies ONLY to pinhole (0) and parallel (1): on the CPU
        // only those projectors are wrapped in a ShiftProjector. Applying it to
        // the spherical/ODS projectors offset the whole rendered image.
        if (cc.projType == 0 || cc.projType == 1) {
            x += cc.shiftX;
            y -= cc.shiftY;
        }

        // Pass cached projector settings as scalars — no __global reads here.
        switch (cc.projType) {
            case 0:
                ray = Camera_pinHole(x, y, random, cc.pset0, cc.pset1, cc.pset2, cc.pset3, apertureMask, apertureMaskWidth);
                break;
            case 1:
                ray = Camera_parallel(x, y, random, cc.pset0, cc.pset1, cc.pset2, cc.pset3, cc.pset4, apertureMask, apertureMaskWidth);
                break;
            case 2:
                ray = Camera_fisheye(x, y, random, cc.pset0, cc.pset1, cc.pset2, cc.pset3, apertureMask, apertureMaskWidth);
                break;
            case 3:
                ray = Camera_stereographic(x, y, random, cc.pset0, cc.pset1, cc.pset2, cc.pset3, apertureMask, apertureMaskWidth);
                break;
            case 4:
                ray = Camera_panoramic(x, y, random, cc.pset0, cc.pset1, cc.pset2, cc.pset3, apertureMask, apertureMaskWidth);
                break;
            case 5:
                ray = Camera_panoramicSlot(x, y, random, cc.pset0, cc.pset1, cc.pset2, cc.pset3, apertureMask, apertureMaskWidth);
                break;
            case 6:
                ray = Camera_ODS(x, y, random, cc.pset0, cc.pset1);
                break;
            case 7:
                ray = Camera_ODSStacked(x, y, random, cc.pset0);
                break;
            default:
                ray = Camera_pinHole(x, y, random, cc.pset0, cc.pset1, cc.pset2, cc.pset3, apertureMask, apertureMaskWidth);
                break;
        }

        ray.direction = normalize((float3) (
                dot(cc.m1s, ray.direction),
                        dot(cc.m2s, ray.direction),
                        dot(cc.m3s, ray.direction)
        ));
        ray.origin = (float3) (
                dot(cc.m1s, ray.origin),
                        dot(cc.m2s, ray.origin),
                        dot(cc.m3s, ray.origin)
        );

        ray.origin += cc.cameraPos;
    } else {
        ray = Camera_preGenerated(cameraSettings, gid);
    }
    return ray;
}

// Trace a shadow ray toward the sun, accumulating attenuation through translucent materials.
// Returns the color attenuation along the path (white = fully lit, black = fully shadowed).
// Matches CPU PathTracer.getDirectLightAttenuation() with RGBA tracking, water fog, and strict direct light.
float3 getDirectLightAttenuation(
    SceneConfig scene, image2d_array_t textureAtlas,
    float3 origin, float3 sunDir, float maxDist,
    bool strictDirectLight
) {
    float3 attenuation = (float3)(1.0f);
    float alphaAtt = 1.0f;  // Alpha channel for overall transparency
    Ray shadow;
    shadow.origin = origin;
    shadow.direction = sunDir;
    shadow.material = 0;
    shadow.flags = RAY_SHADOW;
    shadow.currentIor = AIR_IOR;
    shadow.prevIor = AIR_IOR;
    int wBlock;
    shadow.inWater = isInWaterWithBlock(scene, origin, &wBlock);
    if (shadow.inWater) {
        shadow.currentIor = scene.waterIor;
        // Set material to the water block data so the world octree skips internal
        // water blocks. Without this, the shadow ray hits every water block boundary
        // (full AABB faces), toggling inWater on each hit and never reaching the
        // actual water surface — resulting in almost no water fog attenuation.
        if (wBlock > 0) {
            shadow.material = wBlock;
        }
    }

    // Up to 32 translucent layers (was 8). CPU is unbounded (while attenuation.w
    // > 0); a cap of 8 let too much sun through deep glass/leaf stacks (brighter
    // shadows / god rays than CPU). The loop still exits early on an opaque hit
    // or a clear path, so this only costs extra work on deep translucent stacks.
    for (int i = 0; i < 32; i++) {
        if (alphaAtt <= 0.0f) break;

        IntersectionRecord srec = IntersectionRecord_new();
        srec.distance = maxDist;
        MaterialSample sMat;
        Material sSample;

        if (!closestIntersect(scene, textureAtlas, shadow, &srec, &sMat, &sSample)) {
            break; // Clear path to sun
        }

        if (srec.distance >= maxDist - OFFSET) {
            break; // Past the target distance
        }

        // Opaque material blocks the sun
        if (sMat.color.w > 1.0f - EPS && !sMat.refractive) {
            return (float3)(0.0f);
        }

        // Apply biome tinting for shadow ray materials (e.g. leaves).
        // Tint only affects rgb, which is consumed solely by the attenuation
        // update below — so it can safely run after the break/opaque exits.
        {
            float3 sHitPos = shadow.origin + shadow.direction * srec.distance;
            applyBiomeTint(scene, &sMat, sHitPos);
        }

        // CPU formula: per-channel attenuation with alpha tracking
        float mult = 1.0f - sMat.color.w;
        attenuation.x *= sMat.color.x * sMat.color.w + mult;
        attenuation.y *= sMat.color.y * sMat.color.w + mult;
        attenuation.z *= sMat.color.z * sMat.color.w + mult;
        alphaAtt *= mult;

        // Water fog attenuation in shadow rays.
        // Matches CPU: checks prevMaterial.isWater() — fog is applied for
        // the distance the shadow ray traveled through water to reach this hit.
        if (shadow.inWater) {
            if (scene.waterVisibility <= 0) {
                alphaAtt = 0.0f;
            } else {
                float a = srec.distance / scene.waterVisibility;
                alphaAtt *= exp(-a);
            }
        }

        // Strict direct light: block shadow ray if it crosses an IOR boundary
        if (strictDirectLight && shadow.currentIor != sMat.ior) {
            alphaAtt = 0.0f;
        }

        // Continue tracing past this intersection
        maxDist -= srec.distance + OFFSET;
        shadow.origin = shadow.origin + shadow.direction * (srec.distance + OFFSET);
        shadow.currentIor = sMat.ior;

        // Track water medium transitions for shadow ray
        if (sMat.isWater) {
            shadow.inWater = !shadow.inWater;
        }

        if (maxDist <= 0) break;
    }

    return attenuation * alphaAtt;
}

__kernel void render(
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

    // Total pixel count. We over-launch fewer work-items than pixels and let
    // each work-item iterate over multiple pixels via a grid-stride loop;
    // this keeps warps full when individual paths terminate at very
    // different bounce depths instead of stalling lockstep on the longest
    // path in the warp. Bit-exact parity preserved because each pixel still
    // gets the same RNG seed (gid * constant + iter * constant).
    int pixelCount

) {
    int wid = get_global_id(0);
    int stride = get_global_size(0);

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

    // Grid-stride loop: each work-item processes multiple pixels. When one
    // pixel's path terminates fast (e.g. sky hit) the same work-item picks
    // up the next pixel rather than the warp stalling on the longest-path
    // thread. Empty inner-state per pixel — sumColor/sumAlpha reset.
    for (int gid = wid; gid < pixelCount; gid += stride) {
    float3 sumColor = (float3)(0.0f, 0.0f, 0.0f);
    float sumAlpha = 0.0f;

    for (int it = 0; it < iterations; ++it) {
        unsigned int randomState = randSeed + (unsigned int)(gid * 1664525u) + (unsigned int)(it * 1013904223u);
        Random random = &randomState;
        Random_nextState(random);
        Random_nextState(random);
        Ray ray = ray_to_camera(camCache, cameraSettings, apertureMask, apertureMaskWidth, gid, random);

        ray.material = 0;
        ray.flags = 0;
        ray.currentIor = AIR_IOR;
        ray.prevIor = AIR_IOR;
        ray.inWater = false;

        // Check if camera ray starts inside water (matching CPU PathTracer initial medium setup)
        if (isInWater(scene, ray.origin)) {
            ray.inWater = true;
            ray.currentIor = scene.waterIor;
        }

        float3 color = (float3)(0.0f);
        float alpha = 1.0f;
        float3 cameraOrigin = ray.origin;
        float3 cameraDirection = ray.direction;

        // Find first intersection (depth 0)
        IntersectionRecord firstRecord = IntersectionRecord_new();
        MaterialSample firstSample;
        Material firstMaterial;
        bool firstHit = closestIntersect(scene, textureAtlas, ray, &firstRecord, &firstSample, &firstMaterial);
        if (firstHit) {
            float3 hitPos0 = ray.origin + ray.direction * firstRecord.distance;
            applyBiomeTint(scene, &firstSample, hitPos0);
        }

        if (!firstHit) {
            // Sky hit at depth 0
            if (scene.transparentSky) {
                // CPU: a transparent-sky depth-0 miss does NOTHING — ray.color
                // stays (0,0,0); no sky and no fog are added. The host blend reads
                // only RGB (alpha is tracked separately), so RGB must be black.
                alpha = 0.0f;
                color = (float3)(0.0f);
            } else {
            MaterialSample skySample;
            intersectSky(skyTexture, cachedSkyIntensity, sun, textureAtlas, ray, &skySample, sunDiffuseSun);
            color = skySample.emittance * skySample.color.xyz;

            // CPU: when camera is in water and ray doesn't hit anything,
            // the result is black (full absorption). Matches ray.color.set(0,0,0,1).
            if (ray.inWater) {
                color = (float3)(0.0f);
            }

            if (fogConfig.mode != FOG_MODE_NONE) {
                // CPU: uniform sky fog passes null scatterLight (no sun inscatter).
                // CPU: layered sky fog samples a scatter offset and traces a sun shadow ray.
                float3 skyFogSunAtt = (float3)(0.0f);
                float skyFogSunInt = 0.0f;
                if (fogConfig.mode == FOG_MODE_LAYERED) {
                    float skyScatterOff = Fog_sampleSkyScatterOffset(fogConfig, ray.origin, ray.direction, scene.yMin, scene.yMax, random);
                    float3 skyScatterPos = ray.origin + ray.direction * skyScatterOff;
                    skyFogSunAtt = getDirectLightAttenuation(scene, textureAtlas,
                        skyScatterPos, sun.sw, FOG_LIMIT, strictDirectLight);
                    // CPU fog inscatter is scaled by the sun-visibility alpha
                    // (scatterLight.w), never by sun intensity. That alpha is
                    // already folded into skyFogSunAtt, so the scale is 1.0.
                    skyFogSunInt = 1.0f;
                }
                Fog_addSkyFog(fogConfig, &color, ray.origin, ray.direction,
                              skyFogSunAtt, skyFogSunInt);
            }
            } // end else (!transparentSky)

            // Sky path doesn't enter the branch loop, but avgColor divides by
            // iterations*branchCount.  Scale up so the average is correct.
            color *= (float)branchCount;
        } else {
            // First surface hit - branch count reuses this first intersection
            // Track first-hit air/water distance (shared across all branches).
            // When the camera is outside the octree, subtract the empty-space
            // distance so fog only accounts for in-scene travel, not the void
            // between camera and loaded chunks.
            float emptySpaceDist = 0.0f;
            if (!AABB_inside(scene.octree.bounds, ray.origin)) {
                float3 invDFog = select(1.0f / ray.direction, copysign((float3)(1e30f), ray.direction), fabs(ray.direction) < 1e-30f);
                float dEntry = AABB_quick_intersect(scene.octree.bounds, ray.origin, invDFog);
                if (!isnan(dEntry) && dEntry > 0) emptySpaceDist = dEntry;
            }
            float firstAirDist = 0.0f;
            float firstWaterDist = 0.0f;
            if (ray.inWater) {
                firstWaterDist = fmax(0.0f, firstRecord.distance - emptySpaceDist);
            } else {
                firstAirDist = fmax(0.0f, firstRecord.distance - emptySpaceDist);
            }

            // Branch count loop: reuse same first intersection with different random decisions
            int effectiveBranches = branchCount;
            for (int branch = 0; branch < effectiveBranches; branch++) {
                // Each branch gets its own continuation from the first hit
                Ray bRay = ray;
                float3 throughput = (float3)(1.0f);
                float3 branchColor = (float3)(0.0f);
                float totalWaterDistance = firstWaterDist;
                bool hitAnything = true;
                bool lastWasSpecular = false;  // tracks whether the last bounce was specular (for sky fog)
                IntersectionRecord record = firstRecord;
                MaterialSample sample = firstSample;
                Material material = firstMaterial;

                // CPU PathTracer.java:126 checks `ray.depth + 1 >= rayDepth`
                // BEFORE the bounce, so a rayDepth setting of N produces only
                // N-1 actual NEE-counted bounces (the would-be N-th call enters
                // pathTrace, hits the depth check, and returns black). Use the
                // same effective bound here to match CPU sample contributions
                // bounce-for-bounce.
                int effectiveDepth = cachedRayDepth - 1;
                if (effectiveDepth < 0) effectiveDepth = 0;
                for (int depth = 0; depth < effectiveDepth; depth++) {
                    // For depth > 0, find next intersection
                    if (depth > 0) {
                        record = IntersectionRecord_new();
                        if (!closestIntersect(scene, textureAtlas, bRay, &record, &sample, &material)) {
                            // Sky hit at depth > 0

                            // CPU: when ray is in water and escapes to sky without
                            // hitting anything, result is black (full absorption).
                            if (bRay.inWater) {
                                throughput = (float3)(0.0f);
                            }

                            intersectSky(skyTexture, cachedSkyIntensity, sun, textureAtlas, bRay, &sample, sunDiffuseSun);
                            throughput *= sample.color.xyz;
                            branchColor += sample.emittance * throughput;

                            // CPU: only apply sky fog on specular sky hits (not diffuse).
                            // Diffuse sky hits skip sky fog entirely.
                            if (fogConfig.mode != FOG_MODE_NONE && lastWasSpecular) {
                                float3 skyFogSunAtt2 = (float3)(0.0f);
                                float skyFogSunInt2 = 0.0f;
                                if (fogConfig.mode == FOG_MODE_LAYERED) {
                                    float skyScatterOff2 = Fog_sampleSkyScatterOffset(fogConfig, bRay.origin, bRay.direction, scene.yMin, scene.yMax, random);
                                    float3 skyScatterPos2 = bRay.origin + bRay.direction * skyScatterOff2;
                                    skyFogSunAtt2 = getDirectLightAttenuation(scene, textureAtlas,
                                        skyScatterPos2, sun.sw, FOG_LIMIT, strictDirectLight);
                                    // Inscatter scale is 1.0 (the sun-visibility
                                    // alpha is already in skyFogSunAtt2), not sun
                                    // intensity — matching CPU Fog.addLayeredFog.
                                    skyFogSunInt2 = 1.0f;
                                }
                                Fog_addSkyFog(fogConfig, &branchColor, bRay.origin, bRay.direction,
                                              skyFogSunAtt2, skyFogSunInt2);
                            }
                            break;
                        }

                        // Track water distance (for Beer's-law water fog). Air/
                        // atmospheric fog uses only the first camera->surface
                        // segment (firstAirDist), matching CPU which fogs per
                        // frame over ray.distance — NOT an accumulated multi-bounce
                        // total, which over-darkened diffuse-GI scenes.
                        if (bRay.inWater) {
                            totalWaterDistance += record.distance;
                        }
                    }

                    // Compute hit position once per bounce (reused for biome tint, emitter, sun)
                    float3 hitPos = bRay.origin + bRay.direction * record.distance;

                    // Apply biome tinting (depth > 0 only; depth 0 was already tinted)
                    if (depth > 0) {
                        applyBiomeTint(scene, &sample, hitPos);
                    }

                    // Apply water fog attenuation (Beer's law, matching CPU).
                    if (totalWaterDistance > 0.0f) {
                        float fogAtt = Water_fogAttenuation(totalWaterDistance, scene.waterVisibility);
                        throughput *= fogAtt;
                        totalWaterDistance = 0.0f;
                    }

                    MaterialPdfSample pdfSample = Material_samplePdf(material, record, sample, bRay, random,
                        fancierTranslucency, transmissivityCap);
                    lastWasSpecular = pdfSample.specular;
                    float3 prevThroughput = throughput;
                    throughput *= pdfSample.spectrum;

                    // Importance sampling: steer diffuse bounces toward the sun
                    if (sunIsImportance
                        && !pdfSample.specular && !pdfSample.transmitted) {
                        DiffuseISResult isResult = _Material_diffuseReflectionIS(
                            record, sun.sw,
                            sun.importanceSampleChance, sun.importanceSampleRadius,
                            sun.radius, random);
                        pdfSample.direction = isResult.direction;
                        // Geometric normal correction for importance-sampled direction
                        if (sign(dot(record.geomNormal, pdfSample.direction)) == sign(dot(record.geomNormal, bRay.direction))) {
                            float factor = copysign(1.0f, dot(record.geomNormal, bRay.direction)) * (-EPS) - dot(pdfSample.direction, record.geomNormal);
                            pdfSample.direction += factor * record.geomNormal;
                            pdfSample.direction = normalize(pdfSample.direction);
                        }
                        throughput *= isResult.throughputScale;
                    }

                    // Emitter self-emission. CPU adds emittance ONLY inside
                    // doDiffuseReflection, so gate on a genuine diffuse bounce
                    // (!specular && !transmitted), matching the sun-NEE gate.
                    bool didSelfEmit = false;
                    if (emittersEnabled && sample.emittance > EPS
                        && !pdfSample.specular && !pdfSample.transmitted
                        && (!preventNormalEmitterWithSampling || samplingStrategy == 0 || depth == 0)) {
                        float3 emColor = (float3)(sample.color.x * sample.color.x,
                                                  sample.color.y * sample.color.y,
                                                  sample.color.z * sample.color.z);
                        branchColor += emColor * sample.emittance * cachedEmitterIntensity * prevThroughput;
                        didSelfEmit = true;
                    }

                    // Emitter-grid NEE. CPU does this only on genuine diffuse
                    // bounces, and ONLY when self-emission did not fire this bounce
                    // (if/else-if in doDiffuseReflection). Gate: !specular &&
                    // !transmitted && !didSelfEmit.
                    if (emittersEnabled && samplingStrategy != 0 && gc_sizeX > 0
                        && !pdfSample.specular && !pdfSample.transmitted && !didSelfEmit) {
                        int cellSize = gc_cellSize;
                        int offsetX = gc_offsetX;
                        int sizeX = gc_sizeX;
                        int offsetY = gc_offsetY;
                        int sizeY = gc_sizeY;
                        int offsetZ = gc_offsetZ;
                        int sizeZ = gc_sizeZ;

                        // Select the emitter grid cell from the SURFACE hit point,
                        // not bRay.origin (which is the camera at depth 0 / the
                        // previous bounce otherwise). CPU uses ray.o = the advanced
                        // hit point. Using the ray start sampled emitters from the
                        // wrong cell (usually empty), under-lighting emitter scenes.
                        int gx = (int)floor(hitPos.x / (float)cellSize);
                        int gy = (int)floor(hitPos.y / (float)cellSize);
                        int gz = (int)floor(hitPos.z / (float)cellSize);

                        if (gx >= offsetX && gx < offsetX + sizeX && gy >= offsetY && gy < offsetY + sizeY && gz >= offsetZ && gz < offsetZ + sizeZ) {
                            int idx = (((gy - offsetY) * sizeX) + (gx - offsetX)) * sizeZ + (gz - offsetZ);
                            int eStart = constructedGrid[2*idx];
                            int eCount = constructedGrid[2*idx + 1];
                            if (eCount > 0) {
                                float3 hitPoint = hitPos + record.normal * OFFSET;
                                if (samplingStrategy == 1) {
                                    // ONE: pick one random emitter, sample a random face point
                                    unsigned int r = Random_nextState(random);
                                    int ri = (int)(r % eCount);
                                    int emitterIndex = positionIndexes[eStart + ri];
                                    int ex = emitterPositions[emitterIndex*4 + 0];
                                    int ey = emitterPositions[emitterIndex*4 + 1];
                                    int ez = emitterPositions[emitterIndex*4 + 2];
                                    float avgFaceArea = as_float(emitterPositions[emitterIndex*4 + 3]);
                                    unsigned int faceR = Random_nextState(random);
                                    int face = (int)(faceR % 6);
                                    float ru = Random_nextFloat(random);
                                    float rv = Random_nextFloat(random);
                                    float3 epos;
                                    switch (face) {
                                        case 0: epos = (float3)(ex, ey + ru, ez + rv); break;
                                        case 1: epos = (float3)(ex + 1.0f, ey + ru, ez + rv); break;
                                        case 2: epos = (float3)(ex + ru, ey, ez + rv); break;
                                        case 3: epos = (float3)(ex + ru, ey + 1.0f, ez + rv); break;
                                        case 4: epos = (float3)(ex + ru, ey + rv, ez); break;
                                        default: epos = (float3)(ex + ru, ey + rv, ez + 1.0f); break;
                                    }
                                    float3 toEmitter = epos - hitPoint;
                                    float dist = length(toEmitter) + 1e-6f;
                                    float3 dirToEmitter = toEmitter / dist;
                                    if (dot(record.normal, dirToEmitter) > 0.0f) {
                                        Ray shadow = bRay;
                                        shadow.origin = hitPoint;
                                        shadow.direction = dirToEmitter;
                                        shadow.flags |= RAY_SHADOW;
                                        IntersectionRecord srec = IntersectionRecord_new();
                                        srec.distance = dist;
                                        Material sSample;
                                        MaterialSample sMatSample;
                                        bool shadowClear = !closestIntersect(scene, textureAtlas, shadow, &srec, &sMatSample, &sSample);
                                        if (shadowClear || srec.distance >= dist - 1e-4f) {
                                            if (shadowClear) {
                                                // Emitter invisible to rays (light block): look up material from octree
                                                int bd = Octree_get(&scene.octree, ex, ey, ez);
                                                if (bd > 0) {
                                                    Material em = Material_get(scene.materialPalette, scene.blockPalette.blockPalette[bd + 1]);
                                                    Material_sample(em, textureAtlas, (float2)(ru, rv), &sMatSample);
                                                }
                                                srec.normal = (face < 2) ? (float3)(face == 0 ? -1.0f : 1.0f, 0, 0)
                                                            : (face < 4) ? (float3)(0, face == 2 ? -1.0f : 1.0f, 0)
                                                                         : (float3)(0, 0, face == 4 ? -1.0f : 1.0f);
                                            }
                                            float cosEmitter = fabs(dot(dirToEmitter, srec.normal));
                                            float att = cosEmitter / fmax(dist * dist, 1.0f); // CPU emitter NEE has no receiving-cosine weight (only the visibility gate)
                                            att *= avgFaceArea;
                                            // CPU emitter NEE uses the emitter's LINEAR color (sampleEmitterFace
                                            // scaleAdds the raw color); the quadratic color^2 mapping is for
                                            // direct self-emission ONLY. Squaring here darkened colored emitters.
                                            float3 emitterCol = sMatSample.color.xyz * sMatSample.emittance;
                                            branchColor += throughput * cachedEmitterIntensity * emitterCol * att * (float)M_PI_F;
                                        }
                                    }
                                } else if (samplingStrategy == 2) {
                                    // ONE_BLOCK: pick one random emitter, sample all 6 faces
                                    unsigned int r = Random_nextState(random);
                                    int ri = (int)(r % eCount);
                                    int emitterIndex = positionIndexes[eStart + ri];
                                    int ex = emitterPositions[emitterIndex*4 + 0];
                                    int ey = emitterPositions[emitterIndex*4 + 1];
                                    int ez = emitterPositions[emitterIndex*4 + 2];
                                    float avgFaceArea = as_float(emitterPositions[emitterIndex*4 + 3]);
                                    float3 accCol = (float3)(0.0f);
                                    for (int face = 0; face < 6; face++) {
                                        float ru = Random_nextFloat(random);
                                        float rv = Random_nextFloat(random);
                                        float3 epos;
                                        switch (face) {
                                            case 0: epos = (float3)(ex, ey + ru, ez + rv); break;
                                            case 1: epos = (float3)(ex + 1.0f, ey + ru, ez + rv); break;
                                            case 2: epos = (float3)(ex + ru, ey, ez + rv); break;
                                            case 3: epos = (float3)(ex + ru, ey + 1.0f, ez + rv); break;
                                            case 4: epos = (float3)(ex + ru, ey + rv, ez); break;
                                            default: epos = (float3)(ex + ru, ey + rv, ez + 1.0f); break;
                                        }
                                        float3 toEmitter = epos - hitPoint;
                                        float dist = length(toEmitter) + 1e-6f;
                                        float3 dirToEmitter = toEmitter / dist;
                                        if (dot(record.normal, dirToEmitter) <= 0.0f) continue;
                                        Ray shadow = bRay;
                                        shadow.origin = hitPoint;
                                        shadow.direction = dirToEmitter;
                                        shadow.flags |= RAY_SHADOW;
                                        IntersectionRecord srec = IntersectionRecord_new();
                                        srec.distance = dist;
                                        Material sSample;
                                        MaterialSample sMatSample;
                                        bool shadowClear = !closestIntersect(scene, textureAtlas, shadow, &srec, &sMatSample, &sSample);
                                        if (shadowClear || srec.distance >= dist - 1e-4f) {
                                            if (shadowClear) {
                                                int bd = Octree_get(&scene.octree, ex, ey, ez);
                                                if (bd > 0) {
                                                    Material em = Material_get(scene.materialPalette, scene.blockPalette.blockPalette[bd + 1]);
                                                    Material_sample(em, textureAtlas, (float2)(ru, rv), &sMatSample);
                                                }
                                                srec.normal = (face < 2) ? (float3)(face == 0 ? -1.0f : 1.0f, 0, 0)
                                                            : (face < 4) ? (float3)(0, face == 2 ? -1.0f : 1.0f, 0)
                                                                         : (float3)(0, 0, face == 4 ? -1.0f : 1.0f);
                                            }
                                            float cosEmitter = fabs(dot(dirToEmitter, srec.normal));
                                            float att = cosEmitter / fmax(dist * dist, 1.0f); // CPU emitter NEE has no receiving-cosine weight (only the visibility gate)
                                            att *= avgFaceArea;
                                            float3 emitterCol = sMatSample.color.xyz * sMatSample.emittance; // linear (not color^2)
                                            accCol += emitterCol * att;
                                        }
                                    }
                                    branchColor += throughput * cachedEmitterIntensity * accCol * (1.0f / 6.0f) * (float)M_PI_F;
                                } else if (samplingStrategy == 3) {
                                    // ALL: iterate every emitter in the grid cell, sample all 6 faces each.
                                    float3 accCol = (float3)(0.0f, 0.0f, 0.0f);
                                    for (int si = 0; si < eCount; ++si) {
                                        int emitterIndex = positionIndexes[eStart + si];
                                        int ex = emitterPositions[emitterIndex*4 + 0];
                                        int ey = emitterPositions[emitterIndex*4 + 1];
                                        int ez = emitterPositions[emitterIndex*4 + 2];
                                        float avgFaceArea = as_float(emitterPositions[emitterIndex*4 + 3]);
                                        for (int face = 0; face < 6; face++) {
                                            float ru = Random_nextFloat(random);
                                            float rv = Random_nextFloat(random);
                                            float3 epos;
                                            switch (face) {
                                                case 0: epos = (float3)(ex, ey + ru, ez + rv); break;
                                                case 1: epos = (float3)(ex + 1.0f, ey + ru, ez + rv); break;
                                                case 2: epos = (float3)(ex + ru, ey, ez + rv); break;
                                                case 3: epos = (float3)(ex + ru, ey + 1.0f, ez + rv); break;
                                                case 4: epos = (float3)(ex + ru, ey + rv, ez); break;
                                                default: epos = (float3)(ex + ru, ey + rv, ez + 1.0f); break;
                                            }
                                            float3 toEmitter = epos - hitPoint;
                                            float dist = length(toEmitter) + 1e-6f;
                                            float3 dirToEmitter = toEmitter / dist;
                                            if (dot(record.normal, dirToEmitter) <= 0.0f) continue;
                                            Ray shadow = bRay;
                                            shadow.origin = hitPoint;
                                            shadow.direction = dirToEmitter;
                                            shadow.flags |= RAY_SHADOW;
                                            IntersectionRecord srec = IntersectionRecord_new();
                                            srec.distance = dist;
                                            Material sSample;
                                            MaterialSample sMatSample;
                                            bool shadowClear = !closestIntersect(scene, textureAtlas, shadow, &srec, &sMatSample, &sSample);
                                            if (shadowClear || srec.distance >= dist - 1e-4f) {
                                                if (shadowClear) {
                                                    int bd = Octree_get(&scene.octree, ex, ey, ez);
                                                    if (bd > 0) {
                                                        Material em = Material_get(scene.materialPalette, scene.blockPalette.blockPalette[bd + 1]);
                                                        Material_sample(em, textureAtlas, (float2)(ru, rv), &sMatSample);
                                                    }
                                                    srec.normal = (face < 2) ? (float3)(face == 0 ? -1.0f : 1.0f, 0, 0)
                                                                : (face < 4) ? (float3)(0, face == 2 ? -1.0f : 1.0f, 0)
                                                                             : (float3)(0, 0, face == 4 ? -1.0f : 1.0f);
                                                }
                                                float cosEmitter = fabs(dot(dirToEmitter, srec.normal));
                                                float att = cosEmitter / fmax(dist * dist, 1.0f); // CPU emitter NEE has no receiving-cosine weight (only the visibility gate)
                                                att *= avgFaceArea;
                                                float3 emitterCol = sMatSample.color.xyz * sMatSample.emittance; // linear (not color^2)
                                                accCol += emitterCol * att;
                                            }
                                        }
                                    }
                                    if (eCount > 0) branchColor += throughput * cachedEmitterIntensity * (accCol / (float)(eCount * 6)) * (float)M_PI_F;
                                }
                            }
                        }
                    }

                    // Sun direct light sampling. CPU does this ONLY inside
                    // doDiffuseReflection — never on specular reflections or on
                    // transmission/refraction. Gate on a true diffuse bounce
                    // (!specular && !transmitted) so rays passing straight through
                    // transparent texels don't pick up a spurious sun contribution.
                    // Fires only for strategies with doSunSampling=true (FAST,
                    // HIGH_QUALITY) per chunky's truth table.
                    if (sunDoSampling && !pdfSample.specular && !pdfSample.transmitted) {
                        Ray sunRay;
                        sunRay.origin = hitPos;
                        if (Sun_sampleDirection(sun, &sunRay, random)) {
                            float cosSun = dot(record.normal, sunRay.direction);
                            bool frontLight = cosSun > 0.0f;
                            // CPU PathTracer: backside sun sampling for SSS materials (30% chance)
                            if (frontLight || (sample.sss && Random_nextFloat(random) < F_SUBSURFACE)) {
                                float3 sunHitPoint;
                                if (frontLight) {
                                    sunHitPoint = hitPos + record.normal * OFFSET;
                                } else {
                                    sunHitPoint = hitPos - record.normal * OFFSET;
                                }
                                sunRay.origin = sunHitPoint;
                                float3 sunAtt = getDirectLightAttenuation(scene, textureAtlas,
                                    sunHitPoint, sunRay.direction, FOG_LIMIT, strictDirectLight);
                                if (sunAtt.x + sunAtt.y + sunAtt.z > 0.0f) {
                                    float mult = fabs(cosSun);
                                    // chunky CPU multiplies by 1/luminosity when
                                    // isSunLuminosity()==true. NEE only fires for
                                    // FAST and HIGH_QUALITY; of those FAST has
                                    // sunLuminosity=false (skip) and HIGH_QUALITY
                                    // has sunLuminosity=true (apply).
                                    if (sunIsLuminosity) {
                                        mult *= cachedSunLumInv;
                                    }
                                    float3 sunContrib = sun.color.xyz * cachedSunPower * mult * sunAtt;
                                    branchColor += prevThroughput * sample.color.xyz * sunContrib;
                                }
                            }
                        }
                    }

                    // Update ray position and direction
                    bRay.origin = bRay.origin + bRay.direction * (record.distance - OFFSET);
                    bRay.direction = pdfSample.direction;
                    bRay.origin += bRay.direction * OFFSET;

                    // Update medium tracking
                    if (pdfSample.transmitted) {
                        bRay.prevIor = bRay.currentIor;
                        bRay.currentIor = pdfSample.newIor;
                        bRay.inWater = sample.isWater ? !bRay.inWater : bRay.inWater;
                        if (pdfSample.newIor > AIR_IOR + EPS) {
                            bRay.material = record.blockData;
                        } else {
                            bRay.material = 0;
                        }
                    }

                    // Only DIFFUSE bounces become indirect. CPU doTransmission does
                    // next.set(ray), preserving ray.specular, so a camera/specular
                    // ray that passes straight through a semi-transparent texel
                    // still hits the sky as an APPARENT (specular) sun — not the
                    // ~60x brighter diffuse/luminosity sun. Refraction keeps
                    // specular=true so it is unaffected.
                    if (!pdfSample.specular && !pdfSample.transmitted) {
                        bRay.flags |= RAY_INDIRECT;
                    }

                    // CPU parity: chunky's PathTracer uses a hard depth cutoff,
                    // not Russian roulette. Termination happens via the depth
                    // loop bound (cachedRayDepth).
                    //
                    // Throughput early-exit: if all three channels have decayed
                    // below 1e-30f, any further bounce contributes effectively
                    // zero to branchColor (max possible add < threshold *
                    // emittance, which is far below fp32 precision for any
                    // reasonable accumulator value). Bypasses the closestIntersect
                    // for paths that can no longer contribute. The threshold
                    // is tight enough that the branchColor difference is
                    // sub-fp32-ULP — visually identical to the no-early-exit
                    // version.
                    if (fmax(fmax(throughput.x, throughput.y), throughput.z) < 1e-30f) {
                        break;
                    }
                } // end depth loop

                // Apply remaining water fog if ray ended while still in water
                if (totalWaterDistance > 0.0f) {
                    float fogAtt = Water_fogAttenuation(totalWaterDistance, scene.waterVisibility);
                    branchColor *= fogAtt;
                }

                // Apply ground fog. Use scene entry point as fog origin so
                // empty space between camera and octree doesn't inflate fog.
                //
                // Sun inscatter for fog runs UNCONDITIONALLY in chunky CPU
                // (PathTracer.java ~line 192-200) — the strategy gate only
                // controls direct NEE, not the fog-light approximation. We
                // previously gated this on `sunSamplingStrategy != OFF`,
                // which made fog look brighter on GPU (no sun-occlusion
                // attenuation) than on CPU when the user set strategy to
                // OFF. Match CPU: always compute the attenuation.
                // Atmospheric fog distance = first camera->surface segment. CPU
                // applies it whether that segment is air OR water (prevMat==Air
                // || prevMat.isWater()), so include firstWaterDist for the
                // camera-underwater case (exactly one of the two is nonzero).
                float groundFogDist = firstAirDist + firstWaterDist;
                if (fogConfig.mode != FOG_MODE_NONE && groundFogDist > 0.0f) {
                    float3 fogOrigin = cameraOrigin + cameraDirection * emptySpaceDist;
                    float fogOffset = Fog_sampleScatterOffset(fogConfig, groundFogDist, fogOrigin, cameraDirection, random);
                    float3 sunDir = sun.sw;
                    // CPU Fog.addGroundFog scales inscatter by the sun-visibility
                    // alpha (scatterLight.w ≤ 1), NOT by sun intensity. That alpha
                    // is already folded into fogSunAtt below, so the scalar is 1.0.
                    // (Previously sun.intensity, making god rays 1.25x–50x too bright.)
                    float sunInt = 1.0f;
                    float3 fogSamplePos = fogOrigin + cameraDirection * fogOffset;
                    float3 fogSunAtt = getDirectLightAttenuation(scene, textureAtlas,
                        fogSamplePos, sunDir, FOG_LIMIT, strictDirectLight);
                    Fog_addGroundFog(fogConfig, &branchColor, fogOrigin, cameraDirection,
                                     groundFogDist, fogSunAtt, sunInt, fogOffset);
                }

                color += branchColor;
            } // end branch loop

        }

        sumColor += color;
        sumAlpha += alpha;
    }

    // Each iteration produced branchCount samples, so total samples = iterations * branchCount
    float3 avgColor = sumColor / (float)(iterations * branchCount);
    float avgAlpha = sumAlpha / (float)iterations;

    // Store color + alpha (4 floats per pixel).
    // Alpha = 1 for opaque hits, 0 for transparent sky hits at depth 0.
    vstore4((float4)(avgColor, avgAlpha), gid, res);
    } // end grid-stride loop
}

__kernel void preview(
    __constant const int* projectorType,
    __global const float* cameraSettings,

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

    image2d_t skyTexture,
    __constant const float* skyIntensity,
    __constant const int* sunData,

    __constant const int* canvasConfig,
    __constant const float* waterConfig,
    __global const int* chunkBitmap,
    int chunkBitmapSize,
    __global const int* biomeData,
    int biomeDataSize,
    int biomeYLevels,
    __constant const float* renderConfig,
    __global const int* cloudData,
    __global int* res
) {
    int gid = get_global_id(0);

    int px = gid % canvasConfig[0] + canvasConfig[4];
    int py = gid / canvasConfig[0] + canvasConfig[5];

    // Crosshairs
    if ((px == canvasConfig[2] / 2 && (py >= canvasConfig[3] / 2 - 5 && py <= canvasConfig[3] / 2 + 5)) ||
        (py == canvasConfig[3] / 2 && (px >= canvasConfig[2] / 2 - 5 && px <= canvasConfig[2] / 2 + 5))) {
        res[gid] = 0xFFFFFFFF;
        return;
    }

    SceneConfig scene;
    scene.materialPalette = MaterialPalette_new(matPalette, 0, NULL);
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

    scene.waterPlaneEnabled = (waterConfig[0] > 0.5f);
    scene.waterPlaneHeight = waterConfig[1];
    scene.waterPlaneChunkClip = (waterConfig[2] > 0.5f);
    scene.octreeSize = waterConfig[3];
    scene.waterColor = (float3)(waterConfig[7], waterConfig[8], waterConfig[9]);
    scene.useCustomWaterColor = (waterConfig[10] > 0.5f);
    scene.waterIor = waterConfig[11];
    scene.waterOpacity = waterConfig[16];
    scene.chunkBitmap = chunkBitmap;
    scene.chunkBitmapSize = chunkBitmapSize;
    scene.biomeColorsEnabled = (biomeDataSize > 0);
    scene.biomeData = biomeData;
    scene.biomeDataSize = biomeDataSize;
    scene.biomeYLevels = max(1, biomeYLevels);
    // Clouds — the CPU preview renders them too. Cloud config lives in
    // renderConfig[3,4,5,11,12] (same indices the main render kernel uses).
    scene.cloudsEnabled = (renderConfig[3] > 0.5f);
    scene.cloudHeight = renderConfig[4];
    scene.cloudSize = renderConfig[5];
    scene.cloudOffsetX = renderConfig[11];
    scene.cloudOffsetZ = renderConfig[12];
    scene.cloudData = cloudData;

    Sun sun = Sun_new(sunData);

    unsigned int randomState = 0;
    Random random = &randomState;
    Random_nextState(random);

    CameraCache camCache = CameraCache_load(projectorType, cameraSettings, canvasConfig);
    Ray ray = ray_to_camera(camCache, cameraSettings, NULL, 0, gid, random);

    IntersectionRecord record = IntersectionRecord_new();
    MaterialSample sample;
    Material material;

    ray.material = 0;
    ray.flags = RAY_PREVIEW;
    ray.currentIor = AIR_IOR;
    ray.prevIor = AIR_IOR;
    ray.inWater = false;

    // Check if camera is underwater (matching CPU PreviewRayTracer initial medium setup)
    if (isInWater(scene, ray.origin)) {
        ray.inWater = true;
        ray.currentIor = scene.waterIor;
    }

    float3 color = (float3)(0.0f);
    bool hitSolid = false;

    // Front-to-back alpha accumulators for semi-transparent layers (water, stained glass).
    // C_out += (1 - alphaAccum) * layerAlpha * layerColor at each layer.
    float3 tintAccum = (float3)(0.0f);
    float alphaAccum = 0.0f;

    // Transparency loop with alpha blending.  Fully transparent (alpha==0) hits are
    // skipped, semi-transparent hits are composited as tint layers, opaque hits stop.
    for (int skipIter = 0; skipIter < 8; skipIter++) {
        record = IntersectionRecord_new();
        if (!previewIntersect(scene, textureAtlas, ray, &record, &sample, &material)) {
            break;
        }

        if (sample.color.w <= EPS) {
            // Fully transparent: skip past
            ray.origin = ray.origin + ray.direction * (record.distance + OFFSET);
            ray.material = record.blockData;
            // Toggle water state when passing through water surface
            if (sample.isWater) ray.inWater = !ray.inWater;
            continue;
        }

        // Apply biome tint and flat shading
        float3 previewHitPos = ray.origin + ray.direction * record.distance;
        applyBiomeTint(scene, &sample, previewHitPos);
        // CPU Sun.flatShading multiplies albedo by previewEmittance * shading,
        // where previewEmittance = pow(DEFAULT_INTENSITY=1.25, DEFAULT_GAMMA) is a
        // CONSTANT (NOT the live sun intensity). Omitting it made the GPU preview
        // ~1.6x darker (linear) than the CPU preview.
        float shading = fmax(0.3f, dot(record.normal, sun.sw));
        float3 hitColor = sample.color.xyz * shading * pow(1.25f, DEFAULT_GAMMA);

        if (sample.color.w < 1.0f - EPS) {
            // Semi-transparent: accumulate as tint layer (front-to-back compositing)
            float layerAlpha = sample.color.w;
            float remaining = 1.0f - alphaAccum;
            tintAccum += hitColor * layerAlpha * remaining;
            alphaAccum += layerAlpha * remaining;

            if (alphaAccum >= 0.99f) {
                color = tintAccum;
                hitSolid = true;
                break;
            }

            ray.origin = ray.origin + ray.direction * (record.distance + OFFSET);
            ray.material = record.blockData;
            // Toggle water state when passing through water surface
            if (sample.isWater) ray.inWater = !ray.inWater;
            continue;
        }

        // Opaque hit
        color = hitColor;
        hitSolid = true;
        break;
    }

    // Composite accumulated semi-transparent layers over opaque background
    if (alphaAccum > 0.0f && hitSolid) {
        color = tintAccum + color * (1.0f - alphaAccum);
    }

    if (!hitSolid) {
        // No opaque hit — resolve background (floor grid or sky)
        float3 bgColor = (float3)(0.0f);
        bool gridHit = false;
        if (ray.direction.y < 0) {
            float gridY = 0.0f;
            float gt = (gridY - ray.origin.y) / ray.direction.y;
            if (gt > OFFSET) {
                float3 hitP = ray.origin + ray.direction * gt;
                bool isSubmerged = scene.waterPlaneEnabled;
                int octSize = 1 << (*octreeDepth);
                bool insideOctree = (hitP.x >= 0 && hitP.x <= octSize &&
                                     hitP.z >= 0 && hitP.z <= octSize);
                float xm = fmod(fmod(hitP.x, 16.0f) + 24.0f, 16.0f);
                float zm = fmod(fmod(hitP.z, 16.0f) + 24.0f, 16.0f);
                float linePos = 7.75f;
                float lineEnd = 8.25f;
                bool isLine = (xm >= linePos && xm <= lineEnd) || (zm >= linePos && zm <= lineEnd);
                if (isLine) {
                    bgColor = isSubmerged ? (float3)(0.05f, 0.05f, 0.25f) : (float3)(0.25f, 0.25f, 0.25f);
                } else {
                    bgColor = isSubmerged ? (float3)(0.6f, 0.6f, 0.8f) : (float3)(0.8f, 0.8f, 0.8f);
                }
                if (insideOctree) {
                    bgColor *= 0.75f;
                }
                float gridShading = fmax(0.3f, sun.sw.y); // grid normal is (0,1,0)
                bgColor *= gridShading * pow(1.25f, DEFAULT_GAMMA); // previewEmittance (see surface shading)
                gridHit = true;
            }
        }
        if (!gridHit) {
            intersectSky(skyTexture, *skyIntensity, sun, textureAtlas, ray, &sample, true);
            bgColor = sample.color.xyz;
        }

        // Blend any accumulated semi-transparent layers over the background
        if (alphaAccum > 0.0f) {
            color = tintAccum + bgColor * (1.0f - alphaAccum);
        } else {
            color = bgColor;
        }
    }

    color = sqrt(color);
    int3 rgb = intFloorFloat3(clamp(color * 255.0f, 0.0f, 255.0f));
    res[gid] = 0xFF000000 | (rgb.x << 16) | (rgb.y << 8) | rgb.z;
}

// ---- 2D Map: nearest-neighbour upscale kernel ----
// One work-item per destination pixel.
// Replicates the CPU MapBuffer.drawBuffered() scaling logic on the GPU.
__kernel void mapScale(
    __global const int* src,   // source pixel buffer (chunk-scale)
    __global int* dst,         // destination pixel buffer (screen-scale)
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    float scale,               // view.scale / view.chunkScale
    int srcOffsetX,
    int srcOffsetZ
) {
    int gid = get_global_id(0);
    if (gid >= dstWidth * dstHeight) return;

    int dstX = gid % dstWidth;
    int dstY = gid / dstWidth;

    // Map destination pixel back to source coordinates (nearest-neighbour)
    int srcX = srcOffsetX + (int)(dstX / scale) + 1;
    int srcY = srcOffsetZ + (int)(dstY / scale);

    // Bounds check
    int srcIdx = srcY * srcWidth + srcX;
    if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight
            || srcIdx < 0 || srcIdx >= srcWidth * srcHeight) {
        dst[gid] = 0;
        return;
    }

    dst[gid] = src[srcIdx];
}
