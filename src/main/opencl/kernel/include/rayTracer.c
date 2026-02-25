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

// Check if a ray origin is inside water, matching CPU Scene.isInWater().
// Checks water plane first, then the water octree.
bool isInWater(SceneConfig scene, float3 origin) {
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
    int x = (int)floor(origin.x);
    int y = (int)floor(origin.y);
    int z = (int)floor(origin.z);
    int waterBlock = Octree_get(&scene.waterOctree, x, y, z);
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

// Sun sampling strategies
#define SUN_SAMPLING_OFF         0
#define SUN_SAMPLING_NON_LUMINOUS 1
#define SUN_SAMPLING_FAST        2
#define SUN_SAMPLING_IMPORTANCE  3
#define SUN_SAMPLING_HIGH_QUALITY 4

Ray ray_to_camera(
        const __global int* projectorType,
        const __global float* cameraSettings,
        const __global int* apertureMask,
        int apertureMaskWidth,
        const __global int* canvasConfig,
        int gid,
        Random random
) {
    Ray ray;
    if (*projectorType != -1) {
        float3 cameraPos = vload3(0, cameraSettings);
        float3 m1s = vload3(1, cameraSettings);
        float3 m2s = vload3(2, cameraSettings);
        float3 m3s = vload3(3, cameraSettings);

        int width = canvasConfig[0];
        int height = canvasConfig[1];
        int fullWidth = canvasConfig[2];
        int fullHeight = canvasConfig[3];
        int cropX = canvasConfig[4];
        int cropY = canvasConfig[5];

        float halfWidth = fullWidth / (2.0f * fullHeight);
        float invHeight = 1.0f / fullHeight;
        float x = -halfWidth + ((gid % width) + Random_nextFloat(random) + cropX) * invHeight;
        float y = -0.5f + ((gid / width) + Random_nextFloat(random) + cropY) * invHeight;

        // Camera shift (lens shift / image plane offset) at positions [12] and [13]
        float shiftX = cameraSettings[12];
        float shiftY = cameraSettings[13];
        x += shiftX;
        y -= shiftY;

        __global const float* projSettings = cameraSettings + 14;

        switch (*projectorType) {
            case 0:
                ray = Camera_pinHole(x, y, random, projSettings, apertureMask, apertureMaskWidth);
                break;
            case 1:
                ray = Camera_parallel(x, y, random, projSettings, apertureMask, apertureMaskWidth);
                break;
            case 2:
                ray = Camera_fisheye(x, y, random, projSettings, apertureMask, apertureMaskWidth);
                break;
            case 3:
                ray = Camera_stereographic(x, y, random, projSettings, apertureMask, apertureMaskWidth);
                break;
            case 4:
                ray = Camera_panoramic(x, y, random, projSettings, apertureMask, apertureMaskWidth);
                break;
            case 5:
                ray = Camera_panoramicSlot(x, y, random, projSettings, apertureMask, apertureMaskWidth);
                break;
            case 6:
                ray = Camera_ODS(x, y, random, projSettings);
                break;
            case 7:
                ray = Camera_ODSStacked(x, y, random, projSettings);
                break;
            default:
                ray = Camera_pinHole(x, y, random, projSettings, apertureMask, apertureMaskWidth);
                break;
        }

        ray.direction = normalize((float3) (
                dot(m1s, ray.direction),
                        dot(m2s, ray.direction),
                        dot(m3s, ray.direction)
        ));
        ray.origin = (float3) (
                dot(m1s, ray.origin),
                        dot(m2s, ray.origin),
                        dot(m3s, ray.origin)
        );

        ray.origin += cameraPos;
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
    shadow.flags = 0;
    shadow.currentIor = AIR_IOR;
    shadow.prevIor = AIR_IOR;
    shadow.inWater = isInWater(scene, origin);
    if (shadow.inWater) {
        shadow.currentIor = scene.waterIor;
    }

    for (int i = 0; i < 8; i++) {
        if (alphaAtt <= 0.0f) break;

        IntersectionRecord srec = IntersectionRecord_new();
        srec.distance = maxDist;
        MaterialSample sMat;
        Material sSample;

        if (!closestIntersect(scene, textureAtlas, shadow, &srec, &sMat, &sSample)) {
            break; // Clear path to sun
        }

        // Apply biome tinting for shadow ray materials (e.g. leaves)
        {
            float3 sHitPos = shadow.origin + shadow.direction * srec.distance;
            applyBiomeTint(scene, &sMat, sHitPos);
        }

        if (srec.distance >= maxDist - OFFSET) {
            break; // Past the target distance
        }

        // Opaque material blocks the sun
        if (sMat.color.w > 1.0f - EPS && !sMat.refractive) {
            return (float3)(0.0f);
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
    __global const int* projectorType,
    __global const float* cameraSettings,
    __global const int* apertureMask,
    int apertureMaskWidth,

    __global const int* octreeDepth,
    __global const int* octreeData,

    __global const int* waterOctreeDepth,
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

    __global const float* skyIntensity,
    __global const int* sunData,

    __global const int* dynamicConfig,
    __global const float* emitterIntensity,
    __global const int* emitterPositions,
    __global const int* positionIndexes,
    __global const int* constructedGrid,
    __global const int* gridConfig,
    __global const int* canvasConfig,
    __global const int* rayDepth,
    int iterations,

    // New buffers for expanded features
    __global const float* fogData,
    __global const float* waterConfig,
    __global const float* renderConfig,
    __global const int* cloudData,
    __global const float* waterNormalMap,
    int waterNormalMapW,
    __global const int* biomeData,
    int biomeDataSize,
    __global const int* chunkBitmap,
    int chunkBitmapSize,

    __global float* res

) {
    int gid = get_global_id(0);

    // Cooperative copy of material palette ints into per-work-group local cache.
    if (matCacheWords > 0) {
        int lid = get_local_id(0);
        int lsize = get_local_size(0);
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
    scene.drawDepth = 256;

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
    //                       fancierTranslucency, transmissivityCap, biomeColorsEnabled]
    int sunSamplingStrategy = (int)renderConfig[0];
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
    scene.chunkBitmap = chunkBitmap;
    scene.chunkBitmapSize = chunkBitmapSize;

    // Load fog config
    FogConfig fogConfig = Fog_load(fogData);

    Sun sun = Sun_new(sunData);

    float3 sumColor = (float3)(0.0f, 0.0f, 0.0f);
    float sumAlpha = 0.0f;

    for (int it = 0; it < iterations; ++it) {
        unsigned int randomState = dynamicConfig[0] + (unsigned int)(gid * 1664525u) + (unsigned int)(it * 1013904223u);
        Random random = &randomState;
        Random_nextState(random);
        Random_nextState(random);
        Ray ray = ray_to_camera(projectorType, cameraSettings, apertureMask, apertureMaskWidth, canvasConfig, gid, random);

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
                alpha = 0.0f;
            }
            MaterialSample skySample;
            bool diffuseSun = (sunSamplingStrategy == SUN_SAMPLING_OFF ||
                               sunSamplingStrategy == SUN_SAMPLING_IMPORTANCE);
            intersectSky(skyTexture, *skyIntensity, sun, textureAtlas, ray, &skySample, diffuseSun);
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
                    float skyScatterOff = Fog_sampleSkyScatterOffset(fogConfig, ray.origin, ray.direction, random);
                    float3 skyScatterPos = ray.origin + ray.direction * skyScatterOff;
                    skyFogSunAtt = getDirectLightAttenuation(scene, textureAtlas,
                        skyScatterPos, sun.sw, FOG_LIMIT, strictDirectLight);
                    skyFogSunInt = sun.intensity;
                }
                Fog_addSkyFog(fogConfig, &color, ray.origin, ray.direction,
                              skyFogSunAtt, skyFogSunInt);
            }
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
                float totalAirDistance = firstAirDist;
                float totalWaterDistance = firstWaterDist;
                bool hitAnything = true;
                bool lastWasSpecular = false;  // tracks whether the last bounce was specular (for sky fog)
                IntersectionRecord record = firstRecord;
                MaterialSample sample = firstSample;
                Material material = firstMaterial;

                for (int depth = 0; depth < *rayDepth; depth++) {
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

                            bool diffuseSun = (sunSamplingStrategy == SUN_SAMPLING_OFF ||
                                               sunSamplingStrategy == SUN_SAMPLING_IMPORTANCE);
                            intersectSky(skyTexture, *skyIntensity, sun, textureAtlas, bRay, &sample, diffuseSun);
                            throughput *= sample.color.xyz;
                            branchColor += sample.emittance * throughput;

                            // CPU: only apply sky fog on specular sky hits (not diffuse).
                            // Diffuse sky hits skip sky fog entirely.
                            if (fogConfig.mode != FOG_MODE_NONE && lastWasSpecular) {
                                float3 skyFogSunAtt2 = (float3)(0.0f);
                                float skyFogSunInt2 = 0.0f;
                                if (fogConfig.mode == FOG_MODE_LAYERED) {
                                    float skyScatterOff2 = Fog_sampleSkyScatterOffset(fogConfig, bRay.origin, bRay.direction, random);
                                    float3 skyScatterPos2 = bRay.origin + bRay.direction * skyScatterOff2;
                                    skyFogSunAtt2 = getDirectLightAttenuation(scene, textureAtlas,
                                        skyScatterPos2, sun.sw, FOG_LIMIT, strictDirectLight);
                                    skyFogSunInt2 = sun.intensity;
                                }
                                Fog_addSkyFog(fogConfig, &branchColor, bRay.origin, bRay.direction,
                                              skyFogSunAtt2, skyFogSunInt2);
                            }
                            break;
                        }

                        // Apply biome tinting at the hit point
                        {
                            float3 hitPos = bRay.origin + bRay.direction * record.distance;
                            applyBiomeTint(scene, &sample, hitPos);
                        }

                        // Track distance through air and water (for fog)
                        if (bRay.inWater) {
                            totalWaterDistance += record.distance;
                        } else {
                            totalAirDistance += record.distance;
                        }
                    }

                    // Apply water fog attenuation (Beer's law, matching CPU).
                    // CPU applies ray.color.scale(exp(-distance/waterVisibility)) after
                    // each water segment. Pure absorption, no inscatter.
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
                    if (sunSamplingStrategy == SUN_SAMPLING_IMPORTANCE
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

                    // Emitter self-emission
                    int samplingStrategy = dynamicConfig[3];
                    if (dynamicConfig[2] && sample.emittance > EPS
                        && (!preventNormalEmitterWithSampling || samplingStrategy == 0 || depth == 0)) {
                        float3 emColor = (float3)(sample.color.x * sample.color.x,
                                                  sample.color.y * sample.color.y,
                                                  sample.color.z * sample.color.z);
                        branchColor += emColor * sample.emittance * (*emitterIntensity) * prevThroughput;
                    }

                    // Emitter sampling via emitter grid
                    if (dynamicConfig[2] && samplingStrategy != 0 && gridConfig[2] > 0) {
                        int cellSize = gridConfig[0];
                        int offsetX = gridConfig[1];
                        int sizeX = gridConfig[2];
                        int offsetY = gridConfig[3];
                        int sizeY = gridConfig[4];
                        int offsetZ = gridConfig[5];
                        int sizeZ = gridConfig[6];

                        int gx = (int)floor(bRay.origin.x) / cellSize;
                        int gy = (int)floor(bRay.origin.y) / cellSize;
                        int gz = (int)floor(bRay.origin.z) / cellSize;

                        if (gx >= offsetX && gx < offsetX + sizeX && gy >= offsetY && gy < offsetY + sizeY && gz >= offsetZ && gz < offsetZ + sizeZ) {
                            int idx = (((gy - offsetY) * sizeX) + (gx - offsetX)) * sizeZ + (gz - offsetZ);
                            int eStart = constructedGrid[2*idx];
                            int eCount = constructedGrid[2*idx + 1];
                            if (eCount > 0) {
                                float3 hitPoint = bRay.origin + bRay.direction * record.distance + record.normal * OFFSET;
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
                                        IntersectionRecord srec = IntersectionRecord_new();
                                        Material sSample;
                                        MaterialSample sMatSample;
                                        if (!closestIntersect(scene, textureAtlas, shadow, &srec, &sMatSample, &sSample) || srec.distance >= dist - 1e-4f) {
                                            float cosEmitter = fabs(dot(dirToEmitter, srec.normal));
                                            float att = fmax(0.0f, dot(record.normal, dirToEmitter)) * cosEmitter / fmax(dist * dist, 1.0f);
                                            att *= avgFaceArea;  // Correct for non-cube emitter surface area
                                            float3 emitterCol = (float3)(sMatSample.color.x * sMatSample.color.x * sMatSample.emittance,
                                                                        sMatSample.color.y * sMatSample.color.y * sMatSample.emittance,
                                                                        sMatSample.color.z * sMatSample.color.z * sMatSample.emittance);
                                            branchColor += throughput * (*emitterIntensity) * emitterCol * att * (float)M_PI_F;
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
                                        IntersectionRecord srec = IntersectionRecord_new();
                                        Material sSample;
                                        MaterialSample sMatSample;
                                        if (!closestIntersect(scene, textureAtlas, shadow, &srec, &sMatSample, &sSample) || srec.distance >= dist - 1e-4f) {
                                            float cosEmitter = fabs(dot(dirToEmitter, srec.normal));
                                            float att = fmax(0.0f, dot(record.normal, dirToEmitter)) * cosEmitter / fmax(dist * dist, 1.0f);
                                            att *= avgFaceArea;  // Correct for non-cube emitter surface area
                                            float3 emitterCol = (float3)(sMatSample.color.x * sMatSample.color.x * sMatSample.emittance,
                                                                        sMatSample.color.y * sMatSample.color.y * sMatSample.emittance,
                                                                        sMatSample.color.z * sMatSample.color.z * sMatSample.emittance);
                                            accCol += emitterCol * att;
                                        }
                                    }
                                    branchColor += throughput * (*emitterIntensity) * accCol * (1.0f / 6.0f) * (float)M_PI_F;
                                } else if (samplingStrategy == 3) {
                                    // ALL: iterate every emitter in the grid cell, sample all 6 faces each
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
                                            IntersectionRecord srec = IntersectionRecord_new();
                                            Material sSample;
                                            MaterialSample sMatSample;
                                            if (!closestIntersect(scene, textureAtlas, shadow, &srec, &sMatSample, &sSample) || srec.distance >= dist - 1e-4f) {
                                                float cosEmitter = fabs(dot(dirToEmitter, srec.normal));
                                                float att = fmax(0.0f, dot(record.normal, dirToEmitter)) * cosEmitter / fmax(dist * dist, 1.0f);
                                                att *= avgFaceArea;  // Correct for non-cube emitter surface area
                                                float3 emitterCol = (float3)(sMatSample.color.x * sMatSample.color.x * sMatSample.emittance,
                                                                            sMatSample.color.y * sMatSample.color.y * sMatSample.emittance,
                                                                            sMatSample.color.z * sMatSample.color.z * sMatSample.emittance);
                                                accCol += emitterCol * att;
                                            }
                                        }
                                    }
                                    if (eCount > 0) branchColor += throughput * (*emitterIntensity) * (accCol / (float)(eCount * 6)) * (float)M_PI_F;
                                }
                            }
                        }
                    }

                    // Sun direct light sampling (on diffuse bounces)
                    if ((sunSamplingStrategy == SUN_SAMPLING_FAST || sunSamplingStrategy == SUN_SAMPLING_HIGH_QUALITY)
                        && !pdfSample.specular) {
                        float3 baseHitPoint = bRay.origin + bRay.direction * record.distance;
                        Ray sunRay;
                        sunRay.origin = baseHitPoint;
                        if (Sun_sampleDirection(sun, &sunRay, random)) {
                            float cosSun = dot(record.normal, sunRay.direction);
                            bool frontLight = cosSun > 0.0f;
                            // CPU PathTracer: backside sun sampling for SSS materials (30% chance)
                            if (frontLight || (sample.sss && Random_nextFloat(random) < F_SUBSURFACE)) {
                                float3 hitPoint;
                                if (frontLight) {
                                    hitPoint = baseHitPoint + record.normal * OFFSET;
                                } else {
                                    hitPoint = baseHitPoint - record.normal * OFFSET;
                                }
                                sunRay.origin = hitPoint;
                                float3 sunAtt = getDirectLightAttenuation(scene, textureAtlas,
                                    hitPoint, sunRay.direction, FOG_LIMIT, strictDirectLight);
                                if (sunAtt.x + sunAtt.y + sunAtt.z > 0.0f) {
                                    float sunPower = pow(sun.intensity, DEFAULT_GAMMA);
                                    float mult = fabs(cosSun);
                                    if (sunSamplingStrategy == SUN_SAMPLING_HIGH_QUALITY) {
                                        mult *= (1.0f / sun.luminosity);
                                    }
                                    float3 sunContrib = sun.color.xyz * sunPower * mult * sunAtt;
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

                    if (!pdfSample.specular) {
                        bRay.flags |= RAY_INDIRECT;
                    }

                    // Russian roulette termination
                    if (depth >= RR_START_DEPTH) {
                        float pContinue = fmax(throughput.x, fmax(throughput.y, throughput.z));
                        pContinue = fmin(pContinue, 0.95f);
                        if (Random_nextFloat(random) > pContinue) {
                            break;
                        }
                        throughput /= pContinue;
                    }
                } // end depth loop

                // Apply remaining water fog if ray ended while still in water
                if (totalWaterDistance > 0.0f) {
                    float fogAtt = Water_fogAttenuation(totalWaterDistance, scene.waterVisibility);
                    branchColor *= fogAtt;
                }

                // Apply ground fog. Use scene entry point as fog origin so
                // empty space between camera and octree doesn't inflate fog.
                if (fogConfig.mode != FOG_MODE_NONE && totalAirDistance > 0.0f) {
                    float3 fogOrigin = cameraOrigin + cameraDirection * emptySpaceDist;
                    float fogOffset = Fog_sampleScatterOffset(fogConfig, totalAirDistance, fogOrigin, cameraDirection, random);
                    float3 sunDir = sun.sw;
                    float sunInt = sun.intensity;
                    float3 fogSunAtt = (float3)(1.0f);
                    if (sunSamplingStrategy != SUN_SAMPLING_OFF) {
                        float3 fogSamplePos = fogOrigin + cameraDirection * fogOffset;
                        fogSunAtt = getDirectLightAttenuation(scene, textureAtlas,
                            fogSamplePos, sunDir, FOG_LIMIT, strictDirectLight);
                    }
                    Fog_addGroundFog(fogConfig, &branchColor, fogOrigin, cameraDirection,
                                     totalAirDistance, fogSunAtt, sunInt, fogOffset);
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
}

__kernel void preview(
    __global const int* projectorType,
    __global const float* cameraSettings,

    __global const int* octreeDepth,
    __global const int* octreeData,

    __global const int* waterOctreeDepth,
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
    __global const float* skyIntensity,
    __global const int* sunData,

    __global const int* canvasConfig,
    __global const float* waterConfig,
    __global const int* chunkBitmap,
    int chunkBitmapSize,
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
    scene.drawDepth = 256;

    // Load water config for preview
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
    scene.waterNormalMap = NULL;
    scene.waterNormalMapW = 0;
    scene.cloudsEnabled = false;
    scene.transparentSky = false;
    scene.cloudHeight = 0;
    scene.cloudSize = 128;
    scene.cloudOffsetX = 0;
    scene.cloudOffsetZ = 0;
    scene.cloudData = NULL;
    scene.biomeData = NULL;
    scene.biomeDataSize = 0;
    scene.biomeColorsEnabled = false;
    scene.chunkBitmap = chunkBitmap;
    scene.chunkBitmapSize = chunkBitmapSize;

    Sun sun = Sun_new(sunData);

    unsigned int randomState = 0;
    Random random = &randomState;
    Random_nextState(random);

    Ray ray = ray_to_camera(projectorType, cameraSettings, NULL, 0, canvasConfig, gid, random);

    IntersectionRecord record = IntersectionRecord_new();
    MaterialSample sample;
    Material material;

    ray.material = 0;
    ray.flags = RAY_PREVIEW;
    ray.currentIor = AIR_IOR;
    ray.prevIor = AIR_IOR;
    ray.inWater = false;

    // Check if camera starts inside water for preview too
    if (isInWater(scene, ray.origin)) {
        ray.inWater = true;
        ray.currentIor = scene.waterIor;
    }

    float3 color;
    bool hitSolid = false;
    // Transparency skip loop: matches CPU PreviewRayTracer behavior.
    // CPU: stops when material != Air AND color.w > 0.
    // The GPU Material_sample returns false for transparent texels of non-refractive
    // blocks, so closestIntersect won't even register a hit there. For refractive
    // blocks, transparent texels return true with color.w = 0 (to allow skip-through).
    // We only skip past hits where color.w == 0 (purely transparent samples from
    // refractive material edges). Max 8 iterations to prevent lag.
    for (int skipIter = 0; skipIter < 8; skipIter++) {
        record = IntersectionRecord_new();
        if (!closestIntersect(scene, textureAtlas, ray, &record, &sample, &material)) {
            break;
        }
        if (sample.color.w > EPS) {
            // Solid hit (material is non-air and has opacity)
            float3 previewHitPos = ray.origin + ray.direction * record.distance;
            applyBiomeTint(scene, &sample, previewHitPos);
            float shading = dot(record.normal, (float3)(0.25f, 0.866f, 0.433f));
            shading = fmax(0.3f, shading);
            color = sample.color.xyz * shading;
            hitSolid = true;
            break;
        }
        // Transparent hit (e.g. refractive material with alpha=0 texel): skip past
        ray.origin = ray.origin + ray.direction * (record.distance + OFFSET);
        // Track block data so octree skips the same block type (prevents
        // re-intersecting from inside the block with a negative distance).
        ray.material = record.blockData;
    }
    if (!hitSolid) {
        // Check for floor grid intersection (preview only).
        // The CPU preview draws a chunk-boundary grid pattern at yMin (bottom of octree, Y=0 in local space).
        bool gridHit = false;
        if (ray.direction.y < 0) {
            float gridY = 0.0f; // yMin in octree-local space
            float gt = (gridY - ray.origin.y) / ray.direction.y;
            if (gt > OFFSET) {
                float3 hitP = ray.origin + ray.direction * gt;
                bool isSubmerged = scene.waterPlaneEnabled;
                int octSize = 1 << (*octreeDepth);
                bool insideOctree = (hitP.x >= 0 && hitP.x <= octSize &&
                                     hitP.z >= 0 && hitP.z <= octSize);
                // Chunk grid pattern: 16-block chunks, 0.5-block line width centered at chunk boundaries
                float xm = fmod(fmod(hitP.x, 16.0f) + 24.0f, 16.0f); // positive modulo + offset to center
                float zm = fmod(fmod(hitP.z, 16.0f) + 24.0f, 16.0f);
                float linePos = 7.75f;  // 8 - 0.5/2
                float lineEnd = 8.25f;  // 8 + 0.5/2
                bool isLine = (xm >= linePos && xm <= lineEnd) || (zm >= linePos && zm <= lineEnd);
                if (isLine) {
                    color = isSubmerged ? (float3)(0.05f, 0.05f, 0.25f) : (float3)(0.25f, 0.25f, 0.25f);
                } else {
                    color = isSubmerged ? (float3)(0.6f, 0.6f, 0.8f) : (float3)(0.8f, 0.8f, 0.8f);
                }
                if (insideOctree) {
                    // Darken grid inside octree bounds (matches CPU chunkPatternInsideOctreeColorFactor)
                    color *= 0.75f;
                }
                // Apply simple flat shading (sun direction approximation)
                float gridShading = fmax(0.3f, (float3)(0.25f, 0.866f, 0.433f).y); // normal is (0,1,0)
                color *= gridShading;
                gridHit = true;
            }
        }
        if (!gridHit) {
            if (ray.inWater) {
                // Underwater rays that escape without hitting geometry:
                // show black (full absorption), matching CPU path tracer behavior.
                // Single-precision float can lose the tiny direction.y component
                // for near-horizontal rays, causing Water_planeIntersect and
                // Octree_exitWater to miss the surface.
                color = (float3)(0.0f, 0.0f, 0.0f);
            } else {
                intersectSky(skyTexture, *skyIntensity, sun, textureAtlas, ray, &sample, true);
                color = sample.color.xyz;
            }
        }
    }

    color = sqrt(color);
    int3 rgb = intFloorFloat3(clamp(color * 255.0f, 0.0f, 255.0f));
    res[gid] = 0xFF000000 | (rgb.x << 16) | (rgb.y << 8) | rgb.z;
}
