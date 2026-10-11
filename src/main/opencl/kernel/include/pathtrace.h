// Helpers shared by the megakernel (rayTracer.c) and the wavefront kernels.
#ifndef CHUNKYCLPLUGIN_PATHTRACE_H
#define CHUNKYCLPLUGIN_PATHTRACE_H

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

// Light for an emitter-NEE shadow ray that reached the emitter's cell (ex,ey,ez)
// without hitting its geometry: a light block (invisible to shadow rays), or a
// small model the ray slipped past. It comes from the block's own material; cubes
// are sampled at the chosen face point, models at their texture centre. With no
// usable material (air, or a see-through texel) the emitter contributes nothing:
// the sample is zeroed first, as the caller reads its colour and emittance.
void Emitter_sampleCell(BlockPalette palette, MaterialPalette materialPalette, image2d_array_t atlas,
                        Octree* octree, int ex, int ey, int ez, float2 uv, MaterialSample* sample) {
    sample->color = (float4)(0.0f);
    sample->emittance = 0.0f;
    int block = Octree_get(octree, ex, ey, ez);
    if (block <= 0) return;
    int material = BlockPalette_emitterMaterial(palette, block);
    if (material < 0) return;
    bool isModel = palette.blockPalette[block] == 2 || palette.blockPalette[block] == 3;
    Material m = Material_get(materialPalette, material);
    if (isModel && (m.flags & MATERIAL_HAS_EMISSION_MAP)) {
        Material_meanEmission(m, atlas, sample);
        return;
    }
    if (!Material_sample(m, atlas, isModel ? (float2)(0.5f, 0.5f) : uv, sample)) {
        sample->color = (float4)(0.0f);
        sample->emittance = 0.0f;
    }
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

        if (!traceScene(scene, textureAtlas, shadow, &srec, &sMat, true)) {
            // Escaped without finding a surface. If we are STILL submerged, the
            // path to the sun is an unbounded water column (the unloaded-chunk
            // case again) which transmits nothing — so this is full shadow, not
            // the clear path the plain break would report. Without this the same
            // geometry that used to render black would instead leak FULL sun,
            // and the volumetric in-scatter would amplify it.
            if (shadow.inWater) return (float3)(0.0f);
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

        // Water fog attenuation in shadow rays, now PER CHANNEL: sunlight
        // reaching a deep seabed has had its red end absorbed on the way down,
        // which is what tints everything underwater blue-green.
        if (shadow.inWater) {
            if (scene.waterVisibility <= 0) {
                alphaAtt = 0.0f;
            } else {
                attenuation *= Water_extinction(srec.distance, scene.waterVisibility);
            }
        }

        // CAUSTICS. The sun ray is crossing a water surface, so refract it
        // against the live wave field and weight it by how much that wave
        // focuses light. Shadow rays deliberately skip wave shading for CPU
        // parity (kernel.h), so srec.normal here is FLAT — the wave normal has
        // to be sampled directly, from the same noise that shapes the surface.
        if (sMat.isWater) {
            float3 sHit = shadow.origin + shadow.direction * srec.distance;
            float3 waveN = Water_waveNormal(scene.waterShadingStrategy, scene.animationTime,
                                            sHit.x, sHit.z, scene.waterShaderParams,
                                            scene.waterNormalMap, scene.waterNormalMapW);
            attenuation *= Water_causticFactor(waveN, shadow.direction);
        }

        // Strict direct light: CPU zeroes the NEE ray at ANY IOR boundary
        // (PathTracer.java:581), because a straight shadow ray through a
        // refracting surface is not a valid estimator. Underwater that kills
        // direct sun outright — the "rays die off" report. WATER is now exempt:
        // the caustic term above is what accounts for the refraction, so the ray
        // is kept and weighted instead of discarded. Every other refractive
        // boundary (glass) keeps CPU's hard cut so nothing else shifts.
        if (strictDirectLight && shadow.currentIor != sMat.ior && !sMat.isWater) {
            alphaAtt = 0.0f;
        }

        // Continue tracing past this intersection
        maxDist -= srec.distance + OFFSET;
        shadow.origin = shadow.origin + shadow.direction * (srec.distance + OFFSET);
        shadow.currentIor = sMat.ior;
        // Carry the medium we just entered into the next traversal. Without this
        // the shadow ray re-hit EVERY internal face of a translucent volume and
        // attenuated once per block, so an N-thick glass/ice/leaf wall darkened
        // the sun N times instead of once. The water case was already special-
        // cased above for exactly this reason; this generalises it to full cubes
        // and refractive blocks. Water is deliberately left alone — its medium is
        // seeded once at init above and that path is tuned around it.
        if (!sMat.isWater) {
            shadow.material = (sMat.ior > AIR_IOR + EPS
                    || BlockPalette_isFullCube(scene.blockPalette, srec.blockData))
                ? srec.blockData : 0;
        }

        // Track water medium transitions for shadow ray
        if (sMat.isWater) {
            shadow.inWater = !shadow.inWater;
        }

        if (maxDist <= 0) break;
    }

    return attenuation * alphaAtt;
}

#endif
