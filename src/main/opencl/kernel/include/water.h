#ifndef CHUNKYCLPLUGIN_WATER_H
#define CHUNKYCLPLUGIN_WATER_H

#include "../opencl.h"
#include "rt.h"
#include "noise.h"
#include "constants.h"

// Water shading strategies
#define WATER_SHADING_STILL            0
#define WATER_SHADING_SIMPLEX          1
#define WATER_SHADING_TILED_NORMALMAP  2

// Water shader parameters (from SimplexWaterShader.java)
typedef struct {
    int iterations;       // Number of FBM octaves (default 4)
    float baseFrequency;  // First octave frequency (default 0.4)
    float baseAmplitude;  // First octave amplitude (default 0.025)
    float animationSpeed; // Animation speed multiplier (default 1)
} WaterShaderParams;

// Apply simplex noise water shading to perturb the surface normal.
// Uses 3D simplex noise with analytical derivatives, matching the CPU SimplexWaterShader.
// The noise uses (x*freq, z*freq, time) as the 3D coordinate.
void Water_simplexShading(IntersectionRecord* record, float wx, float wz,
                          float animationTime, WaterShaderParams params) {
    float frequency = params.baseFrequency;
    float amplitude = params.baseAmplitude;
    float time = animationTime * params.animationSpeed;

    float ddx = 0.0f;
    float ddz = 0.0f;

    for (int i = 0; i < params.iterations && i < 8; i++) {
        float fx = wx * frequency;
        float fz = wz * frequency;

        // 3D simplex noise: (x, z, time) with analytical derivatives
        float nddx, nddy, nddz;
        simplexNoise3(fx, fz, time, &nddx, &nddy, &nddz);
        // CPU maps world x -> noise x, world z -> noise y, time -> noise z
        // So noise.ddx = dN/dx, noise.ddy = dN/dz (in world space)
        float ddxNext = ddx - amplitude * nddx;
        float ddzNext = ddz - amplitude * nddy;

        // NaN guard (matching CPU behavior)
        if (isnan(ddxNext + ddzNext)) {
            break;
        }
        ddx = ddxNext;
        ddz = ddzNext;

        frequency *= 2.0f;
        amplitude *= 0.5f;
    }

    // Compute normal from slopes using cross product of tangent vectors
    // xslope = (1, ddx, 0), zslope = (0, ddz, 1)
    // normal = cross(zslope, xslope) = (ddx, 1, ddz)  (unnormalized)
    float3 n = (float3)(ddx, 1.0f, ddz);
    n = normalize(n);

    // Flip the normal if the ray hit from below
    if (record->normal.y < 0) {
        n.y = -n.y;
    }

    record->normal = n;
}

// Tiled normal map water shading, matching CPU WaterModel.doWaterDisplacement().
// Uses a precomputed gradient normal map sampled at two scales (period 16 and 2).
void Water_tiledNormalMapShading(IntersectionRecord* record, float wx, float wz,
                                 __global const float* normalMap, int normalMapW) {
    if (normalMapW <= 0) return;

    float invW = 1.0f / (float)normalMapW;

    // Scale 1: large tiles (period = 16 world units)
    float x1 = wx / 16.0f;
    x1 = x1 - floor(x1);  // frac
    float z1 = wz / 16.0f;
    z1 = z1 - floor(z1);  // frac
    int u1 = clamp((int)(x1 * normalMapW - 1e-5f), 0, normalMapW - 1);
    int v1 = clamp((int)((1.0f - z1) * normalMapW - 1e-5f), 0, normalMapW - 1);
    int idx1 = (u1 * normalMapW + v1) * 2;
    float nx = normalMap[idx1];
    float nz = normalMap[idx1 + 1];

    // Scale 2: small tiles (period = 2 world units) at half weight
    float x2 = wx / 2.0f;
    x2 = x2 - floor(x2);
    float z2 = wz / 2.0f;
    z2 = z2 - floor(z2);
    int u2 = clamp((int)(x2 * normalMapW - 1e-5f), 0, normalMapW - 1);
    int v2 = clamp((int)((1.0f - z2) * normalMapW - 1e-5f), 0, normalMapW - 1);
    int idx2 = (u2 * normalMapW + v2) * 2;
    nx += normalMap[idx2] * 0.5f;
    nz += normalMap[idx2 + 1] * 0.5f;

    // Construct normal: n = (nx, 0.15, nz), then normalize
    float3 n = normalize((float3)(nx, 0.15f, nz));

    // Flip if hit from below
    if (record->normal.y < 0) {
        n.y = -n.y;
    }
    record->normal = n;
}

// Apply water shading to an intersection record using world-space hit coords
void Water_applyShading(IntersectionRecord* record, int waterShadingStrategy,
                        float animationTime, float wx, float wz,
                        WaterShaderParams params,
                        __global const float* waterNormalMap, int waterNormalMapW) {
    switch (waterShadingStrategy) {
        case WATER_SHADING_SIMPLEX:
            Water_simplexShading(record, wx, wz, animationTime, params);
            break;
        case WATER_SHADING_TILED_NORMALMAP:
            Water_tiledNormalMapShading(record, wx, wz, waterNormalMap, waterNormalMapW);
            break;
        case WATER_SHADING_STILL:
        default:
            // No modification - water stays flat
            break;
    }
}

// Check if a ray-plane intersection occurs for the water plane.
// chunkBitmap is a 2D bitfield of loaded chunks (1 bit per 16x16 chunk).
// chunkBitmapSize is the number of chunks per side (octreeSize / 16).
// When chunkClip is true, water is hidden where chunks are loaded (bitmap=1),
// matching the CPU behavior where the water plane fills unloaded areas.
bool Water_planeIntersect(Ray ray, float waterPlaneHeight, float octreeSize, bool chunkClip,
                          __global const int* chunkBitmap, int chunkBitmapSize,
                          IntersectionRecord* record) {
    // Only intersect if ray crosses the Y plane
    if (fabs(ray.direction.y) < EPS) return false;

    float t = (waterPlaneHeight - ray.origin.y) / ray.direction.y;
    if (t < OFFSET || t >= record->distance) return false;

    float3 hitPoint = ray.origin + ray.direction * t;

    // Chunk clipping: hide water where chunks are loaded (bitmap lookup).
    // The CPU's PreviewRayTracer.waterPlaneIntersection checks isChunkLoaded()
    // per-block; we approximate this per-chunk using the exported bitmap.
    if (chunkClip && chunkBitmapSize > 0) {
        int cx = (int)floor(hitPoint.x) >> 4;  // / 16
        int cz = (int)floor(hitPoint.z) >> 4;  // / 16
        if (cx >= 0 && cx < chunkBitmapSize && cz >= 0 && cz < chunkBitmapSize) {
            int bitIndex = cz * chunkBitmapSize + cx;
            int word = chunkBitmap[bitIndex >> 5];  // / 32
            bool isLoaded = (word >> (bitIndex & 31)) & 1;
            if (isLoaded) {
                return false;  // Hide water plane in loaded chunks
            }
        }
        // If outside the bitmap range, chunk is not loaded -> show water
    }

    record->distance = t;
    record->normal = (ray.direction.y < 0) ? (float3)(0, 1, 0) : (float3)(0, -1, 0);
    // Store world-space XZ in texCoord for water shading
    record->texCoord = (float2)(hitPoint.x, hitPoint.z);
    record->material = -1; // Special marker for water plane

    return true;
}

// Apply water fog attenuation (exponential falloff based on distance in water).
// Matches CPU Beer's law: attenuation = exp(-distance / waterVisibility).
// The CPU applies pure absorption scaling — color is simply multiplied by the
// attenuation factor. Water color is applied separately to the water surface
// material (in updateOpacity / closestIntersect), not in the fog itself.
float Water_fogAttenuation(float distance, float waterVisibility) {
    if (waterVisibility <= 0) return 0.0f;
    float a = distance / waterVisibility;
    return exp(-a);
}

#endif
