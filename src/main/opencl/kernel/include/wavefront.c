#include "../opencl.h"
#include "pathtrace.h"
#include "wavefront.h"

// Wavefront path tracing, see wavefront.h. These kernels share the megakernel's
// parameter list (frag_kernel_params.h), followed by the chunk's path state:
//   st         WF_FIELDS * chunkCap floats
//   chunkBase  first pixel of the chunk; the host also passes it as pixelStart
//   chunkCap   pixels the state has room for (>= the chunk's pixel count)
//   step       wfPrimary: the iteration; wfShade: the bounce; others: unused
#define PATH_WAVEFRONT 1

// The camera ray of iteration `step`: its first hit, which every branch starts
// from, or the sky. Also folds the previous iteration into the pixel's sum.
__kernel void wfPrimary(
#include "frag_kernel_params.h"
    ,
    __global float* st,
    int chunkBase,
    int chunkCap,
    int step
) {
#include "frag_kernel_setup.h"
    const size_t cap = (size_t) chunkCap;
    for (int gid = pixelStart + wid; gid < pixelCount; gid += pixelStride) {
        const size_t idx = (size_t) (gid - chunkBase);
        int it = step;
        float3 sumColor;
        float sumAlpha;
        if (it == 0) {
            sumColor = (float3)(0.0f, 0.0f, 0.0f);
            sumAlpha = 0.0f;
        } else {
            // The megakernel's `sumColor += color` at the end of iteration it-1.
            sumColor = wfLoadF3(st, cap, idx, WF_SUM_COLOR);
            sumAlpha = WF(WF_SUM_ALPHA);
            sumColor += wfLoadF3(st, cap, idx, WF_COLOR);
            sumAlpha += WF(WF_ALPHA);
        }
#include "frag_path_camera.h"
        if (!firstHit) {
#include "frag_path_sky.h"
            WF(WF_FIRST_HIT) = as_float(0);
        } else {
#include "frag_path_firsthit.h"
            wfStoreRay(st, cap, idx, WF_RAY, ray);
            wfStoreRecord(st, cap, idx, WF_FIRST_REC, firstRecord);
            wfStoreSample(st, cap, idx, WF_FIRST_SMP, firstSample);
            WF(WF_EMPTY_DIST) = emptySpaceDist;
            WF(WF_AIR_DIST) = firstAirDist;
            WF(WF_WATER_DIST) = firstWaterDist;
            WF(WF_FIRST_HIT) = as_float(1);
        }
        wfStoreF3(st, cap, idx, WF_COLOR, color);
        WF(WF_ALPHA) = alpha;
        wfStoreF3(st, cap, idx, WF_SUM_COLOR, sumColor);
        WF(WF_SUM_ALPHA) = sumAlpha;
        WF(WF_RNG) = as_float(randomState);
    }
}

// Bounce `step` of the current branch: shades the hit wfExtend found (bounce 0:
// the camera ray's first hit) and picks the next ray. Shadow rays, and the colour
// terms that come before them, are left for wfConnect; a branch that ends here is
// finished by wfConnect too.
__kernel void wfShade(
#include "frag_kernel_params.h"
    ,
    __global float* st,
    int chunkBase,
    int chunkCap,
    int step
) {
#include "frag_kernel_setup.h"
    const size_t cap = (size_t) chunkCap;
    const int depth = step;
    for (int gid = pixelStart + wid; gid < pixelCount; gid += pixelStride) {
        const size_t idx = (size_t) (gid - chunkBase);
        if (as_int(WF(WF_FIRST_HIT)) == 0) continue;
        if (depth > 0 && as_int(WF(WF_ALIVE)) == 0) continue;
        unsigned int randomState = as_uint(WF(WF_RNG));
        Random random = &randomState;

        Ray bRay;
        float3 throughput;
        float3 branchColor;
        float totalWaterDistance;
        bool lastWasSpecular;
        IntersectionRecord record;
        MaterialSample sample;
        Material material;  // never read, see closestIntersect
        bool wfHit = true;
        if (depth == 0) {
            // A new branch, as the megakernel's branch loop starts one.
            bRay = wfLoadRay(st, cap, idx, WF_RAY);
            throughput = (float3)(1.0f);
            branchColor = (float3)(0.0f);
            totalWaterDistance = WF(WF_WATER_DIST);
            lastWasSpecular = false;
            record = wfLoadRecord(st, cap, idx, WF_FIRST_REC);
            sample = wfLoadSample(st, cap, idx, WF_FIRST_SMP);
        } else {
            bRay = wfLoadRay(st, cap, idx, WF_BRAY);
            throughput = wfLoadF3(st, cap, idx, WF_THROUGHPUT);
            branchColor = wfLoadF3(st, cap, idx, WF_BRANCH_COLOR);
            totalWaterDistance = WF(WF_WATER_TOTAL);
            lastWasSpecular = as_int(WF(WF_LAST_SPECULAR)) != 0;
            wfHit = as_int(WF(WF_HIT)) != 0;
            record = IntersectionRecord_new();
            if (wfHit) {
                record.distance = WF(WF_HIT_REC);
                record.material = as_int(WF(WF_HIT_REC + 1));
                record.blockData = as_int(WF(WF_HIT_REC + 2));
                record.normal = wfLoadF3(st, cap, idx, WF_HIT_REC + 3);
                record.texCoord = (float2)(WF(WF_HIT_REC + 6), WF(WF_HIT_REC + 7));
                record.hitKind = as_int(WF(WF_HIT_REC + 8));
                traceSceneResolve(scene, textureAtlas, bRay, &record, &sample);
            }
        }

        int effectiveDepth = cachedRayDepth - 1;
        if (effectiveDepth < 0) effectiveDepth = 0;

        // Filled by the step for wfConnect.
        bool wfWaterPending = false;
        float3 wfWaterPos = (float3)(0.0f);
        float3 wfWaterThroughput = (float3)(0.0f);
        float3 wfWaterAtt = (float3)(0.0f);
        float wfWaterCos = 0.0f;
        bool wfEmitPending = false;
        float3 wfEmit = (float3)(0.0f);
        bool wfSunPending = false;
        float3 wfSunOrigin = (float3)(0.0f);
        float3 wfSunDir = (float3)(0.0f);
        float3 wfSunA = (float3)(0.0f);
        float3 wfSunB = (float3)(0.0f);

        // The step's `break`s (sky, escape, dead throughput) leave `terminated` set.
        bool terminated = true;
        if (depth < effectiveDepth) {
            do {
#include "frag_path_step.h"
                terminated = false;
            } while (0);
        }
        bool ended = terminated || depth + 1 >= effectiveDepth;

        wfStoreRay(st, cap, idx, WF_BRAY, bRay);
        wfStoreF3(st, cap, idx, WF_THROUGHPUT, throughput);
        wfStoreF3(st, cap, idx, WF_BRANCH_COLOR, branchColor);
        WF(WF_WATER_TOTAL) = totalWaterDistance;
        WF(WF_LAST_SPECULAR) = as_float((int) lastWasSpecular);
        WF(WF_ALIVE) = as_float((int) !ended);
        WF(WF_ENDED) = as_float((int) ended);
        WF(WF_WATER) = as_float((int) wfWaterPending);
        if (wfWaterPending) {
            wfStoreF3(st, cap, idx, WF_WATER_POS, wfWaterPos);
            wfStoreF3(st, cap, idx, WF_WATER_TP, wfWaterThroughput);
            wfStoreF3(st, cap, idx, WF_WATER_ATT, wfWaterAtt);
            WF(WF_WATER_COS) = wfWaterCos;
        }
        WF(WF_EMIT) = as_float((int) wfEmitPending);
        if (wfEmitPending) wfStoreF3(st, cap, idx, WF_EMIT_COLOR, wfEmit);
        WF(WF_SUN) = as_float((int) wfSunPending);
        if (wfSunPending) {
            wfStoreF3(st, cap, idx, WF_SUN_ORIGIN, wfSunOrigin);
            wfStoreF3(st, cap, idx, WF_SUN_DIR, wfSunDir);
            wfStoreF3(st, cap, idx, WF_SUN_A, wfSunA);
            wfStoreF3(st, cap, idx, WF_SUN_B, wfSunB);
        }
        WF(WF_RNG) = as_float(randomState);
    }
}

// Averages the pixel's iterations and branches into the output, like the end of
// the megakernel's pixel loop.
__kernel void wfFinish(
#include "frag_kernel_params.h"
    ,
    __global float* st,
    int chunkBase,
    int chunkCap,
    int step
) {
#include "frag_kernel_setup.h"
    const size_t cap = (size_t) chunkCap;
    for (int gid = pixelStart + wid; gid < pixelCount; gid += pixelStride) {
        const size_t idx = (size_t) (gid - chunkBase);
        float3 sumColor = wfLoadF3(st, cap, idx, WF_SUM_COLOR);
        float sumAlpha = WF(WF_SUM_ALPHA);
        sumColor += wfLoadF3(st, cap, idx, WF_COLOR);
        sumAlpha += WF(WF_ALPHA);
        float3 avgColor = sumColor / (float)(iterations * branchCount);
        float avgAlpha = sumAlpha / (float)iterations;
        vstore4((float4)(avgColor, avgAlpha), gid, res);
    }
}
