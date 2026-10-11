#include "../opencl.h"
#include "pathtrace.h"
#include "wavefront.h"

// Wavefront path tracing, see wavefront.h: runs after wfShade for each bounce.
// Traces the shadow rays wfShade queued and adds their light in the megakernel's
// order (water in-scatter, emission, sun), then finishes branches that ended at
// this bounce (frag_path_end.h). Built with its own register cap by the host.
#define PATH_WAVEFRONT 1

__kernel void wfConnect(
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
        if (as_int(WF(WF_FIRST_HIT)) == 0) continue;
        // Branches that died at an earlier bounce have all of these cleared.
        bool waterPending = as_int(WF(WF_WATER)) != 0;
        bool emitPending = as_int(WF(WF_EMIT)) != 0;
        bool sunPending = as_int(WF(WF_SUN)) != 0;
        bool ended = as_int(WF(WF_ENDED)) != 0;
        if (!waterPending && !emitPending && !sunPending && !ended) continue;

        unsigned int randomState = as_uint(WF(WF_RNG));
        Random random = &randomState;
        Ray ray = wfLoadRay(st, cap, idx, WF_RAY);
        float3 cameraOrigin = ray.origin;
        float3 cameraDirection = ray.direction;
        float emptySpaceDist = WF(WF_EMPTY_DIST);
        float firstAirDist = WF(WF_AIR_DIST);
        float firstWaterDist = WF(WF_WATER_DIST);

        // The branch end's ground-fog sample: the same random draw and point as in
        // frag_path_end.h, taken first because neither depends on the light added
        // below, so all three shadow rays share one trace call.
        bool fogPending = false;
        float wfFogOffset = 0.0f;
        float3 fogPos = (float3)(0.0f);
        if (ended) {
            float groundFogDist = firstAirDist + firstWaterDist;
            if (fogConfig.mode != FOG_MODE_NONE && groundFogDist > 0.0f) {
                float3 fogOrigin = cameraOrigin + cameraDirection * emptySpaceDist;
                wfFogOffset = Fog_sampleScatterOffset(fogConfig, groundFogDist, fogOrigin, cameraDirection, random);
                fogPos = fogOrigin + cameraDirection * wfFogOffset;
                fogPending = true;
            }
        }

        // One getDirectLightAttenuation call site for the water, sun and fog rays:
        // inlined once, the kernel needs far fewer registers than with three.
        float3 waterAtt = (float3)(0.0f), sunAtt = (float3)(0.0f), wfFogAtt = (float3)(0.0f);
        __attribute__((opencl_unroll_hint(1)))
        for (int q = 0; q < 3; q++) {
            bool active = q == 0 ? waterPending : (q == 1 ? sunPending : fogPending);
            if (!active) continue;
            float3 origin = q == 0 ? wfLoadF3(st, cap, idx, WF_WATER_POS)
                          : (q == 1 ? wfLoadF3(st, cap, idx, WF_SUN_ORIGIN) : fogPos);
            float3 dir = q == 1 ? wfLoadF3(st, cap, idx, WF_SUN_DIR) : sun.sw;
            float3 att = getDirectLightAttenuation(scene, textureAtlas, origin, dir, FOG_LIMIT, strictDirectLight);
            if (q == 0) waterAtt = att; else if (q == 1) sunAtt = att; else wfFogAtt = att;
        }

        float3 branchColor = wfLoadF3(st, cap, idx, WF_BRANCH_COLOR);
        if (waterPending) {
            branchColor += wfLoadF3(st, cap, idx, WF_WATER_TP) * Water_inscatter(
                wfLoadF3(st, cap, idx, WF_WATER_ATT), scene.waterColor,
                Water_scatterLight(waterAtt, WF(WF_WATER_COS), waterSunLight, waterSkyLight));
            WF(WF_WATER) = as_float(0);
        }
        if (emitPending) {
            branchColor += wfLoadF3(st, cap, idx, WF_EMIT_COLOR);
            WF(WF_EMIT) = as_float(0);
        }
        if (sunPending) {
            if (sunAtt.x + sunAtt.y + sunAtt.z > 0.0f) {
                float3 sunContrib = wfLoadF3(st, cap, idx, WF_SUN_A) * sunAtt;
                branchColor += wfLoadF3(st, cap, idx, WF_SUN_B) * sunContrib;
            }
            WF(WF_SUN) = as_float(0);
        }
        if (!ended) {
            wfStoreF3(st, cap, idx, WF_BRANCH_COLOR, branchColor);
            continue;
        }

        WF(WF_ENDED) = as_float(0);
        float3 color = wfLoadF3(st, cap, idx, WF_COLOR);
        Ray bRay = wfLoadRay(st, cap, idx, WF_BRAY);
        float3 throughput = wfLoadF3(st, cap, idx, WF_THROUGHPUT);
        float totalWaterDistance = WF(WF_WATER_TOTAL);
#include "frag_path_end.h"
        wfStoreF3(st, cap, idx, WF_COLOR, color);
        WF(WF_RNG) = as_float(randomState);
    }
}
