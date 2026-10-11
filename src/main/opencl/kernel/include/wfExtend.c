#include "../opencl.h"
#include "pathtrace.h"
#include "wavefront.h"

// Wavefront path tracing, see wavefront.h: traces every live branch's bounce ray
// and stores the surface it hits for wfShade. Only geometry is done here; the
// material sample is built by wfShade (traceSceneResolve), so this kernel stays
// small enough for the host to build it with a low register cap.
__kernel void wfExtend(
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
        if (as_int(WF(WF_FIRST_HIT)) == 0 || as_int(WF(WF_ALIVE)) == 0) continue;
        Ray bRay = wfLoadRay(st, cap, idx, WF_BRAY);
        IntersectionRecord record = IntersectionRecord_new();
        MaterialSample unused;
        bool hit = traceSceneGeometry(scene, textureAtlas, bRay, &record, &unused, false) == TRACE_HIT;
        WF(WF_HIT) = as_float((int) hit);
        if (hit) {
            WF(WF_HIT_REC) = record.distance;
            WF(WF_HIT_REC + 1) = as_float(record.material);
            WF(WF_HIT_REC + 2) = as_float(record.blockData);
            wfStoreF3(st, cap, idx, WF_HIT_REC + 3, record.normal);
            WF(WF_HIT_REC + 6) = record.texCoord.x;
            WF(WF_HIT_REC + 7) = record.texCoord.y;
            WF(WF_HIT_REC + 8) = as_float(record.hitKind);
        }
    }
}
