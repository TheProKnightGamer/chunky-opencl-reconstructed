// Wavefront path tracing: the path state kept between kernel launches.
//
// The megakernel (rayTracer.c: render) traces every bounce of every branch of a
// pixel in one work-item. That kernel needs 128 registers and still spills, so few
// warps fit on an SM, and its warps run with most lanes idle because paths end at
// different bounces. The wavefront kernels split one bounce into three launches,
// each small enough to run at high occupancy:
//
//   wfExtend   trace the bounce ray                  (wfExtend.c, own register cap)
//   wfShade    shade the hit, pick the next ray      (wavefront.c)
//   wfConnect  trace the shadow rays, add the light  (wfConnect.c, own register cap)
//
// plus wfPrimary (camera ray, first hit) and wfFinish (write the pixel). The host
// runs them per chunk of pixels: for each iteration wfPrimary, then for each branch
// and bounce wfExtend / wfShade / wfConnect, then wfFinish.
//
// Every random number is drawn by the same code, in the same order, from the same
// per-pixel RNG state as the megakernel, and every colour term is added in the same
// order, so the image is the megakernel's (up to compiler FMA contraction).
// Emitter-grid sampling and layered fog are not implemented here; scenes that use
// them render with the megakernel (the host checks).
#ifndef CHUNKYCLPLUGIN_WAVEFRONT_H
#define CHUNKYCLPLUGIN_WAVEFRONT_H

#include "../opencl.h"
#include "rt.h"
#include "material.h"

// Field-major layout: field f of pixel gid is st[f * chunkCap + (gid - chunkBase)],
// so a warp's loads of one field are contiguous. Ints and bools are bit-cast.
// WF_FIELDS must match OpenClPathTracingRenderer.WAVEFRONT_STATE_FIELDS.
#define WF_RNG            0   // uint: the pixel's RNG state
#define WF_SUM_COLOR      1   // float3: finished iterations
#define WF_SUM_ALPHA      4
#define WF_COLOR          5   // float3: this iteration (sky, or the branches so far)
#define WF_ALPHA          8
#define WF_FIRST_HIT      9   // int: this iteration's camera ray hit something
#define WF_RAY           10   // Ray (11): the camera ray
#define WF_FIRST_REC     21   // IntersectionRecord (12): its hit
#define WF_FIRST_SMP     33   // MaterialSample (13): its material
#define WF_EMPTY_DIST    46
#define WF_AIR_DIST      47
#define WF_WATER_DIST    48
#define WF_BRAY          49   // Ray (11): the branch's current ray
#define WF_THROUGHPUT    60   // float3
#define WF_BRANCH_COLOR  63   // float3
#define WF_WATER_TOTAL   66   // totalWaterDistance
#define WF_LAST_SPECULAR 67   // int
#define WF_ALIVE         68   // int: the branch continues past this bounce
#define WF_HIT           69   // int: wfExtend found a surface
#define WF_HIT_REC       70   // distance, material, blockData, normal (3), texCoord (2), hitKind
#define WF_SUN           79   // int: a sun shadow query is pending
#define WF_SUN_ORIGIN    80   // float3
#define WF_SUN_DIR       83   // float3
#define WF_SUN_A         86   // float3: sun colour * power * cosine factor
#define WF_SUN_B         89   // float3: throughput * surface colour
#define WF_ENDED         92   // int: the branch ended at this bounce
#define WF_WATER         93   // int: a water in-scatter query is pending
#define WF_WATER_POS     94   // float3
#define WF_WATER_TP      97   // float3: throughput before the segment's extinction
#define WF_WATER_ATT    100   // float3: the segment's extinction
#define WF_WATER_COS    103
#define WF_EMIT         104   // int: an emission term is pending
#define WF_EMIT_COLOR   105   // float3
#define WF_FIELDS       108

// In a wavefront kernel: field f of the current pixel (gid).
#define WF(f) st[(size_t)(f) * (size_t)chunkCap + (size_t)(gid - chunkBase)]

float3 wfLoadF3(__global const float* st, size_t cap, size_t i, int f) {
    return (float3)(st[f * cap + i], st[(f + 1) * cap + i], st[(f + 2) * cap + i]);
}

void wfStoreF3(__global float* st, size_t cap, size_t i, int f, float3 v) {
    st[f * cap + i] = v.x;
    st[(f + 1) * cap + i] = v.y;
    st[(f + 2) * cap + i] = v.z;
}

Ray wfLoadRay(__global const float* st, size_t cap, size_t i, int f) {
    Ray r;
    r.origin = wfLoadF3(st, cap, i, f);
    r.direction = wfLoadF3(st, cap, i, f + 3);
    r.material = as_int(st[(f + 6) * cap + i]);
    r.flags = as_int(st[(f + 7) * cap + i]);
    r.currentIor = st[(f + 8) * cap + i];
    r.prevIor = st[(f + 9) * cap + i];
    r.inWater = as_int(st[(f + 10) * cap + i]) != 0;
    return r;
}

void wfStoreRay(__global float* st, size_t cap, size_t i, int f, Ray r) {
    wfStoreF3(st, cap, i, f, r.origin);
    wfStoreF3(st, cap, i, f + 3, r.direction);
    st[(f + 6) * cap + i] = as_float(r.material);
    st[(f + 7) * cap + i] = as_float(r.flags);
    st[(f + 8) * cap + i] = r.currentIor;
    st[(f + 9) * cap + i] = r.prevIor;
    st[(f + 10) * cap + i] = as_float((int) r.inWater);
}

IntersectionRecord wfLoadRecord(__global const float* st, size_t cap, size_t i, int f) {
    IntersectionRecord r;
    r.distance = st[f * cap + i];
    r.material = as_int(st[(f + 1) * cap + i]);
    r.blockData = as_int(st[(f + 2) * cap + i]);
    r.normal = wfLoadF3(st, cap, i, f + 3);
    r.geomNormal = wfLoadF3(st, cap, i, f + 6);
    r.texCoord = (float2)(st[(f + 9) * cap + i], st[(f + 10) * cap + i]);
    r.hitKind = as_int(st[(f + 11) * cap + i]);
    return r;
}

void wfStoreRecord(__global float* st, size_t cap, size_t i, int f, IntersectionRecord r) {
    st[f * cap + i] = r.distance;
    st[(f + 1) * cap + i] = as_float(r.material);
    st[(f + 2) * cap + i] = as_float(r.blockData);
    wfStoreF3(st, cap, i, f + 3, r.normal);
    wfStoreF3(st, cap, i, f + 6, r.geomNormal);
    st[(f + 9) * cap + i] = r.texCoord.x;
    st[(f + 10) * cap + i] = r.texCoord.y;
    st[(f + 11) * cap + i] = as_float(r.hitKind);
}

MaterialSample wfLoadSample(__global const float* st, size_t cap, size_t i, int f) {
    MaterialSample m;
    m.color = (float4)(wfLoadF3(st, cap, i, f), st[(f + 3) * cap + i]);
    m.emittance = st[(f + 4) * cap + i];
    m.specular = st[(f + 5) * cap + i];
    m.metalness = st[(f + 6) * cap + i];
    m.roughness = st[(f + 7) * cap + i];
    m.ior = st[(f + 8) * cap + i];
    m.refractive = as_int(st[(f + 9) * cap + i]) != 0;
    m.sss = as_int(st[(f + 10) * cap + i]) != 0;
    m.isWater = as_int(st[(f + 11) * cap + i]) != 0;
    m.tintType = as_int(st[(f + 12) * cap + i]);
    return m;
}

void wfStoreSample(__global float* st, size_t cap, size_t i, int f, MaterialSample m) {
    wfStoreF3(st, cap, i, f, m.color.xyz);
    st[(f + 3) * cap + i] = m.color.w;
    st[(f + 4) * cap + i] = m.emittance;
    st[(f + 5) * cap + i] = m.specular;
    st[(f + 6) * cap + i] = m.metalness;
    st[(f + 7) * cap + i] = m.roughness;
    st[(f + 8) * cap + i] = m.ior;
    st[(f + 9) * cap + i] = as_float((int) m.refractive);
    st[(f + 10) * cap + i] = as_float((int) m.sss);
    st[(f + 11) * cap + i] = as_float((int) m.isWater);
    st[(f + 12) * cap + i] = as_float(m.tintType);
}

#endif
