// This includes stuff regarding blocks and block palettes

#ifndef CHUNKYCLPLUGIN_BLOCK_H
#define CHUNKYCLPLUGIN_BLOCK_H

#include "../opencl.h"
#include "rt.h"
#include "utils.h"
#include "constants.h"
#include "textureAtlas.h"
#include "material.h"
#include "primitives.h"

typedef struct {
    __global const int* blockPalette;
    __global const int* quadModels;
    __global const int* aabbModels;
    MaterialPalette* materialPalette;
} BlockPalette;

BlockPalette BlockPalette_new(__global const int* blockPalette, __global const int* quadModels, __global const int* aabbModels, MaterialPalette* materialPalette) {
    BlockPalette p;
    p.blockPalette = blockPalette;
    p.quadModels = quadModels;
    p.aabbModels = aabbModels;
    p.materialPalette = materialPalette;
    return p;
}

// True if this palette entry is a plain full cube (modelType 1).
// Full cubes are the only blocks the octree DDA culls by identity, mirroring
// CPU Octree.java:562 — its isSameMaterial test lives in the non-localIntersect
// branch, so model blocks (panes, fences, slabs) are still intersected when the
// same block repeats and must NOT be treated as a medium.
bool BlockPalette_isFullCube(BlockPalette self, int block) {
    return block != 0 && self.blockPalette[block] == 1;
}

// Material to take an emitter's light from when an emitter-NEE shadow ray reaches the
// emitter's cell without hitting its geometry. Cubes, light blocks and water store
// their material pointer directly; AABB and quad models (torches, lanterns, candles)
// store a MODEL pointer, which is not a material index, so their first primitive's
// material is used. -1 when the block has none.
int BlockPalette_emitterMaterial(BlockPalette self, int block) {
    int modelType = self.blockPalette[block];
    int pointer = self.blockPalette[block + 1];
    switch (modelType) {
        case 1: case 4: case 5: return pointer;
        case 2: return self.aabbModels[pointer] > 0 ? self.aabbModels[pointer + 1 + 7] : -1;
        case 3: return self.quadModels[pointer] > 0 ? self.quadModels[pointer + 1 + 13] : -1;
        default: return -1;
    }
}

// Water height levels matching CPU Water.java height[] array.
// Index 0 = level 0 (fullest), index 7 = level 7 (lowest).
constant float WATER_HEIGHT[8] = {
    14.0f / 16.0f,     // 0.875
    12.25f / 16.0f,    // 0.765625
    10.5f / 16.0f,     // 0.65625
    8.75f / 16.0f,     // 0.546875
    7.0f / 16.0f,      // 0.4375
    5.25f / 16.0f,     // 0.328125
    3.5f / 16.0f,      // 0.21875
    1.75f / 16.0f      // 0.109375
};

// Water data bit layout (matching CPU Water.java):
#define WATER_CORNER_SW  0
#define WATER_CORNER_SE  4
#define WATER_CORNER_NE  8
#define WATER_CORNER_NW  12
#define WATER_FULL_BLOCK 16

// Moller-Trumbore ray-triangle intersection.
// Returns true if hit; sets distance, normal and texCoord in record.
// v0, v1, v2 are triangle vertices in block-local [0,1]^3 space.
bool Water_triangleIntersect(float3 v0, float3 v1, float3 v2, Ray ray,
                             IntersectionRecord* record) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(ray.direction, e2);
    float det = dot(e1, pvec);

    if (fabs(det) < 1e-7f) return false;

    float invDet = 1.0f / det;
    float3 tvec = ray.origin - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    float3 qvec = cross(tvec, e1);
    float v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = dot(e2, qvec) * invDet;
    if (t < OFFSET || t >= record->distance) return false;

    record->distance = t;
    record->normal = normalize(cross(e1, e2));
    record->texCoord = (float2)(u, v);
    return true;
}

// Water block surface mesh, in the exact order WaterModel's triangles were
// tested before (ties between equally distant triangles resolve to the earlier
// one, so the order is part of the result). Each triangle packs three 3-bit
// vertex codes: bit 0 = x, bit 1 = z, bit 2 = "on the surface" (y = corner
// height; otherwise y = 0). The corner heights are SW(x0,z1)=h0, SE(x1,z1)=h1,
// NE(x1,z0)=h2, NW(x0,z0)=h3. The first two are the top surface.
#define WATER_TRI(a, b, c) ((a) | ((b) << 3) | ((c) << 6))
constant ushort WATER_TRIS[10] = {
    WATER_TRI(6, 7, 5),  // top t012: (0,h0,1) (1,h1,1) (1,h2,0)
    WATER_TRI(4, 6, 5),  // top t230: (0,h3,0) (0,h0,1) (1,h2,0)
    WATER_TRI(4, 0, 6),  // west  t
    WATER_TRI(2, 6, 0),  // west  b
    WATER_TRI(5, 7, 1),  // east  t
    WATER_TRI(7, 3, 1),  // east  b
    WATER_TRI(6, 2, 7),  // south t
    WATER_TRI(3, 7, 2),  // south b
    WATER_TRI(4, 5, 0),  // north t
    WATER_TRI(1, 0, 5),  // north b
};
#define WATER_TOP_TRIS 2
#define WATER_ALL_TRIS 10

float3 Water_vertex(uint code, float h0, float h1, float h2, float h3) {
    bool x = code & 1;
    bool z = (code >> 1) & 1;
    float h = z ? (x ? h1 : h0) : (x ? h2 : h3);
    return (float3)(x ? 1.0f : 0.0f, ((code >> 2) & 1) ? h : 0.0f, z ? 1.0f : 0.0f);
}

// Intersects triangles [first, last) of the water mesh in block-local space,
// keeping the closest hit in *record with its normal facing the ray. The loop
// is deliberately NOT unrolled: this runs inside every inlined trace.
bool Water_intersectMesh(int first, int last, float h0, float h1, float h2, float h3,
                         Ray ray, IntersectionRecord* record) {
    bool hit = false;
    __attribute__((opencl_unroll_hint(1)))
    for (int i = first; i < last; i++) {
        uint tri = WATER_TRIS[i];
        float3 v0 = Water_vertex(tri & 7, h0, h1, h2, h3);
        float3 v1 = Water_vertex((tri >> 3) & 7, h0, h1, h2, h3);
        float3 v2 = Water_vertex((tri >> 6) & 7, h0, h1, h2, h3);
        if (Water_triangleIntersect(v0, v1, v2, ray, record)) {
            if (dot(record->normal, ray.direction) > 0)
                record->normal = -record->normal;
            hit = true;
        }
    }
    return hit;
}

// Corner heights of a surface water block.
void Water_cornerHeights(int waterData, float* h0, float* h1, float* h2, float* h3) {
    *h0 = WATER_HEIGHT[((waterData >> WATER_CORNER_SW) & 0xF) % 8];  // SW: x=0, z=1
    *h1 = WATER_HEIGHT[((waterData >> WATER_CORNER_SE) & 0xF) % 8];  // SE: x=1, z=1
    *h2 = WATER_HEIGHT[((waterData >> WATER_CORNER_NE) & 0xF) % 8];  // NE: x=1, z=0
    *h3 = WATER_HEIGHT[((waterData >> WATER_CORNER_NW) & 0xF) % 8];  // NW: x=0, z=0
}

// Flips a full-cube face UV into block texture orientation.
float2 BlockPalette_cubeUv(IntersectionRecord r) {
    float2 uv = r.texCoord;
    if (r.normal.x > 0 || r.normal.z < 0) uv.x = 1 - uv.x;
    if (r.normal.y > 0) uv.y = 1 - uv.y;
    return uv;
}

bool BlockPalette_intersectNormalizedBlock(BlockPalette self, image2d_array_t atlas, MaterialPalette materialPalette, int block, int3 blockPosition, Ray ray, IntersectionRecord* record) {
    // ANY_TYPE. Should not be intersected.
    if (block == 0x7FFFFFFE) {
        return false;
    }

    int modelType = self.blockPalette[block + 0];
    int modelPointer = self.blockPalette[block + 1];

    bool hit = false;
    Ray tempRay = ray;
    tempRay.origin = ray.origin - int3toFloat3(blockPosition);

    IntersectionRecord tempRecord = *record;

    switch (modelType) {
        default:
        case 0: {
            return false;
        }
        case 1: {
            // Full size block (non-water)
            AABB box = AABB_new(0, 1, 0, 1, 0, 1);
            if (!AABB_full_intersect(box, tempRay, &tempRecord)) return false;
            tempRecord.texCoord = BlockPalette_cubeUv(tempRecord);
            if (!Material_alphaTest(Material_get(materialPalette, modelPointer), atlas, tempRecord.texCoord))
                return false;
            tempRecord.material = modelPointer;
            tempRecord.hitKind = HIT_MATERIAL;
            *record = tempRecord;
            return true;
        }
        case 2: {
            int boxes = self.aabbModels[modelPointer];
            for (int i = 0; i < boxes; i++) {
                int offset = modelPointer + 1 + i * TEX_AABB_SIZE;
                TexturedAABB box = TexturedAABB_new(self.aabbModels, offset);
                hit |= TexturedAABB_intersect(box, atlas, materialPalette, tempRay, record);
            }
            return hit;
        }
        case 3: {
            int quads = self.quadModels[modelPointer];
            for (int i = 0; i < quads; i++) {
                int offset = modelPointer + 1 + i * QUAD_SIZE;
                Quad q = Quad_new(self.quadModels, offset);
                hit |= Quad_intersect(q, atlas, materialPalette, tempRay, record);
            }
            return hit;
        }
        case 4: {
            // Light block (minecraft:light) — mirrors CPU LightBlock.intersect():
            //   Preview ray: inset cube textured with Texture.light (opaque
            //     white where the texture is transparent) so the block can be
            //     seen and placed while editing.
            //   Shadow ray (sun/emitter NEE): SKIP entirely (return false), so
            //     emitter-NEE shadow rays reach the emitter face instead of the
            //     inset cube; rayTracer.c's "emitter invisible to rays" branch
            //     then samples the light block's emittance for NEE.
            //   Path-trace ray: invisible to camera/specular rays and to all
            //     rays when emitters are off or the block doesn't emit. Diffuse
            //     indirect rays hit the inset cube as an OPAQUE flat-white
            //     surface, so the self-emission block in rayTracer.c adds
            //     emittance * emitterIntensity.
            if (ray.flags & RAY_SHADOW) {
                return false;
            }
            bool lightPreview = (ray.flags & RAY_PREVIEW) != 0;
            if (!lightPreview
                && !((ray.flags & RAY_EMITTERS) && (ray.flags & RAY_INDIRECT))) {
                return false;
            }
            AABB box = AABB_new(0.125f, 0.875f, 0.125f, 0.875f, 0.125f, 0.875f);
            if (!AABB_full_intersect(box, tempRay, &tempRecord)) {
                return false;
            }
            tempRecord.texCoord = BlockPalette_cubeUv(tempRecord);
            tempRecord.material = modelPointer;
            Material material = Material_get(materialPalette, modelPointer);
            if (!Material_alphaTest(material, atlas, tempRecord.texCoord)) {
                return false;
            }
            // CPU LightBlock: no emission (level 0, or user zeroed it) means not
            // intersectable at all; otherwise a flat opaque white surface (the
            // glow comes from emittance in the path tracer, not the texture).
            if (!lightPreview && Material_emittanceAt(material, atlas, tempRecord.texCoord) <= EPS) {
                return false;
            }
            tempRecord.hitKind = lightPreview ? HIT_MATERIAL : HIT_LIGHT_WHITE;
            *record = tempRecord;
            return true;
        }
        case 5: {
            // Water block with per-corner height data.
            // Word 2 contains water data: bits 0-3=SW, 4-7=SE, 8-11=NE, 12-15=NW, bit 16=full.
            int waterData = self.blockPalette[block + 2];
            bool isFull = (waterData >> WATER_FULL_BLOCK) & 1;

            if (isFull) {
                // Submerged water: full cube (block above is also water)
                AABB box = AABB_new(0, 1, 0, 1, 0, 1);
                if (!AABB_full_intersect(box, tempRay, &tempRecord)) return false;
                tempRecord.texCoord = BlockPalette_cubeUv(tempRecord);
            } else {
                // Surface water block: bottom face plus the triangulated top and
                // sides with per-corner heights.
                float h0, h1, h2, h3;
                Water_cornerHeights(waterData, &h0, &h1, &h2, &h3);

                // Bottom face: (0,0,0) (1,0,0) (0,0,1) — always at y=0
                float3 o = tempRay.origin;
                if (fabs(tempRay.direction.y) > 1e-7f) {
                    float t = (0.0f - o.y) / tempRay.direction.y;
                    if (t > OFFSET && t < tempRecord.distance) {
                        float3 hp = o + tempRay.direction * t;
                        if (hp.x >= 0.0f && hp.x <= 1.0f && hp.z >= 0.0f && hp.z <= 1.0f) {
                            tempRecord.distance = t;
                            tempRecord.normal = (float3)(0, -1, 0);
                            tempRecord.texCoord = (float2)(hp.x, hp.z);
                            hit = true;
                        }
                    }
                }
                hit |= Water_intersectMesh(0, WATER_ALL_TRIS, h0, h1, h2, h3, tempRay, &tempRecord);
                if (!hit) return false;
            }
            if (!Material_alphaTest(Material_get(materialPalette, modelPointer), atlas, tempRecord.texCoord))
                return false;
            tempRecord.material = modelPointer;
            tempRecord.hitKind = HIT_MATERIAL;
            *record = tempRecord;
            return true;
        }
    }
}

#endif
