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

bool BlockPalette_intersectNormalizedBlock(BlockPalette self, image2d_array_t atlas, MaterialPalette materialPalette, int block, int3 blockPosition, Ray ray, IntersectionRecord* record, MaterialSample* sample) {
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
            hit = AABB_full_intersect(box, tempRay, &tempRecord);
            tempRecord.material = modelPointer;
            if (hit) {
                if (tempRecord.normal.x > 0 || tempRecord.normal.z < 0) {
                    tempRecord.texCoord.x = 1 - tempRecord.texCoord.x;
                }
                if (tempRecord.normal.y > 0) {
                    tempRecord.texCoord.y = 1 - tempRecord.texCoord.y;
                }
                Material material = Material_get(materialPalette, modelPointer);
                hit = Material_sample(material, atlas, tempRecord.texCoord, sample);
                if (hit) {
                    *record = tempRecord;
                    return true;
                } else {
                    return false;
                }
            }
            return false;
        }
        case 2: {
            int boxes = self.aabbModels[modelPointer];
            for (int i = 0; i < boxes; i++) {
                int offset = modelPointer + 1 + i * TEX_AABB_SIZE;
                TexturedAABB box = TexturedAABB_new(self.aabbModels, offset);
                hit |= TexturedAABB_intersect(box, atlas, materialPalette, tempRay, record, sample);
            }
            return hit;
        }
        case 3: {
            int quads = self.quadModels[modelPointer];
            for (int i = 0; i < quads; i++) {
                int offset = modelPointer + 1 + i * QUAD_SIZE;
                Quad q = Quad_new(self.quadModels, offset);
                hit |= Quad_intersect(q, atlas, materialPalette, tempRay, record, sample);
            }
            return hit;
        }
        case 4: {
            // Light block - invisible emitter in path tracing, visible in preview.
            // Intersects as a small inset cube (0.125-0.875) matching CPU LightBlockModel.
            // In path tracing: alpha is forced to 0 so Material_samplePdf always chooses
            // doTransmit (pDiffuse=0), making the ray pass straight through with
            // spectrum=(1,1,1). The block is invisible, but self-emission still fires
            // at the hit point because it uses sample.color.xyz (ignoring alpha).
            // In preview: alpha stays at 1 so the block is visible with placeholder texture.
            AABB box = AABB_new(0.125f, 0.875f, 0.125f, 0.875f, 0.125f, 0.875f);
            hit = AABB_full_intersect(box, tempRay, &tempRecord);
            tempRecord.material = modelPointer;
            if (hit) {
                if (tempRecord.normal.x > 0 || tempRecord.normal.z < 0) {
                    tempRecord.texCoord.x = 1 - tempRecord.texCoord.x;
                }
                if (tempRecord.normal.y > 0) {
                    tempRecord.texCoord.y = 1 - tempRecord.texCoord.y;
                }
                Material material = Material_get(materialPalette, tempRecord.material);
                hit = Material_sample(material, atlas, tempRecord.texCoord, sample);
                if (hit) {
                    if (!(ray.flags & RAY_PREVIEW)) {
                        sample->color.w = 0.0f;
                    }
                    *record = tempRecord;
                    return true;
                }
            }
            return false;
        }
        case 5: {
            // Water block with per-corner height data.
            // Word 2 contains water data: bits 0-3=SW, 4-7=SE, 8-11=NE, 12-15=NW, bit 16=full.
            int waterData = self.blockPalette[block + 2];
            bool isFull = (waterData >> WATER_FULL_BLOCK) & 1;
            Material material = Material_get(materialPalette, modelPointer);

            if (isFull) {
                // Submerged water: full cube (block above is also water)
                AABB box = AABB_new(0, 1, 0, 1, 0, 1);
                hit = AABB_full_intersect(box, tempRay, &tempRecord);
                tempRecord.material = modelPointer;
                if (hit) {
                    if (tempRecord.normal.x > 0 || tempRecord.normal.z < 0) {
                        tempRecord.texCoord.x = 1 - tempRecord.texCoord.x;
                    }
                    if (tempRecord.normal.y > 0) {
                        tempRecord.texCoord.y = 1 - tempRecord.texCoord.y;
                    }
                    hit = Material_sample(material, atlas, tempRecord.texCoord, sample);
                    if (hit) {
                        *record = tempRecord;
                        return true;
                    }
                }
                return false;
            }

            // Surface water block: triangulated top surface with per-corner heights.
            // Extract corner height indices (4 bits each, mod 8).
            int c0 = ((waterData >> WATER_CORNER_SW) & 0xF) % 8;  // SW corner
            int c1 = ((waterData >> WATER_CORNER_SE) & 0xF) % 8;  // SE corner
            int c2 = ((waterData >> WATER_CORNER_NE) & 0xF) % 8;  // NE corner
            int c3 = ((waterData >> WATER_CORNER_NW) & 0xF) % 8;  // NW corner

            float h0 = WATER_HEIGHT[c0];  // SW: x=0, z=1
            float h1 = WATER_HEIGHT[c1];  // SE: x=1, z=1
            float h2 = WATER_HEIGHT[c2];  // NE: x=1, z=0
            float h3 = WATER_HEIGHT[c3];  // NW: x=0, z=0

            tempRecord.material = modelPointer;
            IntersectionRecord triRecord;

            // Bottom face
            // Quad: (0,0,0) (1,0,0) (0,0,1) — always at y=0
            {
                AABB bottom = AABB_new(0, 1, 0, 0, 0, 1);
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
            }

            // Top surface triangle 1: t012 — vertices at (0,h0,1), (1,h1,1), (1,h2,0)
            // This is the SE triangle of the water surface.
            {
                float3 v0 = (float3)(0, h0, 1);
                float3 v1 = (float3)(1, h1, 1);
                float3 v2 = (float3)(1, h2, 0);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    // Orient normal to face the ray
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }

            // Top surface triangle 2: t230 — vertices at (0,h3,0), (0,h0,1), (1,h2,0)
            // This is the NW triangle of the water surface.
            {
                float3 v0 = (float3)(0, h3, 0);
                float3 v1 = (float3)(0, h0, 1);
                float3 v2 = (float3)(1, h2, 0);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }

            // West side (x=0): two triangles connecting top edge to bottom
            // westt: (0,h3,0), (0,0,0), (0,h0,1)
            {
                float3 v0 = (float3)(0, h3, 0);
                float3 v1 = (float3)(0, 0, 0);
                float3 v2 = (float3)(0, h0, 1);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }
            // westb: (0,0,1), (0,h0,1), (0,0,0)
            {
                float3 v0 = (float3)(0, 0, 1);
                float3 v1 = (float3)(0, h0, 1);
                float3 v2 = (float3)(0, 0, 0);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }

            // East side (x=1): two triangles
            // eastt: (1,h2,0), (1,h1,1), (1,0,0)
            {
                float3 v0 = (float3)(1, h2, 0);
                float3 v1 = (float3)(1, h1, 1);
                float3 v2 = (float3)(1, 0, 0);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }
            // eastb: (1,h1,1), (1,0,1), (1,0,0)
            {
                float3 v0 = (float3)(1, h1, 1);
                float3 v1 = (float3)(1, 0, 1);
                float3 v2 = (float3)(1, 0, 0);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }

            // South side (z=1): two triangles
            // southt: (0,h0,1), (0,0,1), (1,h1,1)
            {
                float3 v0 = (float3)(0, h0, 1);
                float3 v1 = (float3)(0, 0, 1);
                float3 v2 = (float3)(1, h1, 1);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }
            // southb: (1,0,1), (1,h1,1), (0,0,1)
            {
                float3 v0 = (float3)(1, 0, 1);
                float3 v1 = (float3)(1, h1, 1);
                float3 v2 = (float3)(0, 0, 1);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }

            // North side (z=0): two triangles
            // northt: (0,h3,0), (1,h2,0), (0,0,0)
            {
                float3 v0 = (float3)(0, h3, 0);
                float3 v1 = (float3)(1, h2, 0);
                float3 v2 = (float3)(0, 0, 0);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }
            // northb: (1,0,0), (0,0,0), (1,h2,0)
            {
                float3 v0 = (float3)(1, 0, 0);
                float3 v1 = (float3)(0, 0, 0);
                float3 v2 = (float3)(1, h2, 0);
                triRecord = tempRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &triRecord)) {
                    if (dot(triRecord.normal, tempRay.direction) > 0)
                        triRecord.normal = -triRecord.normal;
                    triRecord.material = modelPointer;
                    tempRecord = triRecord;
                    hit = true;
                }
            }

            // If any triangle hit, sample the material
            if (hit) {
                hit = Material_sample(material, atlas, tempRecord.texCoord, sample);
                if (hit) {
                    *record = tempRecord;
                    return true;
                }
            }
            return false;
        }
    }
}

#endif
