#ifndef CHUNKYCLPLUGIN_OCTREE_H
#define CHUNKYCLPLUGIN_OCTREE_H

#include "../opencl.h"
#include "rt.h"
#include "constants.h"
#include "primitives.h"
#include "block.h"
#include "utils.h"

typedef struct {
    __global const int* treeData;
    AABB bounds;
    int depth;
} Octree;

Octree Octree_create(__global const int* treeData, int depth) {
    Octree octree;
    octree.treeData = treeData;
    octree.depth = depth;
    octree.bounds = AABB_new(0, 1<<depth, 0, 1<<depth, 0, 1<<depth);
    return octree;
}

int Octree_get(Octree* self, int x, int y, int z) {
    int3 bp = (int3) (x, y, z);

    // Check inbounds
    int3 lv = bp >> self->depth;
    if ((lv.x != 0) | (lv.y != 0) | (lv.z != 0))
        return 0;

    int level = self->depth;
    int data = self->treeData[0];
    for (int d = 0; d < self->depth && data > 0; d++) {
        level--;
        lv = 1 & (bp >> level);
        data = self->treeData[data + ((lv.x << 2) | (lv.y << 1) | lv.z)];
    }
    return -data;
}

bool Octree_octreeIntersect(Octree self, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette, int drawDepth, Ray ray, IntersectionRecord* record, MaterialSample* sample) {
    float distMarch = 0;

    // Guard against zero direction components: use large finite value instead of
    // infinity to prevent NaN propagation in AABB intersection calculations.
    float3 invD = select(1.0f / ray.direction, copysign((float3)(1e30f), ray.direction), fabs(ray.direction) < 1e-30f);
    // Use a small direction-dependent offset for stepping past block boundaries.
    // sign(direction) gives +1/-1 per axis; multiply by a tiny epsilon so we
    // always nudge in the travel direction to land inside the next cell.
    float3 signD = sign(ray.direction);
    float3 offsetD = signD * EPS;

    int depth = self.depth;

    // Check if we are in bounds
    if (!AABB_inside(self.bounds, ray.origin)) {
        // Attempt to intersect with the octree
        float dist = AABB_quick_intersect(self.bounds, ray.origin, invD);
        if (isnan(dist) || dist < 0) {
            return false;
        } else {
            // Scale offset with distance to prevent float truncation for far cameras
            float entryOffset = fmax(OFFSET, fabs(dist) * 1e-6f);
            distMarch += dist + entryOffset;
        }
    }

    for (int i = 0; i < drawDepth; i++) {
        if (distMarch > record->distance) {
            // There's already been a closer intersection!
            return false;
        }

        float3 pos = ray.origin + ray.direction * distMarch;
        int3 bp = intFloorFloat3(pos + offsetD);

        // Check inbounds
        int3 lv = bp >> depth;
        if (lv.x != 0 || lv.y != 0 || lv.z != 0) {
            return false;
        }

        // Read the octree with depth (bounded to prevent hang on corrupted data)
        int level = depth;
        int data = self.treeData[0];
        for (int d = 0; d < depth && data > 0; d++) {
            level--;
            lv = 1 & (bp >> level);
            data = self.treeData[data + ((lv.x << 2) | (lv.y << 1) | lv.z)];
        }
        data = -data;
        lv = bp >> level;

        // Get block data if there is an intersection.
        // Skip blocks that match the current medium the ray is traveling through
        // (e.g. adjacent water blocks when the ray is already inside water).
        if (data != ray.material) {
            if (BlockPalette_intersectNormalizedBlock(palette, atlas, materialPalette, data, bp, ray, record, sample)) {
                record->blockData = data;
                return true;
            }
        }

        // Exit the current leaf cell and step past the boundary.
        // Use offsetD to ensure the position is inside the cell for exit calc.
        AABB box = AABB_new(lv.x << level, (lv.x + 1) << level,
                            lv.y << level, (lv.y + 1) << level,
                            lv.z << level, (lv.z + 1) << level);
        float exitDist = AABB_exit(box, pos + offsetD, invD);
        // Guard against NaN/negative exitDist which would cause no progress
        if (isnan(exitDist) || exitDist < 0) return false;
        // Step past the cell boundary with a small offset. Use OFFSET as a
        // minimum but also scale with absolute ray position for float stability.
        // Single-precision floats have ~7 decimal digits of precision, so an
        // offset of ~1e-6 relative to the position keeps us above the ULP.
        float absPos = fmax(fabs(distMarch + exitDist), fmax(fabs(pos.x), fmax(fabs(pos.y), fabs(pos.z))));
        distMarch += exitDist + fmax(OFFSET, absPos * 1e-6f);
    }
    return false;
}

/**
 * Exit water traversal — marches through water blocks as a continuous volume.
 * Matches CPU Octree.exitWater(): skips full water blocks, tests surface water
 * top triangles, and stops at non-water blocks (air or solid).
 *
 * When the ray exits water (hits air or a non-water block), sets record with
 * the exit distance and normal, and sample with isWater=true so the caller
 * knows we left a water volume.
 *
 * When the ray hits a water surface (non-full water block with intersectTop hit),
 * sets record with that intersection and sample with isWater=false (exiting into air).
 *
 * blockPalette layout per block: [modelType, materialPointer, waterData]
 * Water modelType = 5. waterData bit 16 = full block flag.
 */
bool Octree_exitWater(Octree self, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette, int drawDepth, Ray ray, IntersectionRecord* record, MaterialSample* sample) {
    float distMarch = 0;

    // Guard against zero direction components: prevent NaN propagation
    float3 invD = select(1.0f / ray.direction, copysign((float3)(1e30f), ray.direction), fabs(ray.direction) < 1e-30f);
    float3 signD = sign(ray.direction);
    float3 offsetD = signD * EPS;

    int depth = self.depth;

    // Check if we are in bounds
    if (!AABB_inside(self.bounds, ray.origin)) {
        float dist = AABB_quick_intersect(self.bounds, ray.origin, invD);
        if (isnan(dist) || dist < 0) {
            return false;
        } else {
            // Scale offset with distance to prevent float truncation for far cameras
            float entryOffset = fmax(OFFSET, fabs(dist) * 1e-6f);
            distMarch += dist + entryOffset;
        }
    }

    for (int i = 0; i < drawDepth; i++) {
        if (distMarch > record->distance) {
            return false;
        }

        float3 pos = ray.origin + ray.direction * distMarch;
        int3 bp = intFloorFloat3(pos + offsetD);

        // Check inbounds
        int3 lv = bp >> depth;
        if (lv.x != 0 || lv.y != 0 || lv.z != 0) {
            return false;
        }

        // Read the octree with depth (bounded to prevent hang on corrupted data)
        int level = depth;
        int data = self.treeData[0];
        for (int d = 0; d < depth && data > 0; d++) {
            level--;
            lv = 1 & (bp >> level);
            data = self.treeData[data + ((lv.x << 2) | (lv.y << 1) | lv.z)];
        }
        data = -data;
        lv = bp >> level;

        // Check if this block is water by reading the block palette modelType
        int modelType = palette.blockPalette[data + 0];

        if (modelType != 5) {
            // Not water — ray has exited the water volume.
            // Report intersection at the cell entry point (current distMarch).
            if (distMarch < OFFSET) {
                // Ray starts outside water in this octree — no water exit
                return false;
            }
            record->distance = distMarch;
            // Normal is the face we entered through (opposite of ray direction's dominant axis at boundary)
            // Use the cell boundary we just crossed
            record->normal = (float3)(0, 1, 0); // default up
            record->material = -1; // water exit marker
            record->blockData = 0;
            // Set up water material sample for the exit boundary
            sample->color = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
            sample->emittance = 0.0f;
            sample->specular = 0.12f;
            sample->metalness = 0.0f;
            sample->roughness = 0.0f;
            sample->ior = 1.333f;
            sample->refractive = true;
            sample->sss = false;
            sample->isWater = true;
            sample->tintType = 3;  // water biome tint (applied by applyBiomeTint)
            return true;
        }

        // It's a water block (modelType == 5)
        int waterData = palette.blockPalette[data + 2];
        bool isFull = (waterData >> WATER_FULL_BLOCK) & 1;
        int materialPointer = palette.blockPalette[data + 1];

        if (!isFull) {
            // Surface water block — test top triangles (matching CPU WaterModel.intersectTop)
            int c0 = ((waterData >> WATER_CORNER_SW) & 0xF) % 8;
            int c1 = ((waterData >> WATER_CORNER_SE) & 0xF) % 8;
            int c2 = ((waterData >> WATER_CORNER_NE) & 0xF) % 8;
            int c3 = ((waterData >> WATER_CORNER_NW) & 0xF) % 8;

            float h0 = WATER_HEIGHT[c0];  // SW: x=0, z=1
            float h1 = WATER_HEIGHT[c1];  // SE: x=1, z=1
            float h2 = WATER_HEIGHT[c2];  // NE: x=1, z=0
            float h3 = WATER_HEIGHT[c3];  // NW: x=0, z=0

            Ray tempRay = ray;
            tempRay.origin = ray.origin - int3toFloat3(bp);
            IntersectionRecord triRecord = *record;
            triRecord.distance = record->distance - distMarch;
            // Adjust for the march distance in block-local space
            tempRay.origin += ray.direction * distMarch;
            bool topHit = false;

            // Triangle 1: t012 — (0,h0,1), (1,h1,1), (1,h2,0)
            {
                float3 v0 = (float3)(0, h0, 1);
                float3 v1 = (float3)(1, h1, 1);
                float3 v2 = (float3)(1, h2, 0);
                IntersectionRecord tr = triRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &tr)) {
                    if (dot(tr.normal, tempRay.direction) > 0)
                        tr.normal = -tr.normal;
                    triRecord = tr;
                    topHit = true;
                }
            }

            // Triangle 2: t230 — (0,h3,0), (0,h0,1), (1,h2,0)
            {
                float3 v0 = (float3)(0, h3, 0);
                float3 v1 = (float3)(0, h0, 1);
                float3 v2 = (float3)(1, h2, 0);
                IntersectionRecord tr = triRecord;
                if (Water_triangleIntersect(v0, v1, v2, tempRay, &tr)) {
                    if (dot(tr.normal, tempRay.direction) > 0)
                        tr.normal = -tr.normal;
                    triRecord = tr;
                    topHit = true;
                }
            }

            if (topHit) {
                // Hit the water surface from below — exiting water
                record->distance = distMarch + triRecord.distance;
                record->normal = triRecord.normal;
                record->material = materialPointer;
                record->blockData = data;
                Material material = Material_get(materialPalette, materialPointer);
                Material_sample(material, atlas, triRecord.texCoord, sample);
                sample->isWater = true;
                return true;
            }

            // No top hit — skip past this block
        }

        // Full water block or surface block with no top hit — skip to cell boundary
        AABB box = AABB_new(lv.x << level, (lv.x + 1) << level,
                            lv.y << level, (lv.y + 1) << level,
                            lv.z << level, (lv.z + 1) << level);
        float exitDist = AABB_exit(box, pos + offsetD, invD);
        // Guard against NaN/negative exitDist
        if (isnan(exitDist) || exitDist < 0) return false;
        float absPos = fmax(fabs(distMarch + exitDist), fmax(fabs(pos.x), fmax(fabs(pos.y), fabs(pos.z))));
        distMarch += exitDist + fmax(OFFSET, absPos * 1e-6f);
    }
    return false;
}

#endif
