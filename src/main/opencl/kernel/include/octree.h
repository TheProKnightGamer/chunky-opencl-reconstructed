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

// Distance to where a ray leaves octree cell lv (at `level`), i.e. AABB_exit of
// that cell. Only the face on the side the ray is heading toward can be the exit
// on each axis, so it is picked by the sign of invD instead of computing both
// faces and taking fmax: same planes, same arithmetic, same result, half the work
// in the innermost loop of every trace.
float Octree_cellExit(int3 lv, int level, float3 origin, float3 invD) {
    int3 far = lv + select((int3)(0), (int3)(1), isgreater(invD, (float3)(0.0f)));
    float3 planes = convert_float3(far << level);
    float3 t = (planes - origin) * invD;
    return fmin(t.x, fmin(t.y, t.z));
}

bool Octree_octreeIntersect(Octree self, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette, int drawDepth, Ray ray, IntersectionRecord* record) {
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

    const int rootNode = self.treeData[0];

    // Palette index of the PREVIOUS cell the DDA walked through, seeded with the
    // medium the ray is currently inside. Mirrors CPU Octree.enterBlock, which
    // keeps ray.prevMaterial/currentMaterial and skips a cell when
    // currentBlock.isSameMaterial(prevBlock) (Octree.java:562) — that is what
    // hides the shared face between two adjacent blocks of the same type.
    // It MUST advance every step: comparing against a medium that is only
    // updated on bounces would keep culling that block type even after the ray
    // has left the volume and crossed air.
    int prevData = ray.material;

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
        int data = rootNode;
        for (int d = 0; d < depth && data > 0; d++) {
            level--;
            lv = 1 & (bp >> level);
            data = self.treeData[data + ((lv.x << 2) | (lv.y << 1) | lv.z)];
        }
        data = -data;
        lv = bp >> level;

        // Decide whether this cell is intersected at all.
        bool skip;
        if (data == 0) {
            // Air. (Octree.ANY_TYPE never gets here: the loader maps it to stone,
            // as the CPU renderer does, because it is not a palette index.) The GPU form of CPU's `currentBlock != Air.INSTANCE` guard.
            // Short-circuited here so empty space costs no palette read.
            skip = true;
        } else if (data == prevData || data == ray.material) {
            int modelType = palette.blockPalette[data];
            // FULL CUBES (modelType 1) cull against the PREVIOUS cell. This is the
            // face-hiding rule, and it mirrors CPU Octree.java:562, which applies
            // isSameMaterial ONLY in its non-localIntersect branch.
            //
            // EVERY OTHER model type keeps the original medium-skip untouched. Do
            // not "simplify" this into one identity test: CPU still runs
            // intersect() on localIntersect blocks when the block repeats, so
            // skipping those by identity drops real geometry — the far pane of two
            // stacked glass panes, the back rail of a double fence. Water
            // (modelType 5) also depends on the medium-skip staying as it was.
            skip = (modelType == 1) ? (data == prevData) : (data == ray.material);
        } else {
            skip = false;
        }

        if (!skip) {
            if (BlockPalette_intersectNormalizedBlock(palette, atlas, materialPalette, data, bp, ray, record)) {
                record->blockData = data;
                return true;
            }
        }
        prevData = data;

        // Exit the current leaf cell and step past the boundary.
        // Use offsetD to ensure the position is inside the cell for exit calc.
        float exitDist = Octree_cellExit(lv, level, pos + offsetD, invD);
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
bool Octree_exitWater(Octree self, image2d_array_t atlas, BlockPalette palette, MaterialPalette materialPalette, int drawDepth, Ray ray, IntersectionRecord* record) {
    float distMarch = 0;

    // Guard against zero direction components: prevent NaN propagation
    float3 invD = select(1.0f / ray.direction, copysign((float3)(1e30f), ray.direction), fabs(ray.direction) < 1e-30f);
    float3 signD = sign(ray.direction);
    float3 offsetD = signD * EPS;

    int depth = self.depth;

    // Track the most-recently-visited water cell. When the ray finally
    // exits to a non-water cell, we use this to populate record.material
    // with the water material id so downstream water-specific logic in
    // closestIntersect (waterOpacity override, biome tint, water shading)
    // fires correctly. Previously we wrote material=-1 which silently
    // disabled all of those — full water blocks looked unshaded and
    // ignored the user's waterOpacity setting.
    int lastWaterMaterial = -1;
    int lastWaterData = 0;

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

    const int rootNode = self.treeData[0];

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
        int data = rootNode;
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
            // Exit-face normal pointing AGAINST ray direction (chunky's
            // convention: surface normal points INTO the previous medium —
            // here that's the water we just left). For the cell boundary
            // most-recently crossed, the dominant ray-direction axis
            // identifies the exit face. CPU PathTracer.invertNormal()
            // achieves the same effect when currentMat==Air after a water
            // hit; previously we hard-coded (0,1,0) which gave wrong
            // refraction for any ray direction other than straight up
            // (and inverted lighting on lateral water-cell exits).
            float3 absD = fabs(ray.direction);
            if (absD.x >= absD.y && absD.x >= absD.z) {
                record->normal = (float3)(-copysign(1.0f, ray.direction.x), 0.0f, 0.0f);
            } else if (absD.y >= absD.z) {
                record->normal = (float3)(0.0f, -copysign(1.0f, ray.direction.y), 0.0f);
            } else {
                record->normal = (float3)(0.0f, 0.0f, -copysign(1.0f, ray.direction.z));
            }

            if (lastWaterMaterial >= 0) {
                // Use the water material from the last cell we marched
                // through, so colour / emittance / ior / specular come from
                // the actual palette (resource-pack water tints included).
                record->material = lastWaterMaterial;
                record->blockData = lastWaterData;
                record->hitKind = HIT_WATER_EXIT_CENTER;
            } else {
                // Fallback for the rare case the caller invoked
                // exitWater with ray.inWater true but no water cell was
                // actually visited (e.g. floating-point boundary case).
                record->material = -1;
                record->blockData = 0;
                record->hitKind = HIT_WATER_EXIT_PLAIN;
            }
            return true;
        }

        // It's a water block (modelType == 5)
        int waterData = palette.blockPalette[data + 2];
        bool isFull = (waterData >> WATER_FULL_BLOCK) & 1;
        int materialPointer = palette.blockPalette[data + 1];
        // Remember this water cell so a future non-water exit can use it.
        lastWaterMaterial = materialPointer;
        lastWaterData = data;

        if (!isFull) {
            // Surface water block — test top triangles (matching CPU WaterModel.intersectTop)
            float h0, h1, h2, h3;
            Water_cornerHeights(waterData, &h0, &h1, &h2, &h3);

            Ray tempRay = ray;
            tempRay.origin = ray.origin - int3toFloat3(bp);
            IntersectionRecord triRecord = *record;
            triRecord.distance = record->distance - distMarch;
            // Adjust for the march distance in block-local space
            tempRay.origin += ray.direction * distMarch;

            if (Water_intersectMesh(0, WATER_TOP_TRIS, h0, h1, h2, h3, tempRay, &triRecord)) {
                // Hit the water surface from below — exiting water
                record->distance = distMarch + triRecord.distance;
                record->normal = triRecord.normal;
                record->texCoord = triRecord.texCoord;
                record->material = materialPointer;
                record->blockData = data;
                record->hitKind = HIT_WATER_EXIT_UV;
                return true;
            }

            // No top hit — skip past this block
        }

        // Full water block or surface block with no top hit — skip to cell boundary
        float exitDist = Octree_cellExit(lv, level, pos + offsetD, invD);
        // Guard against NaN/negative exitDist
        if (isnan(exitDist) || exitDist < 0) return false;
        float absPos = fmax(fabs(distMarch + exitDist), fmax(fabs(pos.x), fmax(fabs(pos.y), fabs(pos.z))));
        distMarch += exitDist + fmax(OFFSET, absPos * 1e-6f);
    }
    return false;
}

#endif
