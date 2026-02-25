#ifndef CHUNKYCL_KERNEL_H
#define CHUNKYCL_KERNEL_H

#include "../opencl.h"
#include "rt.h"
#include "octree.h"
#include "block.h"
#include "constants.h"
#include "bvh.h"
#include "sky.h"
#include "water.h"


typedef struct {
    Octree octree;
    Octree waterOctree;
    Bvh worldBvh;
    Bvh actorBvh;
    BlockPalette blockPalette;
    MaterialPalette materialPalette;
    int drawDepth;

    // Water config
    bool waterPlaneEnabled;
    float waterPlaneHeight;
    bool waterPlaneChunkClip;
    float octreeSize;
    int waterShadingStrategy;
    float animationTime;
    float waterVisibility;
    float3 waterColor;
    bool useCustomWaterColor;
    float waterIor;
    float waterOpacity;
    WaterShaderParams waterShaderParams;
    __global const float* waterNormalMap;  // Precomputed gradient normal map (dx, dz pairs)
    int waterNormalMapW;                   // Width of the normal map texture (e.g. 512)

    // Cloud config
    bool cloudsEnabled;
    float cloudHeight;
    float cloudSize;
    float cloudOffsetX;  // Pre-computed: origin.x / cloudSize + cloudXOffset
    float cloudOffsetZ;  // Pre-computed: origin.z / cloudSize + cloudZOffset
    __global const int* cloudData;  // Minecraft cloud bitmap (32x32 tiles of 64-bit longs, packed as 2048 ints)

    // Biome tinting
    __global const int* biomeData;  // 4 ints per (x,z): grass, foliage, water, dryFoliage as packed ARGB
    int biomeDataSize;              // octreeSize (0 = no biome data)
    bool biomeColorsEnabled;

    // Chunk bitmap for water plane clipping (1 bit per 16x16 chunk)
    __global const int* chunkBitmap;
    int chunkBitmapSize;  // chunks per side (octreeSize / 16)

    // Scene flags
    bool transparentSky;
} SceneConfig;

// Cloud bitmap lookup: returns 1 if cloud cell at grid position (x, z), else 0.
// The bitmap is a 256x256 repeating grid, stored as 32x32 tiles of 8x8 sub-grids
// packed as 64-bit longs (2 ints each: low 32, high 32).
int Cloud_getCell(__global const int* cloudData, int x, int z) {
    x = ((x % 256) + 256) % 256;
    z = ((z % 256) + 256) % 256;
    int tileX = x >> 3;
    int tileZ = z >> 3;
    int subX = x & 7;
    int subZ = z & 7;
    int idx = (tileX * 32 + tileZ) * 2;
    int bitPos = subZ * 8 + subX;
    int word = (bitPos < 32) ? cloudData[idx] : cloudData[idx + 1];
    return (word >> (bitPos & 31)) & 1;
}

bool Cloud_inCloud(__global const int* cloudData, float x, float z) {
    return Cloud_getCell(cloudData, (int)floor(x), (int)floor(z)) == 1;
}


bool closestIntersect(SceneConfig self, image2d_array_t atlas, Ray ray, IntersectionRecord* record, MaterialSample* sample, Material* mat) {
    float distance = 0;

    for (int i = 0; i < self.drawDepth; i++) {
        IntersectionRecord tempRecord = *record;
        tempRecord.distance = record->distance - distance;

        Ray tempRay = ray;
        tempRay.origin += distance * tempRay.direction;

        if (tempRecord.distance <= 0) {
            return false;
        }

        // --- Trace the WORLD octree (enterBlock equivalent) ---
        bool hit = false;
        IntersectionRecord worldRecord = tempRecord;
        MaterialSample worldSample;
        bool worldHit = Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, &worldRecord, &worldSample);

        // --- Trace BVHs (entities, custom models) ---
        IntersectionRecord bvhRecord = tempRecord;
        MaterialSample bvhSample;
        bool bvhHit = Bvh_intersect(self.worldBvh, atlas, self.materialPalette, tempRay, &bvhRecord, &bvhSample);
        bvhHit |= Bvh_intersect(self.actorBvh, atlas, self.materialPalette, tempRay, &bvhRecord, &bvhSample);

        // Take the closest of world octree and BVH hits
        if (worldHit && (!bvhHit || worldRecord.distance <= bvhRecord.distance)) {
            tempRecord = worldRecord;
            *sample = worldSample;
            hit = true;
        } else if (bvhHit) {
            tempRecord = bvhRecord;
            *sample = bvhSample;
            hit = true;
        }

        // --- Trace the WATER octree (dual-octree, matching CPU Scene.worldIntersection) ---
        {
            IntersectionRecord waterRecord = tempRecord;
            if (!hit) waterRecord.distance = record->distance - distance;
            MaterialSample waterSample;
            bool waterHit = false;

            if (ray.inWater) {
                // Ray is in water -> use exitWater on the water octree.
                // Give exitWater a slight epsilon advantage (matching CPU: ray.t - EPSILON).
                waterRecord.distance = hit ? (tempRecord.distance + EPS) : (record->distance - distance);
                waterHit = Octree_exitWater(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, &waterRecord, &waterSample);
            } else {
                // Ray is not in water -> use enterBlock (octreeIntersect) on the water octree.
                waterHit = Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, tempRay, &waterRecord, &waterSample);
            }

            if (waterHit && (!hit || waterRecord.distance < tempRecord.distance)) {
                tempRecord = waterRecord;
                *sample = waterSample;
                hit = true;
            }
        }

        // Water plane intersection
        if (self.waterPlaneEnabled) {
            IntersectionRecord waterRecord = tempRecord;
            if (Water_planeIntersect(tempRay, self.waterPlaneHeight, self.octreeSize,
                                     self.waterPlaneChunkClip,
                                     self.chunkBitmap, self.chunkBitmapSize,
                                     &waterRecord)) {
                if (!hit || waterRecord.distance < tempRecord.distance) {
                    tempRecord = waterRecord;
                    // Set up water material sample (alpha = waterOpacity, matching CPU)
                    sample->color = (float4)(1.0f, 1.0f, 1.0f, self.waterOpacity);
                    sample->tintType = 0;
                    if (self.useCustomWaterColor) {
                        sample->color.xyz = self.waterColor;
                    } else {
                        // Biome water color will be applied by applyBiomeTint
                        sample->tintType = 3;
                    }
                    sample->emittance = 0.0f;
                    sample->specular = 0.12f;
                    sample->metalness = 0.0f;
                    sample->roughness = 0.0f;
                    sample->ior = self.waterIor;
                    sample->refractive = true;
                    sample->sss = false;
                    sample->isWater = true;
                    // Apply water shading (wave normals) using world-space hit coords
                    // texCoord stores world-space XZ from Water_planeIntersect
                    tempRecord.geomNormal = tempRecord.normal;  // Save geometric normal before wave perturbation
                    Water_applyShading(&tempRecord, self.waterShadingStrategy,
                                       self.animationTime, tempRecord.texCoord.x,
                                       tempRecord.texCoord.y, self.waterShaderParams,
                                       self.waterNormalMap, self.waterNormalMapW);
                    hit = true;
                }
            }
        }

        // Cloud volume intersection using 2D DDA through 5-block-thick slab,
        // matching CPU Sky.cloudIntersection().
        // Cloud slab: cloudHeight to cloudHeight+5 in Y (octree-local).
        // Cloud cells: 256x256 repeating bitmap scaled by cloudSize, offset by cloudOffset.
        if (self.cloudsEnabled) {
            float inv_size = 1.0f / self.cloudSize;
            float cloudTop = self.cloudHeight + 5.0f;
            float oy = tempRay.origin.y;

            int target = 1;  // 1 = looking for cloud entry, 0 = looking for cloud exit
            float t_offset = 0.0f;
            bool cloudHit = false;
            float cloudT = 0.0f;
            float3 cloudNormal = (float3)(0, 0, 0);

            if (oy < self.cloudHeight || oy > cloudTop) {
                // Ray starts outside the cloud slab
                if (fabs(tempRay.direction.y) < 1e-5f) {
                    goto cloud_done;  // Horizontal ray, can't enter slab
                }
                if (tempRay.direction.y > 0) {
                    t_offset = (self.cloudHeight - oy) / tempRay.direction.y;
                } else {
                    t_offset = (cloudTop - oy) / tempRay.direction.y;
                }
                if (t_offset < 0) {
                    goto cloud_done;  // Slab is behind ray
                }
                // Check if entry point is in a cloud cell
                float ex = (tempRay.direction.x * t_offset + tempRay.origin.x) * inv_size + self.cloudOffsetX;
                float ez = (tempRay.direction.z * t_offset + tempRay.origin.z) * inv_size + self.cloudOffsetZ;
                if (Cloud_inCloud(self.cloudData, ex, ez)) {
                    // Direct hit on slab top/bottom
                    cloudT = t_offset;
                    cloudNormal = (float3)(0, -sign(tempRay.direction.y), 0);
                    cloudHit = true;
                    goto cloud_apply;
                }
            } else if (Cloud_inCloud(self.cloudData,
                                     tempRay.origin.x * inv_size + self.cloudOffsetX,
                                     tempRay.origin.z * inv_size + self.cloudOffsetZ)) {
                target = 0;  // Ray starts inside a cloud cell, look for exit
            }

            // Compute tExit: max DDA traverse distance from traversal start
            {
                float tExit;
                if (tempRay.direction.y > 0) {
                    tExit = (cloudTop - oy) / tempRay.direction.y - t_offset;
                } else if (tempRay.direction.y < 0) {
                    tExit = (self.cloudHeight - oy) / tempRay.direction.y - t_offset;
                } else {
                    tExit = 1e30f;  // Horizontal ray through slab
                }
                float maxT = hit ? tempRecord.distance : (record->distance - distance);
                if (maxT - t_offset < tExit) {
                    tExit = maxT - t_offset;
                }

                // DDA setup in cloud-grid coordinates
                float dx = fabs(tempRay.direction.x) * inv_size;
                float dz = fabs(tempRay.direction.z) * inv_size;

                // Handle vertical/near-vertical rays: no horizontal movement, no DDA needed
                if (dx < 1e-10f && dz < 1e-10f) {
                    goto cloud_done;
                }

                float x0 = (tempRay.origin.x + tempRay.direction.x * t_offset) * inv_size + self.cloudOffsetX;
                float z0 = (tempRay.origin.z + tempRay.direction.z * t_offset) * inv_size + self.cloudOffsetZ;
                float xp = x0;
                float zp = z0;
                int ix = (int)floor(xp);
                int iz = (int)floor(zp);
                int xmod = (tempRay.direction.x > 0) ? 1 : ((tempRay.direction.x < 0) ? -1 : 0);
                int zmod = (tempRay.direction.z > 0) ? 1 : ((tempRay.direction.z < 0) ? -1 : 0);
                int xo = (1 + xmod) / 2;
                int zo = (1 + zmod) / 2;
                float t = 0;
                int i = 0;
                int nx = 0, nz = 0;

                if (dx > dz) {
                    float m = dz / dx;
                    float xrem = xmod * (ix + xo - xp);
                    float zlimit = xrem * m;
                    for (int iter = 0; iter < 512 && t < tExit; iter++) {
                        float zrem = zmod * (iz + zo - zp);
                        if (zrem < zlimit) {
                            iz += zmod;
                            if (Cloud_getCell(self.cloudData, ix, iz) == target) {
                                t = i / dx + zrem / dz;
                                nx = 0; nz = -zmod;
                                break;
                            }
                            ix += xmod;
                            if (Cloud_getCell(self.cloudData, ix, iz) == target) {
                                t = (i + xrem) / dx;
                                nx = -xmod; nz = 0;
                                break;
                            }
                        } else {
                            ix += xmod;
                            if (Cloud_getCell(self.cloudData, ix, iz) == target) {
                                t = (i + xrem) / dx;
                                nx = -xmod; nz = 0;
                                break;
                            }
                            if (zrem <= m) {
                                iz += zmod;
                                if (Cloud_getCell(self.cloudData, ix, iz) == target) {
                                    t = i / dx + zrem / dz;
                                    nx = 0; nz = -zmod;
                                    break;
                                }
                            }
                        }
                        t = i / dx;
                        i += 1;
                        zp = z0 + zmod * i * m;
                    }
                } else {
                    float m = (dz > 0) ? dx / dz : 0;
                    float zrem0 = zmod * (iz + zo - zp);
                    float xlimit = zrem0 * m;
                    for (int iter = 0; iter < 512 && t < tExit; iter++) {
                        float xrem = xmod * (ix + xo - xp);
                        if (xrem < xlimit) {
                            ix += xmod;
                            if (Cloud_getCell(self.cloudData, ix, iz) == target) {
                                t = i / dz + xrem / dx;
                                nx = -xmod; nz = 0;
                                break;
                            }
                            iz += zmod;
                            if (Cloud_getCell(self.cloudData, ix, iz) == target) {
                                t = (i + zrem0) / dz;
                                nx = 0; nz = -zmod;
                                break;
                            }
                        } else {
                            iz += zmod;
                            if (Cloud_getCell(self.cloudData, ix, iz) == target) {
                                t = (i + zrem0) / dz;
                                nx = 0; nz = -zmod;
                                break;
                            }
                            if (xrem <= m) {
                                ix += xmod;
                                if (Cloud_getCell(self.cloudData, ix, iz) == target) {
                                    t = i / dz + xrem / dx;
                                    nx = -xmod; nz = 0;
                                    break;
                                }
                            }
                        }
                        t = i / dz;
                        i += 1;
                        xp = x0 + xmod * i * m;
                    }
                }

                // Process DDA result
                int ny = 0;
                if (target == 1) {
                    // Looking for entry
                    if (t > tExit) goto cloud_done;
                    if (nx == 0 && ny == 0 && nz == 0) goto cloud_done;
                    cloudNormal = (float3)((float)nx, (float)ny, (float)nz);
                    cloudT = t + t_offset;
                    cloudHit = true;
                } else {
                    // Inside cloud, looking for exit
                    if (t > tExit) {
                        nx = 0; ny = (tempRay.direction.y > 0) ? 1 : -1; nz = 0;
                        t = tExit;
                    } else {
                        nx = -nx; nz = -nz;
                    }
                    if (nx == 0 && ny == 0 && nz == 0) goto cloud_done;
                    cloudNormal = (float3)((float)nx, (float)ny, (float)nz);
                    cloudT = t + t_offset;
                    cloudHit = true;
                }
            }

            cloud_apply:
            if (cloudHit && cloudT > OFFSET && (!hit || cloudT < tempRecord.distance)) {
                tempRecord.distance = cloudT;
                tempRecord.normal = cloudNormal;
                float3 hitP = tempRay.origin + tempRay.direction * cloudT;
                tempRecord.texCoord = (float2)(hitP.x * inv_size + self.cloudOffsetX - floor(hitP.x * inv_size + self.cloudOffsetX),
                                               hitP.z * inv_size + self.cloudOffsetZ - floor(hitP.z * inv_size + self.cloudOffsetZ));
                sample->color = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
                sample->emittance = 0.0f;
                sample->specular = 0.0f;
                sample->metalness = 0.0f;
                sample->roughness = 1.0f;
                sample->ior = AIR_IOR;
                sample->refractive = false;
                sample->sss = false;
                sample->isWater = false;
                sample->tintType = 0;
                hit = true;
            }
            cloud_done:;
        }

        if (!hit) {
            return false;
        }

        distance += tempRecord.distance;

        *record = tempRecord;
        record->distance = distance;
        record->geomNormal = record->normal;  // Default: geomNormal = normal (overridden below for water)
        if (record->material >= 0) {
            *mat = Material_get(self.materialPalette, record->material);
        }

        // Apply scene water settings to water blocks (matches CPU Scene.updateOpacity).
        // CPU overrides: alpha = waterOpacity, color = custom or biome-tinted.
        if (sample->isWater && record->material >= 0) {
            sample->color.w = self.waterOpacity;
            if (self.useCustomWaterColor) {
                sample->color.xyz = self.waterColor;
                sample->tintType = 0;  // Custom color replaces biome tint
            }
            // Apply wave normals on water faces with non-zero Y component (air-water boundary),
            // matching CPU check: ray.getNormal().y != 0
            if (record->normal.y != 0.0f) {
                record->geomNormal = record->normal;  // Save geometric normal before wave perturbation
                float3 hitPos = ray.origin + ray.direction * record->distance;
                Water_applyShading(record, self.waterShadingStrategy,
                                   self.animationTime, hitPos.x, hitPos.z, self.waterShaderParams,
                                   self.waterNormalMap, self.waterNormalMapW);
            }
        }

        return true;
    }
    return false;
}

// Apply biome tinting to a MaterialSample based on the hit world position.
// Called after intersection when sample->tintType is nonzero (biome-dependent tint).
// Hardcoded fallback colors (linear RGB, matching Biomes.biomesPrePalette[0]):
//   foliage: 0xFF71A74D  grass: 0xFF8EB971  water: 0xFF3F76E4  dryFoliage: 0xFF6A7039
void applyBiomeTint(SceneConfig scene, MaterialSample* sample, float3 hitPos) {
    if (sample->tintType == 0) return;

    float3 tintColor;
    if (scene.biomeColorsEnabled && scene.biomeDataSize > 0) {
        // Look up biome color from the 2D grid
        int bx = (int)floor(hitPos.x);
        int bz = (int)floor(hitPos.z);
        if (bx >= 0 && bx < scene.biomeDataSize && bz >= 0 && bz < scene.biomeDataSize) {
            int idx = (bz * scene.biomeDataSize + bx) * 4 + (sample->tintType - 1);
            int packed = scene.biomeData[idx];
            // Unpack linear ARGB to float3 (values stored as linear, quantized to 8 bits)
            tintColor.x = (float)((packed >> 16) & 0xFF) / 255.0f;
            tintColor.y = (float)((packed >> 8) & 0xFF) / 255.0f;
            tintColor.z = (float)(packed & 0xFF) / 255.0f;
            sample->color.xyz *= tintColor;
            return;
        }
    }
    // Fallback: hardcoded default biome colors
    switch (sample->tintType) {
        case 1: tintColor = colorFromArgb(0xFF71A74D).xyz; break;
        case 2: tintColor = colorFromArgb(0xFF8EB971).xyz; break;
        case 3: tintColor = colorFromArgb(0xFF3F76E4).xyz; break;
        case 4: tintColor = colorFromArgb(0xFF6A7039).xyz; break;
        default: return;
    }
    sample->color.xyz *= tintColor;
}

void intersectSky(image2d_t skyTexture, float skyIntensity, Sun sun, image2d_array_t atlas, Ray ray, MaterialSample* sample, bool diffuseSun) {
    Sky_intersect(skyTexture, skyIntensity, ray, sample);
    Sun_intersect(sun, atlas, ray, sample, diffuseSun);
}

#endif
