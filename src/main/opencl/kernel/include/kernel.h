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
    __global const float* waterNormalMap;
    int waterNormalMapW;
    bool cloudsEnabled;
    float cloudHeight;
    float cloudSize;
    float cloudOffsetX;
    float cloudOffsetZ;
    __global const int* cloudData;
    bool biomeColorsEnabled;
    __global const int* biomeData;
    int biomeDataSize;
    int biomeYLevels;
    __global const int* chunkBitmap;
    int chunkBitmapSize;
    bool transparentSky;
    float yMin;
    float yMax;
} SceneConfig;

inline int Cloud_getCell(__global const int *cloudData, int x, int z) {
    x = x & 255;
    z = z & 255;
    int tileX = x >> 3;
    int tileZ = z >> 3;
    int subX = x & 7;
    int subZ = z & 7;
    int idx = (tileX * 32 + tileZ) * 2;
    int bitPos = subZ * 8 + subX;
    int word = (bitPos < 32) ? cloudData[idx] : cloudData[idx + 1];
    return (word >> (bitPos & 31)) & 1;
}

inline bool Cloud_inCloud(__global const int *cloudData, float x, float z) {
    return Cloud_getCell(cloudData, (int)floor(x), (int)floor(z)) == 1;
}

inline void FillWaterSample(MaterialSample *s, const SceneConfig *cfg) {
    s->color = (float4)(1.0f, 1.0f, 1.0f, cfg->waterOpacity);
    if (cfg->useCustomWaterColor) {
        s->color.xyz = cfg->waterColor;
        s->tintType = 0;
    } else {
        s->tintType = 3;
    }
    s->emittance = 0.0f;
    s->specular = 0.12f;
    s->metalness = 0.0f;
    s->roughness = 0.0f;
    s->ior = cfg->waterIor;
    s->refractive = true;
    s->sss = false;
    s->isWater = true;
}

bool Cloud_intersect(float cloudHeight, float cloudSize, float cloudOffsetX, float cloudOffsetZ,
                     __global const int* cloudData, Ray tempRay,
                     IntersectionRecord* record, MaterialSample* sample, bool hasCloserHit) {
    const float inv_size = 1.0f / cloudSize;
    const float cloudTop = cloudHeight + 5.0f;
    float oy = tempRay.origin.y;
    float t_offset = 0.0f;
    int target = 1;
    float cloudT = 0.0f;
    float3 cloudNormal = (float3)(0.0f);
    if (oy < cloudHeight || oy > cloudTop) {
        if (fabs(tempRay.direction.y) < 1e-5f) return false;
        t_offset = (oy < cloudHeight) ? (cloudHeight - oy) / tempRay.direction.y
                                      : (cloudTop - oy) / tempRay.direction.y;
        if (t_offset < 0.0f) return false;
        float ex = (tempRay.direction.x * t_offset + tempRay.origin.x) * inv_size + cloudOffsetX;
        float ez = (tempRay.direction.z * t_offset + tempRay.origin.z) * inv_size + cloudOffsetZ;
        if (Cloud_inCloud(cloudData, ex, ez)) {
            cloudT = t_offset;
            cloudNormal = (float3)(0.0f, -sign(tempRay.direction.y), 0.0f);
            if (cloudT > OFFSET && (!hasCloserHit || cloudT < record->distance)) {
                record->distance = cloudT;
                record->normal = cloudNormal;
                float3 hitP = tempRay.origin + tempRay.direction * cloudT;
                record->texCoord = (float2)(hitP.x * inv_size + cloudOffsetX - floor(hitP.x * inv_size + cloudOffsetX),
                                            hitP.z * inv_size + cloudOffsetZ - floor(hitP.z * inv_size + cloudOffsetZ));
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
                return true;
            }
            return false;
        }
    } else if (Cloud_inCloud(cloudData,
                             tempRay.origin.x * inv_size + cloudOffsetX,
                             tempRay.origin.z * inv_size + cloudOffsetZ)) {
        target = 0;
    }
    float tExit;
    if (tempRay.direction.y > 0.0f) {
        tExit = (cloudTop - oy) / tempRay.direction.y - t_offset;
    } else if (tempRay.direction.y < 0.0f) {
        tExit = (cloudHeight - oy) / tempRay.direction.y - t_offset;
    } else {
        tExit = 1e30f;
    }
    float maxT = record->distance;
    if (maxT - t_offset < tExit) tExit = maxT - t_offset;
    float dx = fabs(tempRay.direction.x) * inv_size;
    float dz = fabs(tempRay.direction.z) * inv_size;
    if (dx < 1e-10f && dz < 1e-10f) return false;
    float x0 = (tempRay.origin.x + tempRay.direction.x * t_offset) * inv_size + cloudOffsetX;
    float z0 = (tempRay.origin.z + tempRay.direction.z * t_offset) * inv_size + cloudOffsetZ;
    int ix = (int)floor(x0);
    int iz = (int)floor(z0);
    int stepX = (tempRay.direction.x > 0.0f) ? 1 : ((tempRay.direction.x < 0.0f) ? -1 : 0);
    int stepZ = (tempRay.direction.z > 0.0f) ? 1 : ((tempRay.direction.z < 0.0f) ? -1 : 0);
    float invDx = (dx > 0.0f) ? 1.0f / dx : 1e30f;
    float invDz = (dz > 0.0f) ? 1.0f / dz : 1e30f;
    float tMaxX;
    float tMaxZ;
    if (stepX != 0) {
        float nextX = (float)(stepX > 0 ? (ix + 1) : ix);
        tMaxX = (nextX - x0) * invDx;
    } else {
        tMaxX = 1e30f;
    }
    if (stepZ != 0) {
        float nextZ = (float)(stepZ > 0 ? (iz + 1) : iz);
        tMaxZ = (nextZ - z0) * invDz;
    } else {
        tMaxZ = 1e30f;
    }
    float t = 0.0f;
    int nx = 0;
    int nz = 0;
    bool hitCell = false;
    while (t < tExit) {
        if (tMaxX < tMaxZ) {
            ix += stepX;
            t = tMaxX;
            tMaxX += invDx;
            nx = -stepX;
            nz = 0;
        } else {
            iz += stepZ;
            t = tMaxZ;
            tMaxZ += invDz;
            nx = 0;
            nz = -stepZ;
        }
        if (Cloud_getCell(cloudData, ix, iz) == target) {
            hitCell = true;
            break;
        }
    }
    if (target == 1) {
        if (!hitCell) return false;
        if (t > tExit) return false;
        cloudNormal = (float3)((float)nx, 0.0f, (float)nz);
        cloudT = t + t_offset;
    } else {
        if (t > tExit) {
            int ny = (tempRay.direction.y > 0.0f) ? 1 : -1;
            cloudNormal = (float3)(0.0f, (float)ny, 0.0f);
            cloudT = tExit + t_offset;
        } else {
            cloudNormal = (float3)((float)(-nx), 0.0f, (float)(-nz));
            cloudT = t + t_offset;
        }
    }
    if (cloudT > OFFSET && (!hasCloserHit || cloudT < record->distance)) {
        record->distance = cloudT;
        record->normal = cloudNormal;
        float3 hitP = tempRay.origin + tempRay.direction * cloudT;
        record->texCoord = (float2)(hitP.x * inv_size + cloudOffsetX - floor(hitP.x * inv_size + cloudOffsetX),
                                    hitP.z * inv_size + cloudOffsetZ - floor(hitP.z * inv_size + cloudOffsetZ));
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
        return true;
    }
    return false;
}

bool closestIntersect(SceneConfig self, image2d_array_t atlas, Ray ray, IntersectionRecord* record, MaterialSample* sample, Material* mat) {
    IntersectionRecord tempRecord = *record;
    bool hit = false;
    hit |= Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &tempRecord, sample);
    hit |= Bvh_intersect(self.worldBvh, atlas, self.materialPalette, ray, &tempRecord, sample);
    hit |= Bvh_intersect(self.actorBvh, atlas, self.materialPalette, ray, &tempRecord, sample);
    {
        IntersectionRecord waterRecord = tempRecord;
        if (!hit) waterRecord.distance = record->distance;
        MaterialSample waterSample;
        bool waterHit = false;
        if (ray.inWater) {
            waterRecord.distance = hit ? (tempRecord.distance + EPS) : record->distance;
            waterHit = Octree_exitWater(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord, &waterSample);
        } else {
            waterHit = Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord, &waterSample);
        }
        if (waterHit && (!hit || waterRecord.distance < tempRecord.distance)) {
            if (waterSample.isWater) {
                if (self.useCustomWaterColor) {
                    waterSample.color.xyz = self.waterColor;
                    waterSample.tintType = 0;
                } else if (waterSample.tintType == 0) {
                    waterSample.tintType = 3;
                }
            }
            tempRecord = waterRecord;
            *sample = waterSample;
            hit = true;
        }
    }
    if (self.waterPlaneEnabled) {
        IntersectionRecord waterRecord = tempRecord;
        if (!hit) waterRecord.distance = record->distance;
        if (Water_planeIntersect(ray, self.waterPlaneHeight, self.octreeSize,
                                 self.waterPlaneChunkClip,
                                 self.chunkBitmap, self.chunkBitmapSize,
                                 &waterRecord)) {
            if (!hit || waterRecord.distance < tempRecord.distance) {
                tempRecord = waterRecord;
                FillWaterSample(sample, &self);
                tempRecord.geomNormal = tempRecord.normal;
                Water_applyShading(&tempRecord, self.waterShadingStrategy,
                                   self.animationTime, tempRecord.texCoord.x,
                                   tempRecord.texCoord.y, self.waterShaderParams,
                                   self.waterNormalMap, self.waterNormalMapW);
                hit = true;
            }
        }
    }
    if (self.cloudsEnabled) {
        if (Cloud_intersect(self.cloudHeight, self.cloudSize, self.cloudOffsetX, self.cloudOffsetZ,
                            self.cloudData, ray, &tempRecord, sample, hit)) {
            hit = true;
        }
    }
    if (!hit) return false;
    *record = tempRecord;
    record->geomNormal = record->normal;
    if (record->material >= 0) {
        *mat = Material_get(self.materialPalette, record->material);
    }
    if (sample->isWater && record->material >= 0) {
        sample->color.w = self.waterOpacity;
        if (self.useCustomWaterColor) {
            sample->color.xyz = self.waterColor;
            sample->tintType = 0;
        }
        if (record->normal.y != 0.0f) {
            record->geomNormal = record->normal;
            float3 hitPos = ray.origin + ray.direction * record->distance;
            Water_applyShading(record, self.waterShadingStrategy,
                               self.animationTime, hitPos.x, hitPos.z, self.waterShaderParams,
                               self.waterNormalMap, self.waterNormalMapW);
        }
    }
    return true;
}

// Simplified intersection for preview mode, matching CPU PreviewRayTracer:
// - Tests water plane (with chunk clip, covers unloaded areas)
// - Tests octree + BVH (solid geometry)
// - Tests water octree with per-corner heights (covers loaded chunks)
// - Uses Octree_exitWater when ray is underwater, matching main renderer
// - No Water_applyShading, no cloud intersection
bool previewIntersect(SceneConfig self, image2d_array_t atlas, Ray ray,
                      IntersectionRecord* record, MaterialSample* sample, Material* mat) {
    IntersectionRecord tempRecord = *record;
    bool hit = false;
    // Water plane first (covers unloaded chunks, chunk-clipped in loaded chunks)
    if (self.waterPlaneEnabled) {
        IntersectionRecord wpRecord = tempRecord;
        if (Water_planeIntersect(ray, self.waterPlaneHeight, self.octreeSize,
                                 self.waterPlaneChunkClip,
                                 self.chunkBitmap, self.chunkBitmapSize,
                                 &wpRecord)) {
            tempRecord = wpRecord;
            FillWaterSample(sample, &self);
            tempRecord.geomNormal = tempRecord.normal;
            hit = true;
        }
    }
    // Solid geometry (octree + BVH)
    hit |= Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &tempRecord, sample);
    hit |= Bvh_intersect(self.worldBvh, atlas, self.materialPalette, ray, &tempRecord, sample);
    hit |= Bvh_intersect(self.actorBvh, atlas, self.materialPalette, ray, &tempRecord, sample);
    // Water octree (covers loaded chunks where the water plane is clipped)
    {
        IntersectionRecord waterRecord = tempRecord;
        if (!hit) waterRecord.distance = record->distance;
        MaterialSample waterSample;
        bool waterHit = false;
        if (ray.inWater) {
            // Underwater: use exitWater to find where ray leaves the water volume
            waterRecord.distance = hit ? (tempRecord.distance + EPS) : record->distance;
            waterHit = Octree_exitWater(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord, &waterSample);
        } else {
            // Above water: find entry into water blocks
            waterHit = Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord, &waterSample);
        }
        if (waterHit && (!hit || waterRecord.distance < tempRecord.distance)) {
            if (waterSample.isWater) {
                if (self.useCustomWaterColor) {
                    waterSample.color.xyz = self.waterColor;
                    waterSample.tintType = 0;
                } else if (waterSample.tintType == 0) {
                    waterSample.tintType = 3;
                }
                waterSample.color.w = self.waterOpacity;
            }
            tempRecord = waterRecord;
            *sample = waterSample;
            hit = true;
        }
    }
    if (!hit) return false;
    *record = tempRecord;
    record->geomNormal = record->normal;
    if (record->material >= 0) {
        *mat = Material_get(self.materialPalette, record->material);
    }
    return true;
}

void applyBiomeTint(SceneConfig scene, MaterialSample* sample, float3 hitPos) {
    if (sample->tintType == 0) return;
    float3 tintColor;
    if (scene.biomeColorsEnabled && scene.biomeDataSize > 0) {
        int bx = (int)floor(hitPos.x);
        int bz = (int)floor(hitPos.z);
        if (bx >= 0 && bx < scene.biomeDataSize && bz >= 0 && bz < scene.biomeDataSize) {
            // 3D biome lookup: Y level = floor(hitY / 16), clamped to [0, yLevels-1]
            int yl = clamp((int)floor(hitPos.y) >> 4, 0, scene.biomeYLevels - 1);
            int idx = (yl * scene.biomeDataSize * scene.biomeDataSize + bz * scene.biomeDataSize + bx) * 4
                      + (sample->tintType - 1);
            int packed = scene.biomeData[idx];
            tintColor.x = (float)((packed >> 16) & 0xFF) / 255.0f;
            tintColor.y = (float)((packed >> 8) & 0xFF) / 255.0f;
            tintColor.z = (float)(packed & 0xFF) / 255.0f;
            sample->color.xyz *= tintColor;
            return;
        }
    }
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