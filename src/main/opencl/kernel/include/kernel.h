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
    bool hasWaterBlocks;   // the water octree holds at least one block
    AABB waterBounds;      // octree-local bounds of those blocks
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
                     IntersectionRecord* record, bool hasCloserHit) {
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
                record->hitKind = HIT_CLOUD;
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
    // Iteration cap: the cloud grid wraps every 256 cells, so a ray that hasn't
    // hit a cloud cell within a couple of full traversals never will. Without it,
    // an exactly-horizontal ray (tExit ~1e30) with no closer geometry hit would
    // step cell-by-cell almost forever (GPU hang / TDR).
    int cloudSteps = 0;
    while (t < tExit && cloudSteps++ < 2048) {
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
        record->hitKind = HIT_CLOUD;
        return true;
    }
    return false;
}

// Limits a water-octree march to the part of the ray that can meet a water
// block: returns false when the ray cannot reach one before *limit, otherwise
// shrinks *limit to just past the blocks' bounding box. Exact: water hits only
// exist inside the box, so the march finds the same hit or none. Without it every
// ray above the water line (and every shadow ray climbing toward the sun) walked
// the whole water octree to the edge of the scene.
bool Water_clipToBlocks(const SceneConfig* cfg, Ray ray, float* limit) {
    if (!cfg->hasWaterBlocks) return false;
    float3 invD = select(1.0f / ray.direction, copysign((float3)(1e30f), ray.direction), fabs(ray.direction) < 1e-30f);
    AABB b = cfg->waterBounds;
    // A thousandth of a block of padding: keeps a hit computed a rounding step
    // outside the box, or a ray running exactly along a face, inside the clip.
    const float pad = 1e-3f;
    float3 t1 = ((float3)(b.xmin - pad, b.ymin - pad, b.zmin - pad) - ray.origin) * invD;
    float3 t2 = ((float3)(b.xmax + pad, b.ymax + pad, b.zmax + pad) - ray.origin) * invD;
    float3 lo = fmin(t1, t2);
    float3 hi = fmax(t1, t2);
    float tEnter = fmax(lo.x, fmax(lo.y, lo.z));
    float tExit = fmin(hi.x, fmin(hi.y, hi.z));
    if (!(tExit >= fmax(tEnter, 0.0f)) || tEnter > *limit) return false;
    // A block-sized margin keeps DDA stepping offsets well clear of the cut.
    *limit = fmin(*limit, tExit * 1.0001f + 1.0f);
    return true;
}

// Builds the MaterialSample for the winning hit of a trace. Exactly one
// Material_sample per trace, instead of one per primitive tested.
void Hit_resolveSample(IntersectionRecord record, image2d_array_t atlas, MaterialPalette materialPalette,
                       const SceneConfig* cfg, bool previewWaterOpacity, MaterialSample* sample) {
    int kind = record.hitKind & HIT_KIND_MASK;
    if (kind == HIT_WATER_PLANE) {
        FillWaterSample(sample, cfg);
    } else if (kind == HIT_CLOUD) {
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
    } else if (kind == HIT_WATER_EXIT_PLAIN) {
        sample->color = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
        sample->emittance = 0.0f;
        sample->specular = 0.12f;
        sample->metalness = 0.0f;
        sample->roughness = 0.0f;
        sample->ior = 1.333f;
        sample->refractive = true;
        sample->sss = false;
        sample->isWater = true;
        sample->tintType = 3;
    } else {
        float2 uv = (kind == HIT_WATER_EXIT_CENTER) ? (float2)(0.5f, 0.5f) : record.texCoord;
        Material_sample(Material_get(materialPalette, record.material), atlas, uv, sample);
        if (kind == HIT_LIGHT_WHITE) {
            sample->color = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
        } else if (kind == HIT_WATER_EXIT_CENTER || kind == HIT_WATER_EXIT_UV) {
            sample->isWater = true;
        }
    }
    if ((record.hitKind & HIT_FLAG_WATER_BRANCH) && sample->isWater) {
        if (cfg->useCustomWaterColor) {
            sample->color.xyz = cfg->waterColor;
            sample->tintType = 0;
        } else if (sample->tintType == 0) {
            sample->tintType = 3;
        }
        if (previewWaterOpacity) {
            sample->color.w = cfg->waterOpacity;
        }
    }
}

// Finds the closest surface along a ray. With stopAtOpaque (sun / fog shadow
// rays only), a fully opaque world block is reported as soon as the world octree
// finds it, skipping the water march, both entity BVHs, the water plane and the
// clouds: the shadow loop turns ANY opaque occluder into zero light, whatever
// translucent layers lie in front of it, so the answer is the same. Two edge cases
// now return that zero where the old loop could leak light: past its 32-layer cap
// (the uncapped CPU renderer also returns zero), and a closer hit lying within
// OFFSET of the opaque block's face (the loop stepped past it into the block, whose
// face it then rejected). Emitter NEE must not use this: it compares the hit
// distance against the emitter's.
bool traceScene(SceneConfig self, image2d_array_t atlas, Ray ray, IntersectionRecord* record, MaterialSample* sample,
                bool stopAtOpaque) {
    IntersectionRecord tempRecord = *record;
    bool hit = false;
    hit |= Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &tempRecord);
    if (stopAtOpaque && hit && tempRecord.hitKind == HIT_MATERIAL
            && Material_isOpaqueAt(Material_get(self.materialPalette, tempRecord.material), atlas, tempRecord.texCoord)) {
        // Only what the shadow loop reads: the distance, and an opaque sample.
        *record = tempRecord;
        record->geomNormal = record->normal;
        sample->color = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
        sample->refractive = false;
        sample->isWater = false;
        return true;
    }
    hit |= Bvh_intersect(self.worldBvh, atlas, self.materialPalette, ray, &tempRecord);
    hit |= Bvh_intersect(self.actorBvh, atlas, self.materialPalette, ray, &tempRecord);
    {
        IntersectionRecord waterRecord = tempRecord;
        if (!hit) waterRecord.distance = record->distance;
        bool waterHit = false;
        if (ray.inWater) {
            waterRecord.distance = hit ? (tempRecord.distance + EPS) : record->distance;
            waterHit = Octree_exitWater(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord);
        } else if (Water_clipToBlocks(&self, ray, &waterRecord.distance)) {
            waterHit = Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord);
        }
        if (waterHit && (!hit || waterRecord.distance < tempRecord.distance)) {
            tempRecord = waterRecord;
            tempRecord.hitKind |= HIT_FLAG_WATER_BRANCH;
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
                // Wave shading is deferred to the unified water-shading block
                // below (after geomNormal is pinned to the flat normal), so the
                // flat plane normal survives for the anti-leak corrections
                // instead of being clobbered by the blanket geomNormal=normal.
                tempRecord.hitKind = HIT_WATER_PLANE;
                hit = true;
            }
        }
    }
    if (self.cloudsEnabled) {
        if (Cloud_intersect(self.cloudHeight, self.cloudSize, self.cloudOffsetX, self.cloudOffsetZ,
                            self.cloudData, ray, &tempRecord, hit)) {
            hit = true;
        }
    }
    if (!hit) return false;
    *record = tempRecord;
    Hit_resolveSample(tempRecord, atlas, self.materialPalette, &self, false, sample);
    record->geomNormal = record->normal;
    if (sample->isWater) {
        // Water BLOCKS (material >= 0) get opacity / custom color applied here;
        // the water PLANE already did so in FillWaterSample.
        if (record->material >= 0) {
            sample->color.w = self.waterOpacity;
            if (self.useCustomWaterColor) {
                sample->color.xyz = self.waterColor;
                sample->tintType = 0;
            }
        }
        // Wave shading for BOTH water blocks and the water plane. geomNormal is
        // pinned to the FLAT surface normal first (line 283 set it), THEN the
        // shading perturbs record->normal — so the specular/refraction/diffuse
        // anti-leak corrections run against the true flat normal, not the wave.
        // Shadow rays skip wave shading: CPU traces both sun-NEE
        // (PathTracer.getDirectLightAttenuation) and emitter-NEE
        // (sampleEmitterFace) shadow rays with PreviewRayTracer, which never
        // applies water shading. Sun/fog attenuation reads only distance +
        // material sample; emitter NEE reads srec.normal only when the hit
        // grazes the emitter face within 1e-4, where CPU uses the flat normal
        // anyway.
        if (record->normal.y != 0.0f && !(ray.flags & RAY_SHADOW)) {
            record->geomNormal = record->normal;
            float3 hitPos = ray.origin + ray.direction * record->distance;
            Water_applyShading(record, self.waterShadingStrategy,
                               self.animationTime, hitPos.x, hitPos.z, self.waterShaderParams,
                               self.waterNormalMap, self.waterNormalMapW);
        }
    }
    return true;
}

// NOTE: `mat` is NEVER written and must not be read by callers. Its only
// consumer was Material_samplePdf's `self` parameter, which is unused.
bool closestIntersect(SceneConfig self, image2d_array_t atlas, Ray ray, IntersectionRecord* record, MaterialSample* sample, Material* mat) {
    return traceScene(self, atlas, ray, record, sample, false);
}

// Simplified intersection for preview mode, matching CPU PreviewRayTracer:
// - Tests water plane (with chunk clip, covers unloaded areas)
// - Tests octree + BVH (solid geometry)
// - Tests water octree with per-corner heights (covers loaded chunks)
// - Uses Octree_exitWater when ray is underwater, matching main renderer
// - No Water_applyShading, no cloud intersection
// NOTE: `mat` is NEVER written and must not be read by callers. Its only
// consumer was Material_samplePdf's `self` parameter, which is unused.
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
            tempRecord.hitKind = HIT_WATER_PLANE;
            tempRecord.geomNormal = tempRecord.normal;
            hit = true;
        }
    }
    // Solid geometry (octree + BVH)
    hit |= Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &tempRecord);
    hit |= Bvh_intersect(self.worldBvh, atlas, self.materialPalette, ray, &tempRecord);
    hit |= Bvh_intersect(self.actorBvh, atlas, self.materialPalette, ray, &tempRecord);
    // Water octree (covers loaded chunks where the water plane is clipped)
    {
        IntersectionRecord waterRecord = tempRecord;
        if (!hit) waterRecord.distance = record->distance;
        bool waterHit = false;
        if (ray.inWater) {
            // Underwater: use exitWater to find where ray leaves the water volume
            waterRecord.distance = hit ? (tempRecord.distance + EPS) : record->distance;
            waterHit = Octree_exitWater(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord);
        } else if (Water_clipToBlocks(&self, ray, &waterRecord.distance)) {
            // Above water: find entry into water blocks
            waterHit = Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord);
        }
        if (waterHit && (!hit || waterRecord.distance < tempRecord.distance)) {
            tempRecord = waterRecord;
            tempRecord.hitKind |= HIT_FLAG_WATER_BRANCH;
            hit = true;
        }
    }
    // Clouds — the CPU preview (PreviewRayTracer.nextIntersection) renders them too.
    if (self.cloudsEnabled) {
        if (Cloud_intersect(self.cloudHeight, self.cloudSize, self.cloudOffsetX, self.cloudOffsetZ,
                            self.cloudData, ray, &tempRecord, hit)) {
            hit = true;
        }
    }
    if (!hit) return false;
    *record = tempRecord;
    Hit_resolveSample(tempRecord, atlas, self.materialPalette, &self, true, sample);
    record->geomNormal = record->normal;
    return true;
}

void applyBiomeTint(SceneConfig scene, MaterialSample* sample, float3 hitPos) {
    // tintType indexes the biome buffer below, so anything outside 1-4 must never
    // get there (an out-of-range value is an out-of-bounds read, not a wrong tint).
    if (sample->tintType < 1 || sample->tintType > 4) return;
    float3 tintColor;
    if (scene.biomeColorsEnabled && scene.biomeDataSize > 0) {
        int bx = (int)floor(hitPos.x);
        int bz = (int)floor(hitPos.z);
        if (bx >= 0 && bx < scene.biomeDataSize && bz >= 0 && bz < scene.biomeDataSize) {
            // Y interpolation: samples sit at section centers (y mod 16 == 8).
            // Linearly blend between adjacent sections so 3D biome tints don't
            // step every 16 blocks vertically.
            float yIdx = hitPos.y * 0.0625f - 0.5f;
            int yLo = clamp((int)floor(yIdx), 0, scene.biomeYLevels - 1);
            int yHi = clamp(yLo + 1, 0, scene.biomeYLevels - 1);
            float w = clamp(yIdx - (float)yLo, 0.0f, 1.0f);

            int stride = scene.biomeDataSize * scene.biomeDataSize;
            int base = (bz * scene.biomeDataSize + bx) * 4 + (sample->tintType - 1);
            int p0 = scene.biomeData[yLo * stride * 4 + base];
            int p1 = scene.biomeData[yHi * stride * 4 + base];

            float3 c0 = (float3)((float)((p0 >> 16) & 0xFF) / 255.0f,
                                 (float)((p0 >> 8) & 0xFF) / 255.0f,
                                 (float)(p0 & 0xFF) / 255.0f);
            float3 c1 = (float3)((float)((p1 >> 16) & 0xFF) / 255.0f,
                                 (float)((p1 >> 8) & 0xFF) / 255.0f,
                                 (float)(p1 & 0xFF) / 255.0f);
            tintColor = mix(c0, c1, w);
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