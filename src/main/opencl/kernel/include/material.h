// This includes stuff regarding materials

#ifndef CHUNKYCLPLUGIN_MATERIAL_H
#define CHUNKYCLPLUGIN_MATERIAL_H

#include "../opencl.h"
#include "rt.h"
#include "textureAtlas.h"
#include "utils.h"
#include "constants.h"
#include "random.h"

// MaterialPalette contains the global palette pointer and optional per-work-group
// local cache pointer and its word count. `matCache` should point to __local
// memory allocated by the kernel and `matCacheWords` is the number of ints
// copied into that cache. If `matCacheWords` is zero, no local cache is used.
typedef struct {
    __global const int* palette;
    __local unsigned int* matCache;
    int matCacheWords;
} MaterialPalette;

MaterialPalette MaterialPalette_new(__global const int* palette, int matCacheWords, __local unsigned int* matCache) {
    MaterialPalette p;
    p.palette = palette;
    p.matCache = matCache;
    p.matCacheWords = matCacheWords;
    return p;
}

typedef struct {
    unsigned int flags;
    unsigned int tint;
    unsigned int textureSize;
    unsigned int color;
    unsigned int normal_emittance;
    unsigned int specular_metalness_roughness;
    unsigned int ior_and_flags;
} Material;

// Material dword size (must match PackedMaterial.MATERIAL_DWORD_SIZE in Java)
#define MATERIAL_DWORD_SIZE 7

Material Material_get(MaterialPalette self, int material) {
    Material m;
    int base = material;
    // If material index falls within the cached words, use the local cache
    // (faster local memory) otherwise read from global palette.
    if (self.matCacheWords > 0 && base + 6 < self.matCacheWords) {
        m.flags = self.matCache[base + 0];
        m.tint = self.matCache[base + 1];
        m.textureSize = self.matCache[base + 2];
        m.color = self.matCache[base + 3];
        m.normal_emittance = self.matCache[base + 4];
        m.specular_metalness_roughness = self.matCache[base + 5];
        m.ior_and_flags = self.matCache[base + 6];
    } else {
        m.flags = self.palette[base + 0];
        m.tint = self.palette[base + 1];
        m.textureSize = self.palette[base + 2];
        m.color = self.palette[base + 3];
        m.normal_emittance = self.palette[base + 4];
        m.specular_metalness_roughness = self.palette[base + 5];
        m.ior_and_flags = self.palette[base + 6];
    }
    return m;
}

typedef struct {
    float4 color;
    float emittance;
    float specular;
    float metalness;
    float roughness;
    float ior;
    bool refractive;
    bool sss;          // sub-surface scattering
    bool isWater;
    int tintType;      // 0=none, 1=foliage, 2=grass, 3=water, 4=dryFoliage (for biome lookup)
} MaterialSample;

bool Material_sample(Material self, image2d_array_t atlas, float2 uv, MaterialSample* sample) {
    // Color
    float4 color;
    bool isWater = (self.ior_and_flags >> 18) & 1;

    if (isWater) {
        // Match Chunky CPU: water uses a flat base color (not per-pixel texture).
        // CPU calls getAvgColorLinear() which returns the texture average.
        // We use (1,1,1,1) as the base; the actual color is applied later by
        // biome tinting or custom water color, and alpha is overridden to waterOpacity.
        color = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
    } else if (self.flags & 0b100)
        color = Atlas_read_uv(uv.x, uv.y, self.color, self.textureSize, atlas);
    else
        color = colorFromArgb(self.color);

    if (self.tint == 0xFE000000) {
        // Light block
        sample->color.xyz = 0.5f;
        sample->color.w = 1.0f;
    } else if (color.w > EPS) {
        sample->color = color;
    } else {
        // Transparent texel: for refractive materials (glass pane edges), let
        // the ray pass through by returning a fully transparent sample.
        bool isRefractive = (self.ior_and_flags >> 16) & 1;
        if (isRefractive) {
            sample->color = color;
            sample->color.w = 0.0f;
            // Still populate material properties for the pass-through
            sample->ior = (float)(self.ior_and_flags & 0xFFFF) / 1000.0f;
            if (sample->ior < 0.01f) sample->ior = AIR_IOR;
            sample->refractive = true;
            sample->sss = false;
            sample->isWater = false;
            sample->emittance = 0.0f;
            sample->specular = 0.0f;
            sample->metalness = 0.0f;
            sample->roughness = 0.0f;
            return true;
        }
        return false;
    }

    // Tint: apply constant tints directly, defer biome tints to caller
    sample->tintType = 0;
    {
        int tintTag = self.tint >> 24;
        if (tintTag == 0xFF) {
            sample->color *= colorFromArgb(self.tint);
        } else if (tintTag >= 1 && tintTag <= 4) {
            // Store biome tint type; caller applies biome color or hardcoded fallback
            sample->tintType = tintTag;
        }
    }

    // (Normal) emittance
    if (self.flags & 0b010)
        sample->emittance = Atlas_read_uv(uv.x, uv.y, self.normal_emittance, self.textureSize, atlas).w;
    else
        sample->emittance = (self.normal_emittance & 0xFF) / 255.0f;

    // specular, metalness, roughness
    if (self.flags & 0b001) {
        float3 smr = Atlas_read_uv(uv.x, uv.y, self.specular_metalness_roughness, self.textureSize, atlas).xyz;
        sample->specular = smr.x;
        sample->metalness = smr.y;
        sample->roughness = smr.z;
    } else {
        sample->specular = (self.specular_metalness_roughness & 0xFF) / 255.0f;
        sample->metalness = ((self.specular_metalness_roughness >> 8) & 0xFF) / 255.0f;
        sample->roughness = ((self.specular_metalness_roughness >> 16) & 0xFF) / 255.0f;
    }

    // IOR and material flags (word 6)
    // bits 0-15: IOR * 1000, bit 16: refractive, bit 17: SSS, bit 18: isWater
    sample->ior = (float)(self.ior_and_flags & 0xFFFF) / 1000.0f;
    if (sample->ior < 0.01f) sample->ior = AIR_IOR;
    sample->refractive = (self.ior_and_flags >> 16) & 1;
    sample->sss = (self.ior_and_flags >> 17) & 1;
    sample->isWater = (self.ior_and_flags >> 18) & 1;

    return true;
}

float3 _Material_diffuseReflection(IntersectionRecord record, Random random) {
    float x1 = Random_nextFloat(random);
    float x2 = Random_nextFloat(random);
    float r = sqrt(x1);
    float theta = 2 * M_PI_F * x2;

    float tx = r * cos(theta);
    float ty = r * sin(theta);
    float tz = sqrt(1 - x1);

    // Transform from tangent space to world space
    float xx, xy, xz;
    float ux, uy, uz;
    float vx, vy, vz;

    if (fabs(record.normal.x) > 0.1f) {
        xx = 0;
        xy = 1;
    } else {
        xx = 1;
        xy = 0;
    }
    xz = 0;

    ux = xy * record.normal.z - xz * record.normal.y;
    uy = xz * record.normal.x - xx * record.normal.z;
    uz = xx * record.normal.y - xy * record.normal.x;

    r = 1 / sqrt(ux*ux + uy*uy + uz*uz);

    ux *= r;
    uy *= r;
    uz *= r;

    vx = uy * record.normal.z - uz * record.normal.y;
    vy = uz * record.normal.x - ux * record.normal.z;
    vz = ux * record.normal.y - uy * record.normal.x;

    return (float3) (
        ux * tx + vx * ty + record.normal.x * tz,
        uy * tx + vy * ty + record.normal.y * tz,
        uz * tx + vz * ty + record.normal.z * tz
    );
}

// --- Importance Sampling for diffuse reflection ---
// Steers diffuse bounce directions toward the sun for faster convergence.
// Matches the CPU algorithm from Ray.diffuseReflection() / PR #1604.

#define MAX_IMPORTANCE_SAMPLE_CHANCE 0.9f

float _Material_angleDistance(float a1, float a2) {
    float diff = fmod(fabs(a1 - a2), 2.0f * M_PI_F);
    return diff > M_PI_F ? 2.0f * M_PI_F - diff : diff;
}

typedef struct {
    float3 direction;
    float throughputScale;
} DiffuseISResult;

DiffuseISResult _Material_diffuseReflectionIS(IntersectionRecord record, float3 sunDir,
                                               float importanceSampleChance, float importanceSampleRadius,
                                               float sunRadius, Random random) {
    DiffuseISResult result;
    result.throughputScale = 1.0f;

    float x1 = Random_nextFloat(random);
    float x2 = Random_nextFloat(random);
    float r = sqrt(x1);
    float theta = 2.0f * M_PI_F * x2;

    float tx = r * cos(theta);
    float ty = r * sin(theta);

    // Build tangent space basis (same as _Material_diffuseReflection)
    float3 normal = record.normal;
    float xx, xy, xz;
    float ux, uy, uz;
    float vx, vy, vz;

    if (fabs(normal.x) > 0.1f) {
        xx = 0; xy = 1; xz = 0;
    } else {
        xx = 1; xy = 0; xz = 0;
    }

    ux = xy * normal.z - xz * normal.y;
    uy = xz * normal.x - xx * normal.z;
    uz = xx * normal.y - xy * normal.x;

    float invLen = 1.0f / sqrt(ux*ux + uy*uy + uz*uz);
    ux *= invLen;
    uy *= invLen;
    uz *= invLen;

    vx = uy * normal.z - uz * normal.y;
    vy = uz * normal.x - ux * normal.z;
    vz = ux * normal.y - uy * normal.x;

    // Compute sun direction in tangent space via dot products with u, v, n
    float sun_tx = sunDir.x * ux + sunDir.y * uy + sunDir.z * uz;
    float sun_ty = sunDir.x * vx + sunDir.y * vy + sunDir.z * vz;
    float sun_tz = dot(sunDir, normal);

    float circle_radius = sunRadius * importanceSampleRadius;
    float sample_chance = importanceSampleChance;
    float sun_alt_relative = asin(clamp(sun_tz, -1.0f, 1.0f));

    // Check if there is any chance of the sun being visible from this surface
    if (sun_alt_relative + circle_radius > EPS) {
        if (hypot(sun_tx, sun_ty) + circle_radius + EPS < 1.0f) {
            // Case A: sun well above surface — circular region in tangent space
            if (Random_nextFloat(random) < sample_chance) {
                // Sun sampling: steer sample into circle around sun projection
                tx = sun_tx + tx * circle_radius;
                ty = sun_ty + ty * circle_radius;
                result.throughputScale = circle_radius * circle_radius / sample_chance;
            } else {
                // Non-sun sampling: map point out of the sun's circle
                for (int _i = 0; _i < 16; _i++) {
                    if (hypot(tx - sun_tx, ty - sun_ty) >= circle_radius) break;
                    tx -= sun_tx;
                    ty -= sun_ty;
                    if (tx == 0.0f && ty == 0.0f) break;
                    tx /= circle_radius;
                    ty /= circle_radius;
                }
                result.throughputScale = (1.0f - circle_radius * circle_radius) / (1.0f - sample_chance);
            }
        } else {
            // Case B: sun near horizon — rectangular segment approach
            float minr = cos(sun_alt_relative + circle_radius);
            float maxr = cos(fmax(sun_alt_relative - circle_radius, 0.0f));
            float sun_theta = atan2(sun_ty, sun_tx);
            float segment_area_proportion = ((maxr * maxr - minr * minr) * circle_radius) / M_PI_F;
            sample_chance *= segment_area_proportion / (circle_radius * circle_radius);
            sample_chance = fmin(sample_chance, MAX_IMPORTANCE_SAMPLE_CHANCE);
            if (Random_nextFloat(random) < sample_chance) {
                // Sun sampling within rectangular segment
                r = sqrt(minr * minr * x1 + maxr * maxr * (1.0f - x1));
                theta = sun_theta + (2.0f * x2 - 1.0f) * circle_radius;
                tx = r * cos(theta);
                ty = r * sin(theta);
                result.throughputScale = segment_area_proportion / sample_chance;
            } else {
                // Non-sun sampling: rejection sample outside segment
                for (int _i = 0; _i < 64; _i++) {
                    if (!(r > minr && r < maxr && _Material_angleDistance(theta, sun_theta) < circle_radius)) break;
                    x1 = Random_nextFloat(random);
                    x2 = Random_nextFloat(random);
                    r = sqrt(x1);
                    theta = 2.0f * M_PI_F * x2;
                }
                tx = r * cos(theta);
                ty = r * sin(theta);
                result.throughputScale = (1.0f - segment_area_proportion) / (1.0f - sample_chance);
            }
        }
    }

    float tz = sqrt(fmax(0.0f, 1.0f - tx*tx - ty*ty));

    result.direction = (float3) (
        ux * tx + vx * ty + normal.x * tz,
        uy * tx + vy * ty + normal.y * tz,
        uz * tx + vz * ty + normal.z * tz
    );

    return result;
}

float3 _Material_specularReflection(IntersectionRecord record, MaterialSample sample, Ray ray, Random random) {
    float3 direction = ray.direction + (record.normal * (-2 * dot(ray.direction, record.normal)));

    if (sample.roughness > 0) {
        float3 diffuseDirection = _Material_diffuseReflection(record, random);
        diffuseDirection *= sample.roughness;
        direction = diffuseDirection + direction * (1 - sample.roughness);
    }

    // Geometric normal correction: prevent reflection from passing through geometry
    if (signbit(dot(record.geomNormal, direction)) == signbit(dot(record.geomNormal, ray.direction))) {
        float factor = copysign(dot(record.geomNormal, ray.direction), -EPS - dot(record.geomNormal, direction));
        direction += factor * record.geomNormal;
    }

    return normalize(direction);
}

typedef struct {
    float3 direction;
    float3 spectrum;
    bool specular;
    bool transmitted;   // ray went through the surface (refraction/transmission)
    float newIor;       // IOR of medium after interaction
} MaterialPdfSample;

// Schlick Fresnel approximation
float _Material_schlickFresnel(float n1, float n2, float cosTheta) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    float c = 1.0f - cosTheta;
    float c2 = c * c;
    return r0 + (1.0f - r0) * c2 * c2 * c;
}

// Compute refraction direction using Snell's law. Returns false if total internal reflection.
bool _Material_refract(float3 incident, float3 normal, float n1n2, float3* refracted) {
    float cosI = -dot(normal, incident);
    float sinT2 = n1n2 * n1n2 * (1.0f - cosI * cosI);
    if (sinT2 > 1.0f) return false; // total internal reflection
    float cosT = sqrt(1.0f - sinT2);
    *refracted = normalize(n1n2 * incident + (n1n2 * cosI - cosT) * normal);
    return true;
}

// Helper for transmissivity cap redistribution (matches CPU reassignTransmissivity)
float _Material_reassignTransmissivity(float maxChannel, float targetChannel, float otherChannel,
                                        float shouldTrans, float cap) {
    // Redistribute excess from maxChannel to targetChannel
    float excess = maxChannel - cap;
    float sum = targetChannel + otherChannel;
    if (sum > EPS) {
        return targetChannel + excess * (targetChannel / sum);
    }
    return targetChannel + excess * 0.5f;
}

// Compute fancy translucency transmission spectrum (matches CPU translucentRayColor)
float3 _Material_fancyTransmissionSpectrum(float4 sampleColor, float pAbsorb, float transmissivityCap) {
    float shouldTrans = 1.0f - pAbsorb;
    float colorTrans = (sampleColor.x + sampleColor.y + sampleColor.z) / 3.0f;
    float3 rgbTrans = (float3)(shouldTrans);

    if (colorTrans > EPS) {
        rgbTrans = sampleColor.xyz * (shouldTrans / colorTrans);
    }

    float maxTrans = fmax(rgbTrans.x, fmax(rgbTrans.y, rgbTrans.z));
    if (maxTrans > transmissivityCap) {
        if (maxTrans == rgbTrans.x) {
            float gNew = _Material_reassignTransmissivity(rgbTrans.x, rgbTrans.y, rgbTrans.z, shouldTrans, transmissivityCap);
            rgbTrans.z = _Material_reassignTransmissivity(rgbTrans.x, rgbTrans.z, rgbTrans.y, shouldTrans, transmissivityCap);
            rgbTrans.y = gNew;
            rgbTrans.x = transmissivityCap;
        } else if (maxTrans == rgbTrans.y) {
            float rNew = _Material_reassignTransmissivity(rgbTrans.y, rgbTrans.x, rgbTrans.z, shouldTrans, transmissivityCap);
            rgbTrans.z = _Material_reassignTransmissivity(rgbTrans.y, rgbTrans.z, rgbTrans.x, shouldTrans, transmissivityCap);
            rgbTrans.x = rNew;
            rgbTrans.y = transmissivityCap;
        } else {
            float gNew = _Material_reassignTransmissivity(rgbTrans.z, rgbTrans.y, rgbTrans.x, shouldTrans, transmissivityCap);
            rgbTrans.x = _Material_reassignTransmissivity(rgbTrans.z, rgbTrans.x, rgbTrans.y, shouldTrans, transmissivityCap);
            rgbTrans.y = gNew;
            rgbTrans.z = transmissivityCap;
        }
    }

    return rgbTrans;
}

MaterialPdfSample Material_samplePdf(Material self, IntersectionRecord record, MaterialSample sample, Ray ray, Random random,
                                      bool fancierTranslucency, float transmissivityCap) {
    MaterialPdfSample out;
    out.transmitted = false;
    out.newIor = ray.currentIor;

    float n1 = ray.currentIor;
    float n2 = sample.ior;

    // SSS: chance to sample from back side of surface
    if (sample.sss && Random_nextFloat(random) < F_SUBSURFACE) {
        IntersectionRecord flipped = record;
        flipped.normal = -record.normal;
        out.direction = _Material_diffuseReflection(flipped, random);
        out.spectrum = sample.color.xyz;
        out.specular = false;
        return out;
    }

    if (sample.metalness > 0 && sample.metalness > Random_nextFloat(random)) {
        // Metal reflection (tinted by albedo)
        out.direction = _Material_specularReflection(record, sample, ray, random);
        out.spectrum = sample.color.xyz;
        out.specular = true;
        return out;
    }

    if (sample.specular > 0 && sample.specular > Random_nextFloat(random)) {
        // Specular reflection
        out.direction = _Material_specularReflection(record, sample, ray, random);
        out.spectrum = (float3)(1.0f);
        out.specular = true;
        return out;
    }

    // Compute pDiffuse and pAbsorb for translucency
    float alpha = sample.color.w;
    float pDiffuse, pAbsorb;
    if (fancierTranslucency) {
        float maxRGB = fmax(sample.color.x, fmax(sample.color.y, sample.color.z));
        pDiffuse = 1.0f - pow(1.0f - alpha, maxRGB);
        pAbsorb = clamp(1.0f - (1.0f - alpha) / (1.0f - pDiffuse + EPS), 0.0f, 1.0f);
    } else {
        pDiffuse = alpha;
        pAbsorb = alpha;
    }

    // Refractive/transparent materials: glass, stained glass, water blocks, water plane
    bool doRefract = false;
    bool doTransmit = false;  // Non-refractive transparent pass-through
    if (sample.refractive || sample.isWater) {
        bool isTranslucent = alpha < 1.0f - EPS;

        if (isTranslucent) {
            // For translucent materials (alpha < 1), probabilistically choose between
            // diffuse reflection and transmission/refraction based on pDiffuse.
            doRefract = (Random_nextFloat(random) >= pDiffuse);
        } else {
            // Fully opaque refractive material (glass, water plane): always refract
            doRefract = true;
        }
    } else if (alpha < 1.0f - EPS) {
        // Non-refractive transparent blocks (e.g. glass, leaves, etc.):
        // probabilistically pass the ray straight through without refraction.
        doTransmit = (Random_nextFloat(random) >= pDiffuse);
    }

    if (doTransmit) {
        // Simple straight-through transmission (no IOR change, no refraction)
        out.direction = ray.direction;
        if (fancierTranslucency) {
            out.spectrum = _Material_fancyTransmissionSpectrum(sample.color, pAbsorb, transmissivityCap);
        } else {
            float3 rgbTrans = (float3)(1.0f - pAbsorb);
            rgbTrans = rgbTrans + sample.color.xyz * pAbsorb;
            out.spectrum = rgbTrans;
        }
        out.specular = false;
        out.transmitted = true;
        return out;
    }

    if (doRefract) {
        // Transmission/refraction path
        float targetIor = (n1 != n2) ? n2 : AIR_IOR;
        if (n1 != targetIor) {
            // Fresnel refraction
            float cosTheta = fmax(0.0f, -dot(ray.direction, record.normal));
            float fresnel = _Material_schlickFresnel(n1, targetIor, cosTheta);

            if (Random_nextFloat(random) < fresnel) {
                out.direction = _Material_specularReflection(record, sample, ray, random);
                out.spectrum = (float3)(1.0f);
                out.specular = true;
                return out;
            } else {
                float n1n2 = n1 / targetIor;
                float3 refractedDir;
                if (_Material_refract(ray.direction, record.normal, n1n2, &refractedDir)) {
                    out.direction = refractedDir;
                    // Apply fancy translucency spectrum for refraction
                    if (fancierTranslucency) {
                        out.spectrum = _Material_fancyTransmissionSpectrum(sample.color, pAbsorb, transmissivityCap);
                    } else {
                        // Old method: blend color with (1-absorption)
                        float3 rgbTrans = (float3)(1.0f - pAbsorb);
                        rgbTrans = rgbTrans + sample.color.xyz * pAbsorb;
                        out.spectrum = rgbTrans;
                    }
                    out.specular = true;
                    out.transmitted = true;
                    out.newIor = targetIor;
                    return out;
                } else {
                    out.direction = _Material_specularReflection(record, sample, ray, random);
                    out.spectrum = (float3)(1.0f);
                    out.specular = true;
                    return out;
                }
            }
        } else {
            // n1 == targetIor: already in the same medium, pass through
            out.direction = ray.direction;
            // Apply fancy translucency spectrum for transmission
            if (fancierTranslucency) {
                out.spectrum = _Material_fancyTransmissionSpectrum(sample.color, pAbsorb, transmissivityCap);
            } else {
                float3 rgbTrans = (float3)(1.0f - pAbsorb);
                rgbTrans = rgbTrans + sample.color.xyz * pAbsorb;
                out.spectrum = rgbTrans;
            }
            out.specular = false;
            out.transmitted = true;
            return out;
        }
    }

    // Diffuse reflection
    out.direction = _Material_diffuseReflection(record, random);
    // Geometric normal correction: prevent diffuse direction from passing through geometry
    // (matches CPU Ray.diffuseReflection geomN check)
    if (sign(dot(record.geomNormal, out.direction)) == sign(dot(record.geomNormal, ray.direction))) {
        float factor = copysign(1.0f, dot(record.geomNormal, ray.direction)) * (-EPS) - dot(out.direction, record.geomNormal);
        out.direction += factor * record.geomNormal;
        out.direction = normalize(out.direction);
    }
    out.spectrum = sample.color.xyz;
    out.specular = false;
    return out;
}

#endif
