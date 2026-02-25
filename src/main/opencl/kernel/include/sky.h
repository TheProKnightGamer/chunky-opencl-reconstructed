#ifndef CHUNKYCLPLUGIN_SKY_H
#define CHUNKYCLPLUGIN_SKY_H

#include "../opencl.h"
#include "rt.h"
#include "textureAtlas.h"
#include "random.h"
#include "material.h"

typedef struct {
    int flags;
    int textureSize;
    int texture;
    float intensity;
    float luminosity;
    float radius;
    float apparentBrightness;
    int modifySunTexture;
    float3 su;
    float3 sv;
    float3 sw;
    float4 color;
    float4 apparentColor;
    float importanceSampleChance;
    float importanceSampleRadius;
} Sun;

Sun Sun_new(__global const int* data) {
    Sun sun;
    sun.flags = data[0];
    sun.textureSize = data[1];
    sun.texture = data[2];
    sun.intensity = as_float(data[3]);
    sun.luminosity = as_float(data[6]);
    sun.color = colorFromArgb(data[7]);
    sun.radius = as_float(data[8]);
    sun.apparentBrightness = as_float(data[9]);
    sun.apparentColor = colorFromArgb(data[10]);
    sun.modifySunTexture = data[11];
    sun.importanceSampleChance = as_float(data[12]);
    sun.importanceSampleRadius = as_float(data[13]);
    
    float phi = as_float(data[4]);
    float theta = as_float(data[5]);
    float r = fabs(cos(phi));
    
    sun.sw = (float3) (cos(theta) * r, sin(phi), sin(theta) * r);
    if (fabs(sun.sw.x) > 0.1f) {
        sun.su = (float3) (0, 1, 0);
    } else {
        sun.su = (float3) (1, 0, 0);
    }
    sun.sv = normalize(cross(sun.sw, sun.su));
    sun.su = cross(sun.sv, sun.sw);
    
    return sun;
}

// Sun disk intersection.
// diffuseSun: whether to draw the sun disk for diffuse indirect rays.
//   When NEE (next-event estimation) is active (FAST/HIGH_QUALITY modes), the sun
//   contribution is already sampled via shadow rays, so drawing the sun disk for
//   diffuse indirect sky hits would double-count the sun. In those modes, diffuseSun
//   should be false. For OFF and IMPORTANCE modes, diffuseSun should be true.
bool Sun_intersect(Sun self, image2d_array_t atlas, Ray ray, MaterialSample* sample, bool diffuseSun) {
    float3 direction = ray.direction;
    bool isDiffuse = (ray.flags & RAY_INDIRECT) != 0;

    // CPU Sun.intersect checks drawTexture; CPU Sun.intersectDiffuse does not.
    if (isDiffuse) {
        if (!diffuseSun) return false;
    } else {
        if (!(self.flags & 1)) return false;
    }

    if (dot(direction, self.sw) < 0.5f) {
        return false;
    }

    float radius = self.radius;

    float width = radius * 4;
    float width2 = width * 2;
    float a = M_PI_2_F - acos(dot(direction, self.su)) + width;
    if (a >= 0 && a < width2) {
        float b = M_PI_2_F - acos(dot(direction, self.sv)) + width;
        if (b >= 0 && b < width2) {
            if (isDiffuse) {
                // CPU Sun.intersectDiffuse: texture_sample * color * 10
                // Guard against invalid sun texture (textureSize==0 means not loaded)
                if (self.textureSize != 0) {
                    float4 color = Atlas_read_uv(a / width2, b / width2,
                                                 self.texture, self.textureSize, atlas);
                    color.xyz *= self.color.xyz * 10.0f;
                    sample->color += color;
                } else {
                    // Fallback: use flat sun color (no texture available)
                    float4 color = self.color * 10.0f;
                    sample->color += color;
                }
            } else {
                // CPU Sun.intersect: texture_sample * apparentTextureBrightness * 10
                // where apparentTextureBrightness = [apparentColor|white] * apparentBrightness^2.2
                float4 color;
                if (self.textureSize != 0) {
                    color = Atlas_read_uv(a / width2, b / width2,
                                          self.texture, self.textureSize, atlas);
                } else {
                    color = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
                }
                if (self.modifySunTexture) {
                    color.xyz *= self.apparentColor.xyz;
                }
                color *= pow(self.apparentBrightness, DEFAULT_GAMMA) * 10.0f;
                sample->color += color;
            }
            return true;
        }
    }

    return false;
}

bool Sun_sampleDirection(Sun self, Ray* ray, Random random) {
    if (!(self.flags & 1)) {
        return false;
    }

    float radius_cos = cos(self.radius);

    float x1 = Random_nextFloat(random);
    float x2 = Random_nextFloat(random);

    float cos_a = 1 - x1 + x1 * radius_cos;
    float sin_a = sqrt(1 - cos_a * cos_a);
    float phi = 2 * M_PI_F * x2;

    float3 u = self.su * (cos(phi) * sin_a);
    float3 v = self.sv * (sin(phi) * sin_a);
    float3 w = self.sw * cos_a;

    ray->direction = u + v;
    ray->direction += w;
    ray->direction = normalize(ray->direction);

    return true;
}

const sampler_t skySampler = CLK_NORMALIZED_COORDS_TRUE  | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_LINEAR;

void Sky_intersect(image2d_t skyTexture, float skyIntensity, Ray ray, MaterialSample* sample) {
    float3 direction = ray.direction;

    float theta = atan2(direction.z, direction.x);
    theta /= M_PI_F * 2;
    theta = fmod(fmod(theta, 1) + 1, 1);
    float phi = (asin(clamp(direction.y, -1.0f, 1.0f)) + M_PI_2_F) * M_1_PI_F;

    sample->color = read_imagef(skyTexture, skySampler, (float2) (theta, phi)) * skyIntensity;
    sample->emittance = 1.0f;
}

#endif
