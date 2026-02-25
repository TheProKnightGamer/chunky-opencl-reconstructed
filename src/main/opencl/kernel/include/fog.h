#ifndef CHUNKYCLPLUGIN_FOG_H
#define CHUNKYCLPLUGIN_FOG_H

#include "../opencl.h"
#include "constants.h"
#include "random.h"

// Fog modes
#define FOG_MODE_NONE    0
#define FOG_MODE_UNIFORM 1
#define FOG_MODE_LAYERED 2

// Maximum fog layers
#define MAX_FOG_LAYERS 8

typedef struct {
    int mode;
    float uniformDensity;
    float skyFogDensity;
    float3 fogColor;
    int numLayers;
    bool fastFog;
    // Each layer: y, breadth, density
    float layerY[MAX_FOG_LAYERS];
    float layerBreadth[MAX_FOG_LAYERS];
    float layerDensity[MAX_FOG_LAYERS];
} FogConfig;

FogConfig Fog_load(__global const float* fogData) {
    FogConfig fog;
    fog.mode = (int)fogData[0];
    fog.uniformDensity = fogData[1];
    fog.skyFogDensity = fogData[2];
    fog.fogColor = (float3)(fogData[3], fogData[4], fogData[5]);
    fog.numLayers = (int)fogData[6];
    fog.fastFog = (fogData[7 + MAX_FOG_LAYERS * 3] > 0.5f);
    for (int i = 0; i < MAX_FOG_LAYERS && i < fog.numLayers; i++) {
        fog.layerY[i] = fogData[8 + i * 3];
        fog.layerBreadth[i] = fogData[8 + i * 3 + 1];
        fog.layerDensity[i] = fogData[8 + i * 3 + 2];
    }
    return fog;
}

// Clamp dy to prevent division by zero, matching CPU Fog.clampDy()
float Fog_clampDy(float dy) {
    float epsilon = 0.00001f;
    if (dy > 0.0f) {
        return (dy < epsilon) ? epsilon : dy;
    } else {
        return (dy > -epsilon) ? -epsilon : dy;
    }
}

// Logistic CDF: 1 / (1 + exp((y0 - y) / breadth))
// This is the integral of the logistic PDF used for layered fog density.
float Fog_logisticCDF(float y, float y0, float breadthInv) {
    return 1.0f / (1.0f + exp((y0 - y) * breadthInv));
}

// Calculate layered fog transmittance using logistic distribution CDF.
// Matches CPU Fog.addLayeredFog():
//   total += density * (CDF(y1) - CDF(y2))  per layer
//   extinction = exp(total / dy)
float Fog_layeredTransmittance(FogConfig fog, float dy, float y1, float y2) {
    if (fog.numLayers <= 0) return 1.0f;
    float total = 0.0f;
    for (int i = 0; i < fog.numLayers && i < MAX_FOG_LAYERS; i++) {
        float breadth = fog.layerBreadth[i];
        if (breadth <= 0.0f) continue;
        float breadthInv = 1.0f / breadth;
        float y0 = fog.layerY[i];
        total += fog.layerDensity[i] * (Fog_logisticCDF(y1, y0, breadthInv) - Fog_logisticCDF(y2, y0, breadthInv));
    }
    return exp(total / Fog_clampDy(dy));
}

// Sample a scatter offset for layered fog using logistic quantile function.
// Matches CPU Fog.sampleLayeredScatterOffset():
//   middle = CDF(y1) + rand * (CDF(y2) - CDF(y1))
//   offsetY = log(middle / (1 - middle)) * breadth + y0
//   return (offsetY - y1) / dy
float Fog_sampleLayeredScatterOffset(FogConfig fog, float y1, float y2, float dy, Random random) {
    if (fog.numLayers <= 0) return EPS;
    // CPU only uses the first layer for importance sampling
    float breadth = fog.layerBreadth[0];
    if (breadth <= 0.0f) return EPS;
    float breadthInv = 1.0f / breadth;
    float y0 = fog.layerY[0];
    float y1v = Fog_logisticCDF(y1, y0, breadthInv);
    float y2v = Fog_logisticCDF(y2, y0, breadthInv);
    float middle = y1v + Random_nextFloat(random) * (y2v - y1v);
    // Clamp to avoid log(0) or log(inf)
    middle = clamp(middle, 1e-7f, 1.0f - 1e-7f);
    float offsetY = log(middle / (1.0f - middle)) * breadth + y0;
    return (offsetY - y1) / Fog_clampDy(dy);
}

// Sample a random point along the ray for inscatter calculation.
// For uniform fog: uniform random along distance.
// For layered fog: importance-sampled via logistic quantile.
float Fog_sampleScatterOffset(FogConfig fog, float distance, float3 origin, float3 direction, Random random) {
    float d = fmin(distance, FOG_LIMIT);
    if (fog.mode == FOG_MODE_LAYERED) {
        float dy = direction.y;
        float y1 = origin.y;
        float y2 = origin.y + dy * d;  // not used for sky, only ground
        float offset = Fog_sampleLayeredScatterOffset(fog, y1, y2, dy, random);
        return clamp(offset, EPS, d - EPS);
    }
    // Uniform: simple random offset
    return clamp(Random_nextFloat(random) * d, EPS, d - EPS);
}

// Add ground fog contribution (inscatter with sun visibility).
// Matches CPU Fog.addGroundFog().
void Fog_addGroundFog(FogConfig fog, float3* color, float3 origin, float3 direction,
                      float distance, float3 sunAttenuation, float sunIntensity, float offset) {
    if (fog.mode == FOG_MODE_NONE) return;

    float d = fmin(distance, FOG_LIMIT);

    if (fog.mode == FOG_MODE_UNIFORM) {
        // CPU: fogDensity = uniformDensity * EXTINCTION_FACTOR
        //      extinction = exp(-airDistance * fogDensity)
        float fogDensity = fog.uniformDensity * FOG_EXTINCTION;
        float extinction = exp(-d * fogDensity);
        *color *= extinction;
        if (sunIntensity > EPS) {
            float inscatter = sunIntensity;
            if (fog.fastFog) {
                inscatter *= (1.0f - extinction);
            } else {
                inscatter *= d * fogDensity * exp(-offset * fogDensity);
            }
            *color += sunAttenuation * fog.fogColor * inscatter;
        }
    } else {
        // Layered fog: use logistic CDF integration
        float dy = direction.y;
        float y1 = origin.y;
        float y2 = y1 + dy * d;   // approximate end height using first-hit direction
        float extinction = Fog_layeredTransmittance(fog, dy, y1, y2);
        *color *= extinction;
        if (sunIntensity > EPS) {
            float inscatter = (1.0f - extinction) * sunIntensity;
            *color += inscatter * sunAttenuation * fog.fogColor;
        }
    }
}

// Add sky fog (for rays that hit the sky).
// Matches CPU Fog.addSkyFog().
// sunAttenuation: light attenuation at a sampled scatter point (for layered mode).
//   Pass (float3)(0) when no sun scatter should be applied (uniform mode).
void Fog_addSkyFog(FogConfig fog, float3* color, float3 origin, float3 direction,
                   float3 sunAttenuation, float sunIntensity) {
    if (fog.mode == FOG_MODE_NONE) return;

    if (fog.mode == FOG_MODE_UNIFORM) {
        // CPU: fog = (1 - dy)^2 for dy>0, fog = 1 for dy<=0, scaled by skyFogDensity
        if (fog.uniformDensity <= 0.0f) return;
        float fogAmount;
        if (direction.y > 0.0f) {
            float t = 1.0f - direction.y;
            fogAmount = t * t;
        } else {
            fogAmount = 1.0f;
        }
        fogAmount *= fog.skyFogDensity;
        // CPU: color = (1 - fog) * color + fog * fogColor
        // No sun scatter for uniform sky fog (CPU passes null for scatterLight)
        *color = (1.0f - fogAmount) * (*color) + fogAmount * fog.fogColor;
    } else if (fog.mode == FOG_MODE_LAYERED) {
        // CPU: addLayeredFog with y2 = y1 + dy * FOG_LIMIT
        float dy = direction.y;
        float y1 = origin.y;
        float y2 = y1 + dy * FOG_LIMIT;
        float extinction = Fog_layeredTransmittance(fog, dy, y1, y2);
        *color *= extinction;
        // CPU: inscatter = (1 - extinction) * scatterLight.w
        if (sunIntensity > EPS) {
            float inscatter = (1.0f - extinction) * sunIntensity;
            *color += inscatter * sunAttenuation * fog.fogColor;
        }
    }
}

// Sample a scatter offset specifically for sky fog (layered mode only).
// Matches CPU Fog.sampleSkyScatterOffset().
float Fog_sampleSkyScatterOffset(FogConfig fog, float3 origin, float3 direction, Random random) {
    if (fog.mode != FOG_MODE_LAYERED || fog.numLayers <= 0) return EPS;
    float dy = direction.y;
    float y1 = origin.y;
    // CPU: y2 = dy > 0 ? scene.yMax : scene.yMin
    // We approximate yMax/yMin with the octree boundary. For sky fog, FOG_LIMIT
    // is used as a proxy since the exact scene bounds aren't available here.
    // The scatter offset is a distance along the ray, not a y-coordinate.
    float y2 = y1 + dy * FOG_LIMIT;
    return Fog_sampleLayeredScatterOffset(fog, y1, y2, dy, random);
}

#endif
