#ifndef CHUNKYCLPLUGIN_WATER_H
#define CHUNKYCLPLUGIN_WATER_H

#include "../opencl.h"
#include "rt.h"
#include "noise.h"
#include "constants.h"

// Water shading strategies
#define WATER_SHADING_STILL            0
#define WATER_SHADING_SIMPLEX          1
#define WATER_SHADING_TILED_NORMALMAP  2

// Water shader parameters (from SimplexWaterShader.java)
typedef struct {
    int iterations;       // Number of FBM octaves (default 4)
    float baseFrequency;  // First octave frequency (default 0.4)
    float baseAmplitude;  // First octave amplitude (default 0.025)
    float animationSpeed; // Animation speed multiplier (default 1)
} WaterShaderParams;

// Apply simplex noise water shading to perturb the surface normal.
// Uses 3D simplex noise with analytical derivatives, matching the CPU SimplexWaterShader.
// The noise uses (x*freq, z*freq, time) as the 3D coordinate.
void Water_simplexShading(IntersectionRecord* record, float wx, float wz,
                          float animationTime, WaterShaderParams params) {
    float frequency = params.baseFrequency;
    float amplitude = params.baseAmplitude;
    float time = animationTime * params.animationSpeed;

    float ddx = 0.0f;
    float ddz = 0.0f;

    for (int i = 0; i < params.iterations && i < 8; i++) {
        float fx = wx * frequency;
        float fz = wz * frequency;

        // 3D simplex noise: (x, z, time) with analytical derivatives
        float nddx, nddy, nddz;
        simplexNoise3(fx, fz, time, &nddx, &nddy, &nddz);
        // CPU maps world x -> noise x, world z -> noise y, time -> noise z
        // So noise.ddx = dN/dx, noise.ddy = dN/dz (in world space)
        float ddxNext = ddx - amplitude * nddx;
        float ddzNext = ddz - amplitude * nddy;

        // NaN guard (matching CPU behavior)
        if (isnan(ddxNext + ddzNext)) {
            break;
        }
        ddx = ddxNext;
        ddz = ddzNext;

        frequency *= 2.0f;
        amplitude *= 0.5f;
    }

    // Compute normal from slopes using cross product of tangent vectors,
    // matching CPU SimplexWaterShader (normal.cross(zslope, xslope)).
    // xslope = (1, ddx, 0), zslope = (0, ddz, 1)
    // cross(zslope, xslope) = (-ddx, 1, -ddz)  (unnormalized)
    // NOTE: the X and Z components are NEGATED — the previous code used
    // (ddx, 1, ddz), which mirrored every wave's tilt and made simplex-water
    // specular highlights / refraction offsets point the wrong way vs CPU.
    float3 n = (float3)(-ddx, 1.0f, -ddz);
    n = normalize(n);

    // Flip the normal if the ray hit from below
    if (record->normal.y < 0) {
        n.y = -n.y;
    }

    record->normal = n;
}

// Tiled normal map water shading, matching CPU WaterModel.doWaterDisplacement().
// Uses a precomputed gradient normal map sampled at two scales (period 16 and 2).
void Water_tiledNormalMapShading(IntersectionRecord* record, float wx, float wz,
                                 __global const float* normalMap, int normalMapW) {
    if (normalMapW <= 0) return;

    float invW = 1.0f / (float)normalMapW;

    // Scale 1: large tiles (period = 16 world units)
    float x1 = wx / 16.0f;
    x1 = x1 - floor(x1);  // frac
    float z1 = wz / 16.0f;
    z1 = z1 - floor(z1);  // frac
    int u1 = clamp((int)(x1 * normalMapW - 1e-5f), 0, normalMapW - 1);
    int v1 = clamp((int)((1.0f - z1) * normalMapW - 1e-5f), 0, normalMapW - 1);
    int idx1 = (u1 * normalMapW + v1) * 2;
    float nx = normalMap[idx1];
    float nz = normalMap[idx1 + 1];

    // Scale 2: small tiles (period = 2 world units) at half weight
    float x2 = wx / 2.0f;
    x2 = x2 - floor(x2);
    float z2 = wz / 2.0f;
    z2 = z2 - floor(z2);
    int u2 = clamp((int)(x2 * normalMapW - 1e-5f), 0, normalMapW - 1);
    int v2 = clamp((int)((1.0f - z2) * normalMapW - 1e-5f), 0, normalMapW - 1);
    int idx2 = (u2 * normalMapW + v2) * 2;
    nx += normalMap[idx2] * 0.5f;
    nz += normalMap[idx2 + 1] * 0.5f;

    // Construct normal: n = (nx, 0.15, nz), then normalize
    float3 n = normalize((float3)(nx, 0.15f, nz));

    // Flip if hit from below
    if (record->normal.y < 0) {
        n.y = -n.y;
    }
    record->normal = n;
}

// Apply water shading to an intersection record using world-space hit coords
void Water_applyShading(IntersectionRecord* record, int waterShadingStrategy,
                        float animationTime, float wx, float wz,
                        WaterShaderParams params,
                        __global const float* waterNormalMap, int waterNormalMapW) {
    switch (waterShadingStrategy) {
        case WATER_SHADING_SIMPLEX:
            Water_simplexShading(record, wx, wz, animationTime, params);
            break;
        case WATER_SHADING_TILED_NORMALMAP:
            Water_tiledNormalMapShading(record, wx, wz, waterNormalMap, waterNormalMapW);
            break;
        case WATER_SHADING_STILL:
        default:
            // No modification - water stays flat
            break;
    }
}

// Check if a ray-plane intersection occurs for the water plane.
// chunkBitmap is a 2D bitfield of loaded chunks (1 bit per 16x16 chunk).
// chunkBitmapSize is the number of chunks per side (octreeSize / 16).
// When chunkClip is true, water is hidden where chunks are loaded (bitmap=1),
// matching the CPU behavior where the water plane fills unloaded areas.
bool Water_planeIntersect(Ray ray, float waterPlaneHeight, float octreeSize, bool chunkClip,
                          __global const int* chunkBitmap, int chunkBitmapSize,
                          IntersectionRecord* record) {
    // Only intersect if ray crosses the Y plane
    if (fabs(ray.direction.y) < EPS) return false;

    float t = (waterPlaneHeight - ray.origin.y) / ray.direction.y;
    if (t < OFFSET || t >= record->distance) return false;

    float3 hitPoint = ray.origin + ray.direction * t;

    // Chunk clipping: hide water where chunks are loaded (bitmap lookup).
    // The CPU's PreviewRayTracer.waterPlaneIntersection checks isChunkLoaded()
    // per-block; we approximate this per-chunk using the exported bitmap.
    if (chunkClip && chunkBitmapSize > 0) {
        int cx = (int)floor(hitPoint.x) >> 4;  // / 16
        int cz = (int)floor(hitPoint.z) >> 4;  // / 16
        if (cx >= 0 && cx < chunkBitmapSize && cz >= 0 && cz < chunkBitmapSize) {
            int bitIndex = cz * chunkBitmapSize + cx;
            int word = chunkBitmap[bitIndex >> 5];  // / 32
            bool isLoaded = (word >> (bitIndex & 31)) & 1;
            if (isLoaded) {
                return false;  // Hide water plane in loaded chunks
            }
        }
        // If outside the bitmap range, chunk is not loaded -> show water
    }

    record->distance = t;
    record->normal = (ray.direction.y < 0) ? (float3)(0, 1, 0) : (float3)(0, -1, 0);
    // Store world-space XZ in texCoord for water shading
    record->texCoord = (float2)(hitPoint.x, hitPoint.z);
    record->material = -1; // Special marker for water plane

    return true;
}

// ===========================================================================
//  Water volume optics
// ---------------------------------------------------------------------------
//  These DIVERGE from Chunky's CPU renderer on purpose. CPU water fog is a
//  single monochrome Beer's-law term, exp(-d / waterVisibility), with no
//  scattering at all — so the only thing water can ever do to a ray is subtract
//  from it, and every underwater path decays to black. That is the "rays die
//  off" this replaces.
// ===========================================================================

// Per-channel extinction, expressed RELATIVE to the user's waterVisibility.
// Clear water absorbs red several times faster than blue, and that imbalance is
// the entire reason deep water reads blue-green rather than neutral grey.
// RED is pinned to 1.0 so waterVisibility keeps its original meaning — red still
// falls off as exp(-d / waterVisibility) — and only the colour balance is new.
#define WATER_EXTINCTION_R 1.00f
#define WATER_EXTINCTION_G 0.42f
#define WATER_EXTINCTION_B 0.26f

// Strength of scattering: what fraction of extinguished light comes back into
// the path instead of being absorbed. 0.0 reproduces the old pure-absorption
// behaviour (fade to black). This is the master "how murky" knob.
#define WATER_SCATTER_STRENGTH 0.55f

// Henyey-Greenstein anisotropy. Water scatters strongly FORWARD, which is why
// looking toward the sun underwater is a bright haze and looking away is dim and
// flat. 0.0 is isotropic (the dead, uniformly-tinted look).
#define WATER_SCATTER_G 0.35f

// Ceiling on the phase function. Guards against a hot firefly when a path looks
// straight down the sun vector through a long water segment.
#define WATER_PHASE_MAX 3.0f

// Isotropic skylight permeating the water, under the directional sun term.
// This is NOT a token epsilon: it is the only thing lighting water the sun
// cannot reach, so it sets how bright shadowed water AND the deep-water horizon
// are. It is the level for Chunky's default DAYTIME sky: the kernel scales it by
// the scene's actual skylight (Water_skyLight), so night water goes dark instead
// of glowing.
//
// >>> RAISE THIS FIRST if the underwater horizon still reads as too dark. <<<
// Not phase-modulated, by definition — ambient is directionless.
#define WATER_AMBIENT 0.35f

// The light the tuning above was done in: Chunky's defaults. Luminance of the
// skylight its default sky gives an upward-facing surface (simulated sky, sun 60
// degrees up; ClSky.skyAmbient = 0.211, 0.424, 0.691), and the default sun's
// intensity^2.2 (1.25^2.2). A default scene therefore looks exactly as tuned, and
// every other sky, time of day or sun setting scales from it.
#define WATER_DAY_SKY_LUMINANCE 0.398f
#define WATER_DAY_SUN_POWER 1.6338f

// Caustic contrast. 1.0 is the raw projected-area ratio; higher tightens the
// bright bands. Paired with WATER_CAUSTIC_MAX, which caps how much a single wave
// facet may concentrate the sun (energy guard — see Water_causticFactor).
#define WATER_CAUSTIC_SHARPNESS 2.0f
#define WATER_CAUSTIC_MAX       2.5f

// Fallback scattering albedo for a colourless water setting (clear open water).
#define WATER_ALBEDO_FALLBACK ((float3)(0.12f, 0.45f, 0.62f))

// Transmittance of a water segment, per colour channel.
float3 Water_extinction(float distance, float waterVisibility) {
    if (waterVisibility <= 0) return (float3)(0.0f);
    float a = distance / waterVisibility;
    return exp(-a * (float3)(WATER_EXTINCTION_R, WATER_EXTINCTION_G, WATER_EXTINCTION_B));
}

// Scattering albedo, derived from the scene's water colour by keeping its HUE
// and renormalising its MAGNITUDE.
//
// This renormalisation is essential, not cosmetic. Chunky's waterColor is an
// ABSORPTION tint and its default is (0.03, 0.13, 0.16) — very dark. Feeding
// that in directly as a scattering albedo makes in-scattered light about 6x too
// dim and every underwater scene still reads as black, which is the exact bug
// this section exists to fix. Dividing by the max channel keeps the user's
// colour choice meaningful while decoupling brightness from it, so the two
// concerns stay independently tunable.
float3 Water_scatterAlbedo(float3 waterColor) {
    float m = fmax(waterColor.x, fmax(waterColor.y, waterColor.z));
    if (m < 1e-4f) return WATER_ALBEDO_FALLBACK;
    return waterColor / m;
}

// Henyey-Greenstein phase function, normalised so g = 0 returns exactly 1.0.
// That normalisation matters: it keeps overall brightness independent of the
// anisotropy knob, so WATER_SCATTER_G can be tuned for LOOK without also having
// to re-tune WATER_SCATTER_STRENGTH for exposure.
// cosTheta is dot(viewDir, sunDir), so +1 means looking straight at the sun.
float Water_phase(float cosTheta) {
    float g2 = WATER_SCATTER_G * WATER_SCATTER_G;
    float d = 1.0f + g2 - 2.0f * WATER_SCATTER_G * cosTheta;
    d = fmax(d, 1e-4f);
    return clamp((1.0f - g2) / (d * sqrt(d)), 0.0f, WATER_PHASE_MAX);
}

// Radiance scattered INTO a segment with the given transmittance. Whatever the
// segment removed (1 - transmittance) partly comes back, tinted by the water's
// scattering albedo, so thick water settles to a lit haze rather than to black.
//
// `light` is the ACTUAL sunlight reaching a point inside the segment, not a flat
// constant. That is what buys the realism: because the shadow ray that produced
// it was itself attenuated by Water_extinction on the way down, deep water
// darkens on its own, shafts appear behind geometry, and the caustic modulation
// carries into the volume — none of which needs to be faked separately.
float3 Water_inscatter(float3 transmittance, float3 waterColor, float3 light) {
    return ((float3)(1.0f) - transmittance) * Water_scatterAlbedo(waterColor)
         * light * WATER_SCATTER_STRENGTH;
}

// Scene skylight relative to the default day (1.0 = the tuning's sky; ~0 at night).
// skyAmbient is ClSky.skyAmbient, passed in waterConfig[24..26].
float Water_skyLight(float3 skyAmbient) {
    return dot(skyAmbient, (float3)(0.2126f, 0.7152f, 0.0722f)) / WATER_DAY_SKY_LUMINANCE;
}

// Sun colour and power relative to the default sun (white, intensity 1.25).
float3 Water_sunLight(float3 sunColor, float sunPowGamma) {
    return sunColor * (sunPowGamma / WATER_DAY_SUN_POWER);
}

// The light term for Water_inscatter: the sun's light (colour and power, times
// its visibility from inside the water) shaped by the phase function, over the
// isotropic skylight floor scaled to the scene's actual sky.
float3 Water_scatterLight(float3 sunAttenuation, float cosTheta, float3 sunLight, float skyLight) {
    return sunAttenuation * sunLight * Water_phase(cosTheta) + (float3)(WATER_AMBIENT * skyLight);
}

// Wave normal at a world XZ position, without an IntersectionRecord.
// Shadow rays deliberately SKIP wave shading for CPU parity (see kernel.h), so
// the caustic term cannot read a perturbed record->normal and has to sample the
// wave field itself. Reuses Water_applyShading so caustics can never drift out
// of step with the surface the user actually sees.
float3 Water_waveNormal(int waterShadingStrategy, float animationTime,
                        float wx, float wz, WaterShaderParams params,
                        __global const float* waterNormalMap, int waterNormalMapW) {
    IntersectionRecord probe;
    probe.normal = (float3)(0.0f, 1.0f, 0.0f);
    Water_applyShading(&probe, waterShadingStrategy, animationTime, wx, wz,
                       params, waterNormalMap, waterNormalMapW);
    return probe.normal;
}

// Focusing factor for sunlight crossing a wavy surface — the caustic term.
// The surface is a lens: a facet tilted to face the sun gathers a wider slice of
// the beam and concentrates it below, one tilted away spreads it out. Comparing
// the facet's projected area against flat water gives that ratio directly, which
// is why the bright bands track the SAME simplex noise that shapes the surface.
//
// This is a cheap approximation, NOT a photon-mapped caustic: it is stable, and
// clamped so a near-grazing sun cannot divide by ~0 and flare the whole seabed.
float Water_causticFactor(float3 waveNormal, float3 sunDir) {
    float flatCos = fabs(sunDir.y);
    if (flatCos < 1e-3f) return 1.0f;  // sun on the horizon: no meaningful focus
    float waveCos = fabs(dot(sunDir, waveNormal));
    float ratio = clamp(waveCos / flatCos, 0.0f, WATER_CAUSTIC_MAX);
    return pow(ratio, WATER_CAUSTIC_SHARPNESS);
}

#endif
