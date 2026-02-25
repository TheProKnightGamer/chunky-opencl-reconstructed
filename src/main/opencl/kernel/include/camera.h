#ifndef CHUNKYCLPLUGIN_CAMERA_H
#define CHUNKYCLPLUGIN_CAMERA_H

#include "../opencl.h"
#include "rt.h"
#include "random.h"
#include "constants.h"

// Aperture shape IDs
#define APERTURE_CIRCLE   0
#define APERTURE_HEXAGON  1
#define APERTURE_PENTAGON 2
#define APERTURE_STAR     3
#define APERTURE_GAUSSIAN 4
#define APERTURE_CUSTOM   5

Ray Camera_preGenerated(__global const float* rays, int index) {
    Ray ray;
    ray.origin = vload3(index * 2, rays);
    ray.direction = vload3(index * 2 + 1, rays);
    return ray;
}

// Sample a point on the aperture based on the shape
void Camera_sampleAperture(float aperture, int apertureShape, Random random, float* rx, float* ry,
                            __global const int* apertureMask, int apertureMaskWidth) {
    float r, theta;
    switch (apertureShape) {
        case APERTURE_HEXAGON: {
            // Rejection sampling within hexagon
            for (int i = 0; i < 16; i++) {
                float u = (Random_nextFloat(random) * 2.0f - 1.0f) * aperture;
                float v = (Random_nextFloat(random) * 2.0f - 1.0f) * aperture;
                float ax = fabs(u), ay = fabs(v);
                if (ay < aperture * 0.866f && ax + ay * 0.577f < aperture) {
                    *rx = u; *ry = v; return;
                }
            }
            // Fallback to circle
            r = sqrt(Random_nextFloat(random)) * aperture;
            theta = Random_nextFloat(random) * M_PI_F * 2.0f;
            *rx = cos(theta) * r; *ry = sin(theta) * r;
            break;
        }
        case APERTURE_PENTAGON: {
            for (int i = 0; i < 16; i++) {
                float u = (Random_nextFloat(random) * 2.0f - 1.0f) * aperture;
                float v = (Random_nextFloat(random) * 2.0f - 1.0f) * aperture;
                float angle = atan2(v, u);
                float sector = M_PI_F * 2.0f / 5.0f;
                float halfSector = sector * 0.5f;
                float localAngle = fmod(fmod(angle, sector) + sector, sector) - halfSector;
                float maxR = aperture * cos(halfSector) / cos(localAngle);
                if (sqrt(u * u + v * v) < maxR) {
                    *rx = u; *ry = v; return;
                }
            }
            r = sqrt(Random_nextFloat(random)) * aperture;
            theta = Random_nextFloat(random) * M_PI_F * 2.0f;
            *rx = cos(theta) * r; *ry = sin(theta) * r;
            break;
        }
        case APERTURE_STAR: {
            // Star shape via modulated radius
            theta = Random_nextFloat(random) * M_PI_F * 2.0f;
            float starMod = 0.5f + 0.5f * cos(theta * 5.0f);
            r = sqrt(Random_nextFloat(random)) * aperture * starMod;
            *rx = cos(theta) * r; *ry = sin(theta) * r;
            break;
        }
        case APERTURE_GAUSSIAN: {
            // Box-Muller transform for gaussian distribution
            float u1 = fmax(Random_nextFloat(random), 1e-10f);
            float u2 = Random_nextFloat(random);
            float g = sqrt(-2.0f * log(u1));
            *rx = g * cos(2.0f * M_PI_F * u2) * aperture * 0.5f;
            *ry = g * sin(2.0f * M_PI_F * u2) * aperture * 0.5f;
            break;
        }
        case APERTURE_CUSTOM: {
            // Rejection sampling from user-supplied grayscale aperture mask
            if (apertureMask && apertureMaskWidth > 0) {
                for (int i = 0; i < 100; i++) {
                    float u = Random_nextFloat(random);
                    float v = Random_nextFloat(random);
                    int col = (int)(u * apertureMaskWidth);
                    int row = (int)(v * apertureMaskWidth);
                    if (col >= apertureMaskWidth) col = apertureMaskWidth - 1;
                    if (row >= apertureMaskWidth) row = apertureMaskWidth - 1;
                    int color = apertureMask[row * apertureMaskWidth + col];
                    float probability = (float)(color & 0xFF) / 255.0f;
                    if (Random_nextFloat(random) <= probability) {
                        *rx = (u - 0.5f) * 2.0f * aperture;
                        *ry = (v - 0.5f) * 2.0f * aperture;
                        return;
                    }
                }
                // Fallback: no DOF shift
                *rx = 0.0f; *ry = 0.0f;
                break;
            }
            // Fall through to circle if no mask data
        }
        default: // APERTURE_CIRCLE
            r = sqrt(Random_nextFloat(random)) * aperture;
            theta = Random_nextFloat(random) * M_PI_F * 2.0f;
            *rx = cos(theta) * r;
            *ry = sin(theta) * r;
            break;
    }
}

// Projector type 0: Standard pinhole/perspective
Ray Camera_pinHole(float x, float y, Random random, __global const float* projectorSettings,
                   __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float aperture = projectorSettings[0];
    float subjectDistance = projectorSettings[1];
    float fovTan = projectorSettings[2];
    int apertureShape = (int)projectorSettings[3];

    ray.origin = (float3) (0, 0, 0);
    ray.direction = (float3) (fovTan * x, fovTan * y, 1.0f);

    if (aperture > 0) {
        ray.direction *= subjectDistance / ray.direction.z;

        float rx, ry;
        Camera_sampleAperture(aperture, apertureShape, random, &rx, &ry, apertureMask, apertureMaskWidth);

        ray.direction -= (float3) (rx, ry, 0);
        ray.origin += (float3) (rx, ry, 0);
    }

    return ray;
}

// Projector type 1: Parallel/orthographic
Ray Camera_parallel(float x, float y, Random random, __global const float* projectorSettings,
                    __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fovTan = projectorSettings[0];
    float aperture = projectorSettings[1];
    int apertureShape = (int)projectorSettings[3];
    float worldWidth = fovTan * 2.0f;

    ray.origin = (float3) (worldWidth * x, worldWidth * y, 0);
    ray.direction = (float3) (0, 0, 1.0f);

    // For parallel projection, DoF is applied as lateral offset only (no convergence)
    if (aperture > 0) {
        float rx, ry;
        Camera_sampleAperture(aperture, apertureShape, random, &rx, &ry, apertureMask, apertureMaskWidth);
        ray.origin += (float3)(rx, ry, 0);
    }
    return ray;
}

// Apply spherical depth-of-field to a ray (for non-pinhole projections).
// Converts direction to yaw/pitch, applies aperture offset in that space.
void Camera_applySphericalDoF(Ray* ray, float aperture, float subjectDistance, int apertureShape, Random random,
                               __global const int* apertureMask, int apertureMaskWidth) {
    if (aperture <= 0) return;

    // Scale direction to focus at subjectDistance
    float3 focusPoint = ray->origin + ray->direction * subjectDistance;

    // Sample aperture offset
    float rx, ry;
    Camera_sampleAperture(aperture, apertureShape, random, &rx, &ry, apertureMask, apertureMaskWidth);

    // Get yaw/pitch of the ray direction
    float yaw = atan2(ray->direction.x, ray->direction.z);
    float pitch = asin(clamp(ray->direction.y, -1.0f, 1.0f));

    // Build rotation matrix from yaw/pitch
    float cosYaw = cos(-yaw);
    float sinYaw = sin(-yaw);
    float cosPitch = cos(-pitch);
    float sinPitch = sin(-pitch);

    // Apply aperture offset rotated into camera space
    ray->origin.x += cosYaw * rx + sinYaw * sinPitch * ry;
    ray->origin.y += cosPitch * ry;
    ray->origin.z += -sinYaw * rx + cosYaw * sinPitch * ry;

    // Recompute direction toward focus point
    ray->direction = normalize(focusPoint - ray->origin);
}

// Projector type 2: Fisheye (equidistant)
Ray Camera_fisheye(float x, float y, Random random, __global const float* projectorSettings,
                   __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fov = projectorSettings[0];
    float aperture = projectorSettings[1];
    float subjectDistance = projectorSettings[2];
    int apertureShape = (int)projectorSettings[3];

    float r = sqrt(x * x + y * y);
    float theta = r * fov * M_PI_F / 360.0f;

    ray.origin = (float3)(0, 0, 0);
    if (r < EPS) {
        ray.direction = (float3)(0, 0, 1);
    } else {
        float sinTheta = sin(theta);
        ray.direction = (float3)(sinTheta * x / r, sinTheta * y / r, cos(theta));
    }

    Camera_applySphericalDoF(&ray, aperture, subjectDistance, apertureShape, random, apertureMask, apertureMaskWidth);
    return ray;
}

// Projector type 3: Stereographic
Ray Camera_stereographic(float x, float y, Random random, __global const float* projectorSettings,
                         __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fov = projectorSettings[0];
    float aperture = projectorSettings[1];
    float subjectDistance = projectorSettings[2];
    int apertureShape = (int)projectorSettings[3];
    float scale = 2.0f * tan(fov * M_PI_F / 720.0f);

    float xt = x * scale;
    float yt = y * scale;
    float r2 = xt * xt + yt * yt;
    float denom = 1.0f + r2;

    ray.origin = (float3)(0, 0, 0);
    ray.direction = (float3)(2.0f * xt / denom, 2.0f * yt / denom, (1.0f - r2) / denom);
    ray.direction = normalize(ray.direction);

    Camera_applySphericalDoF(&ray, aperture, subjectDistance, apertureShape, random, apertureMask, apertureMaskWidth);
    return ray;
}

// Projector type 4: Panoramic (equirectangular)
Ray Camera_panoramic(float x, float y, Random random, __global const float* projectorSettings,
                     __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fov = projectorSettings[0];
    float aperture = projectorSettings[1];
    float subjectDistance = projectorSettings[2];
    int apertureShape = (int)projectorSettings[3];
    float fovRad = fov * M_PI_F / 360.0f;

    float phi = x * fovRad;
    float theta = y * fovRad;

    ray.origin = (float3)(0, 0, 0);
    float cosT = cos(theta);
    ray.direction = (float3)(sin(phi) * cosT, sin(theta), cos(phi) * cosT);

    Camera_applySphericalDoF(&ray, aperture, subjectDistance, apertureShape, random, apertureMask, apertureMaskWidth);
    return ray;
}

// Projector type 5: Panoramic slot
Ray Camera_panoramicSlot(float x, float y, Random random, __global const float* projectorSettings,
                         __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fov = projectorSettings[0];
    float aperture = projectorSettings[1];
    float subjectDistance = projectorSettings[2];
    int apertureShape = (int)projectorSettings[3];
    float fovRad = fov * M_PI_F / 360.0f;
    float phi = x * fovRad;

    ray.origin = (float3)(0, 0, 0);
    ray.direction = (float3)(sin(phi), y * fovRad, cos(phi));
    ray.direction = normalize(ray.direction);

    Camera_applySphericalDoF(&ray, aperture, subjectDistance, apertureShape, random, apertureMask, apertureMaskWidth);
    return ray;
}

// Projector type 6: Omni-directional stereo (left/right)
Ray Camera_ODS(float x, float y, Random random, __global const float* projectorSettings) {
    Ray ray;
    float ipd = projectorSettings[0]; // inter-pupillary distance
    float side = projectorSettings[1]; // -1 left, +1 right

    float phi = x * M_PI_F;
    float theta = y * M_PI_F * 0.5f;

    float cosT = cos(theta);
    ray.direction = (float3)(sin(phi) * cosT, sin(theta), cos(phi) * cosT);

    // Offset origin for stereo
    float offset = ipd * 0.5f * side;
    ray.origin = (float3)(cos(phi) * offset, 0, -sin(phi) * offset);
    return ray;
}

// Projector type 7: ODS Stacked (both eyes in one image, top=left, bottom=right)
Ray Camera_ODSStacked(float x, float y, Random random, __global const float* projectorSettings) {
    Ray ray;
    float ipd = projectorSettings[0];

    // Top half = left eye, bottom half = right eye
    float side;
    float adjustedY;
    if (y < 0) {
        // Top half (y < 0 in normalized coords) = left eye
        side = -1.0f;
        adjustedY = y * 2.0f + 0.5f; // remap [-0.5, 0] to [-0.5, 0.5]
    } else {
        // Bottom half (y >= 0) = right eye
        side = 1.0f;
        adjustedY = y * 2.0f - 0.5f; // remap [0, 0.5] to [-0.5, 0.5]
    }

    float phi = x * M_PI_F;
    float theta = adjustedY * M_PI_F * 0.5f;

    float cosT = cos(theta);
    ray.direction = (float3)(sin(phi) * cosT, sin(theta), cos(phi) * cosT);

    float offset = ipd * 0.5f * side;
    ray.origin = (float3)(cos(phi) * offset, 0, -sin(phi) * offset);
    return ray;
}

#endif
