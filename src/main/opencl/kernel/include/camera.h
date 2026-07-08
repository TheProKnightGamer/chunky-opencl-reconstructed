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
// Settings: pset0=aperture, pset1=subjectDistance, pset2=fovTan, pset3=apertureShape.
Ray Camera_pinHole(float x, float y, Random random,
                   float pset0, float pset1, float pset2, float pset3,
                   __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float aperture = pset0;
    float subjectDistance = pset1;
    float fovTan = pset2;
    int apertureShape = (int)pset3;

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
// Settings: pset0=fov (RAW world units), pset1=aperture, pset2=worldDiagonalSize,
//           pset3=apertureShape.
Ray Camera_parallel(float x, float y, Random random,
                    float pset0, float pset1, float pset2, float pset3, float pset4,
                    __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fov = pset0;
    float aperture = pset1;
    float worldDiagonalSize = pset2;
    int apertureShape = (int)pset3;
    float subjectDistance = pset4;

    // CPU ParallelProjector sets o = (fov*x, fov*y, 0) using the RAW fov (world
    // units, NOT clampedFovTan). A ForwardDisplacementProjector(-worldDiagonalSize)
    // then pushes the origin back by worldDiagonalSize along +z so the orthographic
    // rays start behind the scene instead of on the camera plane (which clipped
    // geometry in front of it). Direction stays (0,0,1).
    ray.origin = (float3) (fov * x, fov * y, -worldDiagonalSize);
    ray.direction = (float3) (0, 0, 1.0f);

    // CPU parallel DoF: an ApertureProjector with focus distance
    // (subjectDistance + worldDiagonalSize) scales d to that distance, then
    // d -= (rx,ry,0); o += (rx,ry,0) — CONVERGING rays with a sharp subject
    // plane, not a pure lateral origin shift (which blurred every depth equally).
    // ray_to_camera normalizes the direction after the world transform.
    if (aperture > 0) {
        float rx, ry;
        Camera_sampleAperture(aperture, apertureShape, random, &rx, &ry, apertureMask, apertureMaskWidth);
        float focusDist = subjectDistance + worldDiagonalSize;
        ray.direction = (float3) (-rx, -ry, focusDist);
        ray.origin.x += rx;
        ray.origin.y += ry;
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

    // Build rotation matrix from yaw/pitch. CPU SphericalApertureProjector
    // rotates the aperture point with Matrix3.rotate(-pitch, +yaw, 0). cos is
    // even so cos(-yaw)==cos(yaw), but sin(-yaw) flipped the azimuth — mirroring
    // asymmetric bokeh (hexagon/pentagon/star/custom). Use +yaw to match CPU.
    float cosYaw = cos(yaw);
    float sinYaw = sin(yaw);
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
// Settings: pset0=fov, pset1=aperture, pset2=subjectDistance, pset3=apertureShape.
Ray Camera_fisheye(float x, float y, Random random,
                   float pset0, float pset1, float pset2, float pset3,
                   __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fov = pset0;
    float aperture = pset1;
    float subjectDistance = pset2;
    int apertureShape = (int)pset3;

    float r = sqrt(x * x + y * y);
    // CPU FisheyeProjector: angleFromCenter = sqrt((degToRad(x*fov))^2 +
    // (degToRad(y*fov))^2) = r * degToRad(fov). degToRad = *PI/180, NOT /360
    // (the /360 here halved the field of view).
    float theta = r * fov * M_PI_F / 180.0f;

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
// Settings: pset0=fov, pset1=aperture, pset2=subjectDistance, pset3=apertureShape.
Ray Camera_stereographic(float x, float y, Random random,
                         float pset0, float pset1, float pset2, float pset3,
                         __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fov = pset0;
    float aperture = pset1;
    float subjectDistance = pset2;
    int apertureShape = (int)pset3;
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
// Settings: pset0=fov, pset1=aperture, pset2=subjectDistance, pset3=apertureShape.
Ray Camera_panoramic(float x, float y, Random random,
                     float pset0, float pset1, float pset2, float pset3,
                     __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fov = pset0;
    float aperture = pset1;
    float subjectDistance = pset2;
    int apertureShape = (int)pset3;
    // CPU PanoramicProjector maps x->degToRad(x*fov), y->degToRad(y*fov).
    // degToRad = *PI/180 (the /360 halved both yaw and pitch FOV).
    float fovRad = fov * M_PI_F / 180.0f;

    float phi = x * fovRad;
    float theta = y * fovRad;

    ray.origin = (float3)(0, 0, 0);
    float cosT = cos(theta);
    ray.direction = (float3)(sin(phi) * cosT, sin(theta), cos(phi) * cosT);

    Camera_applySphericalDoF(&ray, aperture, subjectDistance, apertureShape, random, apertureMask, apertureMaskWidth);
    return ray;
}

// Projector type 5: Panoramic slot
// Settings: pset0=fov, pset1=aperture, pset2=subjectDistance, pset3=apertureShape.
Ray Camera_panoramicSlot(float x, float y, Random random,
                         float pset0, float pset1, float pset2, float pset3,
                         __global const int* apertureMask, int apertureMaskWidth) {
    Ray ray;
    float fov = pset0;
    float aperture = pset1;
    float subjectDistance = pset2;
    int apertureShape = (int)pset3;
    // CPU PanoramicSlotProjector: horizontal is spherical (degToRad(x*fov),
    // i.e. *PI/180), but VERTICAL is pinhole-style — dy = clampedFovTan(fov)*y
    // where clampedFovTan(f) = 2*tan(degToRad(clamp(f,0,180)/2)). The old code
    // halved the yaw and used a linear vertical angle instead of the tangent.
    float phi = x * fov * M_PI_F / 180.0f;
    float fovTan = 2.0f * tan(clamp(fov, 0.0f, 180.0f) * M_PI_F / 360.0f);

    ray.origin = (float3)(0, 0, 0);
    ray.direction = (float3)(sin(phi), fovTan * y, cos(phi));
    ray.direction = normalize(ray.direction);

    Camera_applySphericalDoF(&ray, aperture, subjectDistance, apertureShape, random, apertureMask, apertureMaskWidth);
    return ray;
}

// Projector type 6: Omni-directional stereo (left/right)
// Settings: pset0=ipd (inter-pupillary distance), pset1=side (-1 left, +1 right).
// Exact replica of CPU ODSSinglePerspectiveProjector -> OmniDirectionalStereoProjector:
// the single-perspective projector feeds apply(x+0.5, y+0.5, side*IPD/2), and
// apply() uses theta = xr*PI - PI/2, phi = PI/2 - yr*PI. The previous code halved
// the vertical span (used y*PI/2 -> ±45° instead of ±90°) and flipped the origin
// z-sign (mirroring the stereo baseline).
Ray Camera_ODS(float x, float y, Random random, float pset0, float pset1) {
    Ray ray;
    float ipd = pset0;
    float side = pset1;

    float xr = x + 0.5f;
    float yr = y + 0.5f;
    float theta = xr * M_PI_F - M_PI_2_F;
    float phi = M_PI_2_F - yr * M_PI_F;
    float scale = ipd * 0.5f * side;

    ray.origin = (float3)(cos(theta) * scale, 0.0f, sin(theta) * scale);
    ray.direction = (float3)(sin(theta) * cos(phi), -sin(phi), cos(theta) * cos(phi));
    return ray;
}

// Projector type 7: ODS Stacked (both eyes in one image, top=left, bottom=right)
// Settings: pset0=ipd (inter-pupillary distance).
// Exact replica of CPU ODSVerticalStackedProjector: left eye uses
// apply(x*2+1, y*2+1, -IPD/2) for y<0, right eye apply(x*2+1, y*2, +IPD/2) for
// y>=0, where apply() uses theta = xr*PI - PI/2, phi = PI/2 - yr*PI. The x*2+1
// horizontal remap yields the full 360° sweep (the previous x*PI gave only 180°),
// and the vertical span is the full ±90°.
Ray Camera_ODSStacked(float x, float y, Random random, float pset0) {
    Ray ray;
    float ipd = pset0;

    float xr = x * 2.0f + 1.0f;
    float yr;
    float side;
    if (y < 0.0f) {
        // Top half = left eye
        yr = y * 2.0f + 1.0f;
        side = -1.0f;
    } else {
        // Bottom half = right eye
        yr = y * 2.0f;
        side = 1.0f;
    }

    float theta = xr * M_PI_F - M_PI_2_F;
    float phi = M_PI_2_F - yr * M_PI_F;
    float scale = ipd * 0.5f * side;

    ray.origin = (float3)(cos(theta) * scale, 0.0f, sin(theta) * scale);
    ray.direction = (float3)(sin(theta) * cos(phi), -sin(phi), cos(theta) * cos(phi));
    return ray;
}

#endif
