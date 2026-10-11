#include "../opencl.h"
#include "pathtrace.h"

// The megakernel: one work-item runs whole paths. The path code lives in the
// frag_* fragments, which the wavefront kernels (wavefront.c) share.
#define PATH_WAVEFRONT 0
__kernel void render(
#include "frag_kernel_params.h"
) {
#include "frag_kernel_setup.h"
    // Grid-stride loop: each work-item processes multiple pixels. When one
    // pixel's path terminates fast (e.g. sky hit) the same work-item picks
    // up the next pixel rather than the warp stalling on the longest-path
    // thread. Empty inner-state per pixel — sumColor/sumAlpha reset.
    for (int gid = pixelStart + wid; gid < pixelCount; gid += pixelStride) {
    float3 sumColor = (float3)(0.0f, 0.0f, 0.0f);
    float sumAlpha = 0.0f;

    for (int it = 0; it < iterations; ++it) {
#include "frag_path_camera.h"
        if (!firstHit) {
#include "frag_path_sky.h"
        } else {
#include "frag_path_firsthit.h"
            // Branch count loop: reuse same first intersection with different random decisions
            int effectiveBranches = branchCount;
            for (int branch = 0; branch < effectiveBranches; branch++) {
                // Each branch gets its own continuation from the first hit
                Ray bRay = ray;
                float3 throughput = (float3)(1.0f);
                float3 branchColor = (float3)(0.0f);
                float totalWaterDistance = firstWaterDist;
                bool hitAnything = true;
                bool lastWasSpecular = false;  // tracks whether the last bounce was specular (for sky fog)
                IntersectionRecord record = firstRecord;
                MaterialSample sample = firstSample;
                Material material = firstMaterial;

                // CPU PathTracer.java:126 checks `ray.depth + 1 >= rayDepth`
                // BEFORE the bounce, so a rayDepth setting of N produces only
                // N-1 actual NEE-counted bounces (the would-be N-th call enters
                // pathTrace, hits the depth check, and returns black). Use the
                // same effective bound here to match CPU sample contributions
                // bounce-for-bounce.
                int effectiveDepth = cachedRayDepth - 1;
                if (effectiveDepth < 0) effectiveDepth = 0;
                for (int depth = 0; depth < effectiveDepth; depth++) {
#include "frag_path_step.h"
                } // end depth loop
#include "frag_path_end.h"
            } // end branch loop

        }

        sumColor += color;
        sumAlpha += alpha;
    }

    // Each iteration produced branchCount samples, so total samples = iterations * branchCount
    float3 avgColor = sumColor / (float)(iterations * branchCount);
    float avgAlpha = sumAlpha / (float)iterations;

    // Store color + alpha (4 floats per pixel).
    // Alpha = 1 for opaque hits, 0 for transparent sky hits at depth 0.
    vstore4((float4)(avgColor, avgAlpha), gid, res);
    } // end grid-stride loop
}

__kernel void preview(
    __constant const int* projectorType,
    __global const float* cameraSettings,

    __constant const int* octreeDepth,
    __global const int* octreeData,

    __constant const int* waterOctreeDepth,
    __global const int* waterOctreeData,

    __global const int* bPalette,
    __global const int* quadModels,
    __global const int* aabbModels,

    __global const int* worldBvhData,
    __global const int* actorBvhData,
    __global const int* bvhTrigs,

    image2d_array_t textureAtlas,
    __global const int* matPalette,

    image2d_t skyTexture,
    __constant const float* skyIntensity,
    __constant const int* sunData,

    __constant const int* canvasConfig,
    __constant const float* waterConfig,
    __global const int* chunkBitmap,
    int chunkBitmapSize,
    __global const int* biomeData,
    int biomeDataSize,
    int biomeYLevels,
    __constant const float* renderConfig,
    __global const int* cloudData,
    __global int* res
) {
    int gid = get_global_id(0);

    int px = gid % canvasConfig[0] + canvasConfig[4];
    int py = gid / canvasConfig[0] + canvasConfig[5];

    // Crosshairs
    if ((px == canvasConfig[2] / 2 && (py >= canvasConfig[3] / 2 - 5 && py <= canvasConfig[3] / 2 + 5)) ||
        (py == canvasConfig[3] / 2 && (px >= canvasConfig[2] / 2 - 5 && px <= canvasConfig[2] / 2 + 5))) {
        res[gid] = 0xFFFFFFFF;
        return;
    }

    SceneConfig scene;
    scene.materialPalette = MaterialPalette_new(matPalette, 0, NULL);
    scene.octree = Octree_create(octreeData, *octreeDepth);
    scene.waterOctree = Octree_create(waterOctreeData, *waterOctreeDepth);
    scene.worldBvh = Bvh_new(worldBvhData, bvhTrigs, &scene.materialPalette);
    scene.actorBvh = Bvh_new(actorBvhData, bvhTrigs, &scene.materialPalette);
    scene.blockPalette = BlockPalette_new(bPalette, quadModels, aabbModels, &scene.materialPalette);
    // Cap the octree DDA at the worst-case number of leaf-cell crossings for a
    // ray traversing the whole octree (the 3D-DDA diagonal bound, 3 * 2^depth),
    // not a flat 256. The old 256 cap made distant geometry along shallow angles
    // in large/wide scenes terminate early and dissolve into sky (CPU is
    // uncapped). Clamped to 8192 so a pathological miss-ray on a huge octree
    // can't stall a dispatch. Normal rays exit early (on hit or out-of-bounds),
    // so this only affects long miss-rays.
    scene.drawDepth = min(3 << max(scene.octree.depth, scene.waterOctree.depth), 8192);

    scene.waterPlaneEnabled = (waterConfig[0] > 0.5f);
    scene.waterPlaneHeight = waterConfig[1];
    scene.waterPlaneChunkClip = (waterConfig[2] > 0.5f);
    scene.octreeSize = waterConfig[3];
    scene.waterColor = (float3)(waterConfig[7], waterConfig[8], waterConfig[9]);
    scene.useCustomWaterColor = (waterConfig[10] > 0.5f);
    scene.waterIor = waterConfig[11];
    scene.waterOpacity = waterConfig[16];
    scene.hasWaterBlocks = (waterConfig[17] > 0.5f);
    scene.waterBounds = AABB_new(waterConfig[18], waterConfig[21], waterConfig[19],
                                 waterConfig[22], waterConfig[20], waterConfig[23]);
    scene.chunkBitmap = chunkBitmap;
    scene.chunkBitmapSize = chunkBitmapSize;
    scene.biomeColorsEnabled = (biomeDataSize > 0);
    scene.biomeData = biomeData;
    scene.biomeDataSize = biomeDataSize;
    scene.biomeYLevels = max(1, biomeYLevels);
    // Clouds — the CPU preview renders them too. Cloud config lives in
    // renderConfig[3,4,5,11,12] (same indices the main render kernel uses).
    scene.cloudsEnabled = (renderConfig[3] > 0.5f);
    scene.cloudHeight = renderConfig[4];
    scene.cloudSize = renderConfig[5];
    scene.cloudOffsetX = renderConfig[11];
    scene.cloudOffsetZ = renderConfig[12];
    scene.cloudData = cloudData;

    Sun sun = Sun_new(sunData);

    unsigned int randomState = 0;
    Random random = &randomState;
    Random_nextState(random);

    CameraCache camCache = CameraCache_load(projectorType, cameraSettings, canvasConfig);
    Ray ray = ray_to_camera(camCache, cameraSettings, NULL, 0, gid, random);

    IntersectionRecord record = IntersectionRecord_new();
    MaterialSample sample;
    Material material;

    ray.material = 0;
    ray.flags = RAY_PREVIEW;
    ray.currentIor = AIR_IOR;
    ray.prevIor = AIR_IOR;
    ray.inWater = false;

    // Check if camera is underwater (matching CPU PreviewRayTracer initial medium setup)
    if (isInWater(scene, ray.origin)) {
        ray.inWater = true;
        ray.currentIor = scene.waterIor;
    }

    float3 color = (float3)(0.0f);
    bool hitSolid = false;

    // Front-to-back alpha accumulators for semi-transparent layers (water, stained glass).
    // C_out += (1 - alphaAccum) * layerAlpha * layerColor at each layer.
    float3 tintAccum = (float3)(0.0f);
    float alphaAccum = 0.0f;

    // Transparency loop with alpha blending.  Fully transparent (alpha==0) hits are
    // skipped, semi-transparent hits are composited as tint layers, opaque hits stop.
    for (int skipIter = 0; skipIter < 8; skipIter++) {
        record = IntersectionRecord_new();
        if (!previewIntersect(scene, textureAtlas, ray, &record, &sample, &material)) {
            break;
        }

        if (sample.color.w <= EPS) {
            // Fully transparent: skip past
            ray.origin = ray.origin + ray.direction * (record.distance + OFFSET);
            ray.material = record.blockData;
            // Toggle water state when passing through water surface
            if (sample.isWater) ray.inWater = !ray.inWater;
            continue;
        }

        // Apply biome tint and flat shading
        float3 previewHitPos = ray.origin + ray.direction * record.distance;
        applyBiomeTint(scene, &sample, previewHitPos);
        // CPU Sun.flatShading multiplies albedo by previewEmittance * shading,
        // where previewEmittance = pow(DEFAULT_INTENSITY=1.25, DEFAULT_GAMMA) is a
        // CONSTANT (NOT the live sun intensity). Omitting it made the GPU preview
        // ~1.6x darker (linear) than the CPU preview.
        float shading = fmax(0.3f, dot(record.normal, sun.sw));
        float3 hitColor = sample.color.xyz * shading * pow(1.25f, DEFAULT_GAMMA);

        if (sample.color.w < 1.0f - EPS) {
            // Semi-transparent: accumulate as tint layer (front-to-back compositing)
            float layerAlpha = sample.color.w;
            float remaining = 1.0f - alphaAccum;
            tintAccum += hitColor * layerAlpha * remaining;
            alphaAccum += layerAlpha * remaining;

            if (alphaAccum >= 0.99f) {
                color = tintAccum;
                hitSolid = true;
                break;
            }

            ray.origin = ray.origin + ray.direction * (record.distance + OFFSET);
            ray.material = record.blockData;
            // Toggle water state when passing through water surface
            if (sample.isWater) ray.inWater = !ray.inWater;
            continue;
        }

        // Opaque hit
        color = hitColor;
        hitSolid = true;
        break;
    }

    // Composite accumulated semi-transparent layers over opaque background
    if (alphaAccum > 0.0f && hitSolid) {
        color = tintAccum + color * (1.0f - alphaAccum);
    }

    if (!hitSolid) {
        // No opaque hit — resolve background (floor grid or sky)
        float3 bgColor = (float3)(0.0f);
        bool gridHit = false;
        if (ray.direction.y < 0) {
            float gridY = 0.0f;
            float gt = (gridY - ray.origin.y) / ray.direction.y;
            if (gt > OFFSET) {
                float3 hitP = ray.origin + ray.direction * gt;
                bool isSubmerged = scene.waterPlaneEnabled;
                int octSize = 1 << (*octreeDepth);
                bool insideOctree = (hitP.x >= 0 && hitP.x <= octSize &&
                                     hitP.z >= 0 && hitP.z <= octSize);
                float xm = fmod(fmod(hitP.x, 16.0f) + 24.0f, 16.0f);
                float zm = fmod(fmod(hitP.z, 16.0f) + 24.0f, 16.0f);
                float linePos = 7.75f;
                float lineEnd = 8.25f;
                bool isLine = (xm >= linePos && xm <= lineEnd) || (zm >= linePos && zm <= lineEnd);
                if (isLine) {
                    bgColor = isSubmerged ? (float3)(0.05f, 0.05f, 0.25f) : (float3)(0.25f, 0.25f, 0.25f);
                } else {
                    bgColor = isSubmerged ? (float3)(0.6f, 0.6f, 0.8f) : (float3)(0.8f, 0.8f, 0.8f);
                }
                if (insideOctree) {
                    bgColor *= 0.75f;
                }
                float gridShading = fmax(0.3f, sun.sw.y); // grid normal is (0,1,0)
                bgColor *= gridShading * pow(1.25f, DEFAULT_GAMMA); // previewEmittance (see surface shading)
                gridHit = true;
            }
        }
        if (!gridHit) {
            intersectSky(skyTexture, *skyIntensity, sun, textureAtlas, ray, &sample, true);
            bgColor = sample.color.xyz;
        }

        // Blend any accumulated semi-transparent layers over the background
        if (alphaAccum > 0.0f) {
            color = tintAccum + bgColor * (1.0f - alphaAccum);
        } else {
            color = bgColor;
        }
    }

    color = sqrt(color);
    int3 rgb = intFloorFloat3(clamp(color * 255.0f, 0.0f, 255.0f));
    res[gid] = 0xFF000000 | (rgb.x << 16) | (rgb.y << 8) | rgb.z;
}

// ---- 2D Map: nearest-neighbour upscale kernel ----
// One work-item per destination pixel.
// Replicates the CPU MapBuffer.drawBuffered() scaling logic on the GPU.
__kernel void mapScale(
    __global const int* src,   // source pixel buffer (chunk-scale)
    __global int* dst,         // destination pixel buffer (screen-scale)
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    float scale,               // view.scale / view.chunkScale
    int srcOffsetX,
    int srcOffsetZ
) {
    int gid = get_global_id(0);
    if (gid >= dstWidth * dstHeight) return;

    int dstX = gid % dstWidth;
    int dstY = gid / dstWidth;

    // Map destination pixel back to source coordinates (nearest-neighbour)
    int srcX = srcOffsetX + (int)(dstX / scale) + 1;
    int srcY = srcOffsetZ + (int)(dstY / scale);

    // Bounds check
    int srcIdx = srcY * srcWidth + srcX;
    if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight
            || srcIdx < 0 || srcIdx >= srcWidth * srcHeight) {
        dst[gid] = 0;
        return;
    }

    dst[gid] = src[srcIdx];
}
