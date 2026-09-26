#ifndef CHUNKYCLPLUGIN_WAVEFRONT_H
#define CHUNKYCLPLUGIN_WAVEFRONT_H

#include "../opencl.h"

#define RAY_INDIRECT 0b001
#define RAY_PREVIEW  0b010
// Ray cast as a shadow / visibility test (sun NEE, emitter NEE). Used to
// gate light-block intersection: light blocks must be invisible to shadow
// rays so emitter-NEE shadow rays reach the actual emitter face instead
// of being blocked by the light block's own inset cube. Without this the
// light block contributes zero light to NEE-sampled hits.
#define RAY_SHADOW   0b100
// Emitters are enabled for this dispatch (dynamicConfig[2]). Set on camera
// rays by the render kernel and inherited by every bounce/copy. Used by the
// light-block intersection (block.h modelType 4), which must only be
// intersectable when emitters are on — mirroring CPU LightBlock.intersect's
// scene.getEmittersEnabled() gate. Never set in preview.
#define RAY_EMITTERS 0b1000

typedef struct {
    float3 origin;
    float3 direction;
    int material;
    int flags;
    float currentIor;   // IOR of medium ray is currently in
    float prevIor;      // IOR of previous medium
    bool inWater;       // whether ray is inside water
} Ray;

typedef struct {
    float distance;
    int material;
    int blockData;     // Octree block palette index (for same-material skip in traversal)

    float3 normal;
    float3 geomNormal; // Geometric (unperturbed) normal for diffuse correction
    float2 texCoord;
    int hitKind;       // HIT_* : how closestIntersect must build the MaterialSample
} IntersectionRecord;

// Intersection routines only record WHERE the closest surface is (distance,
// normal, uv, material pointer) plus how to shade it. The MaterialSample is
// built once, for the winning hit, at the end of closestIntersect. Building it
// inside every primitive test duplicated Material_sample (three texture reads)
// into every inlined trace, which is most of what made the render kernel huge.
#define HIT_MATERIAL          0  // palette material at texCoord
#define HIT_LIGHT_WHITE       1  // light block seen by a path-trace ray: flat white
#define HIT_WATER_EXIT_CENTER 2  // exitWater into a non-water cell: material at (0.5,0.5)
#define HIT_WATER_EXIT_UV     3  // exitWater through the surface triangles
#define HIT_WATER_EXIT_PLAIN  4  // exitWater without a visited water cell: default water
#define HIT_WATER_PLANE       5
#define HIT_CLOUD             6
#define HIT_KIND_MASK      0xFF
#define HIT_FLAG_WATER_BRANCH 0x100 // won through the water-octree branch

IntersectionRecord IntersectionRecord_new() {
    IntersectionRecord record;
    record.distance = HUGE_VALF;
    record.material = 0;
    record.blockData = 0;
    record.normal = (float3) (0, 1, 0);
    record.geomNormal = (float3) (0, 1, 0);
    record.texCoord = (float2)(0.0f, 0.0f);
    record.hitKind = HIT_MATERIAL;
    return record;
}

#endif
