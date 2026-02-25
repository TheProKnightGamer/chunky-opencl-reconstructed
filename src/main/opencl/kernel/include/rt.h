#ifndef CHUNKYCLPLUGIN_WAVEFRONT_H
#define CHUNKYCLPLUGIN_WAVEFRONT_H

#include "../opencl.h"

#define RAY_INDIRECT 0b01
#define RAY_PREVIEW  0b10

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
} IntersectionRecord;

IntersectionRecord IntersectionRecord_new() {
    IntersectionRecord record;
    record.distance = HUGE_VALF;
    record.material = 0;
    record.blockData = 0;
    record.normal = (float3) (0, 1, 0);
    record.geomNormal = (float3) (0, 1, 0);
    record.texCoord = (float2)(0.0f, 0.0f);
    return record;
}

#endif
