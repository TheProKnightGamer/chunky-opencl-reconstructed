#ifndef CHUNKYCL_PLUGIN_UTILS_H
#define CHUNKYCL_PLUGIN_UTILS_H

#include "../opencl.h"

float4 colorFromArgb(unsigned int argb) {
    float4 color;
    color.w = (argb >> 24) & 0xFF;
    color.x = (argb >> 16) & 0xFF;
    color.y = (argb >> 8) & 0xFF;
    color.z = argb & 0xFF;
    // 255, not 256: the host packs with ColorUtil.getRGB (x * 255 + .5), and
    // Chunky unpacks with / 255 too. Dividing by 256 made every packed colour 0.4%
    // dark and capped alpha at 255/256, so nothing untextured or tinted was ever
    // fully opaque.
    color /= 255.0f;
    return color;
}

int3 intFloorFloat3(float3 value) {
    value = floor(value);
    return (int3) (value.x, value.y, value.z);
}

float3 int3toFloat3(int3 value) {
    return (float3) (value.x, value.y, value.z);
}

#endif
