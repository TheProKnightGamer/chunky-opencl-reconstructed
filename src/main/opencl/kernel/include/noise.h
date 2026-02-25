#ifndef CHUNKYCLPLUGIN_NOISE_H
#define CHUNKYCLPLUGIN_NOISE_H

#include "../opencl.h"

// Simplex noise for OpenCL - ported from Chunky's SimplexNoise.java
// Based on Stefan Gustavson's sdnoise1234 (public domain)

// Gradient vectors for 2D simplex noise
constant float2 grad2[] = {
    (float2)(1,1), (float2)(-1,1), (float2)(1,-1), (float2)(-1,-1),
    (float2)(1,0), (float2)(-1,0), (float2)(0,1), (float2)(0,-1)
};

// Gradient vectors for 3D simplex noise (12 cube edge midpoints + 4 repeats = 16)
constant float grad3lut[16][3] = {
    { 1, 0, 1}, { 0, 1, 1}, {-1, 0, 1}, { 0,-1, 1},
    { 1, 0,-1}, { 0, 1,-1}, {-1, 0,-1}, { 0,-1,-1},
    { 1,-1, 0}, { 1, 1, 0}, {-1, 1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 0, 1,-1}, { 0,-1,-1}
};

// Permutation table
constant int perm[] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
    247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
    57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
    60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
    65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
    200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
    52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
    207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
    119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
    218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
    81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
    184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
    222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    // Repeat
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
    247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
    57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
    60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
    65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
    200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
    52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
    207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,
    119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
    218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,
    81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,
    184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
    222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

// 2D simplex noise
float simplexNoise2(float xin, float yin) {
    // Skewing and unskewing factors for 2D
    const float F2 = 0.5f * (sqrt(3.0f) - 1.0f);
    const float G2 = (3.0f - sqrt(3.0f)) / 6.0f;

    float s = (xin + yin) * F2;
    int i = (int)floor(xin + s);
    int j = (int)floor(yin + s);
    float t = (float)(i + j) * G2;
    float X0 = (float)i - t;
    float Y0 = (float)j - t;
    float x0 = xin - X0;
    float y0 = yin - Y0;

    int i1, j1;
    if (x0 > y0) { i1 = 1; j1 = 0; }
    else { i1 = 0; j1 = 1; }

    float x1 = x0 - (float)i1 + G2;
    float y1 = y0 - (float)j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2;
    float y2 = y0 - 1.0f + 2.0f * G2;

    int ii = i & 255;
    int jj = j & 255;
    int gi0 = perm[ii + perm[jj]] & 7;
    int gi1 = perm[ii + i1 + perm[jj + j1]] & 7;
    int gi2 = perm[ii + 1 + perm[jj + 1]] & 7;

    float n0, n1, n2;

    float t0 = 0.5f - x0 * x0 - y0 * y0;
    if (t0 < 0) n0 = 0.0f;
    else { t0 *= t0; n0 = t0 * t0 * dot(grad2[gi0], (float2)(x0, y0)); }

    float t1 = 0.5f - x1 * x1 - y1 * y1;
    if (t1 < 0) n1 = 0.0f;
    else { t1 *= t1; n1 = t1 * t1 * dot(grad2[gi1], (float2)(x1, y1)); }

    float t2 = 0.5f - x2 * x2 - y2 * y2;
    if (t2 < 0) n2 = 0.0f;
    else { t2 *= t2; n2 = t2 * t2 * dot(grad2[gi2], (float2)(x2, y2)); }

    return 70.0f * (n0 + n1 + n2);
}

// 3D simplex noise with analytical derivatives.
// Returns noise value in [-1, 1]. Writes partial derivatives to *out_ddx, *out_ddy, *out_ddz.
// Ported from Chunky's SimplexNoise.java (Stefan Gustavson's sdnoise1234, public domain).
float simplexNoise3(float x, float y, float z, float* out_ddx, float* out_ddy, float* out_ddz) {
    const float F3 = 1.0f / 3.0f;
    const float G3 = 1.0f / 6.0f;

    float n0, n1, n2, n3;
    float gx0, gy0, gz0, gx1, gy1, gz1, gx2, gy2, gz2, gx3, gy3, gz3;

    // Skew input to simplex cell
    float s = (x + y + z) * F3;
    float xs = x + s;
    float ys = y + s;
    float zs = z + s;
    int i = (int)floor(xs);
    int j = (int)floor(ys);
    int k = (int)floor(zs);

    float t = (float)(i + j + k) * G3;
    float X0 = (float)i - t;
    float Y0 = (float)j - t;
    float Z0 = (float)k - t;
    float x0 = x - X0;
    float y0 = y - Y0;
    float z0 = z - Z0;

    // Determine simplex ordering
    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0)      { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else               { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0)       { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0)  { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else               { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    float x1 = x0 - i1 + G3;
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f * G3;
    float y2 = y0 - j2 + 2.0f * G3;
    float z2 = z0 - k2 + 2.0f * G3;
    float x3 = x0 - 1.0f + 3.0f * G3;
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    int ii = i & 255;
    int jj = j & 255;
    int kk = k & 255;

    // Corner 0
    float t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
    float t20, t40;
    if (t0 < 0) {
        t40 = t20 = t0 = n0 = gx0 = gy0 = gz0 = 0.0f;
    } else {
        int gi = perm[ii + perm[jj + perm[kk]]] & 15;
        gx0 = grad3lut[gi][0]; gy0 = grad3lut[gi][1]; gz0 = grad3lut[gi][2];
        t20 = t0 * t0;
        t40 = t20 * t20;
        n0 = t40 * (gx0*x0 + gy0*y0 + gz0*z0);
    }

    // Corner 1
    float t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
    float t21, t41;
    if (t1 < 0) {
        t41 = t21 = t1 = n1 = gx1 = gy1 = gz1 = 0.0f;
    } else {
        int gi = perm[ii+i1 + perm[jj+j1 + perm[kk+k1]]] & 15;
        gx1 = grad3lut[gi][0]; gy1 = grad3lut[gi][1]; gz1 = grad3lut[gi][2];
        t21 = t1 * t1;
        t41 = t21 * t21;
        n1 = t41 * (gx1*x1 + gy1*y1 + gz1*z1);
    }

    // Corner 2
    float t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
    float t22, t42;
    if (t2 < 0) {
        t42 = t22 = t2 = n2 = gx2 = gy2 = gz2 = 0.0f;
    } else {
        int gi = perm[ii+i2 + perm[jj+j2 + perm[kk+k2]]] & 15;
        gx2 = grad3lut[gi][0]; gy2 = grad3lut[gi][1]; gz2 = grad3lut[gi][2];
        t22 = t2 * t2;
        t42 = t22 * t22;
        n2 = t42 * (gx2*x2 + gy2*y2 + gz2*z2);
    }

    // Corner 3
    float t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
    float t23, t43;
    if (t3 < 0) {
        t43 = t23 = t3 = n3 = gx3 = gy3 = gz3 = 0.0f;
    } else {
        int gi = perm[ii+1 + perm[jj+1 + perm[kk+1]]] & 15;
        gx3 = grad3lut[gi][0]; gy3 = grad3lut[gi][1]; gz3 = grad3lut[gi][2];
        t23 = t3 * t3;
        t43 = t23 * t23;
        n3 = t43 * (gx3*x3 + gy3*y3 + gz3*z3);
    }

    // Noise value
    float value = 28.0f * (n0 + n1 + n2 + n3);

    // Analytical derivative
    float temp0 = t20 * t0 * (gx0*x0 + gy0*y0 + gz0*z0);
    float ddx = temp0 * x0;
    float ddy = temp0 * y0;
    float ddz = temp0 * z0;

    float temp1 = t21 * t1 * (gx1*x1 + gy1*y1 + gz1*z1);
    ddx += temp1 * x1;
    ddy += temp1 * y1;
    ddz += temp1 * z1;

    float temp2 = t22 * t2 * (gx2*x2 + gy2*y2 + gz2*z2);
    ddx += temp2 * x2;
    ddy += temp2 * y2;
    ddz += temp2 * z2;

    float temp3 = t23 * t3 * (gx3*x3 + gy3*y3 + gz3*z3);
    ddx += temp3 * x3;
    ddy += temp3 * y3;
    ddz += temp3 * z3;

    ddx *= -8.0f;
    ddy *= -8.0f;
    ddz *= -8.0f;

    ddx += t40*gx0 + t41*gx1 + t42*gx2 + t43*gx3;
    ddy += t40*gy0 + t41*gy1 + t42*gy2 + t43*gy3;
    ddz += t40*gz0 + t41*gz1 + t42*gz2 + t43*gz3;

    *out_ddx = 28.0f * ddx;
    *out_ddy = 28.0f * ddy;
    *out_ddz = 28.0f * ddz;

    return value;
}

#endif
