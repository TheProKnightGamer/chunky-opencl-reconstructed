#ifndef CHUNKYCLPLUGIN_CONSTANTS_H
#define CHUNKYCLPLUGIN_CONSTANTS_H

#define EPS 0.000005f    // Ray epsilon and exit offset
// 1e-4 (vs CPU 1e-6) is required because the kernel uses float32 for ray
// origin/direction; CPU uses double. At distance 100 from the world origin
// float32's relative precision (~7 digits) puts the absolute step error
// near 1e-5, so a smaller offset would leave rays inside the surface they
// just exited and re-hit it. Tradeoff: thin geometry (~1e-4 m) cannot be
// resolved correctly; matches a known limit of single-precision tracers.
#define OFFSET 0.0001f
#define DEFAULT_GAMMA 2.2f // Scene.DEFAULT_GAMMA

// Sub-surface scattering probability
#define F_SUBSURFACE 0.3f

// Fog constants
#define FOG_EXTINCTION 0.04f
#define FOG_LIMIT 30000.0f

// Default IOR values
#define AIR_IOR 1.000293f
#define WATER_IOR 1.333f

#endif
