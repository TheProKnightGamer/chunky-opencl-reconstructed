#ifndef CHUNKYCLPLUGIN_CONSTANTS_H
#define CHUNKYCLPLUGIN_CONSTANTS_H

#define EPS 0.000005f    // Ray epsilon and exit offset
#define OFFSET 0.0001f   // Do not use default Chunky Ray.OFFSET (1e-6) or rays will break.
#define DEFAULT_GAMMA 2.2f // Scene.DEFAULT_GAMMA

// Sub-surface scattering probability
#define F_SUBSURFACE 0.3f

// Fog constants
#define FOG_EXTINCTION 0.04f
#define FOG_LIMIT 30000.0f

// Default IOR values
#define AIR_IOR 1.000293f
#define WATER_IOR 1.333f

// Russian roulette start depth
#define RR_START_DEPTH 4

#endif
