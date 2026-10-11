// FRAGMENT, not a header: included INSIDE kernel bodies (render in
// rayTracer.c and the wavefront kernels), so it has no include guard and is
// meant to be included more than once.
//
// End of a branch: leftover water fog, ground fog, color += branchColor. With
// PATH_WAVEFRONT the fog's random draw and shadow ray were done by wfConnect.

                // Leftover water fog. In practice this only fires when
                // effectiveDepth == 0 (i.e. rayDepth = 1): every other exit from
                // the depth loop has already consumed and zeroed
                // totalWaterDistance in the same iteration that accumulated it.
                // Deliberately AMBIENT-ONLY — no sun sample. Spending another
                // getDirectLightAttenuation (and its register budget, see §14) on
                // a degenerate one-bounce render is not worth it; the trade is
                // that rayDepth = 1 underwater renders darker than rayDepth >= 2.
                if (totalWaterDistance > 0.0f) {
                    float3 fogAtt = Water_extinction(totalWaterDistance, scene.waterVisibility);
                    branchColor *= fogAtt;
                    branchColor += throughput * Water_inscatter(fogAtt, scene.waterColor,
                        Water_scatterLight((float3)(0.0f), dot(bRay.direction, sun.sw), waterSunLight, waterSkyLight));
                }

                // Apply ground fog. Use scene entry point as fog origin so
                // empty space between camera and octree doesn't inflate fog.
                //
                // Sun inscatter for fog runs UNCONDITIONALLY in chunky CPU
                // (PathTracer.java ~line 192-200) — the strategy gate only
                // controls direct NEE, not the fog-light approximation. We
                // previously gated this on `sunSamplingStrategy != OFF`,
                // which made fog look brighter on GPU (no sun-occlusion
                // attenuation) than on CPU when the user set strategy to
                // OFF. Match CPU: always compute the attenuation.
                // Atmospheric fog distance = first camera->surface segment. CPU
                // applies it whether that segment is air OR water (prevMat==Air
                // || prevMat.isWater()), so include firstWaterDist for the
                // camera-underwater case (exactly one of the two is nonzero).
                float groundFogDist = firstAirDist + firstWaterDist;
                if (fogConfig.mode != FOG_MODE_NONE && groundFogDist > 0.0f) {
                    float3 fogOrigin = cameraOrigin + cameraDirection * emptySpaceDist;
#if PATH_WAVEFRONT
                    // Drawn (same random number) and traced by wfConnect before this.
                    float fogOffset = wfFogOffset;
#else
                    float fogOffset = Fog_sampleScatterOffset(fogConfig, groundFogDist, fogOrigin, cameraDirection, random);
#endif
                    float3 sunDir = sun.sw;
                    // CPU Fog.addGroundFog scales inscatter by the sun-visibility
                    // alpha (scatterLight.w ≤ 1), NOT by sun intensity. That alpha
                    // is already folded into fogSunAtt below, so the scalar is 1.0.
                    // (Previously sun.intensity, making god rays 1.25x–50x too bright.)
                    float sunInt = 1.0f;
                    float3 fogSamplePos = fogOrigin + cameraDirection * fogOffset;
#if PATH_WAVEFRONT
                    float3 fogSunAtt = wfFogAtt;
#else
                    float3 fogSunAtt = getDirectLightAttenuation(scene, textureAtlas,
                        fogSamplePos, sunDir, FOG_LIMIT, strictDirectLight);
#endif
                    Fog_addGroundFog(fogConfig, &branchColor, fogOrigin, cameraDirection,
                                     groundFogDist, fogSunAtt, sunInt, fogOffset);
                }

                color += branchColor;
