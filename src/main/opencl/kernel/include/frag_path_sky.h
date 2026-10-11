// FRAGMENT, not a header: included INSIDE kernel bodies (render in
// rayTracer.c and the wavefront kernels), so it has no include guard and is
// meant to be included more than once.
//
// The camera ray missed everything: sky (and sky fog) colour into color.
            // Sky hit at depth 0
            if (scene.transparentSky) {
                // CPU: a transparent-sky depth-0 miss does NOTHING — ray.color
                // stays (0,0,0); no sky and no fog are added. The host blend reads
                // only RGB (alpha is tracked separately), so RGB must be black.
                alpha = 0.0f;
                color = (float3)(0.0f);
            } else {
            MaterialSample skySample;
            intersectSky(skyTexture, cachedSkyIntensity, sun, textureAtlas, ray, &skySample, sunDiffuseSun);
            color = skySample.emittance * skySample.color.xyz;

            // Camera is submerged and the primary ray found nothing — the
            // unloaded-chunk horizon. CPU sets solid black here
            // (PathTracer.java:69); return the infinitely-deep-water colour
            // instead, matching the depth>0 escape above so the horizon and the
            // bounced light agree instead of meeting at a hard black edge.
            if (ray.inWater) {
                color = Water_inscatter((float3)(0.0f), scene.waterColor,
                    Water_scatterLight((float3)(0.0f), dot(ray.direction, sun.sw), waterSunLight, waterSkyLight));
            }

            if (fogConfig.mode != FOG_MODE_NONE) {
                // CPU: uniform sky fog passes null scatterLight (no sun inscatter).
                // CPU: layered sky fog samples a scatter offset and traces a sun shadow ray.
                float3 skyFogSunAtt = (float3)(0.0f);
                float skyFogSunInt = 0.0f;
                if (fogConfig.mode == FOG_MODE_LAYERED) {
                    float skyScatterOff = Fog_sampleSkyScatterOffset(fogConfig, ray.origin, ray.direction, scene.yMin, scene.yMax, random);
                    float3 skyScatterPos = ray.origin + ray.direction * skyScatterOff;
                    skyFogSunAtt = getDirectLightAttenuation(scene, textureAtlas,
                        skyScatterPos, sun.sw, FOG_LIMIT, strictDirectLight);
                    // CPU fog inscatter is scaled by the sun-visibility alpha
                    // (scatterLight.w), never by sun intensity. That alpha is
                    // already folded into skyFogSunAtt, so the scale is 1.0.
                    skyFogSunInt = 1.0f;
                }
                Fog_addSkyFog(fogConfig, &color, ray.origin, ray.direction,
                              skyFogSunAtt, skyFogSunInt);
            }
            } // end else (!transparentSky)

            // Sky path doesn't enter the branch loop, but avgColor divides by
            // iterations*branchCount.  Scale up so the average is correct.
            color *= (float)branchCount;
