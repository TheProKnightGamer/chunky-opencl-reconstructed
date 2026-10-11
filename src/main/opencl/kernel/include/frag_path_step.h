// FRAGMENT, not a header: included INSIDE kernel bodies (render in
// rayTracer.c and the wavefront kernels), so it has no include guard and is
// meant to be included more than once.
//
// One bounce of one branch (the body of the depth loop). With PATH_WAVEFRONT
// the ray was traced by wfExtend (wfHit, record), and the shadow rays plus the
// colour terms that precede them are handed to wfConnect (wf* locals).
                    // For depth > 0, find next intersection
                    if (depth > 0) {
#if PATH_WAVEFRONT
                        // wfExtend already traced this ray; wfHit says whether it hit.
                        if (!wfHit) {
#else
                        record = IntersectionRecord_new();
                        if (!closestIntersect(scene, textureAtlas, bRay, &record, &sample, &material)) {
#endif
                            // Sky hit at depth > 0

                            // Escaped while STILL SUBMERGED: no water surface was
                            // found. That happens over UNLOADED chunks — the water
                            // octree has no blocks there and a level or downward ray
                            // never reaches the water plane. CPU hard-codes this to
                            // solid black (PathTracer.java:69), which is the black
                            // horizon. An infinitely deep water column is not black;
                            // it is the transmittance-0 limit of the fog above, i.e.
                            // fully saturated in-scattered water colour.
                            if (bRay.inWater) {
                                branchColor += throughput * Water_inscatter((float3)(0.0f),
                                    scene.waterColor,
                                    Water_scatterLight((float3)(0.0f), dot(bRay.direction, sun.sw), waterSunLight, waterSkyLight));
                                throughput = (float3)(0.0f);
                                totalWaterDistance = 0.0f;
                                break;
                            }

                            intersectSky(skyTexture, cachedSkyIntensity, sun, textureAtlas, bRay, &sample, sunDiffuseSun);
                            throughput *= sample.color.xyz;
                            branchColor += sample.emittance * throughput;

                            // CPU: only apply sky fog on specular sky hits (not diffuse).
                            // Diffuse sky hits skip sky fog entirely.
                            if (fogConfig.mode != FOG_MODE_NONE && lastWasSpecular) {
                                float3 skyFogSunAtt2 = (float3)(0.0f);
                                float skyFogSunInt2 = 0.0f;
#if !PATH_WAVEFRONT // layered fog renders with the megakernel
                                if (fogConfig.mode == FOG_MODE_LAYERED) {
                                    float skyScatterOff2 = Fog_sampleSkyScatterOffset(fogConfig, bRay.origin, bRay.direction, scene.yMin, scene.yMax, random);
                                    float3 skyScatterPos2 = bRay.origin + bRay.direction * skyScatterOff2;
                                    skyFogSunAtt2 = getDirectLightAttenuation(scene, textureAtlas,
                                        skyScatterPos2, sun.sw, FOG_LIMIT, strictDirectLight);
                                    // Inscatter scale is 1.0 (the sun-visibility
                                    // alpha is already in skyFogSunAtt2), not sun
                                    // intensity — matching CPU Fog.addLayeredFog.
                                    skyFogSunInt2 = 1.0f;
                                }
#endif
                                Fog_addSkyFog(fogConfig, &branchColor, bRay.origin, bRay.direction,
                                              skyFogSunAtt2, skyFogSunInt2);
                            }
                            break;
                        }

                        // Track water distance (for Beer's-law water fog). Air/
                        // atmospheric fog uses only the first camera->surface
                        // segment (firstAirDist), matching CPU which fogs per
                        // frame over ray.distance — NOT an accumulated multi-bounce
                        // total, which over-darkened diffuse-GI scenes.
                        if (bRay.inWater) {
                            totalWaterDistance += record.distance;
                        }
                    }

                    // Compute hit position once per bounce (reused for biome tint, emitter, sun)
                    float3 hitPos = bRay.origin + bRay.direction * record.distance;

                    // Apply biome tinting (depth > 0 only; depth 0 was already tinted)
                    if (depth > 0) {
                        applyBiomeTint(scene, &sample, hitPos);
                    }

                    // Water fog: per-channel absorption PLUS in-scattering.
                    // The absorption half is Beer's law as before but tinted, so
                    // depth reads blue-green. The in-scatter half is the new part
                    // and the reason underwater no longer decays to black: light
                    // the segment removed is partly returned as a water-coloured
                    // haze. Computed inline from values already in SceneConfig —
                    // memory §14 reverted an earlier attempt at this because a
                    // persistent float3 in the depth loop blew the register budget
                    // and crashed with CL_OUT_OF_RESOURCES. Do NOT hoist these.
                    if (totalWaterDistance > 0.0f) {
                        float3 fogAtt = Water_extinction(totalWaterDistance, scene.waterVisibility);
                        // Sample the sunlight actually reaching a random point
                        // inside this water segment, the same way layered ground
                        // fog samples its scatter point. This is what makes depth
                        // darken by itself and puts shafts behind geometry.
                        float3 wScatterPos = bRay.origin + bRay.direction * (record.distance * Random_nextFloat(random));
#if PATH_WAVEFRONT
                        // Traced and added by wfConnect, before the terms below.
                        wfWaterPending = true;
                        wfWaterPos = wScatterPos;
                        wfWaterThroughput = throughput;
                        wfWaterAtt = fogAtt;
                        wfWaterCos = dot(bRay.direction, sun.sw);
#else
                        float3 wSunAtt = getDirectLightAttenuation(scene, textureAtlas,
                            wScatterPos, sun.sw, FOG_LIMIT, strictDirectLight);
                        branchColor += throughput * Water_inscatter(fogAtt, scene.waterColor,
                            Water_scatterLight(wSunAtt, dot(bRay.direction, sun.sw), waterSunLight, waterSkyLight));
#endif
                        throughput *= fogAtt;
                        totalWaterDistance = 0.0f;
                    }

                    MaterialPdfSample pdfSample = Material_samplePdf(material, record, sample, bRay, random,
                        fancierTranslucency, transmissivityCap);
                    lastWasSpecular = pdfSample.specular;
                    float3 prevThroughput = throughput;
                    throughput *= pdfSample.spectrum;

                    // Importance sampling: steer diffuse bounces toward the sun
                    if (sunIsImportance
                        && !pdfSample.specular && !pdfSample.transmitted) {
                        DiffuseISResult isResult = _Material_diffuseReflectionIS(
                            record, sun.sw,
                            sun.importanceSampleChance, sun.importanceSampleRadius,
                            sun.radius, random);
                        pdfSample.direction = isResult.direction;
                        // Geometric normal correction for importance-sampled direction
                        if (sign(dot(record.geomNormal, pdfSample.direction)) == sign(dot(record.geomNormal, bRay.direction))) {
                            float factor = copysign(1.0f, dot(record.geomNormal, bRay.direction)) * (-EPS) - dot(pdfSample.direction, record.geomNormal);
                            pdfSample.direction += factor * record.geomNormal;
                            pdfSample.direction = normalize(pdfSample.direction);
                        }
                        throughput *= isResult.throughputScale;
                    }

                    // Emitter self-emission. CPU adds emittance ONLY inside
                    // doDiffuseReflection, so gate on a genuine diffuse bounce
                    // (!specular && !transmitted), matching the sun-NEE gate.
                    bool didSelfEmit = false;
                    if (emittersEnabled && sample.emittance > EPS
                        && !pdfSample.specular && !pdfSample.transmitted
                        && (!preventNormalEmitterWithSampling || samplingStrategy == 0 || depth == 0)) {
                        float3 emColor = (float3)(sample.color.x * sample.color.x,
                                                  sample.color.y * sample.color.y,
                                                  sample.color.z * sample.color.z);
#if PATH_WAVEFRONT
                        // Added by wfConnect, after the water in-scatter above.
                        wfEmitPending = true;
                        wfEmit = emColor * sample.emittance * cachedEmitterIntensity * prevThroughput;
#else
                        branchColor += emColor * sample.emittance * cachedEmitterIntensity * prevThroughput;
#endif
                        didSelfEmit = true;
                    }

#if PATH_WAVEFRONT // emitter-grid sampling renders with the megakernel
                    (void) didSelfEmit;
#else
                    // Emitter-grid NEE. CPU does this only on genuine diffuse
                    // bounces, and ONLY when self-emission did not fire this bounce
                    // (if/else-if in doDiffuseReflection). Gate: !specular &&
                    // !transmitted && !didSelfEmit.
                    if (emittersEnabled && samplingStrategy != 0 && gc_sizeX > 0
                        && !pdfSample.specular && !pdfSample.transmitted && !didSelfEmit) {
                        int cellSize = gc_cellSize;
                        int offsetX = gc_offsetX;
                        int sizeX = gc_sizeX;
                        int offsetY = gc_offsetY;
                        int sizeY = gc_sizeY;
                        int offsetZ = gc_offsetZ;
                        int sizeZ = gc_sizeZ;

                        // Select the emitter grid cell from the SURFACE hit point,
                        // not bRay.origin (which is the camera at depth 0 / the
                        // previous bounce otherwise). CPU uses ray.o = the advanced
                        // hit point. Using the ray start sampled emitters from the
                        // wrong cell (usually empty), under-lighting emitter scenes.
                        int gx = (int)floor(hitPos.x / (float)cellSize);
                        int gy = (int)floor(hitPos.y / (float)cellSize);
                        int gz = (int)floor(hitPos.z / (float)cellSize);

                        if (gx >= offsetX && gx < offsetX + sizeX && gy >= offsetY && gy < offsetY + sizeY && gz >= offsetZ && gz < offsetZ + sizeZ) {
                            int idx = (((gy - offsetY) * sizeX) + (gx - offsetX)) * sizeZ + (gz - offsetZ);
                            int eStart = constructedGrid[2*idx];
                            int eCount = constructedGrid[2*idx + 1];
                            if (eCount > 0) {
                                float3 hitPoint = hitPos + record.normal * OFFSET;
                                if (samplingStrategy == 1) {
                                    // ONE: pick one random emitter, sample a random face point
                                    unsigned int r = Random_nextState(random);
                                    int ri = (int)(r % eCount);
                                    int emitterIndex = positionIndexes[eStart + ri];
                                    int ex = emitterPositions[emitterIndex*4 + 0];
                                    int ey = emitterPositions[emitterIndex*4 + 1];
                                    int ez = emitterPositions[emitterIndex*4 + 2];
                                    float avgFaceArea = as_float(emitterPositions[emitterIndex*4 + 3]);
                                    unsigned int faceR = Random_nextState(random);
                                    int face = (int)(faceR % 6);
                                    float ru = Random_nextFloat(random);
                                    float rv = Random_nextFloat(random);
                                    float3 epos;
                                    switch (face) {
                                        case 0: epos = (float3)(ex, ey + ru, ez + rv); break;
                                        case 1: epos = (float3)(ex + 1.0f, ey + ru, ez + rv); break;
                                        case 2: epos = (float3)(ex + ru, ey, ez + rv); break;
                                        case 3: epos = (float3)(ex + ru, ey + 1.0f, ez + rv); break;
                                        case 4: epos = (float3)(ex + ru, ey + rv, ez); break;
                                        default: epos = (float3)(ex + ru, ey + rv, ez + 1.0f); break;
                                    }
                                    float3 toEmitter = epos - hitPoint;
                                    float dist = length(toEmitter) + 1e-6f;
                                    float3 dirToEmitter = toEmitter / dist;
                                    if (dot(record.normal, dirToEmitter) > 0.0f) {
                                        Ray shadow = bRay;
                                        shadow.origin = hitPoint;
                                        shadow.direction = dirToEmitter;
                                        shadow.flags |= RAY_SHADOW;
                                        IntersectionRecord srec = IntersectionRecord_new();
                                        srec.distance = dist;
                                        Material sSample;
                                        MaterialSample sMatSample;
                                        bool shadowClear = !closestIntersect(scene, textureAtlas, shadow, &srec, &sMatSample, &sSample);
                                        if (shadowClear || srec.distance >= dist - 1e-4f) {
                                            if (shadowClear) {
                                                Emitter_sampleCell(scene.blockPalette, scene.materialPalette, textureAtlas, &scene.octree,
                                                                   ex, ey, ez, (float2)(ru, rv), &sMatSample);
                                                srec.normal = (face < 2) ? (float3)(face == 0 ? -1.0f : 1.0f, 0, 0)
                                                            : (face < 4) ? (float3)(0, face == 2 ? -1.0f : 1.0f, 0)
                                                                         : (float3)(0, 0, face == 4 ? -1.0f : 1.0f);
                                            }
                                            float cosEmitter = fabs(dot(dirToEmitter, srec.normal));
                                            float att = cosEmitter / fmax(dist * dist, 1.0f); // CPU emitter NEE has no receiving-cosine weight (only the visibility gate)
                                            att *= avgFaceArea;
                                            // CPU emitter NEE uses the emitter's LINEAR color (sampleEmitterFace
                                            // scaleAdds the raw color); the quadratic color^2 mapping is for
                                            // direct self-emission ONLY. Squaring here darkened colored emitters.
                                            float3 emitterCol = sMatSample.color.xyz * sMatSample.emittance;
                                            branchColor += throughput * cachedEmitterIntensity * emitterCol * att * (float)M_PI_F;
                                        }
                                    }
                                } else if (samplingStrategy == 2) {
                                    // ONE_BLOCK: pick one random emitter, sample all 6 faces
                                    unsigned int r = Random_nextState(random);
                                    int ri = (int)(r % eCount);
                                    int emitterIndex = positionIndexes[eStart + ri];
                                    int ex = emitterPositions[emitterIndex*4 + 0];
                                    int ey = emitterPositions[emitterIndex*4 + 1];
                                    int ez = emitterPositions[emitterIndex*4 + 2];
                                    float avgFaceArea = as_float(emitterPositions[emitterIndex*4 + 3]);
                                    float3 accCol = (float3)(0.0f);
                                    for (int face = 0; face < 6; face++) {
                                        float ru = Random_nextFloat(random);
                                        float rv = Random_nextFloat(random);
                                        float3 epos;
                                        switch (face) {
                                            case 0: epos = (float3)(ex, ey + ru, ez + rv); break;
                                            case 1: epos = (float3)(ex + 1.0f, ey + ru, ez + rv); break;
                                            case 2: epos = (float3)(ex + ru, ey, ez + rv); break;
                                            case 3: epos = (float3)(ex + ru, ey + 1.0f, ez + rv); break;
                                            case 4: epos = (float3)(ex + ru, ey + rv, ez); break;
                                            default: epos = (float3)(ex + ru, ey + rv, ez + 1.0f); break;
                                        }
                                        float3 toEmitter = epos - hitPoint;
                                        float dist = length(toEmitter) + 1e-6f;
                                        float3 dirToEmitter = toEmitter / dist;
                                        if (dot(record.normal, dirToEmitter) <= 0.0f) continue;
                                        Ray shadow = bRay;
                                        shadow.origin = hitPoint;
                                        shadow.direction = dirToEmitter;
                                        shadow.flags |= RAY_SHADOW;
                                        IntersectionRecord srec = IntersectionRecord_new();
                                        srec.distance = dist;
                                        Material sSample;
                                        MaterialSample sMatSample;
                                        bool shadowClear = !closestIntersect(scene, textureAtlas, shadow, &srec, &sMatSample, &sSample);
                                        if (shadowClear || srec.distance >= dist - 1e-4f) {
                                            if (shadowClear) {
                                                Emitter_sampleCell(scene.blockPalette, scene.materialPalette, textureAtlas, &scene.octree,
                                                                   ex, ey, ez, (float2)(ru, rv), &sMatSample);
                                                srec.normal = (face < 2) ? (float3)(face == 0 ? -1.0f : 1.0f, 0, 0)
                                                            : (face < 4) ? (float3)(0, face == 2 ? -1.0f : 1.0f, 0)
                                                                         : (float3)(0, 0, face == 4 ? -1.0f : 1.0f);
                                            }
                                            float cosEmitter = fabs(dot(dirToEmitter, srec.normal));
                                            float att = cosEmitter / fmax(dist * dist, 1.0f); // CPU emitter NEE has no receiving-cosine weight (only the visibility gate)
                                            att *= avgFaceArea;
                                            float3 emitterCol = sMatSample.color.xyz * sMatSample.emittance; // linear (not color^2)
                                            accCol += emitterCol * att;
                                        }
                                    }
                                    branchColor += throughput * cachedEmitterIntensity * accCol * (1.0f / 6.0f) * (float)M_PI_F;
                                } else if (samplingStrategy == 3) {
                                    // ALL: iterate every emitter in the grid cell, sample all 6 faces each.
                                    float3 accCol = (float3)(0.0f, 0.0f, 0.0f);
                                    for (int si = 0; si < eCount; ++si) {
                                        int emitterIndex = positionIndexes[eStart + si];
                                        int ex = emitterPositions[emitterIndex*4 + 0];
                                        int ey = emitterPositions[emitterIndex*4 + 1];
                                        int ez = emitterPositions[emitterIndex*4 + 2];
                                        float avgFaceArea = as_float(emitterPositions[emitterIndex*4 + 3]);
                                        for (int face = 0; face < 6; face++) {
                                            float ru = Random_nextFloat(random);
                                            float rv = Random_nextFloat(random);
                                            float3 epos;
                                            switch (face) {
                                                case 0: epos = (float3)(ex, ey + ru, ez + rv); break;
                                                case 1: epos = (float3)(ex + 1.0f, ey + ru, ez + rv); break;
                                                case 2: epos = (float3)(ex + ru, ey, ez + rv); break;
                                                case 3: epos = (float3)(ex + ru, ey + 1.0f, ez + rv); break;
                                                case 4: epos = (float3)(ex + ru, ey + rv, ez); break;
                                                default: epos = (float3)(ex + ru, ey + rv, ez + 1.0f); break;
                                            }
                                            float3 toEmitter = epos - hitPoint;
                                            float dist = length(toEmitter) + 1e-6f;
                                            float3 dirToEmitter = toEmitter / dist;
                                            if (dot(record.normal, dirToEmitter) <= 0.0f) continue;
                                            Ray shadow = bRay;
                                            shadow.origin = hitPoint;
                                            shadow.direction = dirToEmitter;
                                            shadow.flags |= RAY_SHADOW;
                                            IntersectionRecord srec = IntersectionRecord_new();
                                            srec.distance = dist;
                                            Material sSample;
                                            MaterialSample sMatSample;
                                            bool shadowClear = !closestIntersect(scene, textureAtlas, shadow, &srec, &sMatSample, &sSample);
                                            if (shadowClear || srec.distance >= dist - 1e-4f) {
                                                if (shadowClear) {
                                                    Emitter_sampleCell(scene.blockPalette, scene.materialPalette, textureAtlas, &scene.octree,
                                                                       ex, ey, ez, (float2)(ru, rv), &sMatSample);
                                                    srec.normal = (face < 2) ? (float3)(face == 0 ? -1.0f : 1.0f, 0, 0)
                                                                : (face < 4) ? (float3)(0, face == 2 ? -1.0f : 1.0f, 0)
                                                                             : (float3)(0, 0, face == 4 ? -1.0f : 1.0f);
                                                }
                                                float cosEmitter = fabs(dot(dirToEmitter, srec.normal));
                                                float att = cosEmitter / fmax(dist * dist, 1.0f); // CPU emitter NEE has no receiving-cosine weight (only the visibility gate)
                                                att *= avgFaceArea;
                                                float3 emitterCol = sMatSample.color.xyz * sMatSample.emittance; // linear (not color^2)
                                                accCol += emitterCol * att;
                                            }
                                        }
                                    }
                                    if (eCount > 0) branchColor += throughput * cachedEmitterIntensity * (accCol / (float)(eCount * 6)) * (float)M_PI_F;
                                }
                            }
                        }
                    }

#endif
                    // Sun direct light sampling. CPU does this ONLY inside
                    // doDiffuseReflection — never on specular reflections or on
                    // transmission/refraction. Gate on a true diffuse bounce
                    // (!specular && !transmitted) so rays passing straight through
                    // transparent texels don't pick up a spurious sun contribution.
                    // Fires only for strategies with doSunSampling=true (FAST,
                    // HIGH_QUALITY) per chunky's truth table.
                    if (sunDoSampling && !pdfSample.specular && !pdfSample.transmitted) {
                        Ray sunRay;
                        sunRay.origin = hitPos;
                        if (Sun_sampleDirection(sun, &sunRay, random)) {
                            float cosSun = dot(record.normal, sunRay.direction);
                            bool frontLight = cosSun > 0.0f;
                            // CPU PathTracer: backside sun sampling for SSS materials (30% chance)
                            if (frontLight || (sample.sss && Random_nextFloat(random) < F_SUBSURFACE)) {
                                float3 sunHitPoint;
                                if (frontLight) {
                                    sunHitPoint = hitPos + record.normal * OFFSET;
                                } else {
                                    sunHitPoint = hitPos - record.normal * OFFSET;
                                }
                                sunRay.origin = sunHitPoint;
#if PATH_WAVEFRONT
                                // Traced and added by wfConnect, last at this depth:
                                // branchColor += B * (A * sunAtt), the same products.
                                wfSunPending = true;
                                wfSunOrigin = sunHitPoint;
                                wfSunDir = sunRay.direction;
                                {
                                    float mult = fabs(cosSun);
                                    if (sunIsLuminosity) {
                                        mult *= cachedSunLumInv;
                                    }
                                    wfSunA = sun.color.xyz * cachedSunPower * mult;
                                    wfSunB = prevThroughput * sample.color.xyz;
                                }
#else
                                float3 sunAtt = getDirectLightAttenuation(scene, textureAtlas,
                                    sunHitPoint, sunRay.direction, FOG_LIMIT, strictDirectLight);
                                if (sunAtt.x + sunAtt.y + sunAtt.z > 0.0f) {
                                    float mult = fabs(cosSun);
                                    // chunky CPU multiplies by 1/luminosity when
                                    // isSunLuminosity()==true. NEE only fires for
                                    // FAST and HIGH_QUALITY; of those FAST has
                                    // sunLuminosity=false (skip) and HIGH_QUALITY
                                    // has sunLuminosity=true (apply).
                                    if (sunIsLuminosity) {
                                        mult *= cachedSunLumInv;
                                    }
                                    float3 sunContrib = sun.color.xyz * cachedSunPower * mult * sunAtt;
                                    branchColor += prevThroughput * sample.color.xyz * sunContrib;
                                }
#endif
                            }
                        }
                    }

                    // Update ray position and direction
                    bRay.origin = bRay.origin + bRay.direction * (record.distance - OFFSET);
                    bRay.direction = pdfSample.direction;
                    bRay.origin += bRay.direction * OFFSET;

                    // Update medium tracking
                    if (pdfSample.transmitted) {
                        bRay.prevIor = bRay.currentIor;
                        bRay.currentIor = pdfSample.newIor;
                        bRay.inWater = sample.isWater ? !bRay.inWater : bRay.inWater;
                        // Seed the octree DDA with the block we just entered so the
                        // far side of a contiguous run of it is not drawn. This is
                        // NOT gated on IOR: CPU culls by block identity, so a
                        // non-refractive translucent block (leaves, copper grates,
                        // cauldrons, spawner, lava) must cull too. blockData is 0 for
                        // entity/BVH hits (IntersectionRecord_new), which is air —
                        // i.e. no medium — so those need no special case. Restricted
                        // to full cubes (plus the original refractive case, which
                        // covers panes and water): claiming a MODEL block as the
                        // medium would make the DDA skip the rest of its geometry.
                        if (pdfSample.newIor > AIR_IOR + EPS
                                || BlockPalette_isFullCube(scene.blockPalette, record.blockData)) {
                            bRay.material = record.blockData;
                        } else {
                            bRay.material = 0;
                        }
                    }

                    // Only DIFFUSE bounces become indirect. CPU doTransmission does
                    // next.set(ray), preserving ray.specular, so a camera/specular
                    // ray that passes straight through a semi-transparent texel
                    // still hits the sky as an APPARENT (specular) sun — not the
                    // ~60x brighter diffuse/luminosity sun. Refraction keeps
                    // specular=true so it is unaffected.
                    if (!pdfSample.specular && !pdfSample.transmitted) {
                        bRay.flags |= RAY_INDIRECT;
                    }

                    // CPU parity: chunky's PathTracer uses a hard depth cutoff,
                    // not Russian roulette. Termination happens via the depth
                    // loop bound (cachedRayDepth).
                    //
                    // Throughput early-exit: if all three channels have decayed
                    // below 1e-30f, any further bounce contributes effectively
                    // zero to branchColor (max possible add < threshold *
                    // emittance, which is far below fp32 precision for any
                    // reasonable accumulator value). Bypasses the closestIntersect
                    // for paths that can no longer contribute. The threshold
                    // is tight enough that the branchColor difference is
                    // sub-fp32-ULP — visually identical to the no-early-exit
                    // version.
                    if (fmax(fmax(throughput.x, throughput.y), throughput.z) < 1e-30f) {
                        break;
                    }
