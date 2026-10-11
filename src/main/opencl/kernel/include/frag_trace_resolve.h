// FRAGMENT, not a header: included INSIDE functions (kernel.h), so it has no
// include guard and is meant to be included more than once.
//
// traceScene's material sample for the surface in tempRecord (copied to
// *record), plus water opacity / colour and wave shading.
    *record = tempRecord;
    Hit_resolveSample(tempRecord, atlas, self.materialPalette, &self, false, sample);
    record->geomNormal = record->normal;
    if (sample->isWater) {
        // Water BLOCKS (material >= 0) get opacity / custom color applied here;
        // the water PLANE already did so in FillWaterSample.
        if (record->material >= 0) {
            sample->color.w = self.waterOpacity;
            if (self.useCustomWaterColor) {
                sample->color.xyz = self.waterColor;
                sample->tintType = 0;
            }
        }
        // Wave shading for BOTH water blocks and the water plane. geomNormal is
        // pinned to the FLAT surface normal first (line 283 set it), THEN the
        // shading perturbs record->normal — so the specular/refraction/diffuse
        // anti-leak corrections run against the true flat normal, not the wave.
        // Shadow rays skip wave shading: CPU traces both sun-NEE
        // (PathTracer.getDirectLightAttenuation) and emitter-NEE
        // (sampleEmitterFace) shadow rays with PreviewRayTracer, which never
        // applies water shading. Sun/fog attenuation reads only distance +
        // material sample; emitter NEE reads srec.normal only when the hit
        // grazes the emitter face within 1e-4, where CPU uses the flat normal
        // anyway.
        if (record->normal.y != 0.0f && !(ray.flags & RAY_SHADOW)) {
            record->geomNormal = record->normal;
            float3 hitPos = ray.origin + ray.direction * record->distance;
            Water_applyShading(record, self.waterShadingStrategy,
                               self.animationTime, hitPos.x, hitPos.z, self.waterShaderParams,
                               self.waterNormalMap, self.waterNormalMapW);
        }
    }
