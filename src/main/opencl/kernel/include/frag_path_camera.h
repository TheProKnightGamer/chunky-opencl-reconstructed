// FRAGMENT, not a header: included INSIDE kernel bodies (render in
// rayTracer.c and the wavefront kernels), so it has no include guard and is
// meant to be included more than once.
//
// One camera sample: seeds the RNG from (gid, it), builds the camera ray and
// finds the first hit (firstHit, firstRecord, firstSample, color, alpha, ...).
        unsigned int randomState = randSeed + (unsigned int)(gid * 1664525u) + (unsigned int)(it * 1013904223u);
        Random random = &randomState;
        Random_nextState(random);
        Random_nextState(random);
        Ray ray = ray_to_camera(camCache, cameraSettings, apertureMask, apertureMaskWidth, gid, random);

        ray.material = 0;
        ray.flags = emittersEnabled ? RAY_EMITTERS : 0;
        ray.currentIor = AIR_IOR;
        ray.prevIor = AIR_IOR;
        ray.inWater = false;

        // Check if camera ray starts inside water (matching CPU PathTracer initial medium setup)
        if (isInWater(scene, ray.origin)) {
            ray.inWater = true;
            ray.currentIor = scene.waterIor;
        }

        float3 color = (float3)(0.0f);
        float alpha = 1.0f;
        float3 cameraOrigin = ray.origin;
        float3 cameraDirection = ray.direction;

        // Find first intersection (depth 0)
        IntersectionRecord firstRecord = IntersectionRecord_new();
        MaterialSample firstSample;
        Material firstMaterial;
        bool firstHit = closestIntersect(scene, textureAtlas, ray, &firstRecord, &firstSample, &firstMaterial);
        if (firstHit) {
            float3 hitPos0 = ray.origin + ray.direction * firstRecord.distance;
            applyBiomeTint(scene, &firstSample, hitPos0);
        }

