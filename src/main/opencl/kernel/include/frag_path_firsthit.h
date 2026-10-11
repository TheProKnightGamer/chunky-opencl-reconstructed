// FRAGMENT, not a header: included INSIDE kernel bodies (render in
// rayTracer.c and the wavefront kernels), so it has no include guard and is
// meant to be included more than once.
//
// The camera ray hit something: the air / water distances every branch's fog
// uses (emptySpaceDist, firstAirDist, firstWaterDist).
            // First surface hit - branch count reuses this first intersection
            // Track first-hit air/water distance (shared across all branches).
            // When the camera is outside the octree, subtract the empty-space
            // distance so fog only accounts for in-scene travel, not the void
            // between camera and loaded chunks.
            float emptySpaceDist = 0.0f;
            if (!AABB_inside(scene.octree.bounds, ray.origin)) {
                float3 invDFog = select(1.0f / ray.direction, copysign((float3)(1e30f), ray.direction), fabs(ray.direction) < 1e-30f);
                float dEntry = AABB_quick_intersect(scene.octree.bounds, ray.origin, invDFog);
                if (!isnan(dEntry) && dEntry > 0) emptySpaceDist = dEntry;
            }
            float firstAirDist = 0.0f;
            float firstWaterDist = 0.0f;
            if (ray.inWater) {
                firstWaterDist = fmax(0.0f, firstRecord.distance - emptySpaceDist);
            } else {
                firstAirDist = fmax(0.0f, firstRecord.distance - emptySpaceDist);
            }

