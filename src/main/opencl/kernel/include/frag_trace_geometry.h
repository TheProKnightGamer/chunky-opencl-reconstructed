// FRAGMENT, not a header: included INSIDE functions (kernel.h), so it has no
// include guard and is meant to be included more than once.
//
// traceScene's search for the closest surface. The includer defines
// TRACE_GEOMETRY_MISS / TRACE_GEOMETRY_OPAQUE (what to return on a miss / on a
// stopAtOpaque shadow hit); on a hit, tempRecord holds the surface.

    IntersectionRecord tempRecord = *record;
    bool hit = false;
    hit |= Octree_octreeIntersect(self.octree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &tempRecord);
    if (stopAtOpaque && hit && tempRecord.hitKind == HIT_MATERIAL
            && Material_isOpaqueAt(Material_get(self.materialPalette, tempRecord.material), atlas, tempRecord.texCoord)) {
        // Only what the shadow loop reads: the distance, and an opaque sample.
        *record = tempRecord;
        record->geomNormal = record->normal;
        sample->color = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
        sample->refractive = false;
        sample->isWater = false;
        TRACE_GEOMETRY_OPAQUE;
    }
    hit |= Bvh_intersect(self.worldBvh, atlas, self.materialPalette, ray, &tempRecord);
    hit |= Bvh_intersect(self.actorBvh, atlas, self.materialPalette, ray, &tempRecord);
    {
        IntersectionRecord waterRecord = tempRecord;
        if (!hit) waterRecord.distance = record->distance;
        bool waterHit = false;
        if (ray.inWater) {
            waterRecord.distance = hit ? (tempRecord.distance + EPS) : record->distance;
            waterHit = Octree_exitWater(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord);
        } else if (Water_clipToBlocks(&self, ray, &waterRecord.distance)) {
            waterHit = Octree_octreeIntersect(self.waterOctree, atlas, self.blockPalette, self.materialPalette, self.drawDepth, ray, &waterRecord);
        }
        if (waterHit && (!hit || waterRecord.distance < tempRecord.distance)) {
            tempRecord = waterRecord;
            tempRecord.hitKind |= HIT_FLAG_WATER_BRANCH;
            hit = true;
        }
    }
    if (self.waterPlaneEnabled) {
        IntersectionRecord waterRecord = tempRecord;
        if (!hit) waterRecord.distance = record->distance;
        if (Water_planeIntersect(ray, self.waterPlaneHeight, self.octreeSize,
                                 self.waterPlaneChunkClip,
                                 self.chunkBitmap, self.chunkBitmapSize,
                                 &waterRecord)) {
            if (!hit || waterRecord.distance < tempRecord.distance) {
                tempRecord = waterRecord;
                // Wave shading is deferred to the unified water-shading block
                // below (after geomNormal is pinned to the flat normal), so the
                // flat plane normal survives for the anti-leak corrections
                // instead of being clobbered by the blanket geomNormal=normal.
                tempRecord.hitKind = HIT_WATER_PLANE;
                hit = true;
            }
        }
    }
    if (self.cloudsEnabled) {
        if (Cloud_intersect(self.cloudHeight, self.cloudSize, self.cloudOffsetX, self.cloudOffsetZ,
                            self.cloudData, ray, &tempRecord, hit)) {
            hit = true;
        }
    }
    if (!hit) TRACE_GEOMETRY_MISS;
