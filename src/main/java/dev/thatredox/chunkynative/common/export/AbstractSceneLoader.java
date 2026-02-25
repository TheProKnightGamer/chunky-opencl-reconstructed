package dev.thatredox.chunkynative.common.export;

import dev.thatredox.chunkynative.common.export.models.PackedAabbModel;
import dev.thatredox.chunkynative.common.export.models.PackedBvhNode;
import dev.thatredox.chunkynative.common.export.models.PackedQuadModel;
import dev.thatredox.chunkynative.common.export.models.PackedTriangleModel;
import dev.thatredox.chunkynative.common.export.primitives.PackedBlock;
import dev.thatredox.chunkynative.common.export.primitives.PackedMaterial;
import dev.thatredox.chunkynative.common.export.primitives.PackedSun;
import dev.thatredox.chunkynative.common.export.texture.AbstractTextureLoader;
import dev.thatredox.chunkynative.util.Reflection;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.SceneEntities;
import se.llbit.chunky.renderer.scene.sky.Sun;
import se.llbit.log.Log;
import se.llbit.math.Octree;
import se.llbit.math.PackedOctree;
import se.llbit.math.bvh.BVH;
import se.llbit.math.bvh.BinaryBVH;
import se.llbit.math.primitive.TexturedTriangle;

import java.lang.ref.WeakReference;
import java.util.Arrays;

public abstract class AbstractSceneLoader {
    protected int modCount = 0;
    protected WeakReference<BVH> prevWorldBvh = new WeakReference<>(null, null);
    protected WeakReference<BVH> prevActorBvh = new WeakReference<>(null, null);
    protected WeakReference<Octree.OctreeImplementation> prevOctree = new WeakReference<>(null, null);
    protected WeakReference<Octree.OctreeImplementation> prevWaterOctree = new WeakReference<>(null, null);

    protected AbstractTextureLoader texturePalette = null;
    protected ResourcePalette<PackedBlock> blockPalette = null;
    protected CachedResourcePalette<PackedMaterial> materialPalette = null;
    protected ResourcePalette<PackedAabbModel> aabbPalette = null;
    protected ResourcePalette<PackedQuadModel> quadPalette = null;
    protected ResourcePalette<PackedTriangleModel> trigPalette = null;
    protected int[] worldBvh = null;
    protected int[] actorBvh = null;
    protected PackedSun packedSun = null;

    public boolean ensureLoad(Scene scene) {
        return this.ensureLoad(scene, false);
    }

    protected boolean ensureLoad(Scene scene, boolean force) {
        if (force ||
                this.texturePalette == null || this.blockPalette == null || this.materialPalette == null ||
                this.aabbPalette == null || this.quadPalette == null || this.trigPalette == null ||
                this.prevOctree.get() != scene.getWorldOctree().getImplementation()) {
            this.modCount = -1;
            return this.load(0, ResetReason.SCENE_LOADED, scene);
        }
        return true;
    }

    /**
     * @return True if successfully loaded. False if loading failed.
     */
    public boolean load(int modCount, ResetReason resetReason, Scene scene) {
        // Already up to date
        if (this.modCount == modCount) {
            return true;
        }

        if (resetReason == ResetReason.NONE || resetReason == ResetReason.MODE_CHANGE) {
            this.modCount = modCount;
            return true;
        }

        SceneEntities entities = Reflection.getFieldValue(scene, "entities", SceneEntities.class);
        BVH worldBvh = Reflection.getFieldValue(entities, "bvh", BVH.class);
        BVH actorBvh = Reflection.getFieldValue(entities, "actorBvh", BVH.class);

        boolean needTextureLoad = resetReason == ResetReason.SCENE_LOADED ||
                resetReason == ResetReason.MATERIALS_CHANGED ||
                prevWorldBvh.get() != worldBvh ||
                prevActorBvh.get() != actorBvh;

        // Only create palettes when we actually need to rebuild textures/blocks/BVH.
        // For SETTINGS_CHANGED this avoids allocating 6 palette objects that would be
        // immediately discarded.
        int[] blockMapping = null;

        if (needTextureLoad) {
            final AbstractTextureLoader texturePalette = this.createTextureLoader();
            final ResourcePalette<PackedBlock> blockPalette = this.createBlockPalette();
            final CachedResourcePalette<PackedMaterial> materialPalette = new CachedResourcePalette<>(this.createMaterialPalette());
            final ResourcePalette<PackedAabbModel> aabbPalette = this.createAabbModelPalette();
            final ResourcePalette<PackedQuadModel> quadPalette = this.createQuadModelPalette();
            final ResourcePalette<PackedTriangleModel> trigPalette = this.createTriangleModelPalette();

            if (!(worldBvh instanceof BinaryBVH || worldBvh == BVH.EMPTY)) {
                Log.error("BVH implementation must extend BinaryBVH");
                return false;
            }
            if (!(actorBvh instanceof BinaryBVH || actorBvh == BVH.EMPTY)) {
                Log.error("BVH implementation must extend BinaryBVH");
                return false;
            }
            prevWorldBvh = new WeakReference<>(worldBvh, null);
            prevActorBvh = new WeakReference<>(actorBvh, null);

            // Preload textures
            scene.getPalette().getPalette().forEach(b -> PackedBlock.preloadTextures(b, texturePalette));
            if (worldBvh != BVH.EMPTY) preloadBvh((BinaryBVH) worldBvh, texturePalette);
            if (actorBvh != BVH.EMPTY) preloadBvh((BinaryBVH) actorBvh, texturePalette);
            texturePalette.get(Sun.texture);
            texturePalette.build();

            blockMapping = scene.getPalette().getPalette().stream()
                    .mapToInt(block ->
                            blockPalette.put(new PackedBlock(block, texturePalette, materialPalette, aabbPalette, quadPalette)))
                    .toArray();

            int[] packedWorldBvh;
            int[] packedActorBvh;
            if (worldBvh != BVH.EMPTY) {
                packedWorldBvh = loadBvh((BinaryBVH) worldBvh, texturePalette, materialPalette, trigPalette);
            } else {
                packedWorldBvh = PackedBvhNode.EMPTY_NODE.node;
            }
            if (actorBvh != BVH.EMPTY) {
                packedActorBvh = loadBvh((BinaryBVH) actorBvh, texturePalette, materialPalette, trigPalette);
            } else {
                packedActorBvh = PackedBvhNode.EMPTY_NODE.node;
            }
            packedSun = new PackedSun(scene.sun(), texturePalette);

            if (this.texturePalette != null) this.texturePalette.release();
            if (this.blockPalette != null) this.blockPalette.release();
            if (this.materialPalette != null) this.materialPalette.release();
            if (this.aabbPalette != null) this.aabbPalette.release();
            if (this.quadPalette != null) this.quadPalette.release();
            if (this.trigPalette != null) this.trigPalette.release();

            this.texturePalette = texturePalette;
            this.blockPalette = blockPalette;
            this.materialPalette = materialPalette;
            this.aabbPalette = aabbPalette;
            this.quadPalette = quadPalette;
            this.trigPalette = trigPalette;
            this.worldBvh = packedWorldBvh;
            this.actorBvh = packedActorBvh;
        }

        // Load world octree (no merge with water — they are kept separate for dual-octree tracing)
        Octree.OctreeImplementation impl = scene.getWorldOctree().getImplementation();
        if (resetReason == ResetReason.SCENE_LOADED || prevOctree.get() != impl) {
            prevOctree = new WeakReference<>(impl, null);
            if (impl instanceof PackedOctree) {
                assert blockMapping != null;
                int[] worldData = ((PackedOctree) impl).treeData;
                if (!loadOctree(worldData, impl.getDepth(), blockMapping, this.blockPalette))
                    return false;
            } else {
                Log.error("Octree implementation must be PACKED");
                return false;
            }
        }

        // Load water octree separately (CPU keeps world and water octrees independent)
        Octree.OctreeImplementation waterImpl = scene.getWaterOctree().getImplementation();
        if (resetReason == ResetReason.SCENE_LOADED || prevWaterOctree.get() != waterImpl) {
            prevWaterOctree = new WeakReference<>(waterImpl, null);
            if (waterImpl instanceof PackedOctree && waterImpl.getDepth() == impl.getDepth()) {
                assert blockMapping != null;
                int[] waterData = ((PackedOctree) waterImpl).treeData;
                if (!loadWaterOctree(waterData, waterImpl.getDepth(), blockMapping, this.blockPalette))
                    return false;
            } else {
                // No water octree or depth mismatch — upload an empty single-leaf tree
                if (!loadWaterOctree(new int[]{ 0 }, impl.getDepth(), blockMapping != null ? blockMapping : new int[0], this.blockPalette))
                    return false;
            }
        }

        this.modCount = modCount;
        return true;
    }

    protected static void preloadBvh(BinaryBVH bvh, AbstractTextureLoader texturePalette) {
        Arrays.stream(bvh.packedPrimitives).flatMap(Arrays::stream).forEach(primitive -> {
            if (primitive instanceof TexturedTriangle) {
                texturePalette.get(((TexturedTriangle) primitive).material.texture);
            }
        });
    }

    protected static int[] loadBvh(BinaryBVH bvh,
                                  AbstractTextureLoader texturePalette,
                                  ResourcePalette<PackedMaterial> materialPalette,
                                  ResourcePalette<PackedTriangleModel> trigPalette) {
        int[] out = new int[bvh.packed.length];
        for (int i = 0; i < out.length; i += 7) {
            PackedBvhNode node = new PackedBvhNode(bvh.packed, i, bvh.packedPrimitives, texturePalette, materialPalette, trigPalette);
            System.arraycopy(node.pack().elements(), 0, out, i, 7);
        }
        return out;
    }

    protected abstract boolean loadOctree(int[] octree, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette);
    protected abstract boolean loadWaterOctree(int[] waterOctree, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette);

    protected abstract AbstractTextureLoader createTextureLoader();
    protected abstract ResourcePalette<PackedBlock> createBlockPalette();
    protected abstract ResourcePalette<PackedMaterial> createMaterialPalette();
    protected abstract ResourcePalette<PackedAabbModel> createAabbModelPalette();
    protected abstract ResourcePalette<PackedQuadModel> createQuadModelPalette();
    protected abstract ResourcePalette<PackedTriangleModel> createTriangleModelPalette();
}
