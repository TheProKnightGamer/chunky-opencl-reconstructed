package dev.thatredox.chunkynative.common.export.primitives;

import dev.thatredox.chunkynative.common.export.texture.AbstractTextureLoader;
import dev.thatredox.chunkynative.common.export.Packer;
import dev.thatredox.chunkynative.common.export.ResourcePalette;
import dev.thatredox.chunkynative.common.export.models.PackedAabbModel;
import dev.thatredox.chunkynative.common.export.models.PackedQuadModel;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import se.llbit.chunky.block.AbstractModelBlock;
import se.llbit.chunky.block.Block;
import se.llbit.chunky.block.minecraft.LightBlock;
import se.llbit.chunky.block.minecraft.Water;
import se.llbit.chunky.model.AABBModel;
import se.llbit.chunky.model.QuadModel;
import se.llbit.chunky.model.Tint;
import se.llbit.chunky.resources.Texture;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Stream;

public class PackedBlock implements Packer {
    public final int modelType;
    public final int modelPointer;
    public final int waterData; // Water corner heights + full-block flag (only for modelType 5)

    public PackedBlock(Block block, AbstractTextureLoader textureLoader,
                       ResourcePalette<PackedMaterial> materialPalette,
                       ResourcePalette<PackedAabbModel> aabbModels,
                       ResourcePalette<PackedQuadModel> quadModels) {
        if (block instanceof LightBlock) {
            modelType = 4;
            modelPointer = materialPalette.put(new PackedMaterial(Texture.light, 0xFE000000,
                    block.emittance, block.specular, block.metalness, block.roughness, textureLoader));
            waterData = 0;
        } else if (block instanceof Water) {
            // Water blocks use modelType 5 with per-corner height data from OctreeFinalizer.
            // The water data int encodes:
            //   bits 0-3:   CORNER_SW height index (0-7)
            //   bits 4-7:   CORNER_SE height index (0-7)
            //   bits 8-11:  CORNER_NE height index (0-7)
            //   bits 12-15: CORNER_NW height index (0-7)
            //   bit 16:     FULL_BLOCK flag (1 = submerged, full cube)
            modelType = 5;
            Water water = (Water) block;
            modelPointer = materialPalette.put(new PackedMaterial(block.texture, Tint.BIOME_WATER,
                    block.emittance, block.specular, block.metalness, block.roughness,
                    block.ior, block.refractive, block.subSurfaceScattering, true, textureLoader));
            waterData = water.data;
        } else if (block instanceof AbstractModelBlock) {
            AbstractModelBlock b = (AbstractModelBlock) block;
            if (b.getModel() instanceof AABBModel) {
                modelType = 2;
                modelPointer = aabbModels.put(new PackedAabbModel(
                        (AABBModel) b.getModel(), block, textureLoader, materialPalette));
            } else if (b.getModel() instanceof QuadModel) {
                modelType = 3;
                modelPointer = quadModels.put(new PackedQuadModel(
                        (QuadModel) b.getModel(), block, textureLoader, materialPalette));
            } else {
                throw new RuntimeException(String.format(
                        "Unknown model type for block %s: %s", block.name, b.getModel()));
            }
            waterData = 0;
        } else if (block.invisible) {
            modelType = 0;
            modelPointer = 0;
            waterData = 0;
        } else {
            modelType = 1;
            modelPointer = materialPalette.put(new PackedMaterial(block.texture, Tint.NONE,
                    block.emittance, block.specular, block.metalness, block.roughness,
                    block.ior, block.refractive, block.subSurfaceScattering, false, textureLoader));
            waterData = 0;
        }
    }

    public static void preloadTextures(Block block, AbstractTextureLoader textureLoader) {
        if (block instanceof AbstractModelBlock) {
            AbstractModelBlock b = (AbstractModelBlock) block;
            Stream<Texture> textures;
            if (b.getModel() instanceof AABBModel) {
                textures = Arrays.stream(((AABBModel) b.getModel()).getTextures())
                        .flatMap(Arrays::stream)
                        .filter(Objects::nonNull);
            } else if (b.getModel() instanceof QuadModel) {
                textures = Arrays.stream(((QuadModel) b.getModel()).getTextures());
            } else {
                throw new RuntimeException(String.format(
                        "Unknown model type for block %s: %s", block.name, b.getModel()));
            }
            textures.forEach(textureLoader::get);
        } else if (!block.invisible) {
            textureLoader.get(block.texture);
        }
    }

    /**
     * Pack this block into ints. The first integer specifies the type of the model:
     * 0 - Invisible
     * 1 - Full size block
     * 2 - AABB model
     * 3 - Quad model
     * 4 - Light block
     * 5 - Water block (3 ints: type, material pointer, water data)
     * The second integer is a pointer to the model object in its respective palette.
     * For water blocks (type 5), a third integer contains corner height + full-block data.
     */
    @Override
    public IntArrayList pack() {
        IntArrayList out = new IntArrayList(modelType == 5 ? 3 : 2);
        out.add(modelType);
        out.add(modelPointer);
        if (modelType == 5) {
            out.add(waterData);
        }
        return out;
    }
}
