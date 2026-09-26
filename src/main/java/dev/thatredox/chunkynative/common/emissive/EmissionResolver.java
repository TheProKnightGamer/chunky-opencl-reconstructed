package dev.thatredox.chunkynative.common.emissive;

import dev.thatredox.chunkynative.common.export.texture.AbstractTextureLoader;
import se.llbit.chunky.PersistentSettings;
import se.llbit.chunky.block.AbstractModelBlock;
import se.llbit.chunky.block.Block;
import se.llbit.chunky.block.minecraft.LightBlock;
import se.llbit.chunky.block.minecraft.Water;
import se.llbit.chunky.model.AABBModel;
import se.llbit.chunky.model.BlockModel;
import se.llbit.chunky.model.QuadModel;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.resources.AnimatedTexture;
import se.llbit.chunky.resources.Texture;
import se.llbit.chunky.world.Material;

import java.util.*;

/**
 * Decides how each block surface emits light, from the scene's Emissive tab settings
 * and the emission maps of the configured packs. One resolver is made per GPU scene
 * load and memoizes everything, so the texture preload pass and the material packing
 * pass get identical answers and the same mask texture instances.
 *
 * <p>For a block with emission maps, a surface emits {@code emittance * mask(uv)}. Its
 * textures without a map of their own get an all-zero mask, so that, for example, only
 * a furnace's front glows. Every mask of the block also carries the block's mean
 * emission and emission-weighted colour, which emitter sampling of models uses in
 * place of a single texel.
 */
public final class EmissionResolver {
    /** How one material emits. */
    public static final class Emission {
        /** Emittance, multiplied by the mask's alpha when there is a mask. */
        public final float emittance;
        /** Emission map, or null when the whole surface emits {@link #emittance}. */
        public final EmissionMaskTexture mask;
        /** Mean emission (0-1) over the block's textures. Only set with a mask. */
        public final float blockMean;

        Emission(float emittance, EmissionMaskTexture mask, float blockMean) {
            this.emittance = emittance;
            this.mask = mask;
            this.blockMean = blockMean;
        }
    }

    private static final EmissionResolver DISABLED =
            new EmissionResolver(false, null, Collections.emptyMap());

    private final boolean enabled;
    private final EmissiveSettings.SceneSettings settings;
    private final Map<String, EmissionMask> masks;
    private final Map<Texture, Resampled> resampled = new IdentityHashMap<>();
    private final Map<Block, BlockMaps> blocks = new IdentityHashMap<>();

    private EmissionResolver(boolean enabled, EmissiveSettings.SceneSettings settings,
                             Map<String, EmissionMask> masks) {
        this.enabled = enabled;
        this.settings = settings;
        this.masks = masks;
    }

    /** Chunky's own behaviour: every surface emits its material's emittance. */
    public static EmissionResolver disabled() {
        return DISABLED;
    }

    /** May convert packs that have no cached conversion yet. */
    public static EmissionResolver forScene(Scene scene) {
        EmissiveSettings.SceneSettings s = EmissiveSettings.get(scene);
        if (!s.enabled) return DISABLED;
        // Single-colour mode has no per-texel colour texture to lay a mask over.
        Map<String, EmissionMask> masks = PersistentSettings.getSingleColorTextures()
                ? Collections.emptyMap() : EmissiveLibrary.masks();
        return new EmissionResolver(true, s, masks);
    }

    /** A resolver for UI queries such as {@link #hasMap}, ignoring per-block settings. */
    public static EmissionResolver forMasks(Map<String, EmissionMask> masks) {
        return new EmissionResolver(true, new EmissiveSettings.SceneSettings(true, Collections.emptyMap()), masks);
    }

    /**
     * @param material  the block (or other material) the surface belongs to
     * @param texture   the surface's colour texture
     * @param emittance the material's emittance as Chunky has it
     */
    public Emission resolve(Material material, Texture texture, float emittance) {
        if (!enabled || !(material instanceof Block)
                || material instanceof LightBlock || material instanceof Water) {
            return uniform(emittance);
        }
        Block block = (Block) material;
        EmissiveSettings.BlockSetting s = settings.get(block.name);
        boolean hasStrength = !Float.isNaN(s.strength);
        float base;
        switch (s.mode) {
            case OFF:
                return uniform(0);
            case WHOLE:
                return uniform(hasStrength ? s.strength : (emittance > 0 ? emittance : 1));
            case MAP:
                base = hasStrength ? s.strength : (emittance > 0 ? emittance : 1);
                break;
            default:
                if (emittance <= 0) return uniform(emittance);
                base = hasStrength ? s.strength : emittance;
                break;
        }
        BlockMaps bm = blockMaps(block);
        if (!bm.hasMap) return uniform(base);
        if (base <= 0) return uniform(0);
        EmissionMaskTexture mask = maskTexture(bm, texture);
        return mask != null ? new Emission(base, mask, bm.mean) : uniform(0);
    }

    /** Register the mask {@link #resolve} will return, before the loader is built. */
    public void preload(Material material, Texture texture, AbstractTextureLoader loader) {
        Emission e = resolve(material, texture, material.emittance);
        if (e.mask != null) loader.get(e.mask);
    }

    /** Whether any texture of the block has an emission map. */
    public boolean hasMap(Block block) {
        return enabled && blockMaps(block).hasMap;
    }

    private static Emission uniform(float emittance) {
        return new Emission(emittance, null, 0);
    }

    // ---- masks ----

    private static final class Resampled {
        static final Resampled NONE = new Resampled(null, 0, new double[3]);

        /** At the colour texture's size, null if the texture has no map. */
        final byte[] mask;
        /** Mean of mask/255 over the texture. */
        final double mean;
        /** Mean of linear colour * mask/255 over the texture. */
        final double[] colorMean;

        Resampled(byte[] mask, double mean, double[] colorMean) {
            this.mask = mask;
            this.mean = mean;
            this.colorMean = colorMean;
        }
    }

    private static final class BlockMaps {
        boolean hasMap;
        float mean;
        int linearRgb;
        final Map<Texture, EmissionMaskTexture> textures = new IdentityHashMap<>();
    }

    /** Plain block textures only; anything else is composed or drawn at runtime. */
    private static boolean maskable(Texture t) {
        Class<?> c = t.getClass();
        return (c == Texture.class || c == AnimatedTexture.class) && t.getWidth() > 0 && t.getHeight() > 0;
    }

    private static int colorFrames(Texture t) {
        if (!(t instanceof AnimatedTexture)) return 1;
        int frameH = Math.min(t.getWidth(), t.getHeight());
        return Math.max(1, t.getHeight() / frameH);
    }

    private Resampled resampled(Texture t) {
        Resampled r = resampled.get(t);
        if (r != null) return r;
        r = Resampled.NONE;
        if (maskable(t)) {
            EmissionMask m = null;
            for (String path : TexturePaths.of(t)) {
                m = masks.get(path);
                if (m != null) break;
            }
            if (m != null) {
                int w = t.getWidth();
                int h = t.getHeight();
                byte[] mask = m.resample(w, h, colorFrames(t));
                double sum = 0;
                double[] c = new double[3];
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int v = mask[y * w + x] & 0xFF;
                        if (v == 0) continue;
                        double e = v / 255.0;
                        float[] rgba = t.getColor(x, y);
                        sum += e;
                        c[0] += rgba[0] * e;
                        c[1] += rgba[1] * e;
                        c[2] += rgba[2] * e;
                    }
                }
                if (sum > 0) {
                    int n = w * h;
                    r = new Resampled(mask, sum / n, new double[] {c[0] / n, c[1] / n, c[2] / n});
                }
            }
        }
        resampled.put(t, r);
        return r;
    }

    private BlockMaps blockMaps(Block block) {
        BlockMaps bm = blocks.get(block);
        if (bm != null) return bm;
        bm = new BlockMaps();
        List<Texture> textures = texturesOf(block);
        double mean = 0;
        double[] c = new double[3];
        for (Texture t : textures) {
            Resampled r = resampled(t);
            if (r.mask == null) continue;
            bm.hasMap = true;
            mean += r.mean;
            for (int i = 0; i < 3; i++) c[i] += r.colorMean[i];
        }
        if (bm.hasMap) {
            // Faces without a map count as dark: the block's average over all faces.
            bm.mean = (float) (mean / textures.size());
            int rgb = 0;
            for (int i = 0; i < 3; i++) {
                int v = (int) Math.round(Math.min(1.0, c[i] / mean) * 255.0);
                rgb = (rgb << 8) | v;
            }
            bm.linearRgb = rgb;
        }
        blocks.put(block, bm);
        return bm;
    }

    private EmissionMaskTexture maskTexture(BlockMaps bm, Texture t) {
        EmissionMaskTexture m = bm.textures.get(t);
        if (m != null || bm.textures.containsKey(t)) return m;
        if (maskable(t)) {
            Resampled r = resampled(t);
            int w = t.getWidth();
            int h = t.getHeight();
            byte[] data = r.mask != null ? r.mask : new byte[w * h];
            m = new EmissionMaskTexture(w, h, data, bm.linearRgb, t instanceof AnimatedTexture);
        }
        bm.textures.put(t, m);
        return m;
    }

    /** The colour textures a block is drawn with, one entry per box face or quad. */
    public static List<Texture> texturesOf(Block block) {
        List<Texture> out = new ArrayList<>();
        if (block instanceof AbstractModelBlock) {
            BlockModel model = ((AbstractModelBlock) block).getModel();
            if (model instanceof AABBModel) {
                Texture[][] texs = ((AABBModel) model).getTextures();
                if (texs != null) {
                    for (Texture[] box : texs) {
                        if (box == null) continue;
                        for (Texture t : box) if (t != null) out.add(t);
                    }
                }
            } else if (model instanceof QuadModel) {
                Texture[] texs = ((QuadModel) model).getTextures();
                if (texs != null) for (Texture t : texs) if (t != null) out.add(t);
            }
        } else if (!block.invisible && block.texture != null) {
            out.add(block.texture);
        }
        return out;
    }
}
