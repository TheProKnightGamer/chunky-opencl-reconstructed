package dev.thatredox.chunkynative.common.export;

import se.llbit.chunky.block.Block;
import se.llbit.chunky.resources.BitmapImage;
import se.llbit.chunky.resources.Texture;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * GPU-only material overrides.
 *
 * <p><b>These deliberately DIVERGE from Chunky's CPU renderer.</b> Everything else in
 * this plugin chases parity with the CPU path tracer; this class is the one place that
 * intentionally does not, so keep it small and keep the reason written down.
 *
 * <p>It covers the four ice blocks, for two separate reasons:
 *
 * <ul>
 *   <li><b>Packed ice and blue ice</b> are plain {@code MinecraftBlock} upstream, i.e.
 *       {@code opaque = true}, and their vanilla PNGs carry no alpha at all (neither
 *       {@code packed_ice.png} nor {@code blue_ice.png} has a {@code tRNS} chunk). They
 *       render as solid cubes, so every internal face between two of them is drawn.
 *   <li><b>All four</b> are given a common, higher alpha. Vanilla {@code ice.png} is
 *       alpha 190/255, which combined with same-block face culling (a whole ice wall
 *       becomes a single interface) left ice far too easy to see through.
 * </ul>
 *
 * <p>Note that alpha, not IOR, is the lever here. Setting {@code refractive} alone does
 * nothing: {@code Material_samplePdf} gates refraction on {@code random >= pDiffuse}, and
 * {@code pDiffuse == 1} when alpha is 1, so an opaque-textured block always reflects
 * diffusely, never transmits, and therefore never sets the ray medium that face culling
 * depends on.
 */
public final class GpuMaterialOverrides {
    private GpuMaterialOverrides() {
    }

    /**
     * Alpha applied to every ice block, out of 255.
     *
     * <p><b>This is the "how hard to see through" knob.</b> Transmission through one ice
     * surface is roughly {@code (1 - alpha)^maxRGB} — at 235 that is about 11%, at the
     * vanilla 190 it was about 33%. Raise toward 255 for near-solid ice, lower toward 190
     * to go back to vanilla clarity.
     */
    private static final int ICE_ALPHA = 235;

    /** IOR Chunky assigns to {@code minecraft:ice} and {@code minecraft:frosted_ice}. */
    public static final float ICE_IOR = 1.31f;

    /** {@code Material.DEFAULT_IOR}, which is private upstream. */
    private static final float DEFAULT_IOR = 1.000293f;

    /** Ice blocks Chunky builds as fully opaque, which also need IOR and refractive set. */
    private static final Set<String> OPAQUE_ICE = Collections.unmodifiableSet(
            new HashSet<>(Arrays.asList(
                    "minecraft:packed_ice",
                    "minecraft:blue_ice")));

    /** Ice blocks Chunky already marks translucent and refractive; only alpha changes. */
    private static final Set<String> TRANSLUCENT_ICE = Collections.unmodifiableSet(
            new HashSet<>(Arrays.asList(
                    "minecraft:ice",
                    "minecraft:frosted_ice")));

    /**
     * Cache of alpha-adjusted textures.
     *
     * <p>Keyed on the source {@link BitmapImage}, NOT on the {@link Texture}, and that
     * matters: {@code Texture.setTexture} swaps in a brand new {@code BitmapImage} when a
     * resource pack is (re)loaded while the {@code Texture} object itself is a long-lived
     * static. Keying on the Texture would therefore hand back a copy of the *previous*
     * pack's pixels forever. Keying on the bitmap invalidates itself for free.
     *
     * <p>Neither class overrides {@code equals}/{@code hashCode}, so this is an identity
     * map in practice — which is what we want, since one substituted instance per source
     * keeps the atlas de-duplicator from uploading several copies. Concurrent because
     * scene export walks the palette from the chunk-loading threads. At most one entry
     * per ice texture per pack load, so retained stale entries are negligible.
     */
    private static final Map<BitmapImage, Texture> ICE_TEXTURES = new ConcurrentHashMap<>();

    /**
     * @return {@code true} if this block should be packed as dense refractive ice rather
     *         than with its stock Chunky material.
     */
    public static boolean isIce(Block block) {
        boolean opaqueIce = OPAQUE_ICE.contains(block.name);
        if (!opaqueIce && !TRANSLUCENT_ICE.contains(block.name)) {
            return false;
        }
        // Respect the material editor. If the user has moved this block's IOR off the
        // value Chunky ships it with, they are driving and we leave the block alone.
        float expectedIor = opaqueIce ? DEFAULT_IOR : ICE_IOR;
        boolean expectedRefractive = !opaqueIce;
        return block.refractive == expectedRefractive
                && Math.abs(block.ior - expectedIor) < 1e-6f;
    }

    /**
     * Texture to pack for a block {@link #isIce} accepted: a copy of the block's own
     * texture at {@link #ICE_ALPHA}. Fully transparent texels stay fully transparent so
     * cutout textures are unharmed.
     *
     * <p>The result is a distinct {@link Texture} instance, so the texture loader gives it
     * its own atlas entry and the original is untouched for anything else sharing it.
     */
    public static Texture iceTexture(Texture source) {
        return ICE_TEXTURES.computeIfAbsent(source.getBitmap(), src -> {
            BitmapImage out = new BitmapImage(src);
            int[] data = out.data;
            for (int i = 0; i < data.length; i++) {
                int argb = data[i];
                if ((argb >>> 24) == 0) {
                    continue; // keep cutout holes fully transparent
                }
                data[i] = (ICE_ALPHA << 24) | (argb & 0x00FFFFFF);
            }
            return new Texture(out);
        });
    }
}
