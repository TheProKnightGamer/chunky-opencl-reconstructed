package dev.thatredox.chunkynative.common.emissive;

import se.llbit.chunky.resources.BitmapImage;
import se.llbit.chunky.resources.Texture;

/**
 * An emission map laid out exactly like the colour texture it belongs to, so the kernel
 * reads both with the same atlas size and UV.
 *
 * <p>Texel alpha is the emission strength. RGB is the same everywhere: the
 * emission-weighted mean colour of the whole block, stored as linear 8-bit values
 * (not sRGB). Emitter sampling of model blocks reads it instead of sampling one texel.
 *
 * <p>The atlas exporter copies the raw bytes (see {@code EmissionMaskExporter}) and,
 * like it does for the colour texture, keeps a single frame of an animated strip.
 */
public final class EmissionMaskTexture extends Texture {
    /** Whether the colour texture is an animated strip whose frame is picked at upload. */
    public final boolean animated;

    public EmissionMaskTexture(int width, int height, byte[] mask, int linearRgb, boolean animated) {
        super(image(width, height, mask, linearRgb));
        this.animated = animated;
    }

    private static BitmapImage image(int width, int height, byte[] mask, int linearRgb) {
        BitmapImage img = new BitmapImage(width, height);
        int rgb = linearRgb & 0xFFFFFF;
        for (int i = 0; i < mask.length; i++) {
            img.data[i] = ((mask[i] & 0xFF) << 24) | rgb;
        }
        return img;
    }
}
