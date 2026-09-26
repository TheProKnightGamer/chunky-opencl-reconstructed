package dev.thatredox.chunkynative.opencl.renderer.export.textureexporter;

import dev.thatredox.chunkynative.common.emissive.EmissionMaskTexture;

import java.util.Arrays;

/**
 * Exports an {@link EmissionMaskTexture} without gamma conversion: its RGB is already
 * linear and its alpha is the emission strength. An animated mask keeps the same frame
 * {@link AnimatedTextureExporter} picks for the colour texture.
 */
public class EmissionMaskExporter implements TextureExporter {
    private final int width;
    private final int height;
    private final byte[] tex;

    public EmissionMaskExporter(EmissionMaskTexture texture, double animationTime) {
        int w = texture.getWidth();
        int h = texture.getHeight();
        int frameH = h;
        int frame = 0;
        if (texture.animated) {
            frameH = Math.min(h, w);
            int frames = Math.max(1, h / Math.max(1, frameH));
            frame = Math.floorMod(AnimatedTextureExporter.frameAt(animationTime), frames);
        }
        this.width = w;
        this.height = frameH;
        this.tex = new byte[w * frameH * 4];
        int[] data = texture.getData();
        int index = 0;
        for (int y = 0; y < frameH; y++) {
            int row = (y + frame * frameH) * w;
            for (int x = 0; x < w; x++) {
                int argb = data[row + x];
                tex[index] = (byte) (argb >>> 16);
                tex[index + 1] = (byte) (argb >>> 8);
                tex[index + 2] = (byte) argb;
                tex[index + 3] = (byte) (argb >>> 24);
                index += 4;
            }
        }
    }

    @Override
    public int getWidth() {
        return width;
    }

    @Override
    public int getHeight() {
        return height;
    }

    @Override
    public byte[] getTexture() {
        return tex;
    }

    @Override
    public int textureHashCode() {
        return Arrays.hashCode(tex);
    }

    @Override
    public boolean equals(TextureExporter other) {
        return other instanceof EmissionMaskExporter
                && width == other.getWidth() && height == other.getHeight()
                && Arrays.equals(tex, ((EmissionMaskExporter) other).tex);
    }
}
