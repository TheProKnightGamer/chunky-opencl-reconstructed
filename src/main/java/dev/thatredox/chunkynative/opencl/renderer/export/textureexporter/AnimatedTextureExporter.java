package dev.thatredox.chunkynative.opencl.renderer.export.textureexporter;

import se.llbit.chunky.resources.AnimatedTexture;

import java.util.Arrays;

/**
 * Texture exporter for AnimatedTexture that extracts only the first frame.
 * Animated textures store a vertical strip of frames; the GPU needs a single frame.
 */
public class AnimatedTextureExporter implements TextureExporter {
    protected final AnimatedTexture texture;
    private final int frameWidth;
    private final int frameHeight;

    public AnimatedTextureExporter(AnimatedTexture texture) {
        this.texture = texture;
        this.frameWidth = texture.getWidth();
        // Frame height is the same as width for square frames (Minecraft convention)
        this.frameHeight = Math.min(texture.getHeight(), texture.getWidth());
    }

    @Override
    public int getWidth() {
        return frameWidth;
    }

    @Override
    public int getHeight() {
        return frameHeight;
    }

    @Override
    public byte[] getTexture() {
        byte[] out = new byte[frameHeight * frameWidth * 4];
        int index = 0;
        for (int y = 0; y < frameHeight; y++) {
            for (int x = 0; x < frameWidth; x++) {
                // getColor(x, y) reads pixel coordinates from the full bitmap;
                // only read from the first frame (y < frameHeight)
                float[] rgba = texture.getColor(x, y);
                out[index] = (byte) (rgba[0] * 255.0f);
                out[index + 1] = (byte) (rgba[1] * 255.0f);
                out[index + 2] = (byte) (rgba[2] * 255.0f);
                out[index + 3] = (byte) (rgba[3] * 255.0f);
                index += 4;
            }
        }
        return out;
    }

    @Override
    public int textureHashCode() {
        // Hash only the first frame's pixel data
        return Arrays.hashCode(getTexture());
    }

    @Override
    public boolean equals(TextureExporter other) {
        if (this.getWidth() != other.getWidth()) return false;
        if (this.getHeight() != other.getHeight()) return false;
        if (other instanceof AnimatedTextureExporter) {
            return Arrays.equals(this.getTexture(), other.getTexture());
        }
        return false;
    }
}
