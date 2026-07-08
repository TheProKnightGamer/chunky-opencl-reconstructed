package dev.thatredox.chunkynative.opencl.renderer.export.textureexporter;

import se.llbit.chunky.resources.AnimatedTexture;

import java.util.Arrays;

/**
 * Texture exporter for AnimatedTexture. The GPU atlas only holds one frame
 * per texture so we pick the frame matching the scene's current animation
 * time. Animation only updates on scene reset (matches GPU per-render
 * snapshot semantics, and chunky's CPU AnimatedQuadModel sampling formula).
 *
 * <p>The framerate isn't carried by the texture itself — it's set per-model
 * in {@code AnimatedQuadModel} (FireModel uses 20). We use the same default
 * here. Most other animated textures (water surface, lava, magma block) are
 * sampled by their models without specifying a frame, so they fall back to
 * frame 0 on the CPU too — frame 0 is already the right answer for them
 * regardless of animationTime.
 */
public class AnimatedTextureExporter implements TextureExporter {
    /**
     * Frames-per-second matching chunky's FireModel (the only AnimatedQuadModel
     * subclass at time of writing). Other animated textures default to frame 0
     * on the CPU as well, so this constant only changes behaviour for fire.
     */
    private static final int DEFAULT_FRAMERATE = 20;

    protected final AnimatedTexture texture;
    private final int frameWidth;
    private final int frameHeight;
    private final int frameIndex;

    public AnimatedTextureExporter(AnimatedTexture texture) {
        this(texture, 0);
    }

    public AnimatedTextureExporter(AnimatedTexture texture, double animationTime) {
        this.texture = texture;
        this.frameWidth = texture.getWidth();
        // Frame height is the same as width for square frames (Minecraft convention)
        this.frameHeight = Math.min(texture.getHeight(), texture.getWidth());
        int numFrames = Math.max(1, texture.getHeight() / Math.max(1, this.frameHeight));
        int rawFrame = (int) (animationTime * DEFAULT_FRAMERATE);
        this.frameIndex = Math.floorMod(rawFrame, numFrames);
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
        int yOffset = frameIndex * frameHeight;
        for (int y = 0; y < frameHeight; y++) {
            for (int x = 0; x < frameWidth; x++) {
                // The animated bitmap is a vertical strip of frames; offset y
                // into the requested frame.
                float[] rgba = texture.getColor(x, y + yOffset);
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
