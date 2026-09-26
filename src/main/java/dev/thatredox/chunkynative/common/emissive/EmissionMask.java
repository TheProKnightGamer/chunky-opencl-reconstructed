package dev.thatredox.chunkynative.common.emissive;

/**
 * An 8-bit emission mask converted from a resource pack's emissive layer.
 * 0 means the texel does not emit, 255 means it emits at full strength.
 * Animated textures are vertical strips of square frames, like their colour textures.
 */
public final class EmissionMask {
    public final int width;
    public final int height;
    /** Row-major, {@code width * height} bytes. */
    public final byte[] data;

    public EmissionMask(int width, int height, byte[] data) {
        if (data.length != width * height) {
            throw new IllegalArgumentException("Mask data does not match its size");
        }
        this.width = width;
        this.height = height;
        this.data = data;
    }

    private int frames() {
        return height > width ? Math.max(1, height / width) : 1;
    }

    /**
     * Resample this mask onto a colour texture of {@code w * h} texels made of
     * {@code colorFrames} stacked frames. Colour frame {@code f} takes mask frame
     * {@code f % maskFrames}, so a single-frame mask repeats over an animation and a
     * colour texture that only keeps frame 0 gets the mask's frame 0. Downscaling
     * averages, upscaling repeats texels.
     */
    public byte[] resample(int w, int h, int colorFrames) {
        int maskFrames = frames();
        int maskFrameH = height / maskFrames;
        int colorFrameH = Math.max(1, h / Math.max(1, colorFrames));
        byte[] out = new byte[w * h];
        for (int y = 0; y < h; y++) {
            int frame = Math.min(y / colorFrameH, colorFrames - 1);
            int fy = y - frame * colorFrameH;
            int srcFrameY = (frame % maskFrames) * maskFrameH;
            int y0 = srcFrameY + (int) ((long) fy * maskFrameH / colorFrameH);
            int y1 = Math.max(y0 + 1, srcFrameY + (int) ((long) (fy + 1) * maskFrameH / colorFrameH));
            for (int x = 0; x < w; x++) {
                int x0 = (int) ((long) x * width / w);
                int x1 = Math.max(x0 + 1, (int) ((long) (x + 1) * width / w));
                int sum = 0;
                int n = 0;
                for (int sy = y0; sy < y1; sy++) {
                    for (int sx = x0; sx < x1; sx++) {
                        sum += data[sy * width + sx] & 0xFF;
                        n++;
                    }
                }
                out[y * w + x] = (byte) ((sum + n / 2) / n);
            }
        }
        return out;
    }

    static boolean isEmpty(byte[] mask) {
        for (byte b : mask) {
            if (b != 0) return false;
        }
        return true;
    }
}
