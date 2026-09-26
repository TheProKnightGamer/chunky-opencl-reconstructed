package dev.thatredox.chunkynative.opencl.renderer.map;

import javafx.scene.canvas.GraphicsContext;
import se.llbit.chunky.map.MapBuffer;
import se.llbit.chunky.map.WorldMapLoader;
import se.llbit.chunky.world.ChunkPosition;
import se.llbit.chunky.world.ChunkSelectionTracker;
import se.llbit.chunky.world.ChunkView;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Wraps an existing {@link MapBuffer} and intercepts {@code drawBuffered()} to enable
 * GPU-accelerated pixel scaling.  All data operations (tile drawing, view updates) are
 * delegated unchanged to the original MapBuffer.
 *
 * <p>The wrapper also tracks how the delegate's pixel content evolves so the GPU path
 * can skip redundant work. ChunkMap reaches the delegate exclusively through this
 * wrapper, so every mutation of the delegate's pixel buffer passes through here:
 * <ul>
 *   <li>{@code contentVersion} moves whenever the pixel content may have changed,
 *       letting the renderer skip re-uploading the source buffer when only the
 *       sub-chunk pan offset moved.</li>
 *   <li>{@code redrawView()} is skipped entirely when the tile grid is unchanged
 *       (a pan that stays within the same chunk column re-blits byte-identical
 *       pixels in the stock implementation — pure waste on the JavaFX thread).</li>
 * </ul>
 */
public class GpuMapBuffer extends MapBuffer {
    private final MapBuffer delegate;
    private final GpuMapRenderer renderer;

    // Bumped after any mutation of the delegate's pixel content. Bumps happen
    // AFTER the delegate call completes so a concurrent reader can never
    // observe the new version with the old pixels (the reverse — old version,
    // new pixels — only causes one redundant re-upload, never staleness).
    private final AtomicLong contentVersion = new AtomicLong();

    // Whether redrawView() has real work to do. The stock redrawView re-blits
    // every visible tile into the pixel buffer, but the result only differs
    // when the tile grid moved (view origin/zoom crossed a chunk boundary) or
    // tiles were invalidated. Both are detectable here.
    private boolean redrawNeeded = true;

    // Snapshot of the view fields that define the tile grid and tile content.
    private boolean viewKeyValid = false;
    private int keyScale, keyChunkScale;
    private int keyPx0, keyPz0, keyPx1, keyPz1;
    private int keyPrx0, keyPrz0, keyPrx1, keyPrz1;
    private int keyYMin, keyYMax;

    public GpuMapBuffer(MapBuffer delegate, GpuMapRenderer renderer) {
        super();
        this.delegate = delegate;
        this.renderer = renderer;
    }

    /** The delegate MapBuffer, for reflection access to its private fields. */
    MapBuffer getDelegate() {
        return delegate;
    }

    /** Current content version, read by the renderer to decide on re-uploads. */
    long contentVersion() {
        return contentVersion.get();
    }

    @Override
    public synchronized void updateView(ChunkView newView) {
        if (!viewKeyValid
                || newView.scale != keyScale || newView.chunkScale != keyChunkScale
                || newView.px0 != keyPx0 || newView.pz0 != keyPz0
                || newView.px1 != keyPx1 || newView.pz1 != keyPz1
                || newView.prx0 != keyPrx0 || newView.prz0 != keyPrz0
                || newView.prx1 != keyPrx1 || newView.prz1 != keyPrz1
                || newView.yMin != keyYMin || newView.yMax != keyYMax) {
            redrawNeeded = true;
            keyScale = newView.scale;
            keyChunkScale = newView.chunkScale;
            keyPx0 = newView.px0;
            keyPz0 = newView.pz0;
            keyPx1 = newView.px1;
            keyPz1 = newView.pz1;
            keyPrx0 = newView.prx0;
            keyPrz0 = newView.prz0;
            keyPrx1 = newView.prx1;
            keyPrz1 = newView.prz1;
            keyYMin = newView.yMin;
            keyYMax = newView.yMax;
            viewKeyValid = true;
        }
        delegate.updateView(newView);
        if (redrawNeeded) {
            contentVersion.incrementAndGet();
        }
    }

    @Override
    public ChunkView getView() {
        return delegate.getView();
    }

    @Override
    public synchronized void drawTile(WorldMapLoader mapLoader, ChunkPosition chunk,
                                       ChunkSelectionTracker selection) {
        delegate.drawTile(mapLoader, chunk, selection);
        contentVersion.incrementAndGet();
    }

    @Override
    public synchronized void drawTileCached(WorldMapLoader mapLoader, ChunkPosition chunk,
                                             ChunkSelectionTracker selection) {
        delegate.drawTileCached(mapLoader, chunk, selection);
        contentVersion.incrementAndGet();
    }

    @Override
    public synchronized void redrawView(WorldMapLoader mapLoader,
                                         ChunkSelectionTracker selection) {
        if (redrawNeeded) {
            try {
                delegate.redrawView(mapLoader, selection);
                // Cleared only on success. Chunky's tile draw reads the world
                // (getChunk / getRegionWithinRange), so it can throw on an I/O
                // error or a dimension swapped out mid-draw. Clearing the flag up
                // front would strand the never-drawn tiles: updateView reallocates
                // pixels to a ZEROED array when the grid resizes, so every later
                // redraw would be skipped and the map would stay permanently
                // black. Stock chunky just retries on the next repaint; so do we.
                redrawNeeded = false;
            } finally {
                // Bump even on failure: a partial draw did mutate pixels, and the
                // GPU cache must not keep serving the pre-draw image.
                contentVersion.incrementAndGet();
            }
        } else {
            // Pure pan within the same tile grid: the delegate's pixels are
            // already correct, so the full-view tile re-blit is skipped. The
            // delegate's cached-image flag must still be dropped so a CPU
            // fallback re-scales at the new pan offset instead of blitting a
            // stale image.
            renderer.invalidateDelegateImageCache(delegate);
        }
    }

    @Override
    public void copyPixels(int[] data, int srcPos, int x, int z, int size) {
        delegate.copyPixels(data, srcPos, x, z, size);
        contentVersion.incrementAndGet();
    }

    @Override
    public synchronized void drawBuffered(GraphicsContext gc) {
        // GPU-accelerated path; falls back to CPU delegate on error
        if (!renderer.gpuDrawBuffered(gc, this)) {
            delegate.drawBuffered(gc);
        }
    }

    @Override
    public synchronized void clearBuffer() {
        delegate.clearBuffer();
        redrawNeeded = true;
        contentVersion.incrementAndGet();
    }

    @Override
    public synchronized void renderPng(File targetFile) throws IOException {
        delegate.renderPng(targetFile);
    }
}
