package dev.thatredox.chunkynative.opencl.renderer.map;

import javafx.scene.canvas.GraphicsContext;
import se.llbit.chunky.map.MapBuffer;
import se.llbit.chunky.map.WorldMapLoader;
import se.llbit.chunky.world.ChunkPosition;
import se.llbit.chunky.world.ChunkSelectionTracker;
import se.llbit.chunky.world.ChunkView;

import java.io.File;
import java.io.IOException;

/**
 * Wraps an existing {@link MapBuffer} and intercepts {@code drawBuffered()} to enable
 * GPU-accelerated pixel scaling.  All data operations (tile drawing, view updates) are
 * delegated unchanged to the original MapBuffer.
 */
public class GpuMapBuffer extends MapBuffer {
    private final MapBuffer delegate;
    private final GpuMapRenderer renderer;

    public GpuMapBuffer(MapBuffer delegate, GpuMapRenderer renderer) {
        super();
        this.delegate = delegate;
        this.renderer = renderer;
    }

    /** The delegate MapBuffer, for reflection access to its private fields. */
    MapBuffer getDelegate() {
        return delegate;
    }

    @Override
    public void updateView(ChunkView newView) {
        delegate.updateView(newView);
    }

    @Override
    public ChunkView getView() {
        return delegate.getView();
    }

    @Override
    public synchronized void drawTile(WorldMapLoader mapLoader, ChunkPosition chunk,
                                       ChunkSelectionTracker selection) {
        delegate.drawTile(mapLoader, chunk, selection);
    }

    @Override
    public synchronized void drawTileCached(WorldMapLoader mapLoader, ChunkPosition chunk,
                                             ChunkSelectionTracker selection) {
        delegate.drawTileCached(mapLoader, chunk, selection);
    }

    @Override
    public synchronized void redrawView(WorldMapLoader mapLoader,
                                         ChunkSelectionTracker selection) {
        delegate.redrawView(mapLoader, selection);
    }

    @Override
    public void copyPixels(int[] data, int srcPos, int x, int z, int size) {
        delegate.copyPixels(data, srcPos, x, z, size);
    }

    @Override
    public synchronized void drawBuffered(GraphicsContext gc) {
        // GPU-accelerated path; falls back to CPU delegate on error
        if (!renderer.gpuDrawBuffered(gc, delegate)) {
            delegate.drawBuffered(gc);
        }
    }

    @Override
    public synchronized void clearBuffer() {
        delegate.clearBuffer();
    }

    @Override
    public synchronized void renderPng(File targetFile) throws IOException {
        delegate.renderPng(targetFile);
    }
}
