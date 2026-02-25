package dev.thatredox.chunkynative.opencl.renderer;

import dev.thatredox.chunkynative.common.export.AbstractSceneLoader;
import dev.thatredox.chunkynative.common.export.ResourcePalette;
import dev.thatredox.chunkynative.common.export.models.PackedAabbModel;
import dev.thatredox.chunkynative.common.export.models.PackedQuadModel;
import dev.thatredox.chunkynative.common.export.models.PackedTriangleModel;
import dev.thatredox.chunkynative.common.export.primitives.PackedBlock;
import dev.thatredox.chunkynative.common.export.primitives.PackedMaterial;
import dev.thatredox.chunkynative.common.export.primitives.PackedSun;
import dev.thatredox.chunkynative.common.export.texture.AbstractTextureLoader;
import dev.thatredox.chunkynative.common.state.SkyState;
import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.renderer.export.ClPackedResourcePalette;
import dev.thatredox.chunkynative.opencl.renderer.export.ClTextureLoader;
import dev.thatredox.chunkynative.opencl.renderer.scene.ClSky;
import dev.thatredox.chunkynative.opencl.util.ClFloatBuffer;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import dev.thatredox.chunkynative.util.FunctionCache;
import dev.thatredox.chunkynative.util.Reflection;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Fog;
import se.llbit.chunky.renderer.scene.FogLayer;
import se.llbit.chunky.renderer.scene.FogMode;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.SimplexWaterShader;
import se.llbit.chunky.renderer.WaterShadingStrategy;
import se.llbit.chunky.resources.Texture;
import se.llbit.chunky.world.ChunkPosition;
import se.llbit.chunky.world.Clouds;
import se.llbit.math.Vector3;
import se.llbit.math.Vector3i;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

public class ClSceneLoader extends AbstractSceneLoader {
    protected final FunctionCache<int[], ClIntBuffer> clWorldBvh;
    protected final FunctionCache<int[], ClIntBuffer> clActorBvh;
    protected final FunctionCache<PackedSun, ClIntBuffer> clPackedSun;
    protected ClSky clSky = null;
    protected SkyState skyState = null;

    protected ClIntBuffer octreeData = null;
    protected ClIntBuffer octreeDepth = null;
    protected ClIntBuffer waterOctreeData = null;
    protected ClIntBuffer waterOctreeDepth = null;
    protected ClIntBuffer emitterPositions = null;
    protected ClIntBuffer positionIndexes = null;
    protected ClIntBuffer constructedGrid = null;
    protected ClIntBuffer gridConfig = null; // cellSize, offsetX, sizeX, offsetY, sizeY, offsetZ, sizeZ

    // New buffers for expanded features
    protected ClFloatBuffer fogDataBuffer = null;
    protected ClFloatBuffer waterConfigBuffer = null;
    protected ClFloatBuffer renderConfigBuffer = null;
    protected ClIntBuffer cloudDataBuffer = null;
    protected ClFloatBuffer waterNormalMapBuffer = null;
    protected int waterNormalMapW = 0;
    protected ClIntBuffer biomeDataBuffer = null;
    protected int biomeDataSize = 0; // octreeSize for biome 2D grid
    protected ClIntBuffer chunkBitmapBuffer = null;
    protected int chunkBitmapSize = 0; // chunks per side (octreeSize / 16)

    // Cached data for change detection to avoid unnecessary GPU buffer recreation
    private float[] prevFogData = null;
    private float[] prevWaterData = null;
    private float[] prevRenderData = null;
    private int[] prevConstructedGrid = null;
    private int[] prevPosIndexes = null;
    private int[] prevEmitPos = null;
    private int[] prevCloudInts = null;
    private int[] prevChunkBitmap = null;
    private int prevChunkBitmapChunksPerSide = 0;

    // Generation counter incremented on each load() to detect stale kernel args
    private final AtomicInteger resetGeneration = new AtomicInteger(0);

    private final ClContext context;

    public ClSceneLoader(ClContext context) {
        this.context = context;
        this.clWorldBvh = new FunctionCache<>(i -> new ClIntBuffer(i, context), ClIntBuffer::close, null);
        this.clActorBvh = new FunctionCache<>(i -> new ClIntBuffer(i, context), ClIntBuffer::close, null);
        this.clPackedSun = new FunctionCache<>(i -> new ClIntBuffer(i, context), ClIntBuffer::close, null);
    }

    @Override
    public boolean ensureLoad(Scene scene) {
        return this.ensureLoad(scene, clSky == null);
    }

    @Override
    public boolean load(int modCount, ResetReason resetReason, Scene scene) {
        // Remember previous modCount to detect whether super.load() did real work.
        // super.load() now sets this.modCount = modCount on success, so if prevMod
        // differs from this.modCount after the call, actual work was done.
        int prevMod = this.modCount;
        boolean loadSuccess = super.load(modCount, resetReason, scene);

        // If modCount didn't change, super.load() early-returned (already up to date).
        // Skip all exports and do NOT increment resetGeneration — this prevents
        // unnecessarily breaking the render loop (e.g., when both the preview and
        // path tracing renderers call load() with the same resetCount).
        if (prevMod == this.modCount) {
            return loadSuccess;
        }

        // Actual work was done — increment generation so the render loop restarts
        // with the new data.  Only do this for non-trivial reasons (NONE/MODE_CHANGE
        // are already handled by super.load and don't need a render restart).
        if (resetReason != ResetReason.NONE && resetReason != ResetReason.MODE_CHANGE) {
            resetGeneration.incrementAndGet();
        }

        // Check if sky changed
        SkyState newSky = new SkyState(scene.sky(), scene.sun());
        if (!newSky.equals(skyState)) {
            if (clSky != null) clSky.close();
            clSky = new ClSky(scene, context);
            skyState = newSky;
            packedSun = new PackedSun(scene.sun(), getTexturePalette());
        }

        // Emitters: export grid directly from Grid's internal fields via reflection.
        // This avoids the expensive serialize→deserialize round-trip through DataOutputStream.
        // Only check on SCENE_LOADED since settings changes don't affect the emitter grid.
        try {
            if (resetReason == ResetReason.SCENE_LOADED && scene.getEmitterGrid() != null) {
                se.llbit.math.Grid grid = scene.getEmitterGrid();
                int cellSize = Reflection.getFieldValue(grid, "cellSize", Integer.class);
                int offsetX = Reflection.getFieldValue(grid, "offsetX", Integer.class);
                int sizeX = Reflection.getFieldValue(grid, "sizeX", Integer.class);
                int offsetY = Reflection.getFieldValue(grid, "offsetY", Integer.class);
                int sizeY = Reflection.getFieldValue(grid, "sizeY", Integer.class);
                int offsetZ = Reflection.getFieldValue(grid, "offsetZ", Integer.class);
                int sizeZ = Reflection.getFieldValue(grid, "sizeZ", Integer.class);
                int[] gridConstructed = Reflection.getFieldValue(grid, "constructedGrid", int[].class);
                int[] gridPosIndexes = Reflection.getFieldValue(grid, "positionIndexes", int[].class);
                @SuppressWarnings("unchecked")
                java.util.List<se.llbit.math.Grid.EmitterPosition> emitters =
                        Reflection.getFieldValue(grid, "emitterPositions", java.util.List.class);

                // Build emitter position array: 4 ints per emitter (x, y, z, avgFaceArea as float bits)
                int emitterNo = emitters.size();
                int[] emitPos = new int[emitterNo * 4];
                for (int i = 0; i < emitterNo; ++i) {
                    se.llbit.math.Grid.EmitterPosition ep = emitters.get(i);
                    emitPos[i*4 + 0] = ep.x;
                    emitPos[i*4 + 1] = ep.y;
                    emitPos[i*4 + 2] = ep.z;
                    // Compute average face surface area for emitter brightness correction.
                    // CPU uses per-face surface area; GPU approximates with the average.
                    // Full cubes: 6 faces × 1.0 area = avgArea 1.0 (no change).
                    // Non-cube emitters (torches, candles, etc.): avgArea < 1.0 (dimmer).
                    int fCount = ep.block.faceCount();
                    double totalArea = 0;
                    for (int f = 0; f < fCount; f++) {
                        totalArea += ep.block.surfaceArea(f);
                    }
                    float avgFaceArea = (float)(totalArea / fCount);
                    emitPos[i*4 + 3] = Float.floatToIntBits(avgFaceArea);
                }

                // Change detection: compare the already-flat arrays directly
                boolean changed = !Arrays.equals(gridConstructed, prevConstructedGrid)
                        || !Arrays.equals(gridPosIndexes, prevPosIndexes)
                        || !Arrays.equals(emitPos, prevEmitPos);

                if (changed) {
                    prevConstructedGrid = gridConstructed;
                    prevPosIndexes = gridPosIndexes;
                    prevEmitPos = emitPos;

                    if (emitterPositions != null) emitterPositions.close();
                    if (positionIndexes != null) positionIndexes.close();
                    if (constructedGrid != null) constructedGrid.close();
                    if (gridConfig != null) gridConfig.close();

                    emitterPositions = new ClIntBuffer(emitPos, context);
                    positionIndexes = new ClIntBuffer(gridPosIndexes, context);
                    constructedGrid = new ClIntBuffer(gridConstructed, context);
                    gridConfig = new ClIntBuffer(new int[] {cellSize, offsetX, sizeX, offsetY, sizeY, offsetZ, sizeZ}, context);
                }
            } else {
                if (prevConstructedGrid != null) {
                    prevConstructedGrid = null;
                    prevPosIndexes = null;
                    prevEmitPos = null;
                    if (emitterPositions != null) { emitterPositions.close(); emitterPositions = null; }
                    if (positionIndexes != null) { positionIndexes.close(); positionIndexes = null; }
                    if (constructedGrid != null) { constructedGrid.close(); constructedGrid = null; }
                    if (gridConfig != null) { gridConfig.close(); gridConfig = null; }
                }
            }
        } catch (Exception e) {
            // If emitter grid export fails, clear buffers
            prevConstructedGrid = null;
            prevPosIndexes = null;
            prevEmitPos = null;
            if (emitterPositions != null) { emitterPositions.close(); emitterPositions = null; }
            if (positionIndexes != null) { positionIndexes.close(); positionIndexes = null; }
            if (constructedGrid != null) { constructedGrid.close(); constructedGrid = null; }
            if (gridConfig != null) { gridConfig.close(); gridConfig = null; }
        }

        // Export fog, water, render config, cloud data, and water normal map buffers
        exportFogConfig(scene);
        exportWaterConfig(scene);
        exportRenderConfig(scene);
        exportCloudData();
        exportWaterNormalMap();
        exportBiomeData(scene);
        exportChunkBitmap(scene);

        return loadSuccess;
    }

    private void exportFogConfig(Scene scene) {
        // fogData layout: [mode, uniformDensity, skyFogDensity, fogR, fogG, fogB, numLayers,
        //                   fastFog (at index 7 + MAX_FOG_LAYERS * 3),
        //                   layer0_y, layer0_breadth, layer0_density, ...]
        // Layer Y is converted to octree-local space (y - origin.y).
        Fog fog = scene.fog;
        int mode;
        switch (fog.getFogMode()) {
            case UNIFORM: mode = 1; break;
            case LAYERED: mode = 2; break;
            default: mode = 0; break;
        }

        Vector3 fogColor = fog.getFogColor();
        ArrayList<FogLayer> layers = fog.getFogLayers();
        int numLayers = Math.min(layers.size(), 8);

        // Fixed layout: 7 header + MAX_LAYERS*3 layer data + 1 fastFog = 32
        float[] fogData = new float[7 + 8 * 3 + 1];
        fogData[0] = mode;
        fogData[1] = (float) fog.getUniformDensity();
        fogData[2] = (float) fog.getSkyFogDensity();
        fogData[3] = (float) fogColor.x;
        fogData[4] = (float) fogColor.y;
        fogData[5] = (float) fogColor.z;
        fogData[6] = numLayers;
        // fastFog flag at fixed position after layer data
        fogData[7 + 8 * 3] = fog.fastFog() ? 1.0f : 0.0f;
        for (int i = 0; i < numLayers; i++) {
            FogLayer layer = layers.get(i);
            // Convert layer Y from world-space to octree-local space
            fogData[8 + i * 3] = (float) (layer.y - scene.getOrigin().y);
            fogData[8 + i * 3 + 1] = (float) layer.breadth;
            fogData[8 + i * 3 + 2] = (float) layer.density;
        }

        if (!Arrays.equals(fogData, prevFogData)) {
            prevFogData = fogData;
            if (fogDataBuffer != null) fogDataBuffer.close();
            fogDataBuffer = new ClFloatBuffer(fogData, context);
        }
    }

    private void exportWaterConfig(Scene scene) {
        // waterConfig layout: [enabled, height, chunkClip, octreeSize, shadingStrategy,
        //                       animationTime, visibility, r, g, b, useCustomColor, ior,
        //                       shaderIterations, shaderFrequency, shaderAmplitude, shaderSpeed,
        //                       waterOpacity]
        int octreeSize = 1 << scene.getWorldOctree().getDepth();
        WaterShadingStrategy waterStrategy = scene.getWaterShadingStrategy();

        // Extract shader parameters from SimplexWaterShader if available
        int shaderIterations = 4;
        float shaderFrequency = 0.4f;
        float shaderAmplitude = 0.025f;
        float shaderSpeed = 1.0f;
        if (scene.getCurrentWaterShader() instanceof SimplexWaterShader) {
            SimplexWaterShader shader = (SimplexWaterShader) scene.getCurrentWaterShader();
            shaderIterations = shader.iterations;
            shaderFrequency = (float) shader.baseFrequency;
            shaderAmplitude = (float) shader.baseAmplitude;
            shaderSpeed = (float) shader.animationSpeed;
        }

        // Map WaterShadingStrategy to GPU constants: 0=STILL, 1=SIMPLEX, 2=TILED_NORMALMAP
        float gpuShadingStrategy;
        switch (waterStrategy) {
            case SIMPLEX:         gpuShadingStrategy = 1.0f; break;
            case TILED_NORMALMAP: gpuShadingStrategy = 2.0f; break;
            default:              gpuShadingStrategy = 0.0f; break; // STILL
        }

        Vector3 waterColor = scene.getWaterColor();
        float[] waterData = new float[17];
        waterData[0] = scene.isWaterPlaneEnabled() ? 1.0f : 0.0f;
        // Convert water plane height from world-space to octree-local-space.
        // The GPU kernel operates in octree-local coordinates where scene origin
        // has been subtracted from all positions (see ClCamera).
        waterData[1] = (float) (scene.getEffectiveWaterPlaneHeight() - scene.getOrigin().y);
        waterData[2] = scene.getWaterPlaneChunkClip() ? 1.0f : 0.0f;
        waterData[3] = (float) octreeSize;
        waterData[4] = gpuShadingStrategy;
        waterData[5] = (float) scene.getAnimationTime();
        waterData[6] = (float) scene.getWaterVisibility();
        waterData[7] = (float) waterColor.x;
        waterData[8] = (float) waterColor.y;
        waterData[9] = (float) waterColor.z;
        waterData[10] = scene.getUseCustomWaterColor() ? 1.0f : 0.0f;
        waterData[11] = 1.333f; // Water IOR
        waterData[12] = (float) shaderIterations;
        waterData[13] = shaderFrequency;
        waterData[14] = shaderAmplitude;
        waterData[15] = shaderSpeed;
        waterData[16] = (float) scene.getWaterOpacity();

        if (!Arrays.equals(waterData, prevWaterData)) {
            prevWaterData = waterData;
            if (waterConfigBuffer != null) waterConfigBuffer.close();
            waterConfigBuffer = new ClFloatBuffer(waterData, context);
        }
    }

    private void exportRenderConfig(Scene scene) {
        // renderConfig layout: [sunSamplingStrategy, transparentSky, branchCount,
        //                        cloudsEnabled, cloudHeight, cloudSize,
        //                        preventNormalEmitterWithSampling, strictDirectLight,
        //                        fancierTranslucency, transmissivityCap,
        //                        biomeColorsEnabled, cloudOffsetX, cloudOffsetZ]
        int sunStrategy = scene.getSunSamplingStrategy().ordinal();
        float[] renderData = new float[13];
        renderData[0] = (float) sunStrategy;
        renderData[1] = scene.transparentSky() ? 1.0f : 0.0f;
        renderData[2] = 0.0f; // branchCount is now passed via dynamicConfig[4] per-frame
        renderData[3] = scene.sky().cloudsEnabled() ? 1.0f : 0.0f;
        renderData[4] = (float) (scene.sky().cloudYOffset() - scene.getOrigin().y);
        double cloudSize = scene.sky().cloudSize();
        renderData[5] = (float) cloudSize;
        renderData[6] = scene.isPreventNormalEmitterWithSampling() ? 1.0f : 0.0f;
        renderData[7] = scene.getSunSamplingStrategy().isStrictDirectLight() ? 1.0f : 0.0f;
        renderData[8] = scene.getFancierTranslucency() ? 1.0f : 0.0f;
        renderData[9] = (float) scene.getTransmissivityCap();
        renderData[10] = scene.biomeColorsEnabled() ? 1.0f : 0.0f;
        // Pre-compute cloud grid offsets: gridX = worldX * inv_size + cloudOffsetX
        // CPU uses: gridX = (gpuX + origin.x) / cloudSize + cloudXOffset
        //         = gpuX / cloudSize + (origin.x / cloudSize + cloudXOffset)
        renderData[11] = (float) (scene.getOrigin().x / cloudSize + scene.sky().cloudXOffset());
        renderData[12] = (float) (scene.getOrigin().z / cloudSize + scene.sky().cloudZOffset());

        if (!Arrays.equals(renderData, prevRenderData)) {
            prevRenderData = renderData;
            if (renderConfigBuffer != null) renderConfigBuffer.close();
            renderConfigBuffer = new ClFloatBuffer(renderData, context);
        }
    }

    private void exportCloudData() {
        // Export the Minecraft cloud bitmap as a flat int array.
        // The CPU stores clouds as 32x32 longs (each 64 bits = 8x8 sub-grid).
        // We export as 2048 ints (2 ints per long: low 32 bits, high 32 bits).
        int[] cloudInts = new int[32 * 32 * 2];
        for (int tx = 0; tx < 32; tx++) {
            for (int tz = 0; tz < 32; tz++) {
                // Reconstruct the long from individual getCloud calls
                long val = 0;
                for (int sy = 0; sy < 8; sy++) {
                    for (int sx = 0; sx < 8; sx++) {
                        int bit = Clouds.getCloud(tx * 8 + sx, tz * 8 + sy);
                        val |= ((long)(bit & 1)) << (sy * 8 + sx);
                    }
                }
                int idx = (tx * 32 + tz) * 2;
                cloudInts[idx] = (int)(val & 0xFFFFFFFFL);
                cloudInts[idx + 1] = (int)(val >>> 32);
            }
        }
        if (!Arrays.equals(cloudInts, prevCloudInts)) {
            prevCloudInts = cloudInts;
            if (cloudDataBuffer != null) cloudDataBuffer.close();
            cloudDataBuffer = new ClIntBuffer(cloudInts, context);
        }
    }

    private void exportWaterNormalMap() {
        // Only export once - the water-height texture is static and never changes
        if (waterNormalMapBuffer != null) return;

        // Load the built-in water-height heightmap and precompute gradient normal map,
        // matching WaterModel.java's static initializer
        Texture waterHeight = new Texture("water-height");
        int w = waterHeight.getWidth();
        waterNormalMapW = w;
        float[] normalMap = new float[w * w * 2];
        for (int u = 0; u < w; u++) {
            for (int v = 0; v < w; v++) {
                float hx0 = (waterHeight.getColorWrapped(u, v) & 0xFF) / 255.0f;
                float hx1 = (waterHeight.getColorWrapped(u + 1, v) & 0xFF) / 255.0f;
                float hz0 = (waterHeight.getColorWrapped(u, v) & 0xFF) / 255.0f;
                float hz1 = (waterHeight.getColorWrapped(u, v + 1) & 0xFF) / 255.0f;
                normalMap[(u * w + v) * 2] = hx1 - hx0;       // dH/dx
                normalMap[(u * w + v) * 2 + 1] = hz1 - hz0;   // dH/dz
            }
        }
        waterNormalMapBuffer = new ClFloatBuffer(normalMap, context);
    }

    private void exportBiomeData(Scene scene) {
        // Only export once per scene load - biome data is static
        if (biomeDataBuffer != null) return;

        if (!scene.biomeColorsEnabled()) {
            // No biome colors: create a small placeholder buffer
            biomeDataSize = 0;
            biomeDataBuffer = new ClIntBuffer(new int[] {0}, context);
            return;
        }

        // Scene's biome textures (grassTexture, foliageTexture, etc.) are null until
        // loadChunks() has run. The get*Color() methods will NPE if called too early.
        // Test with getGrassColor — if it throws, biome data isn't ready yet; skip and
        // let the next call retry (biomeDataBuffer is still null so we'll try again).
        try {
            scene.getGrassColor(0, 0, 0);
        } catch (NullPointerException e) {
            return;
        }

        int octreeSize = 1 << scene.getWorldOctree().getDepth();
        // Cap export size to avoid excessive memory (2048^2 * 4 ints * 4 bytes = 64MB)
        if (octreeSize > 2048) {
            biomeDataSize = 0;
            biomeDataBuffer = new ClIntBuffer(new int[] {0}, context);
            return;
        }

        biomeDataSize = octreeSize;
        // Layout: 4 ints per (x,z) position: [grass, foliage, water, dryFoliage] as packed linear ARGB
        // Only iterate loaded chunks (typically far fewer than octreeSize²) and fill
        // the rest with a default biome color to avoid the massive O(octreeSize²) loop.
        int defaultBiome = packLinearRgb(scene.getGrassColor(0, 0, 0));
        int defaultFoliage = packLinearRgb(scene.getFoliageColor(0, 0, 0));
        int defaultWater = packLinearRgb(scene.getWaterColor(0, 0, 0));
        int defaultDryFoliage = packLinearRgb(scene.getDryFoliageColor(0, 0, 0));
        int[] biomeData = new int[octreeSize * octreeSize * 4];
        // Fill with defaults
        for (int i = 0; i < biomeData.length; i += 4) {
            biomeData[i]     = defaultBiome;
            biomeData[i + 1] = defaultFoliage;
            biomeData[i + 2] = defaultWater;
            biomeData[i + 3] = defaultDryFoliage;
        }
        // Only query biome colors for positions inside loaded chunks
        Vector3i origin = scene.getOrigin();
        for (ChunkPosition cp : scene.getChunks()) {
            int localBlockX = cp.x * 16 - origin.x;
            int localBlockZ = cp.z * 16 - origin.z;
            for (int dz = 0; dz < 16; dz++) {
                int z = localBlockZ + dz;
                if (z < 0 || z >= octreeSize) continue;
                for (int dx = 0; dx < 16; dx++) {
                    int x = localBlockX + dx;
                    if (x < 0 || x >= octreeSize) continue;
                    int base = (z * octreeSize + x) * 4;
                    float[] grass = scene.getGrassColor(x, 0, z);
                    float[] foliage = scene.getFoliageColor(x, 0, z);
                    float[] water = scene.getWaterColor(x, 0, z);
                    float[] dryFoliage = scene.getDryFoliageColor(x, 0, z);
                    biomeData[base]     = packLinearRgb(grass);
                    biomeData[base + 1] = packLinearRgb(foliage);
                    biomeData[base + 2] = packLinearRgb(water);
                    biomeData[base + 3] = packLinearRgb(dryFoliage);
                }
            }
        }
        biomeDataBuffer = new ClIntBuffer(biomeData, context);
    }

    private static int packLinearRgb(float[] rgb) {
        int r = Math.max(0, Math.min(255, (int)(rgb[0] * 255.0f + 0.5f)));
        int g = Math.max(0, Math.min(255, (int)(rgb[1] * 255.0f + 0.5f)));
        int b = Math.max(0, Math.min(255, (int)(rgb[2] * 255.0f + 0.5f)));
        return 0xFF000000 | (r << 16) | (g << 8) | b;
    }

    private void exportChunkBitmap(Scene scene) {
        int octreeSize = 1 << scene.getWorldOctree().getDepth();
        int chunksPerSide = octreeSize >> 4; // / 16
        if (chunksPerSide <= 0) {
            chunkBitmapSize = 0;
            if (chunkBitmapBuffer != null) chunkBitmapBuffer.close();
            chunkBitmapBuffer = new ClIntBuffer(new int[] {0}, context);
            return;
        }

        int totalChunks = chunksPerSide * chunksPerSide;
        int intsNeeded = (totalChunks + 31) >> 5; // / 32, round up
        int[] bitmap = new int[intsNeeded];

        // Scene origin in world-space (octree-local = world - origin)
        // ChunkPosition x,z are in world chunk coords (multiply by 16 for block coords)
        Vector3i origin = scene.getOrigin();
        int originBlockX = origin.x;
        int originBlockZ = origin.z;

        for (ChunkPosition cp : scene.getChunks()) {
            // Convert world chunk block coords to octree-local block coords
            int localBlockX = cp.x * 16 - originBlockX;
            int localBlockZ = cp.z * 16 - originBlockZ;
            // Convert to chunk index in octree-local space
            int cx = localBlockX >> 4; // / 16
            int cz = localBlockZ >> 4;
            if (cx >= 0 && cx < chunksPerSide && cz >= 0 && cz < chunksPerSide) {
                int bitIndex = cz * chunksPerSide + cx;
                bitmap[bitIndex >> 5] |= (1 << (bitIndex & 31));
            }
        }

        if (!Arrays.equals(bitmap, prevChunkBitmap) || chunksPerSide != prevChunkBitmapChunksPerSide) {
            prevChunkBitmap = bitmap;
            prevChunkBitmapChunksPerSide = chunksPerSide;
            chunkBitmapSize = chunksPerSide;
            if (chunkBitmapBuffer != null) chunkBitmapBuffer.close();
            chunkBitmapBuffer = new ClIntBuffer(bitmap, context);
        }
    }

    @Override
    protected boolean loadOctree(int[] octree, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette) {
        if (octreeData != null) octreeData.close();
        if (octreeDepth != null) octreeDepth.close();

        int[] mappedOctree = Arrays.stream(octree)
                .map(i -> i > 0 || -i >= blockMapping.length ? i : -blockMapping[-i])
                .toArray();
        octreeData = new ClIntBuffer(mappedOctree, context);
        octreeDepth = new ClIntBuffer(depth, context);

        return true;
    }

    @Override
    protected boolean loadWaterOctree(int[] waterOctree, int depth, int[] blockMapping, ResourcePalette<PackedBlock> blockPalette) {
        if (waterOctreeData != null) waterOctreeData.close();
        if (waterOctreeDepth != null) waterOctreeDepth.close();

        int[] mappedOctree = Arrays.stream(waterOctree)
                .map(i -> i > 0 || -i >= blockMapping.length ? i : -blockMapping[-i])
                .toArray();
        waterOctreeData = new ClIntBuffer(mappedOctree, context);
        waterOctreeDepth = new ClIntBuffer(depth, context);

        return true;
    }

    @Override
    protected AbstractTextureLoader createTextureLoader() {
        return new ClTextureLoader(context);
    }

    @Override
    protected ResourcePalette<PackedBlock> createBlockPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedMaterial> createMaterialPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedAabbModel> createAabbModelPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedQuadModel> createQuadModelPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    @Override
    protected ResourcePalette<PackedTriangleModel> createTriangleModelPalette() {
        return new ClPackedResourcePalette<>(context);
    }

    public ClIntBuffer getOctreeData() {
        assert octreeData != null;
        return octreeData;
    }

    public ClIntBuffer getWaterOctreeData() {
        assert waterOctreeData != null;
        return waterOctreeData;
    }

    public ClIntBuffer getEmitterPositions() {
        return emitterPositions;
    }

    public ClIntBuffer getPositionIndexes() {
        return positionIndexes;
    }

    public ClIntBuffer getConstructedGrid() {
        return constructedGrid;
    }

    public ClIntBuffer getGridConfig() {
        return gridConfig;
    }

    public ClIntBuffer getOctreeDepth() {
        assert octreeDepth != null;
        return octreeDepth;
    }

    public ClIntBuffer getWaterOctreeDepth() {
        assert waterOctreeDepth != null;
        return waterOctreeDepth;
    }

    public ClTextureLoader getTexturePalette() {
        assert texturePalette instanceof ClTextureLoader;
        return (ClTextureLoader) texturePalette;
    }

    public ClPackedResourcePalette<PackedBlock> getBlockPalette() {
        assert blockPalette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedBlock>) blockPalette;
    }

    public ClPackedResourcePalette<PackedMaterial> getMaterialPalette() {
        assert materialPalette.palette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedMaterial>) materialPalette.palette;
    }

    public ClPackedResourcePalette<PackedAabbModel> getAabbPalette() {
        assert aabbPalette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedAabbModel>) aabbPalette;
    }

    public ClPackedResourcePalette<PackedQuadModel> getQuadPalette() {
        assert quadPalette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedQuadModel>) quadPalette;
    }

    public ClPackedResourcePalette<PackedTriangleModel> getTrigPalette() {
        assert trigPalette instanceof ClPackedResourcePalette;
        return (ClPackedResourcePalette<PackedTriangleModel>) trigPalette;
    }

    public ClIntBuffer getWorldBvh() {
        return clWorldBvh.apply(this.worldBvh);
    }

    public ClIntBuffer getActorBvh() {
        return clActorBvh.apply(this.actorBvh);
    }

    public ClSky getSky() {
        assert clSky != null;
        return clSky;
    }

    public ClIntBuffer getSun() {
        return clPackedSun.apply(packedSun);
    }

    public ClFloatBuffer getFogData() {
        return fogDataBuffer;
    }

    public ClFloatBuffer getWaterConfig() {
        return waterConfigBuffer;
    }

    public ClFloatBuffer getRenderConfig() {
        return renderConfigBuffer;
    }

    public ClIntBuffer getCloudData() {
        return cloudDataBuffer;
    }

    public ClFloatBuffer getWaterNormalMap() {
        return waterNormalMapBuffer;
    }

    public int getWaterNormalMapWidth() {
        return waterNormalMapW;
    }

    public ClIntBuffer getBiomeData() {
        return biomeDataBuffer;
    }

    public int getBiomeDataSize() {
        return biomeDataSize;
    }

    public int getResetGeneration() {
        return resetGeneration.get();
    }

    public ClIntBuffer getChunkBitmap() {
        return chunkBitmapBuffer;
    }

    public int getChunkBitmapSize() {
        return chunkBitmapSize;
    }
}
