package dev.thatredox.chunkynative.opencl;

import dev.thatredox.chunkynative.opencl.context.ContextManager;
import dev.thatredox.chunkynative.opencl.renderer.OpenClPathTracingRenderer;
import dev.thatredox.chunkynative.opencl.renderer.OpenClPreviewRenderer;
import dev.thatredox.chunkynative.opencl.tonemap.ChunkyImposterGpuPostProcessingFilter;
import dev.thatredox.chunkynative.opencl.tonemap.UE4ToneMappingImposterGpuPostprocessingFilter;
import dev.thatredox.chunkynative.opencl.ui.ChunkyClTab;
import dev.thatredox.chunkynative.opencl.ui.OpenClSettingsLocker;
import se.llbit.chunky.Plugin;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.renderer.scene.SceneFactory;
import se.llbit.chunky.renderer.DefaultRenderManager;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.model.BlockModel;
import se.llbit.chunky.renderer.postprocessing.PostProcessingFilters;
import se.llbit.chunky.renderer.postprocessing.UE4ToneMappingFilter;
import se.llbit.chunky.ui.ChunkyFx;
import se.llbit.chunky.ui.render.RenderControlsTab;
import se.llbit.chunky.ui.render.RenderControlsTabTransformer;
import se.llbit.log.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * This plugin changes the Chunky path tracing renderer to a gpu based path tracer.
 */
public class ChunkyCl implements Plugin {
    @Override
    public void attach(Chunky chunky) {
        // Check if we have block models
        try {
            Class<?> test = BlockModel.class;
        } catch (NoClassDefFoundError e) {
            Log.error("ChunkyCL requires Chunky 2.5.0. Could not load block models.", e);
            return;
        }

        // Verify the OpenCL native library is loadable before registering renderers.
        try {
            Class.forName("org.jocl.CL");
        } catch (UnsatisfiedLinkError | ClassNotFoundException e) {
            Log.error("Failed to load ChunkyCL. Could not load OpenCL native library.", e);
            return;
        }

        // Initialize OpenCL context and compile kernels before registering renderers.
        // The first launch compiles from source (may take a few minutes) and caches
        // the binary to disk. Subsequent launches load the cache near-instantly.
        Log.info("ChunkyCL: Initializing OpenCL context and compiling kernels...");
        try {
            ContextManager.get();
            Log.info("ChunkyCL: OpenCL context initialized successfully.");
        } catch (Exception e) {
            Log.error("Failed to initialize ChunkyCL OpenCL context.", e);
            return;
        }

        Chunky.addRenderer(new OpenClPathTracingRenderer());
        Chunky.addPreviewRenderer(new OpenClPreviewRenderer());

        // Make OpenCL renderer the default for new scenes and set current scene to use it
        chunky.setSceneFactory(new SceneFactory() {
            @Override public se.llbit.chunky.renderer.scene.Scene newScene() {
                se.llbit.chunky.renderer.scene.Scene s = new se.llbit.chunky.renderer.scene.Scene();
                s.setRenderer("ChunkyClRenderer");
                s.setPreviewRenderer("ChunkyClPreviewRenderer");
                return s;
            }

            @Override public se.llbit.chunky.renderer.scene.Scene copyScene(se.llbit.chunky.renderer.scene.Scene scene) {
                se.llbit.chunky.renderer.scene.Scene s = new se.llbit.chunky.renderer.scene.Scene(scene);
                s.setRenderer("ChunkyClRenderer");
                s.setPreviewRenderer("ChunkyClPreviewRenderer");
                return s;
            }
        });

        // Update current scene to use OpenCL renderers if available
        try {
            se.llbit.chunky.renderer.scene.Scene current = chunky.getSceneManager().getScene();
            current.setRenderer("ChunkyClRenderer");
            current.setPreviewRenderer("ChunkyClPreviewRenderer");
            try {
                // Force a redraw so the preview uses the newly selected preview renderer immediately
                Object rm = chunky.getRenderController().getRenderManager();
                if (rm instanceof DefaultRenderManager) {
                    ((DefaultRenderManager) rm).redrawScreen();
                }
            } catch (Exception ignored) {}
        } catch (Exception ignored) {}

        RenderControlsTabTransformer prev = chunky.getRenderControlsTabTransformer();
        chunky.setRenderControlsTabTransformer(tabs -> {
            // First, call the previous transformer (this allows other plugins to work).
            List<RenderControlsTab> transformed = new ArrayList<>(prev.apply(tabs));

            // Wrap existing Lighting / Advanced tabs to lock unsupported settings
            // when the OpenCL renderer is active
            transformed = new ArrayList<>(OpenClSettingsLocker.wrapTabs(transformed));

            // Add the new tab
            transformed.add(new ChunkyClTab(chunky.getSceneManager().getScene()));

            return transformed;
        });

        addImposterFilter("NONE", ChunkyImposterGpuPostProcessingFilter.Filter.NONE);
        addImposterFilter("GAMMA", ChunkyImposterGpuPostProcessingFilter.Filter.GAMMA);
        addImposterFilter("TONEMAP1", ChunkyImposterGpuPostProcessingFilter.Filter.TONEMAP1);
        addImposterFilter("TONEMAP2", ChunkyImposterGpuPostProcessingFilter.Filter.ACES);
        addImposterFilter("TONEMAP3", ChunkyImposterGpuPostProcessingFilter.Filter.HABLE);

        PostProcessingFilters.getPostProcessingFilterFromId("UE4_FILMIC").ifPresent(filter ->
                PostProcessingFilters.addPostProcessingFilter(new UE4ToneMappingImposterGpuPostprocessingFilter((UE4ToneMappingFilter) filter)));
    }

    private static void addImposterFilter(String id, ChunkyImposterGpuPostProcessingFilter.Filter f) {
        PostProcessingFilters.getPostProcessingFilterFromId(id).ifPresent(filter ->
                PostProcessingFilters.addPostProcessingFilter(new ChunkyImposterGpuPostProcessingFilter(filter, f))
        );
    }

    public static void main(String[] args) throws Exception {
        // Start Chunky normally with this plugin attached.
        Chunky.loadDefaultTextures();
        Chunky chunky = new Chunky(ChunkyOptions.getDefaults());
        new ChunkyCl().attach(chunky);
        ChunkyFx.startChunkyUI(chunky);
    }
}
