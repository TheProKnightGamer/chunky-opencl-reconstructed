package dev.thatredox.chunkynative.opencl.context;

import dev.thatredox.chunkynative.opencl.renderer.ClSceneLoader;
import org.jocl.CLException;
import org.jocl.cl_program;
import se.llbit.log.Log;

import java.util.concurrent.CompletableFuture;

public class ContextManager {
    public final Device device;
    public final ClContext context;
    public final Tonemap tonemap;
    public final Renderer renderer;

    public final ClSceneLoader sceneLoader;

    private static volatile ContextManager instance = null;

    private ContextManager(Device device) {
        this.device = device;
        this.context = new ClContext(device);
        // Compile both kernels in parallel — they're independent programs
        CompletableFuture<Tonemap> tonemapFuture = CompletableFuture.supplyAsync(() -> new Tonemap(context));
        CompletableFuture<Renderer> rendererFuture = CompletableFuture.supplyAsync(() -> new Renderer(context));
        this.tonemap = tonemapFuture.join();
        this.renderer = rendererFuture.join();
        this.sceneLoader = new ClSceneLoader(context);
    }

    /**
     * Get the context manager, blocking until initialization is complete if needed.
     */
    public static ContextManager get() {
        if (instance == null) {
            synchronized (ContextManager.class) {
                if (instance == null) {
                    instance = new ContextManager(Device.getPreferredDevice());
                }
            }
        }
        return instance;
    }

    /**
     * Non-blocking check: returns the context manager if already initialized, or null.
     * Use this in render paths to avoid blocking the render thread while kernels compile.
     */
    public static ContextManager tryGet() {
        return instance;
    }

    public static synchronized void setDevice(Device device) {
        try {
            instance = new ContextManager(device);
        } catch (CLException e) {
            Log.error("Failed to set device", e);
        }
    }

    public static synchronized void reload() {
        if (instance != null) {
            setDevice(instance.device);
        } else {
            // Force initialization on reload
            get();
        }
    }

    public static class Tonemap {
        public final cl_program simpleFilter;

        private Tonemap(ClContext context) {
            this.simpleFilter = KernelLoader.loadProgram(context, "tonemap", "post_processing_filter.c");
        }
    }

    public static class Renderer {
        public final cl_program kernel;

        private Renderer(ClContext context) {
            this.kernel = KernelLoader.loadProgram(context, "kernel", "rayTracer.c");
        }
    }
}
