package dev.thatredox.chunkynative.opencl.context;

import dev.thatredox.chunkynative.opencl.renderer.ClSceneLoader;
import org.jocl.CLException;
import org.jocl.cl_program;
import se.llbit.log.Log;

import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public class ContextManager {
    public final Device device;
    public final ClContext context;
    public final Tonemap tonemap;
    public final Renderer renderer;

    public final ClSceneLoader sceneLoader;

    private static volatile ContextManager instance = null;

    /** Latch that is released once background initialization is complete (or failed). */
    private static final CountDownLatch initLatch = new CountDownLatch(1);
    /** If background init failed, the exception is stored here. */
    private static final AtomicReference<Exception> initError = new AtomicReference<>(null);
    /** Guard: ensures only one init thread is started. */
    private static final AtomicBoolean initStarted = new AtomicBoolean(false);

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

    private void close() {
        try {
            if (tonemap != null && tonemap.simpleFilter != null)
                org.jocl.CL.clReleaseProgram(tonemap.simpleFilter);
            if (renderer != null && renderer.kernel != null)
                org.jocl.CL.clReleaseProgram(renderer.kernel);
            if (context != null) {
                if (context.queue != null)
                    org.jocl.CL.clReleaseCommandQueue(context.queue);
                if (context.context != null)
                    org.jocl.CL.clReleaseContext(context.context);
            }
        } catch (Exception e) {
            // Swallow errors during shutdown
        }
    }



    /**
     * Kick off background initialization without blocking the caller.
     * Safe to call multiple times — only the first call triggers init.
     * Subsequent calls are no-ops.
     *
     * Logs periodic progress to the Chunky log panel so the user always
     * sees that compilation is happening, even before the render thread starts.
     */
    public static void initAsync() {
        if (instance != null) return;
        if (!initStarted.compareAndSet(false, true)) return;  // another thread already started init

        final long startTime = System.currentTimeMillis();
        Log.info("ChunkyCL: Compiling OpenCL kernels...");

        // Timer logs every 3 seconds while compilation is running.
        // Runs on a daemon thread so it won't prevent JVM shutdown.
        Timer progressTimer = new Timer("ChunkyCL-CompileProgress", true);
        progressTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (instance != null) {
                    cancel();
                    return;
                }
            }
        }, 3000, 3000);

        // Release OpenCL resources on JVM shutdown
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            ContextManager mgr = instance;
            if (mgr != null) mgr.close();
        }, "ChunkyCL-Shutdown"));

        Thread initThread = new Thread(() -> {
            try {
                ContextManager mgr = new ContextManager(Device.getPreferredDevice());
                instance = mgr;
                long elapsed = (System.currentTimeMillis() - startTime) / 1000;
                Log.info(String.format("ChunkyCL: OpenCL initialization complete (%ds)", elapsed));
            } catch (Exception e) {
                initError.set(e);
                Log.error("ChunkyCL: Background OpenCL initialization failed.", e);
            } finally {
                progressTimer.cancel();
                initLatch.countDown();
            }
        }, "ChunkyCL-Init");
        initThread.setDaemon(true);
        initThread.start();
    }

    /**
     * Get the context manager, blocking until initialization is complete if needed.
     * If initAsync() was called earlier, this will wait for that to finish.
     * Otherwise triggers synchronous initialization.
     */
    public static ContextManager get() {
        if (instance != null) return instance;

        // If async init was started, wait for it
        try {
            initLatch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted waiting for OpenCL initialization", e);
        }

        Exception err = initError.get();
        if (err != null) {
            throw new RuntimeException("OpenCL initialization failed", err);
        }

        // Fallback: if initAsync was never called, init synchronously
        if (instance == null) {
            synchronized (ContextManager.class) {
                if (instance == null) {
                    instance = new ContextManager(Device.getPreferredDevice());
                }
            }
        }
        return instance;
    }

    public static synchronized void setDevice(Device device) {
        try {
            ContextManager old = instance;
            instance = new ContextManager(device);
            if (old != null) old.close();
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
