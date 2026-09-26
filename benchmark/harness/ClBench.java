import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.context.Device;
import dev.thatredox.chunkynative.opencl.renderer.ClSceneLoader;
import dev.thatredox.chunkynative.opencl.renderer.scene.ClCamera;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import org.jocl.*;
import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.main.ChunkyOptions;
import se.llbit.chunky.main.CommandLineOptions;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.SynchronousSceneManager;
import se.llbit.log.Level;
import se.llbit.log.Log;
import se.llbit.log.Receiver;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.regex.*;

import static org.jocl.CL.*;

/**
 * A/B benchmark for the ChunkyCL render kernel. See README.md next to this file.
 *
 * <p>Loads a real Chunky scene through Chunky's own headless path, exports it with the
 * plugin's ClSceneLoader, then builds each kernel VARIANT (a kernel/include directory,
 * optionally with extra build options) and times the render kernel on it. Variants are
 * interleaved round-robin so background GPU load affects them all equally. Every variant
 * then renders the same seeds and is compared pixel-for-pixel against the first one.
 *
 * <p>Args (key=value):
 * <pre>
 *   sceneDir=... scene=...          scene to load (a Chunky scenes directory + name)
 *   variants=V1;V2;...              V = includeDir[|extra build options[|key=val,...]]
 *                                   per-variant keys: wg=N (local size), global=N, matCache=N, diag=1
 *   names=a,b,...                   display names for the variants
 *   ipl=N passes=N rounds=N slices=N wg=N global=N   dispatch shape (defaults 1/4/3/1/0/262144)
 *   width=/height= branch= rayDepth= emitters=0|1 emitterStrategy=N sunStrategy=NAME
 *   refSpp=N out=dir                image-comparison frames; write PNG + raw floats to dir
 *   preview=1                       also run + compare the preview kernel
 *   compileOnly=1                   only build variants into the binary cache (1 core, no GPU work)
 *   cache=dir                       binary cache (keyed by inlined source + options)
 * </pre>
 * A variant with diag=1 must take a trailing {@code __global uint*} argument; the kernel
 * accumulates (active lanes, warp count) pairs per trace site into it.
 */
public class ClBench {
    static final Pattern INCLUDE = Pattern.compile("^\\h*#include\\h*\"([^\"]+)\"\\h*$", Pattern.MULTILINE);
    static final String BASE_OPTS = "-cl-std=CL1.2 -Werror -cl-mad-enable -cl-no-signed-zeros";
    static final int RENDER_ARGS = 44;

    static final Map<String, String> A = new HashMap<>();
    // Everything that owns cl_mem handles is pinned here: the plugin's NativeCleaner
    // releases a buffer as soon as its Java wrapper becomes unreachable, and a local
    // only used to set kernel args is dead to the JIT right after that.
    static final List<Object> KEEP_ALIVE = new ArrayList<>();

    static String arg(String k, String d) { return A.getOrDefault(k, d); }
    static int iarg(String k, int d) { return Integer.parseInt(arg(k, Integer.toString(d))); }

    static class Variant {
        String name, dir, opts;
        long wg = -1;         // local size override (-1 = global setting)
        int matCache = -1;    // LDS material cache words override (-1 = global setting)
        boolean diag = false; // kernel takes a trailing SIMT-diagnostic buffer
        long global = -1;     // global work size cap override (-1 = global setting)
        cl_program prog;
        cl_kernel kernel;
        final List<Double> msPerPass = new ArrayList<>();
        float[] image;
    }

    public static void main(String[] args) {
        int code = 0;
        try {
            run(args);
        } catch (Throwable t) {
            t.printStackTrace();
            code = 1;
        }
        // Chunky's worker pools are non-daemon threads: never let a failure leave the JVM
        // (and any GPU lock held around it) hanging.
        System.exit(code);
    }

    static List<Variant> parseVariants() {
        List<Variant> variants = new ArrayList<>();
        String[] names = A.containsKey("names") ? arg("names", "").split(",") : new String[0];
        for (String v : arg("variants", "").split(";")) {
            if (v.isEmpty()) continue;
            Variant var = new Variant();
            String[] p = v.split("\\|", 3);
            var.dir = p[0];
            var.opts = p.length > 1 ? p[1] : "";
            if (p.length > 2) {
                for (String kv : p[2].split(",")) {
                    String[] q = kv.split("=");
                    if (q[0].equals("wg")) var.wg = Long.parseLong(q[1]);
                    if (q[0].equals("matCache")) var.matCache = Integer.parseInt(q[1]);
                    if (q[0].equals("diag")) var.diag = true;
                    if (q[0].equals("global")) var.global = Long.parseLong(q[1]);
                }
            }
            var.name = variants.size() < names.length ? names[variants.size()] : var.dir + (var.opts.isEmpty() ? "" : "[" + var.opts + "]");
            variants.add(var);
        }
        return variants;
    }

    static void run(String[] args) throws Exception {
        for (String s : args) {
            int i = s.indexOf('=');
            A.put(s.substring(0, i), s.substring(i + 1));
        }
        System.setProperty("java.awt.headless", "true");
        final boolean quiet = !A.containsKey("verbose");
        Receiver rcv = new Receiver() {
            @Override public void logEvent(Level level, String message) {
                if (quiet && level == Level.INFO) return;
                System.out.println("[" + level + "] " + message);
            }
        };
        Log.setReceiver(rcv, Level.INFO, Level.WARNING, Level.ERROR);
        CL.setExceptionsEnabled(true);

        List<Variant> variants = parseVariants();
        if (A.containsKey("compileOnly")) {
            ClContext cctx = new ClContext(Device.getPreferredDevice());
            for (Variant var : variants) {
                if (build(cctx, var) == null) throw new RuntimeException("build failed: " + var.name);
            }
            return;
        }

        String sceneDir = arg("sceneDir", null), sceneName = arg("scene", null);
        CommandLineOptions cmd = new CommandLineOptions(new String[]{"-scene-dir", sceneDir, "-render", sceneName, "-f"});
        Log.setReceiver(rcv, Level.INFO, Level.WARNING, Level.ERROR);
        Field of = CommandLineOptions.class.getDeclaredField("options");
        of.setAccessible(true);
        Chunky chunky = new Chunky((ChunkyOptions) of.get(cmd));
        Field h = Chunky.class.getDeclaredField("headless");
        h.setAccessible(true);
        h.set(chunky, true);
        SynchronousSceneManager sm = (SynchronousSceneManager) chunky.getRenderController().getSceneManager();
        long t0 = System.nanoTime();
        sm.loadScene(new File(sceneDir, sceneName), sceneName);
        Scene scene = sm.getScene();
        if (A.containsKey("width")) scene.setCanvasSize(iarg("width", 0), iarg("height", 0));
        if (A.containsKey("branch")) scene.setBranchCount(iarg("branch", 10));
        if (A.containsKey("emitters")) scene.setEmittersEnabled(iarg("emitters", 0) != 0);
        if (A.containsKey("emitterStrategy"))
            scene.setEmitterSamplingStrategy(se.llbit.chunky.renderer.EmitterSamplingStrategy.values()[iarg("emitterStrategy", 0)]);
        if (A.containsKey("sunStrategy"))
            scene.setSunSamplingStrategy(se.llbit.chunky.renderer.SunSamplingStrategy.valueOf(arg("sunStrategy", "FAST")));
        if (A.containsKey("rayDepth")) scene.setRayDepth(iarg("rayDepth", 5));
        System.out.printf("scene %s loaded in %.1fs: %dx%d branch=%d rayDepth=%d emitters=%b/%s sun=%s%n", sceneName,
                (System.nanoTime() - t0) / 1e9, scene.canvasConfig.getWidth(), scene.canvasConfig.getHeight(),
                scene.getBranchCount(), scene.getRayDepth(), scene.getEmittersEnabled(),
                scene.getEmitterSamplingStrategy(), scene.getSunSamplingStrategy());

        Device device = Device.getPreferredDevice();
        ClContext ctx = new ClContext(device);
        ClSceneLoader loader = new ClSceneLoader(ctx);
        t0 = System.nanoTime();
        if (!loader.ensureLoad(scene)) throw new RuntimeException("scene export failed");
        System.out.printf("GPU export in %.1fs; matPalette words=%d; biome data %d wide x %d levels; sky ambient %s%n",
                (System.nanoTime() - t0) / 1e9, loader.getMaterialPalette().wordCount(),
                loader.getBiomeDataSize(), loader.getBiomeYLevels(), Arrays.toString(loader.getSky().skyAmbient));

        if (A.containsKey("validate")) {
            validate(ctx, loader);
            if (variants.isEmpty()) return;
        }

        for (Variant var : variants) {
            var.prog = build(ctx, var);
            if (var.prog == null) throw new RuntimeException("build failed: " + var.name);
            var.kernel = clCreateKernel(var.prog, "render", null);
            long[] v = new long[1];
            clGetKernelWorkGroupInfo(var.kernel, device.device, CL_KERNEL_PRIVATE_MEM_SIZE, Sizeof.cl_ulong, Pointer.to(v), null);
            long priv = v[0];
            clGetKernelWorkGroupInfo(var.kernel, device.device, CL_KERNEL_WORK_GROUP_SIZE, Sizeof.size_t, Pointer.to(v), null);
            int[] na = new int[1];
            clGetKernelInfo(var.kernel, CL_KERNEL_NUM_ARGS, Sizeof.cl_uint, Pointer.to(na), null);
            System.out.printf("variant %-28s args=%d maxWG=%d private=%dB%n", var.name, na[0], v[0], priv);
        }

        // ---- shared buffers ----
        int w = scene.canvasConfig.getWidth(), hgt = scene.canvasConfig.getHeight();
        int pixelCount = w * hgt;
        ClCamera camera = new ClCamera(scene, ctx);
        camera.generate(null, true);
        ClIntBuffer canvas = new ClIntBuffer(new int[]{w, hgt,
                scene.canvasConfig.getCropWidth(), scene.canvasConfig.getCropHeight(),
                scene.canvasConfig.getCropX(), scene.canvasConfig.getCropY()}, ctx);
        ClIntBuffer rayDepthBuf = new ClIntBuffer(scene.getRayDepth(), ctx);
        ClIntBuffer empty = new ClIntBuffer(new int[]{0, 0, 0, 0, 0, 0, 0}, ctx);
        int branch = Math.max(1, scene.getBranchCount());
        int[] cfg = {12345, 0, scene.getEmittersEnabled() ? 1 : 0, scene.getEmitterSamplingStrategy().ordinal(), branch};
        ClIntBuffer dyn = new ClIntBuffer(cfg, ctx);
        cl_mem emitterI = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float,
                Pointer.to(new float[]{(float) scene.getEmitterIntensity()}), null);
        cl_mem out = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, (long) Sizeof.cl_float * 4 * pixelCount, null, null);
        cl_mem diagBuf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, 64 * 4, null, null);
        KEEP_ALIVE.addAll(Arrays.asList(ctx, loader, camera, canvas, rayDepthBuf, empty, dyn, emitterI, out, diagBuf, scene));
        int matCacheWords = Math.min(iarg("matCache", 8192), loader.getMaterialPalette().wordCount());

        int ipl = iarg("ipl", 1);
        int passes = iarg("passes", 4);
        int rounds = iarg("rounds", 3);
        int slices = iarg("slices", 1);
        long wg = iarg("wg", 0);
        long gcap = iarg("global", 262144);

        for (Variant var : variants) {
            int mc = var.matCache >= 0 ? Math.min(var.matCache, loader.getMaterialPalette().wordCount()) : matCacheWords;
            setArgs(var.kernel, camera, loader, empty, dyn, emitterI, canvas, rayDepthBuf, mc, out, ipl, pixelCount);
            if (var.diag) mem(var.kernel, RENDER_ARGS, diagBuf);
        }

        // Warm-up each variant (materialise buffers, clocks).
        for (Variant var : variants) {
            long t = System.nanoTime();
            renderFrame(ctx, var.kernel, pixelCount, slices, var.wg >= 0 ? var.wg : wg, var.global > 0 ? var.global : gcap);
            clFinish(ctx.queue);
            System.out.printf("warm-up %-28s %.0f ms%n", var.name, (System.nanoTime() - t) / 1e6);
        }

        for (int r = 0; r < rounds; r++) {
            for (Variant var : variants) {
                long t = System.nanoTime();
                for (int p = 0; p < passes; p++) {
                    renderFrame(ctx, var.kernel, pixelCount, slices, var.wg >= 0 ? var.wg : wg, var.global > 0 ? var.global : gcap);
                }
                clFinish(ctx.queue);
                var.msPerPass.add((System.nanoTime() - t) / 1e6 / passes);
            }
        }
        System.out.println();
        double baseMed = -1;
        for (Variant var : variants) {
            List<Double> s = new ArrayList<>(var.msPerPass);
            Collections.sort(s);
            double med = s.get(s.size() / 2);
            if (baseMed < 0) baseMed = med;
            double samples = (double) pixelCount * ipl * branch;
            System.out.printf("RESULT %-28s median %8.1f ms/pass  min %8.1f  %7.2f Msamples/s  speedup x%.3f   all=%s%n",
                    var.name, med, s.get(0), samples / med / 1e3, baseMed / med, fmt(var.msPerPass));
        }

        // ---- SIMT diagnostic: average active lanes per trace site over one frame ----
        String[] siteNames = {"camera", "bounce", "shadowSeg", "sunNEE", "waterNEE", "s5", "s6", "s7"};
        for (Variant var : variants) {
            if (!var.diag) continue;
            clEnqueueFillBuffer(ctx.queue, diagBuf, Pointer.to(new int[]{0}), 4, 0, 64 * 4, 0, null, null);
            renderFrame(ctx, var.kernel, pixelCount, slices, var.wg >= 0 ? var.wg : wg, var.global > 0 ? var.global : gcap);
            int[] d = new int[16];
            clEnqueueReadBuffer(ctx.queue, diagBuf, CL_TRUE, 0, 64, Pointer.to(d), 0, null, null);
            StringBuilder sb = new StringBuilder("DIAG " + var.name + ":");
            for (int k = 0; k < 8; k++) {
                long lanes = d[2 * k] & 0xFFFFFFFFL, warps = d[2 * k + 1] & 0xFFFFFFFFL;
                if (warps > 0) sb.append(String.format("  %s %.1f/32 lanes (%d warp-traces)", siteNames[k], (double) lanes / warps, warps));
            }
            System.out.println(sb);
        }

        // ---- image comparison: identical seeds across variants ----
        int refSpp = iarg("refSpp", 8);
        String outDir = arg("out", null);
        for (Variant var : variants) {
            float[] acc = new float[pixelCount * 4];
            ByteBuffer bb = ByteBuffer.allocateDirect(pixelCount * 16).order(ByteOrder.nativeOrder());
            FloatBuffer fb = bb.asFloatBuffer();
            for (int f = 0; f < refSpp; f++) {
                cfg[0] = 777 + f * 7919;
                clEnqueueWriteBuffer(ctx.queue, dyn.get(), CL_TRUE, 0, Sizeof.cl_int * 5, Pointer.to(cfg), 0, null, null);
                renderFrame(ctx, var.kernel, pixelCount, slices, var.wg >= 0 ? var.wg : wg, var.global > 0 ? var.global : gcap);
                clEnqueueReadBuffer(ctx.queue, out, CL_TRUE, 0, (long) pixelCount * 16, Pointer.to(bb), 0, null, null);
                for (int i = 0; i < pixelCount * 4; i++) acc[i] += fb.get(i) / refSpp;
            }
            var.image = acc;
            if (outDir != null) writePng(acc, w, hgt, Paths.get(outDir, safe(var.name) + ".png"));
        }

        if (A.containsKey("preview")) comparePreview(ctx, variants, camera, loader, canvas, pixelCount, w, hgt, outDir);

        float[] ref = variants.get(0).image;
        for (Variant var : variants) {
            double sumAbs = 0, maxAbs = 0, meanRef = 0, meanV = 0;
            long exact = 0, bad = 0;
            for (int i = 0; i < pixelCount; i++) {
                for (int c = 0; c < 3; c++) {
                    float a = ref[i * 4 + c], b = var.image[i * 4 + c];
                    double d = Math.abs(a - b);
                    if (Float.isNaN(b) || Float.isInfinite(b)) bad++;
                    sumAbs += d;
                    maxAbs = Math.max(maxAbs, d);
                    meanRef += a;
                    meanV += b;
                    if (a == b) exact++;
                }
            }
            int n = pixelCount * 3;
            System.out.printf("IMAGE %-28s mean=%.5f (ref %.5f, %+.3f%%)  meanAbsDiff=%.6f  maxAbs=%.4f  exact=%.2f%%  nan/inf=%d%n",
                    var.name, meanV / n, meanRef / n, 100 * (meanV - meanRef) / Math.max(1e-9, meanRef),
                    sumAbs / n, maxAbs, 100.0 * exact / n, bad);
        }
    }

    // Preview kernel: one work-item per pixel, packed ARGB out. Every variant's preview is
    // compared against the first variant's, pixel for pixel.
    static void comparePreview(ClContext ctx, List<Variant> variants, ClCamera camera, ClSceneLoader loader,
                               ClIntBuffer canvas, int pixelCount, int w, int hgt, String outDir) throws Exception {
        cl_mem pout = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, (long) 4 * pixelCount, null, null);
        KEEP_ALIVE.add(pout);
        int[] refPrev = null;
        for (Variant var : variants) {
            cl_kernel pk = clCreateKernel(var.prog, "preview", null);
            int a = 0;
            mem(pk, a++, camera.projectorType.get());
            mem(pk, a++, camera.cameraSettings.get());
            mem(pk, a++, loader.getOctreeDepth().get());
            mem(pk, a++, loader.getOctreeData().get());
            mem(pk, a++, loader.getWaterOctreeDepth().get());
            mem(pk, a++, loader.getWaterOctreeData().get());
            mem(pk, a++, loader.getBlockPalette().get());
            mem(pk, a++, loader.getQuadPalette().get());
            mem(pk, a++, loader.getAabbPalette().get());
            mem(pk, a++, loader.getWorldBvh().get());
            mem(pk, a++, loader.getActorBvh().get());
            mem(pk, a++, loader.getTrigPalette().get());
            clSetKernelArg(pk, a++, Sizeof.cl_mem, Pointer.to(loader.getTexturePalette().getAtlas()));
            mem(pk, a++, loader.getMaterialPalette().get());
            mem(pk, a++, loader.getSky().skyTexture.get());
            mem(pk, a++, loader.getSky().skyIntensity.get());
            mem(pk, a++, loader.getSun().get());
            mem(pk, a++, canvas.get());
            mem(pk, a++, loader.getWaterConfig().get());
            mem(pk, a++, loader.getChunkBitmap().get());
            setIntArg(pk, a++, loader.getChunkBitmapSize());
            mem(pk, a++, loader.getBiomeData().get());
            setIntArg(pk, a++, loader.getBiomeDataSize());
            setIntArg(pk, a++, loader.getBiomeYLevels());
            mem(pk, a++, loader.getRenderConfig().get());
            mem(pk, a++, loader.getCloudData().get());
            mem(pk, a++, pout);
            long t = System.nanoTime();
            clEnqueueNDRangeKernel(ctx.queue, pk, 1, null, new long[]{pixelCount}, null, 0, null, null);
            clFinish(ctx.queue);
            double ms = (System.nanoTime() - t) / 1e6;
            int[] px = new int[pixelCount];
            clEnqueueReadBuffer(ctx.queue, pout, CL_TRUE, 0, 4L * pixelCount, Pointer.to(px), 0, null, null);
            if (refPrev == null) refPrev = px;
            int diff = 0;
            for (int i = 0; i < pixelCount; i++) if (px[i] != refPrev[i]) diff++;
            System.out.printf("PREVIEW %-28s %.1f ms  differing pixels %d%n", var.name, ms, diff);
            if (outDir != null) {
                BufferedImage bi = new BufferedImage(w, hgt, BufferedImage.TYPE_INT_RGB);
                bi.setRGB(0, 0, w, hgt, px, 0, w);
                Files.createDirectories(Paths.get(outDir));
                ImageIO.write(bi, "png", Paths.get(outDir, "preview_" + safe(var.name) + ".png").toFile());
            }
            clReleaseKernel(pk);
        }
    }

    static int[] readInts(ClContext ctx, cl_mem m) {
        long[] sz = new long[1];
        clGetMemObjectInfo(m, CL_MEM_SIZE, Sizeof.size_t, Pointer.to(sz), null);
        int[] a = new int[(int) (sz[0] / 4)];
        clEnqueueReadBuffer(ctx.queue, m, CL_TRUE, 0, 4L * a.length, Pointer.to(a), 0, null, null);
        return a;
    }

    /**
     * Walks the uploaded scene buffers the way the kernel does and reports every index
     * that would read outside its buffer. Catches bad data before it becomes an MMU fault.
     */
    static void validate(ClContext ctx, ClSceneLoader L) {
        int[] tree = readInts(ctx, L.getOctreeData().get());
        int[] wtree = readInts(ctx, L.getWaterOctreeData().get());
        int[] bp = readInts(ctx, L.getBlockPalette().get());
        int[] quads = readInts(ctx, L.getQuadPalette().get());
        int[] aabbs = readInts(ctx, L.getAabbPalette().get());
        int[] mat = readInts(ctx, L.getMaterialPalette().get());
        int[] trigs = readInts(ctx, L.getTrigPalette().get());
        int[] wbvh = readInts(ctx, L.getWorldBvh().get());
        int[] abvh = readInts(ctx, L.getActorBvh().get());
        System.out.printf("VALIDATE sizes: tree=%d water=%d blockPalette=%d quads=%d aabbs=%d materials=%d trigs=%d worldBvh=%d actorBvh=%d%n",
                tree.length, wtree.length, bp.length, quads.length, aabbs.length, mat.length, trigs.length, wbvh.length, abvh.length);
        Map<String, Integer> problems = new TreeMap<>();
        java.util.function.BiConsumer<String, Integer> bad = (what, v) -> {
            problems.merge(what, 1, Integer::sum);
            if (problems.get(what) <= 3) System.out.println("  VALIDATE bad " + what + " value=" + v);
        };
        java.util.function.IntPredicate matOk = m -> m >= 0 && m + 6 < mat.length;
        Set<Integer> blocks = new TreeSet<>();
        for (int[] t : new int[][]{tree, wtree}) {
            for (int i = 0; i < t.length; i++) {
                int v = t[i];
                if (v > 0) { if (v + 7 >= t.length + 1) bad.accept("octree child offset", v); }
                else if (v != 0) blocks.add(-v);
            }
        }
        for (int b : blocks) {
            if (b == 0x7FFFFFFE) { bad.accept("octree ANY_TYPE leaf", b); continue; }
            if (b < 0 || b + 2 >= bp.length) { bad.accept("octree leaf -> block palette index", b); continue; }
            int type = bp[b], ptr = bp[b + 1];
            switch (type) {
                case 1: case 4: case 5:
                    if (!matOk.test(ptr)) bad.accept("block type " + type + " material", ptr);
                    break;
                case 2: {
                    if (ptr < 0 || ptr >= aabbs.length) { bad.accept("aabb model pointer", ptr); break; }
                    int n = aabbs[ptr];
                    for (int k = 0; k < n; k++) {
                        int off = ptr + 1 + k * 13;
                        if (off + 12 >= aabbs.length) { bad.accept("aabb model box overrun", ptr); break; }
                        for (int f = 7; f <= 12; f++) if (!matOk.test(aabbs[off + f])) bad.accept("aabb face material", aabbs[off + f]);
                    }
                    break;
                }
                case 3: {
                    if (ptr < 0 || ptr >= quads.length) { bad.accept("quad model pointer", ptr); break; }
                    int n = quads[ptr];
                    for (int k = 0; k < n; k++) {
                        int off = ptr + 1 + k * 15;
                        if (off + 14 >= quads.length) { bad.accept("quad model overrun", ptr); break; }
                        if (!matOk.test(quads[off + 13])) bad.accept("quad material", quads[off + 13]);
                    }
                    break;
                }
                case 0: break;
                default: bad.accept("unknown model type", type);
            }
        }
        for (int[] bvh : new int[][]{wbvh, abvh}) {
            for (int n = 0; n + 6 < bvh.length; n += 7) {
                int h = bvh[n];
                if (h > 0) {
                    if (h + 6 >= bvh.length) bad.accept("bvh child offset", h);
                    if (n + 13 >= bvh.length) bad.accept("bvh left child overrun", n);
                } else {
                    int prim = -h;
                    if (prim < 0 || prim >= trigs.length) { bad.accept("bvh leaf -> trig index", prim); continue; }
                    int cnt = Math.min(trigs[prim], 256);
                    for (int k = 0; k < cnt; k++) {
                        int off = prim + 1 + 20 * k;
                        if (off + 19 >= trigs.length) { bad.accept("triangle overrun", prim); break; }
                        if (!matOk.test(trigs[off + 19])) bad.accept("triangle material", trigs[off + 19]);
                    }
                }
            }
        }
        System.out.println("VALIDATE " + (problems.isEmpty() ? "all indices in bounds" : "PROBLEMS " + problems));
        if (L.getBiomeDataSize() > 0) {
            int[] biome = readInts(ctx, L.getBiomeData().get());
            long differ = 0, n = biome.length / 4;
            Map<String, Integer> pairs = new TreeMap<>();
            for (int i = 0; i + 3 < biome.length; i += 4) {
                if (biome[i] != biome[i + 1]) differ++;
                pairs.merge(String.format("%06X/%06X", biome[i] & 0xFFFFFF, biome[i + 1] & 0xFFFFFF), 1, Integer::sum);
            }
            System.out.printf("VALIDATE biome: %d positions, slot0 != slot1 at %d; most common slot0/slot1 pairs: %s%n",
                    n, differ, pairs.entrySet().stream().sorted((x, y) -> y.getValue() - x.getValue()).limit(4)
                            .map(e -> e.getKey() + " x" + e.getValue()).toList());
        }
    }

    static String safe(String s) { return s.replaceAll("[^A-Za-z0-9._-]", "_"); }

    static String fmt(List<Double> l) {
        StringBuilder sb = new StringBuilder("[");
        for (double d : l) sb.append(String.format("%.0f ", d));
        return sb.toString().trim() + "]";
    }

    static void writePng(float[] img, int w, int h, Path p) throws Exception {
        BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = (y * w + x) * 4;
                int r = (int) Math.min(255, Math.max(0, Math.pow(img[i], 1 / 2.2) * 255));
                int g = (int) Math.min(255, Math.max(0, Math.pow(img[i + 1], 1 / 2.2) * 255));
                int b = (int) Math.min(255, Math.max(0, Math.pow(img[i + 2], 1 / 2.2) * 255));
                bi.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        Files.createDirectories(p.getParent());
        ImageIO.write(bi, "png", p.toFile());
        // Raw little-endian RGBA floats too, for offline comparison.
        ByteBuffer bb = ByteBuffer.allocate(img.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : img) bb.putFloat(f);
        Files.write(Paths.get(p.toString().replace(".png", ".f32")), bb.array());
    }

    static final int[] scratch = new int[1];

    static void setIntArg(cl_kernel k, int idx, int v) {
        scratch[0] = v;
        clSetKernelArg(k, idx, Sizeof.cl_int, Pointer.to(scratch));
    }

    static void mem(cl_kernel k, int idx, cl_mem m) {
        clSetKernelArg(k, idx, Sizeof.cl_mem, Pointer.to(m));
    }

    // Mirrors OpenClPathTracingRenderer's render-kernel arg order exactly.
    static void setArgs(cl_kernel k, ClCamera camera, ClSceneLoader L, ClIntBuffer empty, ClIntBuffer dyn,
                        cl_mem emitterI, ClIntBuffer canvas, ClIntBuffer rayDepth, int matCacheWords, cl_mem out,
                        int ipl, int pixelCount) {
        int a = 0;
        mem(k, a++, camera.projectorType.get());
        mem(k, a++, camera.cameraSettings.get());
        mem(k, a++, camera.apertureMaskBuffer != null ? camera.apertureMaskBuffer.get() : empty.get());
        setIntArg(k, a++, camera.apertureMaskWidth);
        mem(k, a++, L.getOctreeDepth().get());
        mem(k, a++, L.getOctreeData().get());
        mem(k, a++, L.getWaterOctreeDepth().get());
        mem(k, a++, L.getWaterOctreeData().get());
        mem(k, a++, L.getBlockPalette().get());
        mem(k, a++, L.getQuadPalette().get());
        mem(k, a++, L.getAabbPalette().get());
        mem(k, a++, L.getWorldBvh().get());
        mem(k, a++, L.getActorBvh().get());
        mem(k, a++, L.getTrigPalette().get());
        clSetKernelArg(k, a++, Sizeof.cl_mem, Pointer.to(L.getTexturePalette().getAtlas()));
        mem(k, a++, L.getMaterialPalette().get());
        setIntArg(k, a++, matCacheWords);
        clSetKernelArg(k, a++, Sizeof.cl_uint * Math.max(matCacheWords, 1), null);
        mem(k, a++, L.getSky().skyTexture.get());
        mem(k, a++, L.getSky().skyIntensity.get());
        mem(k, a++, L.getSun().get());
        mem(k, a++, dyn.get());
        mem(k, a++, emitterI);
        mem(k, a++, L.getEmitterPositions() != null ? L.getEmitterPositions().get() : empty.get());
        mem(k, a++, L.getPositionIndexes() != null ? L.getPositionIndexes().get() : empty.get());
        mem(k, a++, L.getConstructedGrid() != null ? L.getConstructedGrid().get() : empty.get());
        mem(k, a++, L.getGridConfig() != null ? L.getGridConfig().get() : empty.get());
        mem(k, a++, canvas.get());
        mem(k, a++, rayDepth.get());
        setIntArg(k, a++, ipl);
        mem(k, a++, L.getFogData().get());
        mem(k, a++, L.getWaterConfig().get());
        mem(k, a++, L.getRenderConfig().get());
        mem(k, a++, L.getCloudData().get());
        mem(k, a++, L.getWaterNormalMap().get());
        setIntArg(k, a++, L.getWaterNormalMapWidth());
        mem(k, a++, L.getBiomeData().get());
        setIntArg(k, a++, L.getBiomeDataSize());
        mem(k, a++, L.getChunkBitmap().get());
        setIntArg(k, a++, L.getChunkBitmapSize());
        mem(k, a++, out);
        setIntArg(k, a++, 0);          // pixelStart
        setIntArg(k, a++, 0);          // pixelStride (set per frame)
        setIntArg(k, a++, pixelCount); // pixelCount
        if (a != RENDER_ARGS) throw new IllegalStateException("arg count " + a);
    }

    // Same interleaved slicing as OpenClPathTracingRenderer.dispatchFrame.
    static void renderFrame(ClContext ctx, cl_kernel k, int pixelCount, int slices, long wg, long gcap) {
        long ideal = Math.max(1L, (long) pixelCount / Math.max(1L, (long) slices * 16));
        long g = Math.min(ideal, gcap);
        long gran = wg > 0 ? wg : 256;
        g = Math.max(gran, ((g + gran - 1) / gran) * gran);
        setIntArg(k, 42, (int) (slices * g));
        long[] global = {g};
        long[] local = wg > 0 ? new long[]{wg} : null;
        for (int s = 0; s < slices; s++) {
            setIntArg(k, 41, (int) (s * g));
            clEnqueueNDRangeKernel(ctx.queue, k, 1, null, global, local, 0, null, null);
        }
    }

    static String inline(Path dir, String file, Set<String> seen) throws Exception {
        String src = new String(Files.readAllBytes(dir.resolve(file)));
        Matcher m = INCLUDE.matcher(src);
        StringBuilder sb = new StringBuilder();
        while (m.find()) {
            String inc = m.group(1);
            String rep;
            if (inc.equals("../opencl.h")) rep = "// (opencl.h stripped)";
            else if (seen.add(inc)) rep = inline(dir, inc, seen);
            else rep = "// (already included " + inc + ")";
            m.appendReplacement(sb, Matcher.quoteReplacement(rep));
        }
        m.appendTail(sb);
        return sb.toString();
    }

    // Program build with a binary cache keyed by (inlined source, options). NVIDIA builds of
    // the render kernel take 30-120 s, so timing runs should only ever hit this cache.
    static cl_program build(ClContext ctx, Variant var) throws Exception {
        String src = inline(Paths.get(var.dir), "rayTracer.c", new HashSet<>());
        String opts = BASE_OPTS + (var.opts.isEmpty() ? "" : " " + var.opts);
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        md.update(src.getBytes());
        md.update(opts.getBytes());
        md.update(ctx.device.name().getBytes());  // binaries are device-specific
        String key = HexFormat.of().formatHex(md.digest()).substring(0, 24);
        Path cacheDir = Paths.get(arg("cache", System.getProperty("java.io.tmpdir") + "/clbench-cache"));
        Files.createDirectories(cacheDir);
        Path bin = cacheDir.resolve(key + ".bin");
        cl_device_id[] devs = ctx.deviceArray;
        if (Files.exists(bin)) {
            byte[] b = Files.readAllBytes(bin);
            int[] st = new int[1];
            cl_program p = clCreateProgramWithBinary(ctx.context, 1, devs, new long[]{b.length}, new byte[][]{b}, st, null);
            clBuildProgram(p, 1, devs, opts, null, null);
            System.out.println("[cache] " + var.name + " -> " + key);
            return p;
        }
        long t = System.nanoTime();
        cl_program p = clCreateProgramWithSource(ctx.context, 1, new String[]{src}, null, null);
        CL.setExceptionsEnabled(false);
        String verbose = ctx.device.supportsNvCompilerOptions() ? " -cl-nv-verbose" : "";
        int code = clBuildProgram(p, 1, devs, opts + verbose, null, null);
        CL.setExceptionsEnabled(true);
        long[] sz = new long[1];
        clGetProgramBuildInfo(p, devs[0], CL_PROGRAM_BUILD_LOG, 0, null, sz);
        byte[] log = new byte[(int) sz[0]];
        clGetProgramBuildInfo(p, devs[0], CL_PROGRAM_BUILD_LOG, log.length, Pointer.to(log), null);
        String logs = new String(log).trim();
        StringBuilder brief = new StringBuilder();
        boolean inRender = false;
        for (String line : logs.split("\n")) {
            if (line.contains("entry function")) inRender = line.contains("'render'");
            if (inRender && (line.contains("stack frame") || line.contains("Used "))) brief.append("   ").append(line.trim()).append('\n');
            if (line.contains("error")) brief.append("   ").append(line.trim()).append('\n');
        }
        System.out.printf("[build] %s in %.0fs (code %d)%n%s", var.name, (System.nanoTime() - t) / 1e9, code, brief);
        if (code != CL_SUCCESS) {
            System.out.println(logs);
            return null;
        }
        Files.write(cacheDir.resolve(key + ".log"), logs.getBytes());
        clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, Sizeof.size_t, Pointer.to(sz), null);
        byte[] b = new byte[(int) sz[0]];
        clGetProgramInfo(p, CL_PROGRAM_BINARIES, Sizeof.POINTER, Pointer.to(new Pointer[]{Pointer.to(b)}), null);
        Files.write(bin, b);
        return p;
    }
}
