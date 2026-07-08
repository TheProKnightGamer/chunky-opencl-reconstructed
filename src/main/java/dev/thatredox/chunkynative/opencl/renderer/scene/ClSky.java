package dev.thatredox.chunkynative.opencl.renderer.scene;


import static org.jocl.CL.*;

import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import org.apache.commons.math3.util.FastMath;
import org.jocl.*;

import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.renderer.scene.sky.Sky;
import se.llbit.chunky.renderer.scene.sky.SkyCache;
import se.llbit.chunky.resources.Texture;
import se.llbit.log.Log;
import se.llbit.math.Ray;

import java.lang.reflect.Field;

public class ClSky implements AutoCloseable {
    public final ClMemory skyTexture;
    public final ClMemory skyIntensity;
    private final ClContext context;

    public ClSky(Scene scene, ClContext context) {
        this.context = context;
        int textureResolution = getTextureResolution(scene);

        // The kernel's Sky_intersect multiplies the baked sky by this value. The
        // bake (getSkyColor below) already folds in skyExposure*skyLightModifier,
        // and the CPU never scales the sky by sun intensity — so this MUST be 1.0.
        // It used to be sun.getIntensity() (default 1.25), which made the whole
        // sky and all sky-lit ambient ~1.25x (up to 50x) brighter than the CPU.
        this.skyIntensity = new ClMemory(clCreateBuffer(context.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float,
                Pointer.to(new float[] {1.0f}), null));

        cl_image_format fmt = new cl_image_format();
        fmt.image_channel_data_type = CL_FLOAT;
        fmt.image_channel_order = CL_RGBA;

        cl_image_desc desc = new cl_image_desc();
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = textureResolution;
        desc.image_height = textureResolution;

        float[] texture = new float[textureResolution * textureResolution * 4];
        Ray ray = new Ray();
        for (int i = 0; i < textureResolution; i++) {
            for (int j = 0; j < textureResolution; j++) {
                int offset = 4 * (j * textureResolution + i);

                double theta = ((double) i / textureResolution) * 2 * FastMath.PI;
                double phi = ((double) j / textureResolution) * FastMath.PI - FastMath.PI / 2;
                double r = FastMath.cos(phi);
                ray.d.set(FastMath.cos(theta) * r, FastMath.sin(phi), FastMath.sin(theta) * r);

                // Bake the sky WITHOUT the sun disk. The sun is drawn analytically
                // at runtime by the kernel's Sun_intersect (apparent brightness for
                // direct/specular rays, luminosity-scaled for diffuse rays), exactly
                // mirroring CPU addSunColor / addSunColorDiffuseSun. Baking the sun
                // here double-counted it against the runtime sun AND baked the wrong
                // (apparent, not luminosity) brightness for diffuse rays, which broke
                // sun lighting in the OFF / IMPORTANCE / HIGH_QUALITY modes.
                // KNOWN LIMITATION (accepted): getSkyColor scales by
                // skyLightModifier; the CPU uses apparentSkyLightModifier for the
                // DIRECT-view (depth-0 camera) sky only. We bake ONE texture
                // (skyLightModifier) and reuse it for all ray types. This is
                // default-invisible (both modifiers default to 1.0) and only
                // diverges if the user sets "Apparent Sky Light" != "Sky Light".
                // Matching it exactly would need a 2nd baked texture or per-ray
                // scaling tangled with the runtime sun compositing — too much for a
                // near-never-visible difference, so it is left as a limitation.
                scene.sky().getSkyColor(ray, false);
                texture[offset + 0] = (float) ray.color.x;
                texture[offset + 1] = (float) ray.color.y;
                texture[offset + 2] = (float) ray.color.z;
                texture[offset + 3] = 1.0f;
            }
        }

        this.skyTexture = new ClMemory(clCreateImage(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                fmt, desc, Pointer.to(texture), null));
    }

    /**
     * Default fallback when reflection access to chunky's private SkyCache fails
     * (e.g. after an upstream Chunky rename). Matches chunky's default sky cache
     * resolution so behavior degrades gracefully instead of crashing.
     */
    private static final int DEFAULT_SKY_RESOLUTION = 128;

    /** Cap for the skymap-mode bake to keep memory bounded (4K^2 RGBA float = 256 MB). */
    private static final int MAX_SKY_RESOLUTION = 4096;

    private static int getTextureResolution(Scene scene) {
        Sky sky = scene.sky();

        // For image-based sky modes (SKYMAP_*, SKYBOX), match the source image
        // dimensions so we don't downsample a 4K skymap to 128 px. For procedural
        // modes (SIMULATED/GRADIENT/SOLID_COLOR) the chunky SkyCache resolution
        // is the right target — anything larger is wasted memory.
        int imageRes = readSkymapResolution(sky);
        if (imageRes > 0) {
            return Math.min(MAX_SKY_RESOLUTION, Math.max(DEFAULT_SKY_RESOLUTION, imageRes));
        }

        try {
            Field skyCacheField = sky.getClass().getDeclaredField("skyCache");
            skyCacheField.setAccessible(true);
            SkyCache skyCache = (SkyCache) skyCacheField.get(sky);

            return skyCache.getSkyResolution();
        } catch (NoSuchFieldException | IllegalAccessException e) {
            Log.warn("ChunkyCL: could not read chunky's SkyCache resolution via "
                    + "reflection (chunky API may have changed); falling back to "
                    + DEFAULT_SKY_RESOLUTION + "px sky bake.", e);
            return DEFAULT_SKY_RESOLUTION;
        }
    }

    /**
     * Returns the largest image-based source dimension if the sky is in a SKYMAP
     * or SKYBOX mode, else 0. Uses reflection because chunky's Sky exposes neither
     * the mode nor the loaded textures publicly.
     */
    private static int readSkymapResolution(Sky sky) {
        try {
            Field modeField = sky.getClass().getDeclaredField("mode");
            modeField.setAccessible(true);
            Object modeVal = modeField.get(sky);
            String modeName = modeVal == null ? "" : modeVal.toString();

            if ("SKYMAP_EQUIRECTANGULAR".equals(modeName) || "SKYMAP_ANGULAR".equals(modeName)) {
                Field skymapField = sky.getClass().getDeclaredField("skymap");
                skymapField.setAccessible(true);
                Texture t = (Texture) skymapField.get(sky);
                return t == null ? 0 : Math.max(t.getWidth(), t.getHeight());
            }
            if ("SKYBOX".equals(modeName)) {
                Field skyboxField = sky.getClass().getDeclaredField("skybox");
                skyboxField.setAccessible(true);
                Texture[] faces = (Texture[]) skyboxField.get(sky);
                int max = 0;
                if (faces != null) {
                    for (Texture face : faces) {
                        if (face != null) max = Math.max(max, Math.max(face.getWidth(), face.getHeight()));
                    }
                }
                // Skybox needs ~4x equirectangular resolution to capture full detail
                return max > 0 ? max * 4 : 0;
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            // Field layout changed; fall back to SkyCache resolution.
        }
        return 0;
    }

    @Override
    public void close() {
        skyTexture.close();
        skyIntensity.close();
    }
}
