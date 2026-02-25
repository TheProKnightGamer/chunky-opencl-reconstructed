package dev.thatredox.chunkynative.opencl.renderer.scene;

import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.util.ClMemory;
import dev.thatredox.chunkynative.util.Reflection;
import dev.thatredox.chunkynative.util.Util;
import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.floats.FloatList;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import se.llbit.chunky.main.Chunky;
import dev.thatredox.chunkynative.opencl.util.ClIntBuffer;
import se.llbit.chunky.renderer.ApertureShape;
import se.llbit.chunky.renderer.projection.ProjectionMode;
import se.llbit.chunky.renderer.scene.Camera;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.math.Matrix3;
import se.llbit.math.Ray;
import se.llbit.math.Vector3;
import se.llbit.resources.ImageLoader;
import se.llbit.chunky.resources.BitmapImage;

import java.io.File;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.locks.Lock;
import java.util.stream.IntStream;

import static org.jocl.CL.*;

public class ClCamera implements AutoCloseable {
    public ClMemory projectorType;
    public ClMemory cameraSettings;
    public ClIntBuffer apertureMaskBuffer;
    public int apertureMaskWidth;
    public final boolean needGenerate;

    private final Scene scene;
    private final ClContext context;


    public ClCamera(Scene scene, ClContext context) {
        this.scene = scene;
        this.context = context;
        Camera camera = scene.camera();

        int projType = -1;
        Vector3 pos = new Vector3(camera.getPosition());
        pos.sub(scene.getOrigin());

        FloatArrayList settings = new FloatArrayList();
        settings.addAll(FloatList.of(Util.vector3ToFloat(pos)));
        settings.addAll(FloatList.of(Util.matrix3ToFloat(Reflection.getFieldValue(camera, "transform", Matrix3.class))));

        // Camera shift (lens shift / image plane offset) at fixed positions [12] and [13]
        settings.add((float) camera.getShiftX());
        settings.add((float) camera.getShiftY());

        // Map aperture shapes to GPU IDs
        int apertureShapeId = 0; // CIRCLE
        ApertureShape shape = camera.getApertureShape();
        if (shape != null) {
            switch (shape) {
                case CIRCLE:   apertureShapeId = 0; break;
                case HEXAGON:  apertureShapeId = 1; break;
                case PENTAGON: apertureShapeId = 2; break;
                case STAR:     apertureShapeId = 3; break;
                case GAUSSIAN: apertureShapeId = 4; break;
                case CUSTOM:   apertureShapeId = 5; break;
                default:       apertureShapeId = 0; break;
            }
        }

        // Load custom aperture mask if needed
        apertureMaskBuffer = null;
        apertureMaskWidth = 0;
        if (shape == ApertureShape.CUSTOM) {
            try {
                String filename = Reflection.getFieldValueNullable(camera, "apertureMaskFilename", String.class);
                if (filename != null && !filename.isEmpty()) {
                    BitmapImage mask = ImageLoader.read(new File(filename));
                    if (mask != null) {
                        apertureMaskWidth = mask.width;
                        apertureMaskBuffer = new ClIntBuffer(mask.data, context);
                    }
                }
            } catch (Exception e) {
                // Fall back to circle aperture if mask loading fails
                apertureShapeId = 0;
            }
        }

        switch (camera.getProjectionMode()) {
            case PINHOLE:
                projType = 0;
                settings.add(camera.infiniteDoF() ? 0 : (float) (camera.getSubjectDistance() / camera.getDof()));
                settings.add((float) camera.getSubjectDistance());
                settings.add((float) Camera.clampedFovTan(camera.getFov()));
                settings.add((float) apertureShapeId);
                break;
            case PARALLEL:
                projType = 1;
                settings.add((float) Camera.clampedFovTan(camera.getFov()));
                settings.add(camera.infiniteDoF() ? 0 : (float) (camera.getSubjectDistance() / camera.getDof()));
                settings.add((float) camera.getSubjectDistance());
                settings.add((float) apertureShapeId);
                break;
            case FISHEYE:
                projType = 2;
                settings.add((float) camera.getFov());
                settings.add(camera.infiniteDoF() ? 0 : (float) (camera.getSubjectDistance() / camera.getDof()));
                settings.add((float) camera.getSubjectDistance());
                settings.add((float) apertureShapeId);
                break;
            case STEREOGRAPHIC:
                projType = 3;
                settings.add((float) camera.getFov());
                settings.add(camera.infiniteDoF() ? 0 : (float) (camera.getSubjectDistance() / camera.getDof()));
                settings.add((float) camera.getSubjectDistance());
                settings.add((float) apertureShapeId);
                break;
            case PANORAMIC:
                projType = 4;
                settings.add((float) camera.getFov());
                settings.add(camera.infiniteDoF() ? 0 : (float) (camera.getSubjectDistance() / camera.getDof()));
                settings.add((float) camera.getSubjectDistance());
                settings.add((float) apertureShapeId);
                break;
            case PANORAMIC_SLOT:
                projType = 5;
                settings.add((float) camera.getFov());
                settings.add(camera.infiniteDoF() ? 0 : (float) (camera.getSubjectDistance() / camera.getDof()));
                settings.add((float) camera.getSubjectDistance());
                settings.add((float) apertureShapeId);
                break;
            case ODS_LEFT:
                projType = 6;
                settings.add(0.069f); // default IPD ~69mm
                settings.add(-1.0f);  // left eye
                break;
            case ODS_RIGHT:
                projType = 6;
                settings.add(0.069f);
                settings.add(1.0f);   // right eye
                break;
            case ODS_STACKED:
                projType = 7;
                settings.add(0.069f); // default IPD ~69mm
                break;
            default:
                // Fall through to pre-generated rays
                break;
        }

        needGenerate = projType == -1;

        projectorType = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_int, Pointer.to(new int[] {projType}), null));

        if (needGenerate) {
            cameraSettings = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY,
                    (long) Sizeof.cl_float * scene.canvasConfig.getWidth() * scene.canvasConfig.getHeight() * 3 * 2, null, null));
        } else {
            cameraSettings = new ClMemory(clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * settings.size(), Pointer.to(settings.toFloatArray()), null));
        }
    }

    public void generate(Lock renderLock, boolean jitter) {
        if (!needGenerate) return;

        int width = scene.canvasConfig.getWidth();
        int height = scene.canvasConfig.getHeight();
        int fullWidth = scene.canvasConfig.getCropWidth();
        int fullHeight = scene.canvasConfig.getCropHeight();
        int cropX = scene.canvasConfig.getCropX();
        int cropY = scene.canvasConfig.getCropY();

        float[] rays = new float[width * height * 3 * 2];

        double halfWidth = fullWidth / (2.0 * fullHeight);
        double invHeight = 1.0 / fullHeight;

        Camera cam = scene.camera();

        Chunky.getCommonThreads().submit(() -> IntStream.range(0, width).parallel().forEach(i -> {
            Ray ray = new Ray();
            Random random = jitter ? ThreadLocalRandom.current() : null;
            for (int j = 0; j < height; j++) {
                int offset = (j * width + i) * 3 * 2;

                float ox = jitter ? random.nextFloat(): 0.5f;
                float oy = jitter ? random.nextFloat(): 0.5f;

                cam.calcViewRay(ray, -halfWidth + (i + ox + cropX) * invHeight, -0.5 + (j + oy + cropY) * invHeight);
                ray.o.sub(scene.getOrigin());

                System.arraycopy(Util.vector3ToFloat(ray.o), 0, rays, offset, 3);
                System.arraycopy(Util.vector3ToFloat(ray.d), 0, rays, offset+3, 3);
            }
        })).join();

        if (renderLock != null) renderLock.lock();
        clEnqueueWriteBuffer(context.queue, this.cameraSettings.get(), CL_TRUE, 0,
                (long) Sizeof.cl_float * rays.length, Pointer.to(rays), 0,
                null, null);
        if (renderLock != null) renderLock.unlock();
    }

    @Override
    public void close() {
        this.projectorType.close();
        this.cameraSettings.close();
        if (this.apertureMaskBuffer != null) this.apertureMaskBuffer.close();
    }
}
