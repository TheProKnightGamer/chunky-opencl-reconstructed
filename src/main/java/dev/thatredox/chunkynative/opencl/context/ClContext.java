package dev.thatredox.chunkynative.opencl.context;

import se.llbit.chunky.PersistentSettings;
import se.llbit.log.Log;
import org.jocl.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Function;

import static org.jocl.CL.*;

public class ClContext {
    public final Device device;
    public final cl_context context;
    public final cl_command_queue queue;
    public final cl_device_id[] deviceArray;

    public ClContext(Device device) {
        this.device = device;
        this.deviceArray = new cl_device_id[] { device.device };

        cl_context_properties contextProperties = new cl_context_properties();
        this.context = clCreateContext(contextProperties, 1, deviceArray,
                null, null, null);


        int[] version = device.version();
        if (version[0] >= 2) {
            cl_queue_properties queueProperties = new cl_queue_properties();
            queue = clCreateCommandQueueWithProperties(context, device.device, queueProperties, null);
        } else {
            queue = createCommandQueueOld(0);
        }

        // Check if version is behind
        if (version[0] <= 1 && version[1] < 2) {
            Log.error("OpenCL 1.2+ required.");
        }
    }

    @SuppressWarnings("deprecation")
    private cl_command_queue createCommandQueueOld(long properties) {
        return clCreateCommandQueue(context, device.device, properties, null);
    }

    /**
     * Load an OpenCL program, using a binary cache for faster subsequent launches.
     * On first run, compiles from source and caches the binary to disk.
     * On subsequent runs, loads the cached binary in milliseconds.
     *
     * @param sourceReader  Function to read source files from filenames.
     * @param kernelName    Kernel entrypoint filename.
     * @return OpenCL program.
     */
    public cl_program loadProgram(Function<String, String> sourceReader, String kernelName) {
        // --- Phase 1: Collect all source strings ---
        // TreeMap gives deterministic iteration order for hashing
        TreeMap<String, String> allSources = new TreeMap<>();
        String kernel = sourceReader.apply(kernelName);
        allSources.put(kernelName, kernel);

        // Recursively resolve all #include headers
        Set<String> pendingHeaders = new LinkedHashSet<>();
        collectHeaderNames(kernel, pendingHeaders);

        while (!pendingHeaders.isEmpty()) {
            Iterator<String> it = pendingHeaders.iterator();
            String headerName = it.next();
            it.remove();
            if (allSources.containsKey(headerName)) continue;
            if (headerName.endsWith(".h")) {
                String headerSource = sourceReader.apply(headerName);
                allSources.put(headerName, headerSource);
                collectHeaderNames(headerSource, pendingHeaders);
            }
        }

        // --- Phase 2: Check binary cache ---
        String cacheKey = computeCacheKey(allSources);
        cl_program cached = loadCachedBinary(cacheKey);
        if (cached != null) {
            Log.warn("ChunkyCL: Loaded cached OpenCL binary for " + kernelName);
            return cached;
        }

        String compileMsg = "ChunkyCL: Compiling " + kernelName + " from source (first launch may take a few minutes)...";
        Log.warn(compileMsg);
        // Flush directly to console so the message is visible before the blocking compile,
        // even if the GUI log receiver can't update (e.g. JavaFX thread is blocked).
        System.err.println(compileMsg);
        System.err.flush();

        // --- Phase 3: Compile from source ---
        cl_program kernelProgram = clCreateProgramWithSource(context, 1, new String[] { kernel }, null, null);

        HashMap<String, cl_program> headerPrograms = new HashMap<>();
        for (Map.Entry<String, String> entry : allSources.entrySet()) {
            if (!entry.getKey().equals(kernelName) && entry.getKey().endsWith(".h")) {
                headerPrograms.put(entry.getKey(),
                    clCreateProgramWithSource(context, 1, new String[] { entry.getValue() }, null, null));
            }
        }

        String[] includeNames = headerPrograms.keySet().toArray(new String[0]);
        cl_program[] includePrograms = new cl_program[includeNames.length];
        Arrays.setAll(includePrograms, i -> headerPrograms.get(includeNames[i]));

        CL.setExceptionsEnabled(false);
        int code = clCompileProgram(kernelProgram, 1, deviceArray, "-cl-std=CL1.2 -Werror",
                includePrograms.length, includePrograms, includeNames, null, null);
        if (code != CL_SUCCESS) {
            String error;
            switch (code) {
                case CL_INVALID_PROGRAM:
                    error = "CL_INVALID_PROGRAM";
                    break;
                case CL_INVALID_VALUE:
                    error = "CL_INVALID_VALUE";
                    break;
                case CL_INVALID_DEVICE:
                    error = "CL_INVALID_DEVICE";
                    break;
                case CL_INVALID_COMPILER_OPTIONS:
                    error = "CL_INVALID_COMPILER_OPTIONS";
                    break;
                case CL_INVALID_OPERATION:
                    error = "CL_INVALID_OPERATION";
                    break;
                case CL_COMPILER_NOT_AVAILABLE:
                    error = "CL_COMPILER_NOT_AVAILABLE";
                    break;
                case CL_COMPILE_PROGRAM_FAILURE:
                    error = "CL_COMPILE_PROGRAM_FAILURE";
                    break;
                case CL_OUT_OF_RESOURCES:
                    error = "CL_OUT_OF_RESOURCES";
                    break;
                case CL_OUT_OF_HOST_MEMORY:
                    error = "CL_OUT_OF_HOST_MEMORY";
                    break;
                default:
                    error = "Code " + code;
                    break;
            }
            CL.setExceptionsEnabled(true);
            Log.error("Failed to build CL program: " + error);

            long[] size = new long[1];
            clGetProgramBuildInfo(kernelProgram, deviceArray[0], CL.CL_PROGRAM_BUILD_LOG, 0, null, size);

            byte[] buffer = new byte[(int)size[0]];
            clGetProgramBuildInfo(kernelProgram, deviceArray[0], CL.CL_PROGRAM_BUILD_LOG, buffer.length, Pointer.to(buffer), null);

            throw new RuntimeException("Failed to build CL program with error: " + error + "\n" + new String(buffer, 0, buffer.length-1));
        }
        CL.setExceptionsEnabled(true);

        cl_program linked = clLinkProgram(context, 1, deviceArray, "", 1,
                new cl_program[] { kernelProgram }, null, null, null);

        // --- Phase 4: Cache the compiled binary ---
        saveBinaryToCache(linked, cacheKey, kernelName);

        return linked;
    }

    // ---- Binary Cache Helpers ----

    private String computeCacheKey(TreeMap<String, String> allSources) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(device.name().getBytes(StandardCharsets.UTF_8));
            digest.update(device.versionString().getBytes(StandardCharsets.UTF_8));
            digest.update(getDriverVersion().getBytes(StandardCharsets.UTF_8));
            for (Map.Entry<String, String> entry : allSources.entrySet()) {
                digest.update(entry.getKey().getBytes(StandardCharsets.UTF_8));
                digest.update(entry.getValue().getBytes(StandardCharsets.UTF_8));
            }
            byte[] hash = digest.digest();
            StringBuilder sb = new StringBuilder();
            for (byte b : hash) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }

    private String getDriverVersion() {
        try {
            long[] size = new long[1];
            clGetDeviceInfo(device.device, CL_DRIVER_VERSION, 0, null, size);
            byte[] buf = new byte[(int) size[0]];
            clGetDeviceInfo(device.device, CL_DRIVER_VERSION, buf.length, Pointer.to(buf), null);
            return new String(buf, 0, Math.max(0, buf.length - 1));
        } catch (Exception e) {
            return "unknown";
        }
    }

    public Path getCacheDir() {
        return PersistentSettings.settingsDirectory().toPath().resolve("opencl-cache");
    }

    private cl_program loadCachedBinary(String cacheKey) {
        try {
            Path cacheFile = getCacheDir().resolve(cacheKey + ".bin");
            if (!Files.exists(cacheFile)) return null;

            byte[] binary = Files.readAllBytes(cacheFile);

            CL.setExceptionsEnabled(false);
            int[] binaryStatus = new int[1];
            cl_program program = clCreateProgramWithBinary(context, 1, deviceArray,
                    new long[] { binary.length }, new byte[][] { binary }, binaryStatus, null);

            if (binaryStatus[0] != CL_SUCCESS) {
                CL.setExceptionsEnabled(true);
                Log.warn("ChunkyCL: Cached binary invalid (status " + binaryStatus[0] + "), recompiling...");
                Files.deleteIfExists(cacheFile);
                return null;
            }

            int code = clBuildProgram(program, 1, deviceArray, "", null, null);
            CL.setExceptionsEnabled(true);

            if (code != CL_SUCCESS) {
                Log.warn("ChunkyCL: Failed to build cached binary (code " + code + "), recompiling...");
                Files.deleteIfExists(cacheFile);
                return null;
            }

            return program;
        } catch (Exception e) {
            CL.setExceptionsEnabled(true);
            Log.warn("ChunkyCL: Error loading cached binary: " + e.getMessage());
            return null;
        }
    }

    private void saveBinaryToCache(cl_program program, String cacheKey, String kernelName) {
        try {
            Path cacheDir = getCacheDir();
            Files.createDirectories(cacheDir);

            long[] binarySize = new long[1];
            clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, Sizeof.size_t, Pointer.to(binarySize), null);

            if (binarySize[0] <= 0) {
                Log.warn("ChunkyCL: Binary size is 0, skipping cache for " + kernelName);
                return;
            }

            byte[] binary = new byte[(int) binarySize[0]];
            Pointer[] binaryPointers = new Pointer[] { Pointer.to(binary) };
            clGetProgramInfo(program, CL_PROGRAM_BINARIES, Sizeof.POINTER, Pointer.to(binaryPointers), null);

            Files.write(cacheDir.resolve(cacheKey + ".bin"), binary);
            Log.info("ChunkyCL: Cached compiled binary for " + kernelName + " (" + binary.length + " bytes)");
        } catch (Exception e) {
            Log.warn("ChunkyCL: Failed to cache binary: " + e.getMessage());
        }
    }

    private static void collectHeaderNames(String source, Set<String> headerNames) {
        Arrays.stream(source.split("\\n"))
              .filter(line -> line.startsWith("#include"))
              .forEach(line -> {
                  String stripped = line.substring("#include".length()).trim();
                  if (stripped.length() >= 2) {
                      String header = stripped.substring(1, stripped.length() - 1);
                      headerNames.add(header);
                  }
              });
    }
}
