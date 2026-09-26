package dev.thatredox.chunkynative.common.emissive;

import se.llbit.chunky.PersistentSettings;
import se.llbit.log.Log;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Reads the emissive layers of a resource pack and converts them to {@link EmissionMask}s.
 *
 * <p>Two formats are understood:
 * <ul>
 *   <li>LabPBR: the alpha channel of {@code <texture>_s.png}. 0-254 is the emission
 *       strength, 255 means no emission.</li>
 *   <li>OptiFine: the alpha of the {@code <texture><suffix>.png} overlay, where the suffix
 *       comes from {@code assets/minecraft/optifine/emissive.properties} (usually
 *       {@code _e}).</li>
 * </ul>
 *
 * <p>Decoding every PNG of a high resolution pack takes seconds, so the converted masks
 * are cached, one gzip file per pack, and reused until the pack file changes. Masks
 * that are empty everywhere are dropped.
 */
public final class EmissivePacks {
    private static final int MAGIC = 0x43434C45; // "CCLE"
    private static final int VERSION = 1;
    private static final String LABPBR_SUFFIX = "_s";
    private static final String OPTIFINE_PROPERTIES = "assets/minecraft/optifine/emissive.properties";

    private EmissivePacks() {}

    /**
     * The emission masks of one pack, keyed by texture path without extension, for
     * example {@code assets/minecraft/textures/block/redstone_ore}.
     */
    public static Map<String, EmissionMask> load(File pack) throws IOException {
        String signature = signature(pack);
        Path cacheFile = cacheFile(pack);
        Map<String, EmissionMask> masks = readCache(cacheFile, pack, signature);
        if (masks != null) {
            return masks;
        }

        long start = System.nanoTime();
        masks = convert(pack);
        Log.info(String.format("ChunkyCL: converted %d emission maps from %s in %d ms",
                masks.size(), pack.getName(), (System.nanoTime() - start) / 1_000_000));
        try {
            writeCache(cacheFile, pack, signature, masks);
        } catch (IOException e) {
            Log.warn("ChunkyCL: could not cache the emission maps of " + pack, e);
        }
        return masks;
    }

    /** Delete every cached conversion, so packs are read again on next use. */
    public static void clearCache() {
        Path dir = cacheDir();
        if (!Files.isDirectory(dir)) return;
        try (Stream<Path> files = Files.list(dir)) {
            for (Path f : files.collect(Collectors.toList())) {
                Files.deleteIfExists(f);
            }
        } catch (IOException e) {
            Log.warn("ChunkyCL: could not clear the emission map cache", e);
        }
    }

    private static Path cacheDir() {
        return PersistentSettings.cacheDirectory().toPath().resolve("chunkycl-emissive");
    }

    private static Path cacheFile(File pack) throws IOException {
        return cacheDir().resolve(sha1(pack.getCanonicalPath()).substring(0, 20) + ".bin");
    }

    // ---- pack access ----

    /** A resource pack, zipped or unpacked, with file names relative to its root. */
    private interface PackSource extends Closeable {
        /** Normalized name ("assets/...") to the pack's own entry. */
        Map<String, String> files();

        byte[] read(String entry) throws IOException;
    }

    private static PackSource open(File pack) throws IOException {
        if (pack.isDirectory()) {
            Path root = pack.toPath();
            Map<String, String> files = new HashMap<>();
            try (Stream<Path> walk = Files.walk(root)) {
                walk.filter(Files::isRegularFile).forEach(p -> {
                    String rel = root.relativize(p).toString().replace(File.separatorChar, '/');
                    String name = normalize(rel);
                    if (name != null) files.put(name, rel);
                });
            }
            return new PackSource() {
                @Override public Map<String, String> files() { return files; }
                @Override public byte[] read(String entry) throws IOException {
                    return Files.readAllBytes(root.resolve(entry));
                }
                @Override public void close() {}
            };
        }

        ZipFile zip = new ZipFile(pack);
        Map<String, String> files = new HashMap<>();
        Enumeration<? extends ZipEntry> entries = zip.entries();
        while (entries.hasMoreElements()) {
            ZipEntry e = entries.nextElement();
            if (e.isDirectory()) continue;
            String name = normalize(e.getName());
            if (name != null) files.put(name, e.getName());
        }
        return new PackSource() {
            @Override public Map<String, String> files() { return files; }
            @Override public byte[] read(String entry) throws IOException {
                ZipEntry e = zip.getEntry(entry);
                if (e == null) throw new FileNotFoundException(entry);
                try (InputStream in = zip.getInputStream(e)) {
                    return in.readAllBytes();
                }
            }
            @Override public void close() throws IOException { zip.close(); }
        };
    }

    /** Strip anything before "assets/", as some packs are zipped inside a folder. */
    private static String normalize(String name) {
        if (name.startsWith("assets/")) return name;
        int i = name.indexOf("/assets/");
        return i >= 0 ? name.substring(i + 1) : null;
    }

    private static boolean isTexture(String name) {
        return name.endsWith(".png") && name.indexOf("/textures/") > 0;
    }

    // ---- conversion ----

    private static final class Job {
        final String key;
        final String entry;
        final boolean labPbr;

        Job(String key, String entry, boolean labPbr) {
            this.key = key;
            this.entry = entry;
            this.labPbr = labPbr;
        }
    }

    private static Map<String, EmissionMask> convert(File pack) throws IOException {
        try (PackSource src = open(pack)) {
            Map<String, String> files = src.files();
            String optifineSuffix = optifineSuffix(src, files);

            // LabPBR first: if a pack has both layers for a texture, _s wins.
            Map<String, Job> jobs = new LinkedHashMap<>();
            for (Map.Entry<String, String> f : files.entrySet()) {
                String name = f.getKey();
                if (!isTexture(name)) continue;
                String base = name.substring(0, name.length() - 4);
                if (base.endsWith(LABPBR_SUFFIX)) {
                    String key = base.substring(0, base.length() - LABPBR_SUFFIX.length());
                    jobs.put(key, new Job(key, f.getValue(), true));
                }
            }
            if (optifineSuffix != null) {
                for (Map.Entry<String, String> f : files.entrySet()) {
                    String name = f.getKey();
                    if (!isTexture(name)) continue;
                    String base = name.substring(0, name.length() - 4);
                    if (base.endsWith(optifineSuffix)) {
                        String key = base.substring(0, base.length() - optifineSuffix.length());
                        jobs.putIfAbsent(key, new Job(key, f.getValue(), false));
                    }
                }
            }

            Map<String, EmissionMask> masks = new HashMap<>();
            List<Job> list = new ArrayList<>(jobs.values());
            List<EmissionMask> converted = list.parallelStream()
                    .map(job -> convert(src, job))
                    .collect(Collectors.toList());
            for (int i = 0; i < list.size(); i++) {
                if (converted.get(i) != null) masks.put(list.get(i).key, converted.get(i));
            }
            return masks;
        }
    }

    private static EmissionMask convert(PackSource src, Job job) {
        try {
            BufferedImage img = ImageIO.read(new ByteArrayInputStream(src.read(job.entry)));
            if (img == null) return null;
            int w = img.getWidth();
            int h = img.getHeight();
            int[] argb = img.getRGB(0, 0, w, h, null, 0, w);
            byte[] mask = new byte[w * h];
            for (int i = 0; i < argb.length; i++) {
                int a = argb[i] >>> 24;
                int m = job.labPbr ? (a == 255 ? 0 : (a * 255 + 127) / 254) : a;
                mask[i] = (byte) m;
            }
            return EmissionMask.isEmpty(mask) ? null : new EmissionMask(w, h, mask);
        } catch (IOException | RuntimeException e) {
            Log.warn("ChunkyCL: could not read emission map " + job.entry + ": " + e.getMessage());
            return null;
        }
    }

    private static String optifineSuffix(PackSource src, Map<String, String> files) {
        String entry = files.get(OPTIFINE_PROPERTIES);
        if (entry == null) return null;
        try {
            Properties p = new Properties();
            p.load(new StringReader(new String(src.read(entry), StandardCharsets.UTF_8)));
            String suffix = p.getProperty("suffix.emissive", "").trim();
            return suffix.isEmpty() ? null : suffix;
        } catch (IOException e) {
            return null;
        }
    }

    // ---- cache ----

    /** Changes whenever the pack's emissive layers may have changed. */
    private static String signature(File pack) throws IOException {
        if (!pack.isDirectory()) {
            return "zip|" + pack.length() + "|" + pack.lastModified();
        }
        StringBuilder sb = new StringBuilder();
        Path root = pack.toPath();
        try (Stream<Path> walk = Files.walk(root)) {
            List<Path> files = walk.filter(Files::isRegularFile).sorted().collect(Collectors.toList());
            for (Path p : files) {
                String rel = root.relativize(p).toString().replace(File.separatorChar, '/');
                if (!rel.endsWith(".png") && !rel.endsWith(".properties")) continue;
                sb.append(rel).append('|').append(Files.size(p)).append('|')
                        .append(Files.getLastModifiedTime(p).toMillis()).append('\n');
            }
        }
        return "dir|" + sha1(sb.toString());
    }

    private static Map<String, EmissionMask> readCache(Path file, File pack, String signature) {
        if (!Files.isRegularFile(file)) return null;
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(
                new GZIPInputStream(Files.newInputStream(file), 1 << 16)))) {
            if (in.readInt() != MAGIC || in.readInt() != VERSION) return null;
            if (!in.readUTF().equals(pack.getCanonicalPath())) return null;
            if (!in.readUTF().equals(signature)) return null;
            int count = in.readInt();
            Map<String, EmissionMask> masks = new HashMap<>(count * 2);
            for (int i = 0; i < count; i++) {
                String key = in.readUTF();
                int w = in.readInt();
                int h = in.readInt();
                byte[] data = new byte[w * h];
                in.readFully(data);
                masks.put(key, new EmissionMask(w, h, data));
            }
            return masks;
        } catch (IOException | RuntimeException e) {
            Log.warn("ChunkyCL: ignoring unreadable emission map cache " + file + ": " + e.getMessage());
            return null;
        }
    }

    private static void writeCache(Path file, File pack, String signature,
                                   Map<String, EmissionMask> masks) throws IOException {
        Files.createDirectories(file.getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(
                new GZIPOutputStream(Files.newOutputStream(tmp), 1 << 16)))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeUTF(pack.getCanonicalPath());
            out.writeUTF(signature);
            out.writeInt(masks.size());
            for (Map.Entry<String, EmissionMask> e : masks.entrySet()) {
                out.writeUTF(e.getKey());
                out.writeInt(e.getValue().width);
                out.writeInt(e.getValue().height);
                out.write(e.getValue().data);
            }
        }
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private static String sha1(String s) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-1").digest(s.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder();
            for (byte b : d) hex.append(String.format("%02x", b));
            return hex.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
