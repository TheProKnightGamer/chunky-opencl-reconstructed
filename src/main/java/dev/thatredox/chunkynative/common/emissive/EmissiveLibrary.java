package dev.thatredox.chunkynative.common.emissive;

import se.llbit.log.Log;

import java.io.File;
import java.util.*;

/**
 * The emission masks of the configured packs, merged by priority, kept in memory
 * between scene loads. Reloaded when the pack list or a pack's modification time
 * changes. Conversions are cached on disk by {@link EmissivePacks}.
 */
public final class EmissiveLibrary {
    private static List<String> loadedKey = null;
    private static Map<String, EmissionMask> masks = Collections.emptyMap();

    private EmissiveLibrary() {}

    /** Masks keyed by texture path, for {@link EmissiveSettings#packs()}. May convert packs. */
    public static Map<String, EmissionMask> masks() {
        return masks(EmissiveSettings.packs());
    }

    public static synchronized Map<String, EmissionMask> masks(List<File> packs) {
        List<String> key = new ArrayList<>();
        for (File f : packs) key.add(f.getPath() + "|" + f.length() + "|" + f.lastModified());
        if (key.equals(loadedKey)) {
            return masks;
        }

        Map<String, EmissionMask> merged = new HashMap<>();
        for (File pack : packs) {
            try {
                // The first pack to provide a texture's layer wins.
                EmissivePacks.load(pack).forEach(merged::putIfAbsent);
            } catch (Exception e) {
                Log.warn("ChunkyCL: could not read emission maps from " + pack, e);
            }
        }
        masks = Collections.unmodifiableMap(merged);
        loadedKey = key;
        return masks;
    }

    /** Forget the in-memory masks, for example after the disk cache was cleared. */
    public static synchronized void invalidate() {
        loadedKey = null;
        masks = Collections.emptyMap();
    }
}
