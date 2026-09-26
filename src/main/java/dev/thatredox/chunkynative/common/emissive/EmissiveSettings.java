package dev.thatredox.chunkynative.common.emissive;

import se.llbit.chunky.PersistentSettings;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.json.Json;
import se.llbit.json.JsonArray;
import se.llbit.json.JsonMember;
import se.llbit.json.JsonObject;
import se.llbit.json.JsonValue;

import java.io.File;
import java.util.*;

/**
 * Emissive settings. The emission packs are global (like Chunky's resource packs); the
 * per-block choices belong to the scene and are saved in its JSON under
 * {@value #SCENE_KEY}:
 *
 * <pre>{"enabled": true, "blocks": {"minecraft:torch": {"mode": "map", "strength": 2.0}}}</pre>
 *
 * Blocks left on {@link Mode#DEFAULT} with the default strength are not stored.
 */
public final class EmissiveSettings {
    public static final String SCENE_KEY = "chunkyclEmissive";
    private static final String PACKS_KEY = "chunkyclEmissivePacks";
    private static final String USE_CHUNKY_PACKS_KEY = "chunkyclEmissiveUseChunkyPacks";

    public enum Mode {
        /** Emission map if the block glows in Chunky and has one, otherwise Chunky's behaviour. */
        DEFAULT("default", "Default"),
        /** Only the texels the emission map marks glow. */
        MAP("map", "Emission map"),
        /** The whole block glows (Chunky's own behaviour). */
        WHOLE("whole", "Whole block"),
        /** No emission. */
        OFF("off", "Off");

        public final String id;
        public final String label;

        Mode(String id, String label) {
            this.id = id;
            this.label = label;
        }

        static Mode fromId(String id) {
            for (Mode m : values()) {
                if (m.id.equals(id)) return m;
            }
            return DEFAULT;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    /** A block's choice. A NaN strength means "use the block's Chunky emittance". */
    public static final class BlockSetting {
        public static final BlockSetting DEFAULT = new BlockSetting(Mode.DEFAULT, Float.NaN);

        public final Mode mode;
        public final float strength;

        public BlockSetting(Mode mode, float strength) {
            this.mode = mode;
            this.strength = strength;
        }

        public boolean isDefault() {
            return mode == Mode.DEFAULT && Float.isNaN(strength);
        }
    }

    /** One scene's settings, parsed once per GPU scene load. */
    public static final class SceneSettings {
        public final boolean enabled;
        public final Map<String, BlockSetting> blocks;

        SceneSettings(boolean enabled, Map<String, BlockSetting> blocks) {
            this.enabled = enabled;
            this.blocks = blocks;
        }

        public BlockSetting get(String block) {
            return blocks.getOrDefault(block, BlockSetting.DEFAULT);
        }
    }

    private EmissiveSettings() {}

    // ---- global: packs ----

    /** Packs added in the Emissive tab, highest priority first. */
    public static synchronized List<File> extraPacks() {
        List<File> packs = new ArrayList<>();
        JsonValue v = PersistentSettings.settings.get(PACKS_KEY);
        if (v.isArray()) {
            for (JsonValue e : v.array()) {
                String path = e.stringValue("");
                if (!path.isEmpty()) packs.add(new File(path));
            }
        }
        return packs;
    }

    public static synchronized void setExtraPacks(List<File> packs) {
        JsonArray arr = new JsonArray();
        for (File f : packs) arr.add(Json.of(f.getAbsolutePath()));
        PersistentSettings.settings.set(PACKS_KEY, arr);
        PersistentSettings.save();
    }

    public static boolean useChunkyPacks() {
        return PersistentSettings.settings.getBool(USE_CHUNKY_PACKS_KEY, true);
    }

    public static void setUseChunkyPacks(boolean value) {
        PersistentSettings.settings.setBool(USE_CHUNKY_PACKS_KEY, value);
        PersistentSettings.save();
    }

    /**
     * Every pack emission maps are read from, highest priority first: the Emissive tab's
     * packs, then (optionally) Chunky's enabled resource packs. Missing files are skipped.
     */
    public static List<File> packs() {
        LinkedHashSet<File> packs = new LinkedHashSet<>(extraPacks());
        if (useChunkyPacks()) {
            try {
                packs.addAll(PersistentSettings.getEnabledResourcePacks());
            } catch (RuntimeException ignored) {
                // No resource pack setting yet.
            }
        }
        List<File> existing = new ArrayList<>();
        for (File f : packs) {
            if (f.exists()) existing.add(f.getAbsoluteFile());
        }
        return existing;
    }

    // ---- per scene: blocks ----

    public static SceneSettings get(Scene scene) {
        JsonObject root = sceneObject(scene);
        boolean enabled = root.get("enabled").boolValue(true);
        Map<String, BlockSetting> blocks = new HashMap<>();
        JsonValue b = root.get("blocks");
        if (b.isObject()) {
            for (JsonMember m : b.object()) {
                JsonObject o = m.value.object();
                blocks.put(m.name, new BlockSetting(
                        Mode.fromId(o.get("mode").stringValue("default")),
                        o.get("strength").isUnknown() ? Float.NaN : o.get("strength").floatValue(1)));
            }
        }
        return new SceneSettings(enabled, Collections.unmodifiableMap(blocks));
    }

    public static void setEnabled(Scene scene, boolean enabled) {
        JsonObject root = sceneObject(scene).copy();
        root.set("enabled", Json.of(enabled));
        scene.setAdditionalData(SCENE_KEY, root);
    }

    /** Store several block settings at once; default settings are removed. */
    public static void setBlocks(Scene scene, Map<String, BlockSetting> settings) {
        JsonObject root = sceneObject(scene).copy();
        JsonObject blocks = root.get("blocks").isObject() ? root.get("blocks").object() : new JsonObject();
        for (Map.Entry<String, BlockSetting> e : settings.entrySet()) {
            BlockSetting s = e.getValue();
            if (s.isDefault()) {
                blocks.remove(e.getKey());
            } else {
                JsonObject o = new JsonObject();
                o.add("mode", s.mode.id);
                if (!Float.isNaN(s.strength)) o.add("strength", s.strength);
                blocks.set(e.getKey(), o);
            }
        }
        root.set("blocks", blocks);
        // Replace the whole object rather than editing it in place: the render thread's
        // copy of the scene shares additionalData with the UI's scene.
        scene.setAdditionalData(SCENE_KEY, root);
    }

    public static void clearBlocks(Scene scene) {
        JsonObject root = sceneObject(scene).copy();
        root.remove("blocks");
        scene.setAdditionalData(SCENE_KEY, root);
    }

    private static JsonObject sceneObject(Scene scene) {
        JsonValue v = scene.getAdditionalData(SCENE_KEY);
        return v != null && v.isObject() ? v.object() : new JsonObject();
    }
}
