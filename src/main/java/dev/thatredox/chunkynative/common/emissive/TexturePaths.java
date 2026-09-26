package dev.thatredox.chunkynative.common.emissive;

import se.llbit.chunky.resources.Texture;
import se.llbit.chunky.resources.TexturePackLoader;
import se.llbit.chunky.resources.texturepack.AnimatedTextureLoader;
import se.llbit.chunky.resources.texturepack.SimpleTexture;
import se.llbit.chunky.resources.texturepack.TextureLoader;
import se.llbit.log.Log;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.*;

/**
 * Maps Chunky's block {@link Texture} objects back to the resource pack paths they are
 * loaded from, so the matching emissive layer can be found.
 *
 * <p>Chunky registers every texture in {@link TexturePackLoader#ALL_TEXTURES} as a
 * loader holding a path and the Texture it fills. Only plain single-file loaders
 * ({@link SimpleTexture}, {@link AnimatedTextureLoader}) are paired: textures that are
 * composed, rotated or cut out of a larger image would need the same transform applied
 * to their mask. Wrappers such as alternatives are searched, primary path first.
 */
public final class TexturePaths {
    private static Map<Texture, List<String>> paths = null;

    private TexturePaths() {}

    /** Resource paths without extension, most preferred first. Empty if unknown. */
    public static synchronized List<String> of(Texture texture) {
        if (paths == null) {
            paths = build();
        }
        return paths.getOrDefault(texture, Collections.emptyList());
    }

    private static Map<Texture, List<String>> build() {
        Map<Texture, List<String>> map = new IdentityHashMap<>();
        Set<Object> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (TextureLoader loader : TexturePackLoader.ALL_TEXTURES.values()) {
            walk(loader, map, seen);
        }
        return map;
    }

    private static void walk(Object loader, Map<Texture, List<String>> map, Set<Object> seen) {
        if (loader == null || !seen.add(loader)) return;
        Class<?> cls = loader.getClass();
        if (cls == SimpleTexture.class || cls == AnimatedTextureLoader.class) {
            String file = field(loader, "file", String.class);
            Texture texture = field(loader, "texture", Texture.class);
            if (file != null && texture != null) {
                List<String> list = map.computeIfAbsent(texture, t -> new ArrayList<>(1));
                if (!list.contains(file)) list.add(file);
            }
            return;
        }
        // Wrappers (alternatives, conditionals, ...): recurse into their loaders in
        // declaration order, which puts the primary path first.
        for (Class<?> c = cls; c != null && c != Object.class; c = c.getSuperclass()) {
            for (Field f : c.getDeclaredFields()) {
                if (Modifier.isStatic(f.getModifiers())) continue;
                try {
                    if (TextureLoader.class.isAssignableFrom(f.getType())) {
                        f.setAccessible(true);
                        walk(f.get(loader), map, seen);
                    } else if (f.getType().isArray()
                            && TextureLoader.class.isAssignableFrom(f.getType().getComponentType())) {
                        f.setAccessible(true);
                        Object[] arr = (Object[]) f.get(loader);
                        if (arr != null) for (Object o : arr) walk(o, map, seen);
                    }
                } catch (ReflectiveOperationException | RuntimeException e) {
                    Log.warn("ChunkyCL: could not inspect texture loader " + cls.getName() + ": " + e);
                }
            }
        }
    }

    private static <T> T field(Object obj, String name, Class<T> type) {
        for (Class<?> c = obj.getClass(); c != null; c = c.getSuperclass()) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                Object v = f.get(obj);
                return type.isInstance(v) ? type.cast(v) : null;
            } catch (NoSuchFieldException e) {
                // try the superclass
            } catch (ReflectiveOperationException | RuntimeException e) {
                return null;
            }
        }
        return null;
    }
}
