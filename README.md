# OpenCL Plugin

This is a plugin for [Chunky](https://github.com/chunky-dev/chunky), which harnesses the power of the GPU with OpenCL 1.2+ to accelerate rendering.

#### Disclaimer: This is not the original; for the original chunky-opencl, go to [chunky-opencl](https://github.com/chunky-dev/chunky-opencl). I, myself, do not know much Java, so this was largely coded by AI through a lot of trial and error.

The goal of this fork is for the GPU renderer to produce the same image as Chunky's CPU path tracer, only much faster.

## Features currently implemented:
 - Full water rendering + settings! Volumetric water, water optics and ice. Water in-scattering follows the scene's sky and sun, so water no longer glows at night.
 - Glass rendering!
 - Emitter Sampling Strategies!
 - Sun Sampling Strategies!
 - Faster Map Using GPU!
 - Emission maps from resource packs, with an `Emissive` tab (see below)!
 - Light blocks (`minecraft:light`) light the scene when emitters are on.
 - Subsurface scattering (leaves), animated textures (fire, lava, portals and so on), biome colours with 3D biomes, and the Hable and UE4 tone-mapping settings.
 - Skymaps and skyboxes baked at their own resolution.
 - Settings that do nothing on the GPU (render threads, CPU load, octree and BVH implementation) are greyed out while the GPU renderer is selected.

### Emission maps (`Emissive` tab)

Blocks can glow only where a resource pack's emission map says so, instead of over the whole texture. A torch's flame glows, its stick doesn't.

* Reads LabPBR maps (alpha of `<texture>_s.png`) and OptiFine maps (`<texture>_e.png`, or the suffix from `optifine/emissive.properties`).
* Add packs (zips or folders) in the `Emissive` tab, or use Chunky's own resource packs.
* Per block, choose `Default`, `Emission map`, `Whole block` or `Off`, and optionally override its strength. The choices are saved with the scene.
* Each pack's maps are converted once and cached, so loading a cached pack is nearly instant.

## Installation

### Note: This requires the `2.5.0` snapshots.
Download `chunky-opencl.jar` from the [latest release](https://github.com/TheProKnightGamer/chunky-opencl-reconstructed/releases/latest). In the Chunky Launcher, expand `Advanced Settings` and click on `Manage plugins`. In the `Plugin Manager` window, click on `Add` and select the downloaded `.jar` file. Click on `Save` and start Chunky as usual.

![image](https://user-images.githubusercontent.com/42661490/116319916-28ef2580-a76c-11eb-9f93-86d444a349fd.png)

New and loaded scenes use `ChunkyClRenderer` (render) and `Chunky CL Preview Renderer` (preview) automatically. You can switch back to the CPU renderer in the `Advanced` tab.

![image](https://user-images.githubusercontent.com/42661490/122492084-fc040580-cf99-11eb-9b08-b166dc25db41.png)

The first launch takes a few minutes while the OpenCL kernels compile. The compiled kernels are cached, so later launches are much faster. A new plugin version that changes the kernels recompiles once.

## Performance

Rough performance with a Blackwell RTX 6000 Pro(I got temporary access to one :D) vs 14700k rendering four regions with all emitter sampling at full HD in march 25, 2026:
<img width="1920" height="1011" alt="Screenshot from 2026-03-25 21-48-53" src="https://github.com/user-attachments/assets/4124bd21-b984-4f1e-8dfc-8009ccc6310f" />
<img width="1920" height="1011" alt="Screenshot from 2026-03-25 21-46-48" src="https://github.com/user-attachments/assets/b7edfa75-ef16-43a0-96cb-cd401eec8825" />
That's 84k samples per second vs 63 million samples per second, making complex scenes take mere minutes instead of several hours at 750x the speed of a high-end CPU!
(I cut the CPU early as it took forever to render just four samples.)

<img width="169" height="100" alt="comparison" src="https://github.com/user-attachments/assets/1c2be74d-f3cf-4911-a16f-f62c10e53a75" />

The map used in the test renders can be found here at [Tax' Future City](https://www.curseforge.com/minecraft/worlds/taxfuturecity)

### Since then

The render now uses a wavefront path tracer: each bounce is split into small kernels (trace, shade, shadow rays) instead of one large kernel per path, which keeps the GPU busier. It produces the same image as the single-kernel renderer.

Measured speedups for v1.0.23 against v1.0.22 (which was already 1.56–1.77x faster than v1.0.19):

| GPU | Scene | Speedup |
| --- | --- | --- |
| RTX 3070, 1080p | 8-region terrain | 1.14x |
| RTX 3070, 1080p | entity-heavy test scene | 1.17x |
| RTX 3070, 1080p | underwater | 1.58x |
| Intel UHD 770 | 8-region terrain | 1.59x |
| Intel UHD 770 | entity-heavy test scene | 1.07x |

Long renders stay under the GPU watchdog (Windows TDR, Linux display GPUs): launches are timed and resized as the render runs, and heavy frames are split across several launches.

Tips:
* Indoor scenes should disable sunlight under `Lighting`
* Draw depth may be adjusted under `Advanced`
* Draw entities may be unchecked under `Advanced`
* The OpenCL device can be selected in the `OpenCL` tab

## Settings (JVM options)

Add these as JVM options in the Chunky Launcher. None are needed for normal use.

| Option | Default | Effect |
| --- | --- | --- |
| `-Dchunkycl.wavefront=false` | `true` | Render with the single-kernel renderer instead of the wavefront one. |
| `-Dchunkycl.wavefront.stateMB=N` | `1024` | GPU memory budget for wavefront path state (432 bytes per pixel, also capped at a sixth of GPU memory). Larger frames render in chunks. |
| `-Dchunkycl.nvMaxRegisters=N` | `128` | NVIDIA only: register cap for the render kernel. `0` leaves it to the compiler. |

## Compatibility

* Not compatible with the Denoising Plugin.
* Scenes with emitter sampling (strategy other than `NONE`) or layered fog use the single-kernel renderer. The log says which renderer is used.
* Blocks switched on in the `Emissive` tab that Chunky doesn't treat as lights (an ore, for example) light their surroundings through path tracing alone, which is noisier.
* Small differences from the CPU renderer remain, mostly from the GPU's single-precision floats.

## Troubleshooting

* **`CL_OUT_OF_RESOURCES` or `CL_INVALID_COMMAND_QUEUE` when a render starts:** on the first render of a scene the log prints a `[ChunkyCL] GPU memory` report listing what each buffer uses. Compare the total with your GPU's free memory, then try fewer loaded chunks, a lower resolution, or a smaller `-Dchunkycl.wavefront.stateMB`.
* **`GPU fp64 sample accumulator unavailable on this device` or `device lacks cl_khr_fp64`:** harmless. Samples are blended on the CPU in full precision instead, with the same result.

---

## Development
This project is setup to work with IntelliJ and CLion. The base directory is intended to be opened in IntelliJ and the `src/main/opencl` directory in CLion.

Build the plugin with `./gradlew build`. It needs Chunky `2.5.0-SNAPSHOT` in your local Maven repository (`./gradlew :chunky:publishToMavenLocal` in a Chunky checkout, as the CI does). The jar is written to `build/libs/chunky-opencl.jar`.

For hot reloading, add `-DchunkyClHotReload="<src/main/opencl directory>"` as a JVM option.

`benchmark/harness` times kernel changes on real scenes and checks that they don't change the image. See its [README](benchmark/harness/README.md).

## Copyright & License
ChunkyCL is Copyright (c) 2021 - 2024, [ThatRedox](https://github.com/ThatRedox) and contributors.

Permission to modify and redistribute is granted under the terms of the GPLv3 license. See the file `LICENSE` for the full license.

ChunkyCL uses the following 3rd party libraries:
* [Chunky](https://github.com/chunky-dev/chunky/)
* [JOCL](http://www.jocl.org/)
* [OpenCL header from the LLVM Project](https://llvm.org)
