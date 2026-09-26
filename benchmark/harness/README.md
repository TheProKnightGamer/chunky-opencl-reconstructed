# Kernel A/B benchmark harness

`ClBench` times the render kernel on a real Chunky scene and checks that a change did
not alter the image. It is how every kernel optimisation in this plugin was measured.

It loads a scene through Chunky's own headless path, uploads it with the plugin's
`ClSceneLoader`, and then, for each kernel **variant** (a copy of
`src/main/opencl/kernel/include`, optionally with extra build options):

- times full frames, **interleaving** the variants round-robin, so background load on a
  shared GPU affects every variant equally. Compare variants within one run, not across
  runs.
- renders the same seeds with every variant and compares the images pixel for pixel
  against the first variant. `exact=100.00%` means bit-identical output.
- optionally (`preview=1`) runs and compares the preview kernel too.

## Setup (once)

Needs Java 17, Chunky's jars in `~/.chunky/lib`, and a built plugin jar
(`./gradlew jar`, which the harness loads the plugin classes from).

```
benchmark/harness/setup.sh big4k=$HOME/.chunky/scenes/<scene dir> water=...
```

Each `name=dir` links a saved Chunky scene (its `.octree2` dump; `spp` is reset to 0).
The repo's `benchmark/OpenCL_test` scene is always added as `cltest`. Everything the
harness writes lives in `BENCH_DIR` (default `~/.cache/chunkycl-bench`), including an
isolated Chunky home, so your own Chunky settings are never touched.

## Use

Make a variant by copying the kernel sources somewhere and editing the copy. Then build
the variants into the binary cache. This uses one core and no GPU time, because NVIDIA
builds of the render kernel take 30-120 s:

```
V=~/.cache/chunkycl-bench/variants
benchmark/harness/compile.sh "variants=$V/a/kernel/include;$V/b/kernel/include|-cl-nv-maxrregcount=128" names=a,b_r128
```

Time and compare them (the arguments must match the compile step, or it rebuilds):

```
benchmark/harness/run.sh big4k "variants=...same as above..." names=a,b_r128 \
    width=1920 height=1080 passes=2 rounds=3 refSpp=2 preview=1 out=/tmp/imgs
```

Output:

```
RESULT a        median 1718.1 ms/pass ... speedup x1.000
RESULT b_r128   median 1146.7 ms/pass ... speedup x1.498
IMAGE  b_r128   mean=... meanAbsDiff=0.000000 maxAbs=0.0000 exact=100.00% nan/inf=0
```

A variant spec is `includeDir[|extra build options[|key=val,...]]`, where the keys are
`wg=N` (explicit work-group size), `matCache=N` (LDS material-cache words) and `diag=1`.
See the header of `ClBench.java` for every argument.

### SIMT diagnostic (`diag=1`)

A `diag=1` variant must take one extra trailing `__global uint*` argument. Its kernel
can count how many of a warp's 32 lanes are active at a point, with NVIDIA inline PTX:

```c
#define DIAG(site) do { \
    uint _m; asm volatile("activemask.b32 %0;" : "=r"(_m)); \
    uint _l; asm volatile("mov.u32 %0, %%laneid;" : "=r"(_l)); \
    if (_l == (31 - clz(_m & (0u - _m)))) { atomic_add(&diagBuf[(site)*2], popcount(_m)); \
                                           atomic_add(&diagBuf[(site)*2+1], 1u); } \
} while (0)
```

The harness then prints the average active lanes per site. This is how the low SIMD
utilisation of the render kernel's trace calls was found.

## Checking what the plugin really builds

The harness builds each variant with one `clBuildProgram` call. The plugin instead
compiles, links and caches (`ClContext.loadProgram`), and NVIDIA only applies
code-generation options such as `-cl-nv-maxrregcount` at link time and when a cached
binary is built. `RegCheck` goes through the plugin's own path and prints the render
kernel's private memory, which shows whether the register cap took effect (about
1.65 KB when capped at 128, about 1.1 KB at 255). Run it twice: the second run loads
from the plugin's binary cache.

```
source benchmark/harness/env.sh
$JAVA_HOME/bin/javac -d $BENCH_DIR/classes -cp "$CP" benchmark/harness/RegCheck.java
$JAVA_HOME/bin/java -Dchunky.home=$BENCH_HOME -DchunkyClHotReload=$REPO/src/main/opencl \
    -cp "$CP" RegCheck
```

## Notes

- On NVIDIA, `-cl-nv-verbose` is added to every source build, and the render kernel's
  register and spill counts are printed (`ptxas ... bytes spill stores`).
- Binaries are cached per device, so the same variants can be built for another device
  by pointing `BENCH_HOME` at a Chunky home with a different `clDevice`, for example an
  Intel iGPU. Leave out the NVIDIA-only `-cl-nv-*` options there.
- The harness runs from a snapshot of the plugin jar taken by `setup.sh`. Rerun
  `setup.sh` after rebuilding the jar.
- A crashing variant (for example, an out-of-bounds read) surfaces as
  `CL_INVALID_COMMAND_QUEUE`. The kernel log (`journalctl -k | grep Xid`) says whether it
  was an MMU fault (Xid 31) rather than a watchdog timeout.
- On a shared machine, wrap `run.sh` in whatever GPU lock is in use.

## Emission maps

`make_emissive_testpack.py <minecraft client jar> <out.zip>` builds a small synthetic
LabPBR/OptiFine emission pack from vanilla textures. To render with it, add the zip to
`chunkyclEmissivePacks` in `$BENCH_HOME/chunky.json`, and put per-block choices in a
scene's `additionalData.chunkyclEmissive` (the format is documented in
`EmissiveSettings`). Emission is easiest to judge at night, so copy a `*_night` scene
and run it with `emitters=1`.
