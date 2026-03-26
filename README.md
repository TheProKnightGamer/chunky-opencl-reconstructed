# OpenCL Plugin

This is a plugin for [Chunky](https://github.com/chunky-dev/chunky), which harnesses the power of the GPU with OpenCL 1.2+ to accelerate rendering.

#### This is not the original, but a side project. I, myself, do not know much Java, so the majority of this is a test to see how well AI can move features from chunky to chunky-opencl

## Features currently implemented:
 - Full water rendering + settings!
 - Glass rendering!
 - Emitter Sampling Strategies!
 - Sun Sampling Strategies!
 - Faster Map Using GPU!

## Installation

### Note: This requires the `2.5.0` snapshots.
Download the [latest plugin release](https://github.com/chunky-dev/chunky-opencl-reconstructed/releases) and extract it. In the Chunky Launcher, expand `Advanced Settings` and click on `Manage plugins`. In the `Plugin Manager` window, click on `Add` and select the `.jar` file in the extracted zip file. Click on `Save` and start Chunky as usual.

![image](https://user-images.githubusercontent.com/42661490/116319916-28ef2580-a76c-11eb-9f93-86d444a349fd.png)

Select `ChunkyCL` as your renderer for the scene in the `Advanced` tab.


![image](https://user-images.githubusercontent.com/42661490/122492084-fc040580-cf99-11eb-9b08-b166dc25db41.png)

## Performance

Rough performance with an RTX 6000(I got temporary access to one :D) vs 14700k rendering four regions with all emitter sampling at full HD in march 25 2026:

[TO BE INSERTED VERY SOON]

I cut the CPU early as it took forever to render just four samples.


## Compatibility

* Not compatible with the Denoising Plugin.

---

## Copyright & License
ChunkyCL is Copyright (c) 2021 - 2024, [ThatRedox](https://github.com/ThatRedox) and contributors.

Permission to modify and redistribute is granted under the terms of the GPLv3 license. See the file `LICENSE` for the full license.

ChunkyCL uses the following 3rd party libraries:
* [Chunky](https://github.com/chunky-dev/chunky/)
* [JOCL](http://www.jocl.org/)
* [OpenCL header from the LLVM Project](https://llvm.org)
