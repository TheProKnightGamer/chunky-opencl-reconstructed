// GPU-side fp64 sample accumulator.
//
// This kernel replaces the per-iteration CPU blend that used to read the
// fp32 pass buffer back to host memory and accumulate into a Java
// double[] sampleBuffer. By keeping the accumulator on the GPU, we
// eliminate the per-iteration host transfer and the CPU multiply-add
// pass over millions of pixels. The host only reads the accumulator
// when it needs to redraw the screen (every ~2 s, see the redraw
// cadence logic in OpenClPathTracingRenderer).
//
// PARITY: each pixel's blend is `accum * w1 + pass * w2` performed
// in fp64. Per the OpenCL spec, +, -, *, / on fp64 must be
// correctly rounded round-to-nearest-even, the same as Java's
// fp64 semantics. We disable FP_CONTRACT to prevent the compiler
// from fusing `a*w1 + b*w2` into a single fma, which would round
// differently from the two-step CPU sequence and silently break
// bit-exact parity with the previous code path.
//
// PRECISION: passBuffer is fp32 (the path tracer's natural output);
// the cast `(double)passBuffer[i]` is exact (every fp32 value has a
// unique fp64 representation), and the multiply by w2 is then a
// correctly-rounded fp64 op.
//
// SUPPORT: the kernel uses the cl_khr_fp64 extension which is
// supported by all desktop GPUs from ~2010 onwards. If a device
// doesn't support it, this file's compilation fails and the host
// falls back to the CPU-blend path. The main rayTracer.c kernel
// is unaffected because it lives in a separate program.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#pragma OPENCL FP_CONTRACT OFF

__kernel void accumulate(
    __global double* accum,            // pixelCount * 3 doubles (R, G, B)
    __global const float* passBuffer,  // pixelCount * 4 floats  (R, G, B, alpha)
    const double prevWeight,
    const double passWeight,
    const int pixelCount               // explicit guard so out-of-bounds work-items no-op
) {
    int gid = get_global_id(0);
    if (gid >= pixelCount) return;

    int accIdx = gid * 3;
    int passIdx = gid * 4;

    // Read into named locals so the compiler can't pattern-match the
    // expression to a fused multiply-add even if FP_CONTRACT were
    // ignored. These reads also let the compiler issue them in
    // parallel rather than serialised behind the multiplies.
    double r = accum[accIdx];
    double g = accum[accIdx + 1];
    double b = accum[accIdx + 2];

    // Cast fp32 → fp64 is exact for finite values.
    double pr = (double) passBuffer[passIdx];
    double pg = (double) passBuffer[passIdx + 1];
    double pb = (double) passBuffer[passIdx + 2];

    // Two distinct fp64 multiplies + one fp64 add. Bit-identical to
    // the previous Java loop:
    //   sampleBuffer[i] = sampleBuffer[i] * prevWeight + passBuffer[i] * passWeight;
    accum[accIdx]     = r * prevWeight + pr * passWeight;
    accum[accIdx + 1] = g * prevWeight + pg * passWeight;
    accum[accIdx + 2] = b * prevWeight + pb * passWeight;
}

#pragma OPENCL FP_CONTRACT DEFAULT
