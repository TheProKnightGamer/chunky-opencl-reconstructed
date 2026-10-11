import dev.thatredox.chunkynative.opencl.context.ClContext;
import dev.thatredox.chunkynative.opencl.context.Device;
import dev.thatredox.chunkynative.opencl.context.KernelLoader;
import org.jocl.*;

import static org.jocl.CL.*;

/**
 * Builds the render program through the plugin's own ClContext/KernelLoader path (run
 * with -DchunkyClHotReload=<repo>/src/main/opencl) and prints the render kernel's
 * private memory, i.e. what the driver really generated. Run twice: the second run
 * loads from the plugin's binary cache, which must give the same result.
 *
 * <p>Also builds the wavefront programs with the plugin's register caps (80 for
 * wfExtend, 96 for wfConnect, as ContextManager.Wavefront): at 80 registers
 * wfExtend has a 560-byte stack frame, uncapped 528, so the private memory shows
 * whether a cap took effect.
 */
public class RegCheck {
    public static void main(String[] args) {
        CL.setExceptionsEnabled(true);
        Device device = Device.getPreferredDevice();
        ClContext ctx = new ClContext(device);
        long t = System.nanoTime();
        cl_program prog = KernelLoader.loadProgram(ctx, "kernel", "rayTracer.c");
        cl_kernel k = clCreateKernel(prog, "render", null);
        long[] v = new long[1];
        clGetKernelWorkGroupInfo(k, device.device, CL_KERNEL_PRIVATE_MEM_SIZE, Sizeof.cl_ulong, Pointer.to(v), null);
        long priv = v[0];
        clGetKernelWorkGroupInfo(k, device.device, CL_KERNEL_WORK_GROUP_SIZE, Sizeof.size_t, Pointer.to(v), null);
        System.out.printf("REGCHECK %s: render private=%dB maxWG=%d (%.0f s)%n", device.name(), priv, v[0],
                (System.nanoTime() - t) / 1e9);
        report(device, KernelLoader.loadProgram(ctx, "kernel", "wfExtend.c", 80), "wfExtend");
        report(device, KernelLoader.loadProgram(ctx, "kernel", "wfConnect.c", 96), "wfConnect");
        report(device, KernelLoader.loadProgram(ctx, "kernel", "wavefront.c"), "wfShade");
        System.exit(0);
    }

    static void report(Device device, cl_program prog, String kernel) {
        cl_kernel k = clCreateKernel(prog, kernel, null);
        long[] v = new long[1];
        clGetKernelWorkGroupInfo(k, device.device, CL_KERNEL_PRIVATE_MEM_SIZE, Sizeof.cl_ulong, Pointer.to(v), null);
        System.out.printf("REGCHECK %s: %s private=%dB%n", device.name(), kernel, v[0]);
    }
}
