package dev.thatredox.chunkynative.opencl.util;

import dev.thatredox.chunkynative.util.NativeCleaner;
import org.jocl.CL;
import org.jocl.cl_mem;

public class ClMemory implements AutoCloseable {
    protected final NativeCleaner.Cleaner cleaner;
    protected final cl_mem memory;
    protected volatile boolean valid;

    public ClMemory(cl_mem memory) {
        this.cleaner = NativeCleaner.INSTANCE.register(this, () -> CL.clReleaseMemObject(memory));
        this.memory = memory;
        this.valid = true;
    }

    public cl_mem get() {
        if (valid) {
            return memory;
        } else {
            throw new NullPointerException("Attempted to access invalid CL_MEM: " + this.memory.toString());
        }
    }

    @Override
    public void close() {
        // Invalidate before releasing so a concurrent get() can never observe
        // a valid flag paired with an already-released cl_mem.
        this.valid = false;
        // Clean native resource and unregister the cleaner to avoid retaining it
        // in the NativeCleaner list (prevents a memory leak when many resources
        // are allocated and closed explicitly).
        this.cleaner.clean();
        NativeCleaner.INSTANCE.unregister(this.cleaner);
    }
}
