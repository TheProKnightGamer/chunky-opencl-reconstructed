package dev.thatredox.chunkynative.opencl.util;

import dev.thatredox.chunkynative.opencl.context.ClContext;

import static org.jocl.CL.*;

import org.jocl.*;

public class ClFloatBuffer implements AutoCloseable {
    private final ClMemory buffer;
    private final ClContext context;

    public ClFloatBuffer(float[] buffer, int length, ClContext context) {
        if (length == 0) {
            buffer = new float[1];
            length = 1;
        }
        assert buffer.length >= length;

        this.context = context;
        this.buffer = new ClMemory(
                clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * length, Pointer.to(buffer), null));
    }

    public ClFloatBuffer(float[] buffer, ClContext context) {
        this(buffer, buffer.length, context);
    }

    public cl_mem get() {
        return buffer.get();
    }

    public void set(float[] values, int offset) {
        clEnqueueWriteBuffer(context.queue, this.get(), CL_TRUE, (long) Sizeof.cl_float * offset,
                (long) Sizeof.cl_float * values.length, Pointer.to(values),
                0, null, null);
    }

    @Override
    public void close() {
        buffer.close();
    }
}
