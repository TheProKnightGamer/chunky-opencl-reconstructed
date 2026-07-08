package dev.thatredox.chunkynative.opencl.tonemap;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;
import se.llbit.chunky.renderer.postprocessing.HableToneMappingFilter;
import se.llbit.chunky.renderer.postprocessing.PostProcessingFilter;
import se.llbit.chunky.resources.BitmapImage;
import se.llbit.util.Configurable;
import se.llbit.util.TaskTracker;

import static org.jocl.CL.clSetKernelArg;

public class HableToneMappingImposterGpuPostprocessingFilter extends HableToneMappingFilter
        implements PostProcessingFilter, Configurable {
    private class Inner extends SimpleGpuPostProcessingFilter {
        public Inner(String name, String description, String id, String entryPoint) {
            super(name, description, id, entryPoint);
        }

        @Override
        protected void addArguments(cl_kernel kernel) {
            HableToneMappingImposterGpuPostprocessingFilter outer =
                    HableToneMappingImposterGpuPostprocessingFilter.this;
            float hA = outer.getShoulderStrength();
            float hB = outer.getLinearStrength();
            float hC = outer.getLinearAngle();
            float hD = outer.getToeStrength();
            float hE = outer.getToeNumerator();
            float hF = outer.getToeDenominator();
            float hW = outer.getLinearWhitePointValue();
            float whiteScale = 1.0f / (((hW * (hA * hW + hC * hB) + hD * hE)
                    / (hW * (hA * hW + hB) + hD * hF)) - hE / hF);

            int arg = 5;
            setFloat(kernel, arg++, hA);
            setFloat(kernel, arg++, hB);
            setFloat(kernel, arg++, hC);
            setFloat(kernel, arg++, hD);
            setFloat(kernel, arg++, hE);
            setFloat(kernel, arg++, hF);
            setFloat(kernel, arg, whiteScale);
        }

        private void setFloat(cl_kernel kernel, int arg, float value) {
            clSetKernelArg(kernel, arg, Sizeof.cl_float, Pointer.to(new float[] { value }));
        }
    }

    private final Inner inner;

    public HableToneMappingImposterGpuPostprocessingFilter(HableToneMappingFilter imposter) {
        this.inner = new Inner(imposter.getName(), imposter.getDescription(), imposter.getId(), "hable_filter");
        this.setShoulderStrength(imposter.getShoulderStrength());
        this.setLinearStrength(imposter.getLinearStrength());
        this.setLinearAngle(imposter.getLinearAngle());
        this.setToeStrength(imposter.getToeStrength());
        this.setToeNumerator(imposter.getToeNumerator());
        this.setToeDenominator(imposter.getToeDenominator());
        this.setLinearWhitePointValue(imposter.getLinearWhitePointValue());
    }

    @Override
    public void processFrame(int width, int height, double[] input, BitmapImage output, double exposure, TaskTracker.Task task) {
        inner.processFrame(width, height, input, output, exposure, task);
    }

    @Override
    public String getName() {
        return inner.getName();
    }

    @Override
    public String getDescription() {
        return inner.getDescription();
    }

    @Override
    public String getId() {
        return inner.getId();
    }
}
