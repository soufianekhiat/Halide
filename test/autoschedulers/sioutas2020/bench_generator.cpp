/**
 * bench_generator.cpp
 *
 * A representative GPU image-processing workload used to benchmark
 * Sioutas2020 against Anderson2021.
 *
 * Pipeline (inspired by the TACO benchmark suite from the paper):
 *   input
 *     └─ 5-tap separable Gaussian blur  (blur_x → blur_y)
 *          └─ Sobel gradient magnitude  (gx, gy → magnitude)
 *
 * The pipeline exercises:
 *   - Multi-stage stencils (separable kernel)
 *   - Intermediate storage decisions (fuse vs. tile)
 *   - Simple pointwise stages (gx, gy as element-wise functions)
 *
 * Compiled twice by CMake:
 *   - autoschedule with Sioutas2020   → sioutas2020_bench_s
 *   - autoschedule with Anderson2021  → sioutas2020_bench_a
 * Both are then benchmarked with RunGenMain --benchmarks=all.
 */

#include "Halide.h"

namespace {

class GaussianSobel : public Halide::Generator<GaussianSobel> {
public:
    Input<Buffer<float, 2>>  input{"input"};
    Output<Buffer<float, 2>> output{"output"};

    void generate() {
        Var x("x"), y("y");

        // ---- 5-tap separable Gaussian blur ---------------------------------
        // Horizontal pass
        Func blur_x("blur_x");
        blur_x(x, y) = (    input(x - 2, y)
                       + 4.0f * input(x - 1, y)
                       + 6.0f * input(x,     y)
                       + 4.0f * input(x + 1, y)
                       +        input(x + 2, y)) * (1.0f / 16.0f);

        // Vertical pass
        Func blur_y("blur_y");
        blur_y(x, y) = (    blur_x(x, y - 2)
                       + 4.0f * blur_x(x, y - 1)
                       + 6.0f * blur_x(x, y    )
                       + 4.0f * blur_x(x, y + 1)
                       +        blur_x(x, y + 2)) * (1.0f / 16.0f);

        // ---- Sobel gradient magnitude --------------------------------------
        Func gx("gx"), gy("gy");
        gx(x, y) = (-1.0f * blur_y(x - 1, y - 1) + blur_y(x + 1, y - 1)
                   - 2.0f * blur_y(x - 1, y    ) + 2.0f * blur_y(x + 1, y    )
                   - 1.0f * blur_y(x - 1, y + 1) + blur_y(x + 1, y + 1));

        gy(x, y) = (-1.0f * blur_y(x - 1, y - 1) - 2.0f * blur_y(x, y - 1) - blur_y(x + 1, y - 1)
                   +        blur_y(x - 1, y + 1) + 2.0f * blur_y(x, y + 1) + blur_y(x + 1, y + 1));

        output(x, y) = sqrt(gx(x, y) * gx(x, y) + gy(x, y) * gy(x, y));
    }

    void schedule() {
        // Default (no-op) schedule; the autoscheduler fills this in.
        // Estimates drive the cost model / tile-size decisions.
        input.set_estimates({{0, 1920}, {0, 1080}});
        output.set_estimates({{0, 1920}, {0, 1080}});
    }
};

// ---------------------------------------------------------------------------
// A heavier workload: bilateral-grid-style local statistics
// (mean + variance in a 5×5 neighbourhood).  Tests pipelines with multiple
// independent reduction stages that the schedulers must decide to fuse or not.
// ---------------------------------------------------------------------------

class LocalStats : public Halide::Generator<LocalStats> {
public:
    Input<Buffer<float, 2>>  input{"input"};
    Output<Buffer<float, 2>> variance{"variance"};

    void generate() {
        Var x("x"), y("y");
        RDom r(-2, 5, -2, 5);

        Func mean("mean"), sq_mean("sq_mean");
        mean(x, y) = sum(input(x + r.x, y + r.y)) * (1.0f / 25.0f);
        sq_mean(x, y) = sum(input(x + r.x, y + r.y) *
                            input(x + r.x, y + r.y)) * (1.0f / 25.0f);

        variance(x, y) = sq_mean(x, y) - mean(x, y) * mean(x, y);
    }

    void schedule() {
        input.set_estimates({{0, 1920}, {0, 1080}});
        variance.set_estimates({{0, 1920}, {0, 1080}});
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(GaussianSobel, gaussian_sobel)
HALIDE_REGISTER_GENERATOR(LocalStats,    local_stats)
