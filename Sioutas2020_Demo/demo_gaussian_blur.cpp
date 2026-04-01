/**
 * Demo 1: Separable Gaussian Blur
 *
 * A classic 2-stage stencil pipeline: blur_x then blur_y.
 * Scheduling challenge: should blur_x be inlined or computed separately?
 *
 * Report: gaussian_blur_report.txt (or argv[4])
 */

#include "demo_common.h"

// ---- Schedule display (uses ImageParam, GPU target) -----------------------

static Pipeline make_gaussian_blur(const std::string &suffix) {
    ImageParam input(Float(32), 2, "gb_input" + suffix);
    // Estimates on input are required by Mullapudi2016; harmless for others.
    input.set_estimates({{0, 1924}, {0, 1084}});
    Var x("x"), y("y");

    Func blur_x("gb_blur_x" + suffix);
    blur_x(x, y) = (        input(x - 2, y) +
                    4.0f *  input(x - 1, y) +
                    6.0f *  input(x,     y) +
                    4.0f *  input(x + 1, y) +
                            input(x + 2, y)) * (1.0f / 16.0f);

    Func blur_y("gb_blur_y" + suffix);
    blur_y(x, y) = (        blur_x(x, y - 2) +
                    4.0f *  blur_x(x, y - 1) +
                    6.0f *  blur_x(x, y    ) +
                    4.0f *  blur_x(x, y + 1) +
                            blur_x(x, y + 2)) * (1.0f / 16.0f);

    blur_y.set_estimate(x, 2, 1916)
          .set_estimate(y, 2, 1076);

    return Pipeline(blur_y);
}

// ---- CPU benchmark (Buffer-based, no ImageParam binding needed) -----------

static double bench_gaussian_blur(const AutoschedulerParams &params,
                                  const Target &target) {
    const int W = 1920, H = 1080, PAD = 2;

    // Input with border padding so stencil accesses are in-bounds
    Buffer<float> in_buf(W + 2 * PAD, H + 2 * PAD);
    in_buf.set_min(-PAD, -PAD);
    in_buf.fill(0.5f);

    Var x("x"), y("y");
    Func input("gb_b_in");
    input(x, y) = in_buf(x, y);

    Func blur_x("gb_b_bx");
    blur_x(x, y) = (        input(x - 2, y) +
                    4.0f *  input(x - 1, y) +
                    6.0f *  input(x,     y) +
                    4.0f *  input(x + 1, y) +
                            input(x + 2, y)) * (1.0f / 16.0f);

    Func blur_y("gb_b_by");
    blur_y(x, y) = (        blur_x(x, y - 2) +
                    4.0f *  blur_x(x, y - 1) +
                    6.0f *  blur_x(x, y    ) +
                    4.0f *  blur_x(x, y + 1) +
                            blur_x(x, y + 2)) * (1.0f / 16.0f);

    blur_y.set_estimate(x, 0, W).set_estimate(y, 0, H);
    Pipeline(blur_y).apply_autoscheduler(target, params);

    Buffer<float> out(W, H);
    blur_y.realize(out, target);  // warmup + JIT compile

    return time_ms([&]() { blur_y.realize(out, target); });
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    load_all_plugins(argc, argv);
    Report rpt(default_report_path(argc, argv, "gaussian_blur"));

    run_five_way_with_bench(
        rpt,
        "Separable Gaussian Blur 5-tap (1920x1080)",
        make_gaussian_blur,
        bench_gaussian_blur);

    return 0;
}
