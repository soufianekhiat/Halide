/**
 * Demo 6: Conv2D + ReLU  (4D convolution layer)
 *
 * conv(c,x,y,n) = bias(c) + sum_over(CI,3,3) filter * input
 * relu(c,x,y,n) = max(0, conv)
 *
 * Tests high-dimensional tiling strategies.
 * Sioutas2020 only tiles args[0] and args[1] (c, x) -- y and n are untiled.
 * Report: conv2d_relu_report.txt (or argv[4])
 */

#include "demo_common.h"

// ---- Schedule display -----------------------------------------------------

static Pipeline make_conv2d_relu(const std::string &suffix) {
    const int N = 5, CI = 120, CO = 24, W = 100, H = 80;

    ImageParam input (Float(32), 4, "cr_input"  + suffix);
    ImageParam filter(Float(32), 4, "cr_filter" + suffix);
    ImageParam bias  (Float(32), 1, "cr_bias"   + suffix);
    input.set_estimates({{0, CI}, {0, W+2}, {0, H+2}, {0, N}});
    filter.set_estimates({{0, CO}, {0, 3}, {0, 3}, {0, CI}});
    bias.set_estimates({{0, CO}});

    Var c("c"), x("x"), y("y"), n("n");
    RDom r(0, CI, 0, 3, 0, 3, "r");  // r.x = input channel, r.y/r.z = kernel

    Func conv("cr_conv" + suffix);
    conv(c, x, y, n)  = bias(c);
    conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * input(r.x, x + r.y, y + r.z, n);

    Func relu("cr_relu" + suffix);
    relu(c, x, y, n) = max(0.0f, conv(c, x, y, n));

    relu.set_estimate(c, 0, CO)
        .set_estimate(x, 0, W)
        .set_estimate(y, 0, H)
        .set_estimate(n, 0, N);

    return Pipeline(relu);
}

// ---- CPU benchmark --------------------------------------------------------

static double bench_conv2d_relu(const AutoschedulerParams &params,
                                 const Target &target) {
    // Sioutas2020's CPU scheduler leaves the reduction update unscheduled,
    // causing a thread-pool deadlock (parallel consumer + sequential reduction).
    // On GPU the reduction update is properly gpu_tile'd, so GPU is fine.
    if (params.name == "Sioutas2020" && !target.has_gpu_feature()) return -1.0;
    // Mullapudi2016 crashes (STATUS_STACK_BUFFER_OVERRUN) on this 4D pipeline.
    if (params.name == "Mullapudi2016") return -1.0;

    const int N = 1, CI = 16, CO = 8, W = 128, H = 32;

    // input(r.x, x+r.y, y+r.z, n): dims (CI, W+2, H+2, N)
    Buffer<float> input_buf (CI, W + 2, H + 2, N);
    Buffer<float> filter_buf(CO, 3, 3, CI);
    Buffer<float> bias_buf  (CO);
    input_buf.fill(1.0f);
    filter_buf.fill(1.0f);
    bias_buf.fill(0.0f);

    Var c("c"), x("x"), y("y"), n("n");
    RDom r(0, CI, 0, 3, 0, 3, "r");

    Func conv("cr_b_conv");
    conv(c, x, y, n)  = bias_buf(c);
    conv(c, x, y, n) += filter_buf(c, r.y, r.z, r.x) * input_buf(r.x, x + r.y, y + r.z, n);

    Func relu("cr_b_relu");
    relu(c, x, y, n) = max(0.0f, conv(c, x, y, n));

    relu.set_estimate(c, 0, CO)
        .set_estimate(x, 0, W)
        .set_estimate(y, 0, H)
        .set_estimate(n, 0, N);

    Pipeline(relu).apply_autoscheduler(target, params);

    Buffer<float> out(CO, W, H, N);
    relu.realize(out, target);

    return time_ms([&]() { relu.realize(out, target); });
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    load_all_plugins(argc, argv);
    Report rpt(default_report_path(argc, argv, "conv2d_relu"));

    run_five_way_with_bench(
        rpt,
        "Conv2D + ReLU  CO=24 W=100 H=80 N=5 CI=120 k=3x3",
        make_conv2d_relu,
        bench_conv2d_relu);

    return 0;
}
