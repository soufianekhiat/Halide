/**
 * Demo 5: Unsharp Mask
 *
 * 5x5 box blur -> sharpen -> clamp.
 * Tests stencil + pointwise fusion decisions.
 * Report: unsharp_mask_report.txt (or argv[4])
 */

#include "demo_common.h"

// ---- Schedule display -----------------------------------------------------

static Pipeline make_unsharp_mask(const std::string &suffix) {
    ImageParam input(Float(32), 2, "um_input" + suffix);
    input.set_estimates({{0, 1924}, {0, 1084}});
    Var x("x"), y("y");

    // 5x5 box blur via reduction (pure-identity LHS)
    Func blurred("um_blurred" + suffix);
    RDom r(-2, 5, -2, 5, "r");
    blurred(x, y)  = 0.0f;
    blurred(x, y) += input(x + r.x, y + r.y) * (1.0f / 25.0f);

    // Sharpen: add high-frequency detail back
    Func sharpened("um_sharpened" + suffix);
    sharpened(x, y) = input(x, y) + 1.5f * (input(x, y) - blurred(x, y));

    Func output("um_output" + suffix);
    output(x, y) = clamp(sharpened(x, y), 0.0f, 1.0f);

    output.set_estimate(x, 2, 1916).set_estimate(y, 2, 1076);

    return Pipeline(output);
}

// ---- CPU benchmark --------------------------------------------------------

static double bench_unsharp_mask(const AutoschedulerParams &params,
                                  const Target &target) {
    // Mullapudi2016 crashes (STATUS_STACK_BUFFER_OVERRUN) on this pipeline:
    // the negative-min RDom r(-2,5,-2,5) triggers a bug in its scheduler.
    if (params.name == "Mullapudi2016") return -1.0;

    const int W = 1920, H = 1080, PAD = 2;

    Buffer<float> in_buf(W + 2 * PAD, H + 2 * PAD);
    in_buf.set_min(-PAD, -PAD);
    in_buf.fill(0.5f);

    Var x("x"), y("y");
    Func input("um_b_in");
    input(x, y) = in_buf(x, y);

    Func blurred("um_b_blur");
    RDom r(-2, 5, -2, 5, "r");
    blurred(x, y)  = 0.0f;
    blurred(x, y) += input(x + r.x, y + r.y) * (1.0f / 25.0f);

    Func sharpened("um_b_sharp");
    sharpened(x, y) = input(x, y) + 1.5f * (input(x, y) - blurred(x, y));

    Func output("um_b_out");
    output(x, y) = clamp(sharpened(x, y), 0.0f, 1.0f);

    output.set_estimate(x, 0, W).set_estimate(y, 0, H);
    Pipeline(output).apply_autoscheduler(target, params);

    Buffer<float> out(W, H);
    output.realize(out, target);

    return time_ms([&]() { output.realize(out, target); });
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    load_all_plugins(argc, argv);
    Report rpt(default_report_path(argc, argv, "unsharp_mask"));

    run_five_way_with_bench(
        rpt,
        "Unsharp Mask 5x5 (1920x1080, blur+sharpen+clamp)",
        make_unsharp_mask,
        bench_unsharp_mask);

    return 0;
}
