/**
 * Demo 8: Local Laplacian Filters
 *
 * Multi-scale image processing using Laplacian pyramids:
 *   remap LUT -> gray -> Gaussian pyramids (input + processed)
 *   -> Laplacian pyramids -> blend -> reconstruct -> color restore
 *
 * 8-level pyramid with levels*256 LUT entries.
 * From Halide apps/local_laplacian and the Sioutas2020 TACO paper.
 * Tests deep multi-stage pipelines with many intermediates.
 *
 * Report: local_laplacian_report.txt (or argv[6])
 */

#include "demo_common.h"

// ---- Helpers: downsample / upsample (match the Halide app) ----------------

static Func downsample(Func f) {
    using Halide::_;
    Var x("x"), y("y");
    Func downx, downy;
    downy(x, y, _) = (f(x, 2 * y - 1, _) +
                       3.0f * (f(x, 2 * y, _) + f(x, 2 * y + 1, _)) +
                       f(x, 2 * y + 2, _)) / 8.0f;
    downx(x, y, _) = (downy(2 * x - 1, y, _) +
                       3.0f * (downy(2 * x, y, _) + downy(2 * x + 1, y, _)) +
                       downy(2 * x + 2, y, _)) / 8.0f;
    return downx;
}

static Func upsample(Func f) {
    using Halide::_;
    Var x("x"), y("y");
    Func upx, upy;
    upx(x, y, _) = lerp(f((x + 1) / 2, y, _), f((x - 1) / 2, y, _),
                         ((x % 2) * 2 + 1) / 4.0f);
    upy(x, y, _) = lerp(upx(x, (y + 1) / 2, _), upx(x, (y - 1) / 2, _),
                         ((y % 2) * 2 + 1) / 4.0f);
    return upy;
}

// ---- Schedule display -----------------------------------------------------

static const int J = 4;  // pyramid levels (reduced for schedule display speed)

static Pipeline make_local_laplacian(const std::string &suffix) {
    const int W = 384, H = 640;
    const int LEVELS = 8;

    ImageParam input(UInt(16), 3, "ll_input" + suffix);
    input.set_estimates({{0, W}, {0, H}, {0, 3}});

    Var x("x"), y("y"), c("c"), k("k");

    // Remap LUT
    Func remap("ll_remap" + suffix);
    Expr fx = cast<float>(x) / 256.0f;
    float alpha = 1.0f, beta = 1.0f;
    remap(x) = alpha * fx * exp(-fx * fx / 2.0f);

    // Boundary condition + float conversion
    Func clamped("ll_clamped" + suffix);
    clamped = BoundaryConditions::repeat_edge(input);

    Func floating("ll_floating" + suffix);
    floating(x, y, c) = clamped(x, y, c) / 65535.0f;

    // Luminance
    Func gray("ll_gray" + suffix);
    gray(x, y) = 0.299f * floating(x, y, 0) +
                 0.587f * floating(x, y, 1) +
                 0.114f * floating(x, y, 2);

    // Processed Gaussian pyramid
    Func gPyramid[J];
    Expr level = k * (1.0f / (LEVELS - 1));
    Expr idx = gray(x, y) * cast<float>(LEVELS - 1) * 256.0f;
    idx = clamp(cast<int>(idx), 0, (LEVELS - 1) * 256);

    gPyramid[0] = Func("ll_gPyr0" + suffix);
    gPyramid[0](x, y, k) = beta * (gray(x, y) - level) + level + remap(idx - 256 * k);
    for (int j = 1; j < J; j++) {
        gPyramid[j] = Func("ll_gPyr" + std::to_string(j) + suffix);
        gPyramid[j](x, y, k) = downsample(gPyramid[j - 1])(x, y, k);
    }

    // Laplacian pyramid
    Func lPyramid[J];
    lPyramid[J - 1] = Func("ll_lPyr" + std::to_string(J - 1) + suffix);
    lPyramid[J - 1](x, y, k) = gPyramid[J - 1](x, y, k);
    for (int j = J - 2; j >= 0; j--) {
        lPyramid[j] = Func("ll_lPyr" + std::to_string(j) + suffix);
        lPyramid[j](x, y, k) = gPyramid[j](x, y, k) - upsample(gPyramid[j + 1])(x, y, k);
    }

    // Input Gaussian pyramid
    Func inGPyramid[J];
    inGPyramid[0] = Func("ll_inGPyr0" + suffix);
    inGPyramid[0](x, y) = gray(x, y);
    for (int j = 1; j < J; j++) {
        inGPyramid[j] = Func("ll_inGPyr" + std::to_string(j) + suffix);
        inGPyramid[j](x, y) = downsample(inGPyramid[j - 1])(x, y);
    }

    // Output Laplacian pyramid (blend)
    Func outLPyramid[J];
    for (int j = 0; j < J; j++) {
        outLPyramid[j] = Func("ll_outLPyr" + std::to_string(j) + suffix);
        Expr lev = inGPyramid[j](x, y) * cast<float>(LEVELS - 1);
        Expr li = clamp(cast<int>(lev), 0, LEVELS - 2);
        Expr lf = lev - cast<float>(li);
        outLPyramid[j](x, y) = (1.0f - lf) * lPyramid[j](x, y, li) +
                                 lf * lPyramid[j](x, y, li + 1);
    }

    // Output Gaussian pyramid (reconstruct)
    Func outGPyramid[J];
    outGPyramid[J - 1] = Func("ll_outGPyr" + std::to_string(J - 1) + suffix);
    outGPyramid[J - 1](x, y) = outLPyramid[J - 1](x, y);
    for (int j = J - 2; j >= 0; j--) {
        outGPyramid[j] = Func("ll_outGPyr" + std::to_string(j) + suffix);
        outGPyramid[j](x, y) = upsample(outGPyramid[j + 1])(x, y) + outLPyramid[j](x, y);
    }

    // Color restoration
    Func color("ll_color" + suffix);
    float eps = 0.01f;
    color(x, y, c) = clamped(x, y, c) * (outGPyramid[0](x, y) + eps) / (gray(x, y) + eps);

    // Convert back to uint16
    Func output("ll_output" + suffix);
    output(x, y, c) = cast<uint16_t>(clamp(color(x, y, c), 0.0f, 65535.0f));

    output.set_estimate(x, 0, W)
          .set_estimate(y, 0, H)
          .set_estimate(c, 0, 3);

    return Pipeline(output);
}

// ---- CPU benchmark --------------------------------------------------------

static double bench_local_laplacian(const AutoschedulerParams &params,
                                     const Target &target) {
    const int W = 256, H = 256;
    const int LEVELS = 8;

    Buffer<uint16_t> input_buf(W, H, 3);
    input_buf.fill((uint16_t)32768);

    Var x("x"), y("y"), c("c"), k("k");

    Func clamped("ll_b_clamped");
    clamped = BoundaryConditions::repeat_edge(input_buf);

    Func floating("ll_b_floating");
    floating(x, y, c) = clamped(x, y, c) / 65535.0f;

    Func gray("ll_b_gray");
    gray(x, y) = 0.299f * floating(x, y, 0) +
                 0.587f * floating(x, y, 1) +
                 0.114f * floating(x, y, 2);

    float alpha = 1.0f, beta = 1.0f;
    Func remap("ll_b_remap");
    Expr bfx = cast<float>(x) / 256.0f;
    remap(x) = alpha * bfx * exp(-bfx * bfx / 2.0f);

    Func gPyramid[J];
    Expr level = k * (1.0f / (LEVELS - 1));
    Expr idx = gray(x, y) * cast<float>(LEVELS - 1) * 256.0f;
    idx = clamp(cast<int>(idx), 0, (LEVELS - 1) * 256);

    gPyramid[0] = Func("ll_b_gPyr0");
    gPyramid[0](x, y, k) = beta * (gray(x, y) - level) + level + remap(idx - 256 * k);
    for (int j = 1; j < J; j++) {
        gPyramid[j] = Func("ll_b_gPyr" + std::to_string(j));
        gPyramid[j](x, y, k) = downsample(gPyramid[j - 1])(x, y, k);
    }

    Func lPyramid[J];
    lPyramid[J - 1] = Func("ll_b_lPyr" + std::to_string(J - 1));
    lPyramid[J - 1](x, y, k) = gPyramid[J - 1](x, y, k);
    for (int j = J - 2; j >= 0; j--) {
        lPyramid[j] = Func("ll_b_lPyr" + std::to_string(j));
        lPyramid[j](x, y, k) = gPyramid[j](x, y, k) - upsample(gPyramid[j + 1])(x, y, k);
    }

    Func inGPyramid[J];
    inGPyramid[0] = Func("ll_b_inGPyr0");
    inGPyramid[0](x, y) = gray(x, y);
    for (int j = 1; j < J; j++) {
        inGPyramid[j] = Func("ll_b_inGPyr" + std::to_string(j));
        inGPyramid[j](x, y) = downsample(inGPyramid[j - 1])(x, y);
    }

    Func outLPyramid[J];
    for (int j = 0; j < J; j++) {
        outLPyramid[j] = Func("ll_b_outLPyr" + std::to_string(j));
        Expr lev = inGPyramid[j](x, y) * cast<float>(LEVELS - 1);
        Expr li = clamp(cast<int>(lev), 0, LEVELS - 2);
        Expr lf = lev - cast<float>(li);
        outLPyramid[j](x, y) = (1.0f - lf) * lPyramid[j](x, y, li) +
                                 lf * lPyramid[j](x, y, li + 1);
    }

    Func outGPyramid[J];
    outGPyramid[J - 1] = Func("ll_b_outGPyr" + std::to_string(J - 1));
    outGPyramid[J - 1](x, y) = outLPyramid[J - 1](x, y);
    for (int j = J - 2; j >= 0; j--) {
        outGPyramid[j] = Func("ll_b_outGPyr" + std::to_string(j));
        outGPyramid[j](x, y) = upsample(outGPyramid[j + 1])(x, y) + outLPyramid[j](x, y);
    }

    Func color("ll_b_color");
    float eps = 0.01f;
    color(x, y, c) = clamped(x, y, c) * (outGPyramid[0](x, y) + eps) / (gray(x, y) + eps);

    Func output("ll_b_output");
    output(x, y, c) = cast<uint16_t>(clamp(color(x, y, c), 0.0f, 65535.0f));

    output.set_estimate(x, 0, W)
          .set_estimate(y, 0, H)
          .set_estimate(c, 0, 3);

    Pipeline(output).apply_autoscheduler(target, params);

    Buffer<uint16_t> out(W, H, 3);
    output.realize(out, target);  // warmup + JIT compile

    return time_ms([&]() { output.realize(out, target); });
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    load_all_plugins(argc, argv);
    Report rpt(default_report_path(argc, argv, "local_laplacian"));

    run_five_way_with_bench(
        rpt,
        "Local Laplacian Filters (4-level pyramid, levels=8)",
        make_local_laplacian,
        bench_local_laplacian);

    return 0;
}
