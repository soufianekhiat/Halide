/**
 * Demo 4: Histogram Equalization
 *
 * Phase 1: scatter accumulation  -- hist(pixel_value) += 1
 * Phase 2: CDF prefix scan       -- cdf(i) = cdf(i-1) + hist(i)
 * Phase 3: gather remap          -- out(x,y) = cdf(lum(x,y)) / total
 *
 * Tests how each scheduler handles scatter LHS patterns.
 * Report: histogram_eq_report.txt (or argv[4])
 */

#include "demo_common.h"

// ---- Schedule display -----------------------------------------------------

static Pipeline make_histogram_eq(const std::string &suffix) {
    ImageParam input(UInt(8), 2, "heq_input" + suffix);
    input.set_estimates({{0, 1920}, {0, 1080}});
    Var x("x"), y("y"), i("i");

    Func lum("heq_lum" + suffix);
    lum(x, y) = cast<int32_t>(input(x, y));

    // Scatter accumulation (non-identity LHS)
    Func hist("heq_hist" + suffix);
    RDom r(0, 1920, 0, 1080, "r");
    hist(i) = cast<int32_t>(0);
    hist(clamp(lum(r.x, r.y), 0, 255)) += cast<int32_t>(1);
    hist.set_estimate(i, 0, 256);

    // CDF prefix scan (sequential dependency)
    Func cdf("heq_cdf" + suffix);
    RDom b(1, 255, "b");
    cdf(i) = hist(i);
    cdf(b.x) = cdf(b.x - 1) + hist(b.x);
    cdf.set_estimate(i, 0, 256);

    // Equalize
    Func eq("heq_eq" + suffix);
    const float scale = 255.0f / (1920.0f * 1080.0f);
    eq(x, y) = clamp(cast<float>(cdf(clamp(lum(x, y), 0, 255))) * scale,
                     0.0f, 255.0f);
    eq.set_estimate(x, 0, 1920).set_estimate(y, 0, 1080);

    return Pipeline(eq);
}

// ---- CPU benchmark --------------------------------------------------------

static double bench_histogram_eq(const AutoschedulerParams &params,
                                  const Target &target) {
    const int W = 1920, H = 1080;

    Buffer<uint8_t> in_buf(W, H);
    // Fill with a simple gradient pattern
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            in_buf(xx, yy) = static_cast<uint8_t>((xx + yy) % 256);

    Var x("x"), y("y"), i("i");
    Func input("heq_b_in");
    input(x, y) = in_buf(x, y);

    Func lum("heq_b_lum");
    lum(x, y) = cast<int32_t>(input(x, y));

    Func hist("heq_b_hist");
    RDom r(0, W, 0, H, "r");
    hist(i) = cast<int32_t>(0);
    hist(clamp(lum(r.x, r.y), 0, 255)) += cast<int32_t>(1);
    hist.set_estimate(i, 0, 256);

    Func cdf("heq_b_cdf");
    RDom b(1, 255, "b");
    cdf(i) = hist(i);
    cdf(b.x) = cdf(b.x - 1) + hist(b.x);
    cdf.set_estimate(i, 0, 256);

    Func eq("heq_b_eq");
    const float scale = 255.0f / (float)(W * H);
    eq(x, y) = clamp(cast<float>(cdf(clamp(lum(x, y), 0, 255))) * scale,
                     0.0f, 255.0f);
    eq.set_estimate(x, 0, W).set_estimate(y, 0, H);

    Pipeline(eq).apply_autoscheduler(target, params);

    Buffer<float> out(W, H);
    eq.realize(out, target);

    return time_ms([&]() { eq.realize(out, target); });
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    load_all_plugins(argc, argv);
    Report rpt(default_report_path(argc, argv, "histogram_eq"));

    run_five_way_with_bench(
        rpt,
        "Histogram Equalization (1920x1080, scatter+scan+gather)",
        make_histogram_eq,
        bench_histogram_eq);

    return 0;
}
