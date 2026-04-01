/**
 * Demo 7: Non-Local Means Denoising
 *
 * dc(x,y,dx,dy,c) = squared difference images
 * d(x,y,dx,dy) = sum across color channels
 * blur_d_y, blur_d = separable blur of difference images (patch comparison)
 * w(x,y,dx,dy) = exp(-blur_d * inv_sigma_sq)  (similarity weights)
 * non_local_means_sum += w * shifted input over search area
 * non_local_means = sum / alpha  (normalize)
 *
 * Multi-stage stencil with 5D intermediates and dual reduction domains.
 * Pipeline from Halide apps/nl_means and the Sioutas2020 TACO paper.
 *
 * Report: nl_means_report.txt (or argv[6])
 */

#include "demo_common.h"

// ---- Schedule display (uses ImageParam, GPU target) -----------------------

static Pipeline make_nl_means(const std::string &suffix) {
    const int W = 192, H = 320;  // smaller for schedule display speed
    const int PATCH = 7, SEARCH = 7;

    ImageParam input(Float(32), 3, "nlm_input" + suffix);
    input.set_estimates({{0, W}, {0, H}, {0, 3}});

    Var x("x"), y("y"), c("c");

    // Parameters as Expr (fixed for schedule display)
    Expr patch_size = PATCH;
    Expr search_area = SEARCH;
    Expr sigma = 0.12f;
    Expr inv_sigma_sq = -1.0f / (sigma * sigma * patch_size * patch_size);

    // Boundary condition
    Func clamped("nlm_clamped" + suffix);
    clamped = BoundaryConditions::repeat_edge(input);

    // Squared difference images
    Var dx("dx"), dy("dy");
    Func dc("nlm_dc" + suffix);
    dc(x, y, dx, dy, c) = pow(clamped(x, y, c) - clamped(x + dx, y + dy, c), 2);

    // Sum across color channels
    RDom channels(0, 3, "channels");
    Func d("nlm_d" + suffix);
    d(x, y, dx, dy) = sum(dc(x, y, dx, dy, channels));

    // Separable blur of difference images
    RDom patch_dom(-(PATCH / 2), PATCH, "patch_dom");
    Func blur_d_y("nlm_blur_d_y" + suffix);
    blur_d_y(x, y, dx, dy) = sum(d(x, y + patch_dom, dx, dy));

    Func blur_d("nlm_blur_d" + suffix);
    blur_d(x, y, dx, dy) = sum(blur_d_y(x + patch_dom, y, dx, dy));

    // Similarity weights
    Func w("nlm_w" + suffix);
    w(x, y, dx, dy) = fast_exp(blur_d(x, y, dx, dy) * inv_sigma_sq);

    // Add alpha channel
    Func clamped_with_alpha("nlm_cwa" + suffix);
    clamped_with_alpha(x, y, c) = mux(c, {clamped(x, y, 0),
                                           clamped(x, y, 1),
                                           clamped(x, y, 2),
                                           1.0f});

    // Weighted sum over search area
    RDom s_dom(-(SEARCH / 2), SEARCH, -(SEARCH / 2), SEARCH, "s_dom");
    Func nlm_sum("nlm_sum" + suffix);
    nlm_sum(x, y, c) += w(x, y, s_dom.x, s_dom.y)
                        * clamped_with_alpha(x + s_dom.x, y + s_dom.y, c);

    // Normalize
    Func nlm_out("nlm_out" + suffix);
    nlm_out(x, y, c) = clamp(nlm_sum(x, y, c) / nlm_sum(x, y, 3), 0.0f, 1.0f);

    nlm_out.set_estimate(x, 0, W)
           .set_estimate(y, 0, H)
           .set_estimate(c, 0, 3);

    return Pipeline(nlm_out);
}

// ---- CPU benchmark --------------------------------------------------------

static double bench_nl_means(const AutoschedulerParams &params,
                              const Target &target) {
    const int W = 128, H = 128;
    const int PATCH = 7, SEARCH = 7;
    const float SIGMA = 0.12f;

    Buffer<float> input_buf(W + SEARCH, H + SEARCH, 3);
    input_buf.fill(0.5f);

    Var x("x"), y("y"), c("c"), dx("dx"), dy("dy");

    Func clamped("nlm_b_clamped");
    clamped = BoundaryConditions::repeat_edge(input_buf);

    Expr inv_sigma_sq = -1.0f / (SIGMA * SIGMA * PATCH * PATCH);

    Func dc("nlm_b_dc");
    dc(x, y, dx, dy, c) = pow(clamped(x, y, c) - clamped(x + dx, y + dy, c), 2);

    RDom channels(0, 3, "channels");
    Func d("nlm_b_d");
    d(x, y, dx, dy) = sum(dc(x, y, dx, dy, channels));

    RDom patch_dom(-(PATCH / 2), PATCH, "patch_dom");
    Func blur_d_y("nlm_b_bdy");
    blur_d_y(x, y, dx, dy) = sum(d(x, y + patch_dom, dx, dy));

    Func blur_d("nlm_b_bd");
    blur_d(x, y, dx, dy) = sum(blur_d_y(x + patch_dom, y, dx, dy));

    Func w("nlm_b_w");
    w(x, y, dx, dy) = fast_exp(blur_d(x, y, dx, dy) * inv_sigma_sq);

    Func clamped_with_alpha("nlm_b_cwa");
    clamped_with_alpha(x, y, c) = mux(c, {clamped(x, y, 0),
                                           clamped(x, y, 1),
                                           clamped(x, y, 2),
                                           1.0f});

    RDom s_dom(-(SEARCH / 2), SEARCH, -(SEARCH / 2), SEARCH, "s_dom");
    Func nlm_sum("nlm_b_sum");
    nlm_sum(x, y, c) += w(x, y, s_dom.x, s_dom.y)
                        * clamped_with_alpha(x + s_dom.x, y + s_dom.y, c);

    Func nlm_out("nlm_b_out");
    nlm_out(x, y, c) = clamp(nlm_sum(x, y, c) / nlm_sum(x, y, 3), 0.0f, 1.0f);

    nlm_out.set_estimate(x, 0, W)
           .set_estimate(y, 0, H)
           .set_estimate(c, 0, 3);

    Pipeline(nlm_out).apply_autoscheduler(target, params);

    Buffer<float> out(W, H, 3);
    nlm_out.realize(out, target);  // warmup + JIT compile

    return time_ms([&]() { nlm_out.realize(out, target); });
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    load_all_plugins(argc, argv);
    Report rpt(default_report_path(argc, argv, "nl_means"));

    run_five_way_with_bench(
        rpt,
        "Non-Local Means Denoising (patch=7 search=7 sigma=0.12)",
        make_nl_means,
        bench_nl_means);

    return 0;
}
