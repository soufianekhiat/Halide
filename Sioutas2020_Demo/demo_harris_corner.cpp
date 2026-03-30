/**
 * Demo 2: Harris Corner Detector
 *
 * Deep 7-stage pipeline: gradients -> products -> windowed sums -> response.
 * Report: harris_corner_report.txt (or argv[4])
 */

#include "demo_common.h"

// ---- Schedule display -----------------------------------------------------

static Pipeline make_harris_corner(const std::string &suffix) {
    ImageParam input(Float(32), 2, "hc_input" + suffix);
    input.set_estimates({{0, 1924}, {0, 1084}});
    Var x("x"), y("y");

    Func Ix("hc_Ix" + suffix), Iy("hc_Iy" + suffix);
    Ix(x, y) = (input(x + 1, y) - input(x - 1, y)) * 0.5f;
    Iy(x, y) = (input(x, y + 1) - input(x, y - 1)) * 0.5f;

    Func Ixx("hc_Ixx" + suffix), Ixy("hc_Ixy" + suffix), Iyy("hc_Iyy" + suffix);
    Ixx(x, y) = Ix(x, y) * Ix(x, y);
    Ixy(x, y) = Ix(x, y) * Iy(x, y);
    Iyy(x, y) = Iy(x, y) * Iy(x, y);

    RDom r(-1, 3, -1, 3);
    Func Sxx("hc_Sxx" + suffix), Sxy("hc_Sxy" + suffix), Syy("hc_Syy" + suffix);
    Sxx(x, y) = sum(Ixx(x + r.x, y + r.y));
    Sxy(x, y) = sum(Ixy(x + r.x, y + r.y));
    Syy(x, y) = sum(Iyy(x + r.x, y + r.y));

    Expr det   = Sxx(x, y) * Syy(x, y) - Sxy(x, y) * Sxy(x, y);
    Expr trace = Sxx(x, y) + Syy(x, y);
    Func response("hc_response" + suffix);
    response(x, y) = det - 0.04f * trace * trace;

    response.set_estimate(x, 2, 1916).set_estimate(y, 2, 1076);

    return Pipeline(response);
}

// ---- CPU benchmark --------------------------------------------------------

static double bench_harris_corner(const AutoschedulerParams &params,
                                  const Target &target) {
    const int W = 1920, H = 1080, PAD = 2;  // 2 for Ix/Iy stencil + 3x3 sum box

    Buffer<float> in_buf(W + 2 * PAD, H + 2 * PAD);
    in_buf.set_min(-PAD, -PAD);
    in_buf.fill(0.5f);

    Var x("x"), y("y");
    Func input("hc_b_in");
    input(x, y) = in_buf(x, y);

    Func Ix("hc_b_Ix"), Iy("hc_b_Iy");
    Ix(x, y) = (input(x + 1, y) - input(x - 1, y)) * 0.5f;
    Iy(x, y) = (input(x, y + 1) - input(x, y - 1)) * 0.5f;

    Func Ixx("hc_b_Ixx"), Ixy("hc_b_Ixy"), Iyy("hc_b_Iyy");
    Ixx(x, y) = Ix(x, y) * Ix(x, y);
    Ixy(x, y) = Ix(x, y) * Iy(x, y);
    Iyy(x, y) = Iy(x, y) * Iy(x, y);

    RDom r(-1, 3, -1, 3);
    Func Sxx("hc_b_Sxx"), Sxy("hc_b_Sxy"), Syy("hc_b_Syy");
    Sxx(x, y) = sum(Ixx(x + r.x, y + r.y));
    Sxy(x, y) = sum(Ixy(x + r.x, y + r.y));
    Syy(x, y) = sum(Iyy(x + r.x, y + r.y));

    Expr det   = Sxx(x, y) * Syy(x, y) - Sxy(x, y) * Sxy(x, y);
    Expr trace = Sxx(x, y) + Syy(x, y);
    Func response("hc_b_resp");
    response(x, y) = det - 0.04f * trace * trace;

    response.set_estimate(x, 0, W).set_estimate(y, 0, H);
    Pipeline(response).apply_autoscheduler(target, params);

    Buffer<float> out(W, H);
    response.realize(out, target);

    return time_ms([&]() { response.realize(out, target); });
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    load_all_plugins(argc, argv);
    Report rpt(default_report_path(argc, argv, "harris_corner"));

    run_five_way_with_bench(
        rpt,
        "Harris Corner Detector (1920x1080, 7 stages)",
        make_harris_corner,
        bench_harris_corner);

    return 0;
}
