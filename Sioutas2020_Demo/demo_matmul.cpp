/**
 * Demo 3: Matrix Multiply
 *
 * C(x,y) += A(x,r) * B(r,y)  N=1024
 * Tests large-reduction parallelization strategies.
 * Report: matmul_report.txt (or argv[4])
 */

#include "demo_common.h"

// ---- Schedule display -----------------------------------------------------

static Pipeline make_matmul(const std::string &suffix) {
    const int N = 1024;
    ImageParam A(Float(32), 2, "mm_A" + suffix);
    ImageParam B(Float(32), 2, "mm_B" + suffix);
    A.set_estimates({{0, N}, {0, N}});
    B.set_estimates({{0, N}, {0, N}});
    Var x("x"), y("y");
    RDom r(0, N, "r");

    Func prod("mm_prod" + suffix);
    prod(x, y)  = 0.0f;
    prod(x, y) += A(x, r) * B(r, y);

    prod.set_estimate(x, 0, N).set_estimate(y, 0, N);

    return Pipeline(prod);
}

// ---- CPU benchmark --------------------------------------------------------

static double bench_matmul(const AutoschedulerParams &params,
                            const Target &target) {
    const int N = 1024;

    Buffer<float> A_buf(N, N), B_buf(N, N);
    A_buf.fill(1.0f);
    B_buf.fill(1.0f);

    Var x("x"), y("y");
    RDom r(0, N, "r");
    Func A("mm_b_A"), B("mm_b_B");
    A(x, y) = A_buf(x, y);
    B(x, y) = B_buf(x, y);

    Func prod("mm_b_prod");
    prod(x, y)  = 0.0f;
    prod(x, y) += A(x, r) * B(r, y);

    prod.set_estimate(x, 0, N).set_estimate(y, 0, N);
    Pipeline(prod).apply_autoscheduler(target, params);

    Buffer<float> out(N, N);
    prod.realize(out, target);  // warmup

    // MatMul is expensive; use fewer iterations
    return time_ms([&]() { prod.realize(out, target); }, /*warmup=*/1, /*iters=*/3);
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    load_all_plugins(argc, argv);
    Report rpt(default_report_path(argc, argv, "matmul"));

    run_five_way_with_bench(
        rpt,
        "Matrix Multiply 1024x1024 (reduction N=1024)",
        make_matmul,
        bench_matmul);

    return 0;
}
