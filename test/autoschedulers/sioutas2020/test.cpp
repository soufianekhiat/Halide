/**
 * Sioutas2020 autoscheduler — correctness test
 *
 * Mirrors the pattern of test/autoschedulers/anderson2021/test.cpp.
 * Each case calls apply_autoscheduler() and checks the resulting
 * schedule_source string for expected / forbidden sub-strings.
 *
 * No GPU hardware is required: apply_autoscheduler only constructs the
 * schedule IR, it never compiles or runs anything.
 */

#include "Halide.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

using namespace Halide;

// ---------------------------------------------------------------------------
// Lightweight assertion helpers
// ---------------------------------------------------------------------------

static void expect_contains(const std::string &haystack,
                              const std::string &needle,
                              const char *ctx) {
    if (haystack.find(needle) == std::string::npos) {
        fprintf(stderr,
                "[%s] Schedule should contain '%s' but does not.\n"
                "Schedule was:\n%s\n",
                ctx, needle.c_str(), haystack.c_str());
        exit(1);
    }
}

static void expect_not_contains(const std::string &haystack,
                                  const std::string &needle,
                                  const char *ctx) {
    if (haystack.find(needle) != std::string::npos) {
        fprintf(stderr,
                "[%s] Schedule should NOT contain '%s' but does.\n"
                "Schedule was:\n%s\n",
                ctx, needle.c_str(), haystack.c_str());
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Targets
// ---------------------------------------------------------------------------

// x86-64 + CUDA: exercises the GPU scheduling path without needing GPU HW.
static const Target kGPU{"x86-64-linux-sse41-avx-avx2-cuda"};
// CPU-only: exercises the CPU scheduling path.
static const Target kCPU{"x86-64-linux-sse41-avx-avx2"};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc != 2 || !strlen(argv[1])) {
        fprintf(stderr, "Usage: %s <sioutas2020-autoscheduler-lib>\n", argv[0]);
        return 1;
    }

#ifdef HALIDE_WITH_EXCEPTIONS
    try {
#endif
        load_plugin(argv[1]);

        AutoschedulerParams gpu{"Sioutas2020"};
        AutoschedulerParams cpu{"Sioutas2020"};

        Var x("x"), y("y"), c("c"), i("i");

        // ----------------------------------------------------------------
        // GPU tests
        // ----------------------------------------------------------------

        // 1. Pointwise pipeline: trivial intermediate inlined, output gpu_tiled.
        {
            Func f("ptw_f"), g("ptw_g"), h("ptw_h");
            f(x, y) = cast<float>(x + y);
            g(x, y) = f(x, y) * 2.0f + 1.0f;
            h(x, y) = sqrt(g(x, y) * g(x, y) + 1.0f);   // non-trivial
            h.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

            auto r = Pipeline(h).apply_autoscheduler(kGPU, gpu);
            expect_contains(r.schedule_source, "gpu_tile", "pointwise GPU");
            printf("PASS  1: pointwise GPU\n");
        }

        // 2. 3×3 stencil with an expensive producer.
        {
            Func f("stc_f"), h("stc_h");
            // Make f expensive so it is NOT inlined.
            f(x, y) = sqrt(cast<float>(x * x + y * y + x * y + 1));
            h(x, y) = (f(x-1,y-1) + f(x,y-1) + f(x+1,y-1) +
                       f(x-1,y)   + f(x,y)   + f(x+1,y)   +
                       f(x-1,y+1) + f(x,y+1) + f(x+1,y+1));
            h.set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

            auto r = Pipeline(h).apply_autoscheduler(kGPU, gpu);
            expect_contains(r.schedule_source, "gpu_tile", "stencil GPU");
            printf("PASS  2: 3×3 stencil GPU\n");
        }

        // 3. Stencil chain (6 stages).
        {
            const int N = 6;
            Func f[N];
            f[0](x, y) = sqrt(cast<float>(x + y + 1));
            for (int k = 1; k < N; k++) {
                f[k](x, y) = f[k-1](x-1, y) + f[k-1](x, y) + f[k-1](x+1, y);
            }
            f[N-1].set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

            auto r = Pipeline(f[N-1]).apply_autoscheduler(kGPU, gpu);
            expect_contains(r.schedule_source, "gpu_tile", "stencil chain GPU");
            printf("PASS  3: stencil chain (6 stages) GPU\n");
        }

        // 4. Separable Gaussian blur.
        {
            ImageParam input(Float(32), 2, "sep_input");
            Func bx("sep_bx"), by("sep_by");
            bx(x, y) = (input(x-1,y) + input(x,y)*2.0f + input(x+1,y)) * 0.25f;
            by(x, y) = (bx(x,y-1) + bx(x,y)*2.0f + bx(x,y+1)) * 0.25f;
            by.set_estimate(x, 1, 2046).set_estimate(y, 1, 2046);

            auto r = Pipeline(by).apply_autoscheduler(kGPU, gpu);
            expect_contains(r.schedule_source, "gpu_tile", "separable blur GPU");
            printf("PASS  4: separable blur GPU\n");
        }

        // 5. 3-channel (c) image — should tile x and y.
        {
            ImageParam input(Float(32), 3, "rgb_in");
            Func out("rgb_out");
            out(x, y, c) = input(x, y, c) * 1.5f + input(x+1, y, c) * 0.5f;
            out.set_estimate(x, 0, 1920)
               .set_estimate(y, 0, 1080)
               .set_estimate(c, 0, 3);

            auto r = Pipeline(out).apply_autoscheduler(kGPU, gpu);
            expect_contains(r.schedule_source, "gpu_tile", "3-channel image GPU");
            printf("PASS  5: 3-channel image GPU\n");
        }

        // 6. Reduction with pure-variable LHS (gather pattern):
        //    f(x,y) += input(x+r.x, y+r.y).
        //    Init stage and update stage should both be gpu_tiled.
        {
            ImageParam input(Float(32), 2, "rdx_input");
            Func f("rdx_f");
            RDom r(-2, 5, -2, 5);
            f(x, y) = 0.0f;
            f(x, y) += input(x + r.x, y + r.y);
            f.set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

            auto result = Pipeline(f).apply_autoscheduler(kGPU, gpu);
            // Both init and update tiled.
            expect_contains(result.schedule_source, "gpu_tile", "gather reduction GPU");
            printf("PASS  6: gather reduction GPU\n");
        }

        // 7. Histogram (scatter pattern): complex LHS, update must use
        //    gpu_single_thread to avoid data races.
        {
            ImageParam input(Int(32), 2, "hist_input");
            Func f("hist_f"), hist("hist"), out("hist_out");
            f(x, y) = clamp(input(x, y), 0, 255);
            RDom r(0, 1024, 0, 1024);
            hist(i) = cast<uint32_t>(0);
            hist(f(r.x, r.y)) += cast<uint32_t>(1);
            out(i) = hist(i);
            f.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);
            out.set_estimate(i, 0, 256);

            auto result = Pipeline(out).apply_autoscheduler(kGPU, gpu);
            // Scatter update must fall back to single-thread on GPU.
            expect_contains(result.schedule_source, "gpu_single_thread",
                             "histogram scatter GPU");
            printf("PASS  7: histogram (scatter) GPU\n");
        }

        // 8. Multi-output pipeline.
        {
            ImageParam input(Float(32), 2, "mo_input");
            Func a("mo_a"), b("mo_b");
            a(x, y) = input(x, y) + input(x+1, y);
            b(x, y) = input(x, y) + input(x, y+1);
            a.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);
            b.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

            auto r = Pipeline({a, b}).apply_autoscheduler(kGPU, gpu);
            expect_contains(r.schedule_source, "gpu_tile", "multi-output GPU");
            printf("PASS  8: multi-output GPU\n");
        }

        // 9. 1-D function.
        {
            Func f("f_1d"), g("g_1d");
            f(x) = sqrt(cast<float>(x + 1));
            g(x) = f(x) + f(x + 1);
            g.set_estimate(x, 0, 65536);

            auto r = Pipeline(g).apply_autoscheduler(kGPU, gpu);
            expect_contains(r.schedule_source, "gpu_tile", "1D GPU");
            printf("PASS  9: 1-D function GPU\n");
        }

        // 10. Scalar (0-D) reduction.
        //     The scalar output itself cannot be GPU-tiled; its schedule must
        //     be compute_root() only.  (The ImageParam wrapper IS a 2-D pure
        //     function and will legitimately receive gpu_tile — that is fine.)
        {
            ImageParam input(Float(32), 2, "sc_input");
            Func total("total");
            RDom r(0, 1024, 0, 1024);
            total() = 0.0f;
            total() += input(r.x, r.y);

            auto result = Pipeline(total).apply_autoscheduler(kGPU, gpu);
            // The scalar function itself must be compute_root, not gpu_tiled.
            expect_contains(result.schedule_source, "total.compute_root()",
                             "scalar reduction GPU");
            expect_not_contains(result.schedule_source, "total.gpu_tile",
                                 "scalar reduction GPU");
            printf("PASS 10: scalar (0-D) reduction GPU\n");
        }

        // 11. Scan — prefix sum along y.  Update has complex LHS
        //     (r is the index, not the pure var y), so update falls back.
        {
            ImageParam input(Float(32), 2, "scan_input");
            Func scan("scan");
            RDom r(1, 1023);
            scan(x, y) = input(x, y);
            scan(x, r) += scan(x, r - 1);   // LHS arg is r, not y
            scan.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

            auto result = Pipeline(scan).apply_autoscheduler(kGPU, gpu);
            // Init should be tiled; update should fall back (scatter-like).
            expect_contains(result.schedule_source, "gpu_tile",
                             "prefix scan init GPU");
            expect_contains(result.schedule_source, "gpu_single_thread",
                             "prefix scan update GPU");
            printf("PASS 11: prefix scan GPU\n");
        }

        // 12. Multiple update definitions on one function.
        {
            ImageParam input(Float(32), 2, "mu_input");
            Func f("mu_f");
            RDom r(-1, 3, -1, 3);
            f(x, y) = 0.0f;
            f(x, y) += input(x + r.x, y + r.y);   // gather — pure LHS
            f(x, y) *= 1.0f / 9.0f;                 // scale  — pure LHS
            f.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

            auto result = Pipeline(f).apply_autoscheduler(kGPU, gpu);
            expect_contains(result.schedule_source, "gpu_tile",
                             "multiple updates GPU");
            printf("PASS 12: multiple update definitions GPU\n");
        }

        // 13. Large outer product (Mullapudi2016 / Adams2019 classic).
        {
            Buffer<float> a(2048), b(2048);
            a.fill(1.0f);
            b.fill(1.0f);
            Func outer("outer_prod");
            outer(x, y) = a(x) * b(y);
            outer.set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

            auto r = Pipeline(outer).apply_autoscheduler(kGPU, gpu);
            expect_contains(r.schedule_source, "gpu_tile", "outer product GPU");
            printf("PASS 13: outer product GPU\n");
        }

        // ----------------------------------------------------------------
        // CPU tests
        // ----------------------------------------------------------------

        // 14. Stencil on CPU — must NOT use gpu_tile; must use compute_root.
        {
            Func f("cpu_f"), g("cpu_g");
            f(x, y) = sqrt(cast<float>(x * x + y * y + 1));
            g(x, y) = f(x-1,y) + f(x,y) + f(x+1,y);
            g.set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

            auto r = Pipeline(g).apply_autoscheduler(kCPU, cpu);
            expect_contains(r.schedule_source, "compute_root", "stencil CPU");
            expect_not_contains(r.schedule_source, "gpu_tile",  "stencil CPU");
            printf("PASS 14: stencil CPU (no gpu_tile)\n");
        }

        // 15. 1-D function on CPU.
        {
            Func f("cpu_1d");
            f(x) = sqrt(cast<float>(x * x + 1));
            f.set_estimate(x, 0, 65536);

            auto r = Pipeline(f).apply_autoscheduler(kCPU, cpu);
            expect_contains(r.schedule_source, "compute_root", "1D CPU");
            expect_not_contains(r.schedule_source, "gpu_tile",  "1D CPU");
            printf("PASS 15: 1-D function CPU\n");
        }

        // 16. Reduction on CPU.
        {
            ImageParam input(Float(32), 2, "cpu_rdx_in");
            Func f("cpu_rdx");
            RDom r(-1, 3, -1, 3);
            f(x, y) = 0.0f;
            f(x, y) += input(x + r.x, y + r.y);
            f.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

            auto result = Pipeline(f).apply_autoscheduler(kCPU, cpu);
            expect_contains(result.schedule_source, "compute_root", "reduction CPU");
            expect_not_contains(result.schedule_source, "gpu_tile",  "reduction CPU");
            printf("PASS 16: reduction CPU\n");
        }

        // 17. Custom tile sizes via params.
        {
            Func f("ts_f"), g("ts_g");
            f(x, y) = sqrt(cast<float>(x * x + y * y + 1));
            g(x, y) = f(x-1,y) + f(x,y) + f(x+1,y);
            g.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

            AutoschedulerParams custom{"Sioutas2020",
                                       {{"gpu_tile_x", "32"}, {"gpu_tile_y", "8"}}};
            auto r = Pipeline(g).apply_autoscheduler(kGPU, custom);
            // Custom sizes must appear in the schedule string.
            expect_contains(r.schedule_source, "32", "custom tile 32");
            expect_contains(r.schedule_source, "8",  "custom tile 8");
            printf("PASS 17: custom tile sizes\n");
        }

#ifdef HALIDE_WITH_EXCEPTIONS
    } catch (const Halide::Error &e) {
        std::cerr << "Halide error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "std error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception\n";
        return 1;
    }
#endif

    std::cout << "Success!\n";
    return 0;
}
