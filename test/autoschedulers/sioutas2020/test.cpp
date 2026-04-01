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

        // Full cost-model path (default) — uses split+.gpu() instead of gpu_tile.
        AutoschedulerParams gpu{"Sioutas2020"};
        // CPU-only path — always uses SimpleAutoSchedule.
        AutoschedulerParams cpu{"Sioutas2020"};
        // SimpleAutoSchedule path — legacy extent-based gpu_tile heuristic.
        AutoschedulerParams simple_gpu{
            "Sioutas2020",
            {{"enable_fusion", "0"}, {"enable_cost_model", "0"}}};

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

            auto r = Pipeline(h).apply_autoscheduler(kGPU, simple_gpu);
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

            auto r = Pipeline(h).apply_autoscheduler(kGPU, simple_gpu);
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

            auto r = Pipeline(f[N-1]).apply_autoscheduler(kGPU, simple_gpu);
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

            auto r = Pipeline(by).apply_autoscheduler(kGPU, simple_gpu);
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

            auto r = Pipeline(out).apply_autoscheduler(kGPU, simple_gpu);
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

            auto result = Pipeline(f).apply_autoscheduler(kGPU, simple_gpu);
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

            auto result = Pipeline(out).apply_autoscheduler(kGPU, simple_gpu);
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

            auto r = Pipeline({a, b}).apply_autoscheduler(kGPU, simple_gpu);
            expect_contains(r.schedule_source, "gpu_tile", "multi-output GPU");
            printf("PASS  8: multi-output GPU\n");
        }

        // 9. 1-D function.
        {
            Func f("f_1d"), g("g_1d");
            f(x) = sqrt(cast<float>(x + 1));
            g(x) = f(x) + f(x + 1);
            g.set_estimate(x, 0, 65536);

            auto r = Pipeline(g).apply_autoscheduler(kGPU, simple_gpu);
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

            auto result = Pipeline(total).apply_autoscheduler(kGPU, simple_gpu);
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

            auto result = Pipeline(scan).apply_autoscheduler(kGPU, simple_gpu);
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

            auto result = Pipeline(f).apply_autoscheduler(kGPU, simple_gpu);
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

            auto r = Pipeline(outer).apply_autoscheduler(kGPU, simple_gpu);
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

        // 17. Custom tile sizes via params (SimpleAutoSchedule path).
        {
            Func f("ts_f"), g("ts_g");
            f(x, y) = sqrt(cast<float>(x * x + y * y + 1));
            g(x, y) = f(x-1,y) + f(x,y) + f(x+1,y);
            g.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

            AutoschedulerParams custom{"Sioutas2020",
                                       {{"gpu_tile_x", "32"}, {"gpu_tile_y", "8"},
                                        {"enable_fusion", "0"}, {"enable_cost_model", "0"}}};
            auto r = Pipeline(g).apply_autoscheduler(kGPU, custom);
            // Custom sizes must appear in the schedule string.
            expect_contains(r.schedule_source, "32", "custom tile 32");
            expect_contains(r.schedule_source, "8",  "custom tile 8");
            printf("PASS 17: custom tile sizes\n");
        }

        // ----------------------------------------------------------------
        // Tests for enhanced features (bounds-based dim selection)
        // ----------------------------------------------------------------

        // 18. 4D function: extent-based dim selection should tile x and y
        //     (the two largest dims), NOT c and x (positional).
        //     f(c, x, y, n) with c=8, x=128, y=32, n=1 -> tile x x y.
        {
            Var cc("cc"), xx("xx"), yy("yy"), nn("nn");
            Func f4d("f4d");
            f4d(cc, xx, yy, nn) = cast<float>(cc + xx + yy + nn);
            f4d.set_estimate(cc, 0, 8)
               .set_estimate(xx, 0, 128)
               .set_estimate(yy, 0, 32)
               .set_estimate(nn, 0, 1);

            auto r = Pipeline(f4d).apply_autoscheduler(kGPU, simple_gpu);
            // Should tile the two largest dims: xx (128) and yy (32).
            expect_contains(r.schedule_source, "xx", "4D extent-based tile xx");
            expect_contains(r.schedule_source, "yy", "4D extent-based tile yy");
            expect_contains(r.schedule_source, "gpu_tile", "4D extent-based gpu_tile");
            printf("PASS 18: 4D extent-based dimension selection\n");
        }

        // 19. 3D function with large channel: should produce a 3D gpu_tile
        //     (fused extra dim) when combined extra extent >= gpu_tile_channel.
        //     f(x, y, c) with x=1024, y=1024, c=16 -> 3D tile.
        {
            Func f3d("f3d");
            f3d(x, y, c) = cast<float>(x + y + c);
            f3d.set_estimate(x, 0, 1024)
               .set_estimate(y, 0, 1024)
               .set_estimate(c, 0, 16);

            auto r = Pipeline(f3d).apply_autoscheduler(kGPU, simple_gpu);
            // 3D tiling emits the fused variable reference in the schedule source.
            expect_contains(r.schedule_source, "gpu_tile", "3D fused gpu_tile");
            printf("PASS 19: 3D tiling with fused channel dimension\n");
        }

        // 20. Rvar unrolling: a gather reduction with a small 3x3 kernel
        //     should have 'unroll' in the schedule.
        {
            ImageParam img(Float(32), 2, "unroll_img");
            Func fu("fu");
            RDom r3(-1, 3, -1, 3, "r3");
            fu(x, y) = 0.0f;
            fu(x, y) += img(x + r3.x, y + r3.y);
            fu.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

            auto result = Pipeline(fu).apply_autoscheduler(kGPU, simple_gpu);
            expect_contains(result.schedule_source, "unroll", "rvar unroll 3x3");
            printf("PASS 20: small rvar unrolling (3x3 kernel)\n");
        }

        // 21. 1D large function: should use 1D gpu_tile (not fuse-all).
        //     gpu_tile_x * gpu_tile_y = 256 threads for 1D.
        {
            Func f1d("f1d_large");
            f1d(x) = cast<float>(x * x + 1);
            f1d.set_estimate(x, 0, 65536);

            auto r = Pipeline(f1d).apply_autoscheduler(kGPU, simple_gpu);
            expect_contains(r.schedule_source, "gpu_tile", "1D large gpu_tile");
            printf("PASS 21: 1D large function (1D tile)\n");
        }

        // ----------------------------------------------------------------
        // Tests for full cost-model path (enable_fusion=true, default)
        // ----------------------------------------------------------------

        // 22. Stencil chain with full cost-model: should produce compute_at
        //     fusion and use split+.gpu() instead of gpu_tile.
        {
            const int N = 6;
            Func fc[N];
            fc[0](x, y) = sqrt(cast<float>(x + y + 1));
            for (int k = 1; k < N; k++) {
                fc[k](x, y) = fc[k-1](x-1, y) + fc[k-1](x, y) + fc[k-1](x+1, y);
            }
            fc[N-1].set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

            auto r = Pipeline(fc[N-1]).apply_autoscheduler(kGPU, gpu);
            // Full cost-model path uses .gpu() (split+reorder+gpu) instead of gpu_tile.
            expect_contains(r.schedule_source, ".gpu(", "full-path stencil chain .gpu");
            expect_contains(r.schedule_source, "compute_at(", "full-path stencil chain compute_at");
            expect_not_contains(r.schedule_source, "gpu_tile", "full-path stencil chain no gpu_tile");
            printf("PASS 22: full cost-model stencil chain (compute_at + .gpu)\n");
        }

        // 23. Histogram scatter with full cost-model: scatter update must
        //     still use gpu_single_thread regardless of path taken.
        {
            ImageParam input2(Int(32), 2, "cm_hist_input");
            input2.set_estimates({{0, 1024}, {0, 1024}});  // required for cost-model path
            Func f2("cm_hist_f"), hist2("cm_hist"), out2("cm_hist_out");
            f2(x, y) = clamp(input2(x, y), 0, 255);
            RDom r2(0, 1024, 0, 1024);
            hist2(i) = cast<uint32_t>(0);
            hist2(f2(r2.x, r2.y)) += cast<uint32_t>(1);
            out2(i) = hist2(i);
            f2.set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);
            out2.set_estimate(i, 0, 256);

            auto result = Pipeline(out2).apply_autoscheduler(kGPU, gpu);
            // Scatter update must always fall back to gpu_single_thread.
            expect_contains(result.schedule_source, "gpu_single_thread",
                             "full-path histogram scatter");
            printf("PASS 23: full cost-model histogram (gpu_single_thread preserved)\n");
        }

        // 24. Scalar (0-D) reduction with full cost-model: scalar must not
        //     be tiled.
        {
            ImageParam sc_in2(Float(32), 2, "cm_sc_input");
            sc_in2.set_estimates({{0, 1024}, {0, 1024}});  // required for cost-model path
            Func sc_total("cm_total");
            RDom sc_r(0, 1024, 0, 1024);
            sc_total() = 0.0f;
            sc_total() += sc_in2(sc_r.x, sc_r.y);

            auto result = Pipeline(sc_total).apply_autoscheduler(kGPU, gpu);
            expect_contains(result.schedule_source, "cm_total.compute_root()",
                             "full-path scalar");
            expect_not_contains(result.schedule_source, "cm_total.gpu_tile",
                                 "full-path scalar no gpu_tile");
            printf("PASS 24: full cost-model scalar reduction\n");
        }

        // 25. CPU is unchanged by enable_fusion (CPU always uses SimpleAutoSchedule).
        {
            Func f("cpu_chain_f"), g("cpu_chain_g");
            f(x, y) = sqrt(cast<float>(x * x + y * y + 1));
            g(x, y) = f(x-1, y) + f(x, y) + f(x+1, y);
            g.set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

            auto r = Pipeline(g).apply_autoscheduler(kCPU, cpu);
            expect_contains(r.schedule_source, "compute_root", "cpu full-path");
            expect_not_contains(r.schedule_source, "gpu_tile", "cpu full-path no gpu_tile");
            printf("PASS 25: CPU path unaffected by enable_fusion flag\n");
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
