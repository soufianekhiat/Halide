/**
 * schedule_compare.cpp
 *
 * Diagnostic tool: applies both Sioutas2020 and Anderson2021 to the
 * Gaussian+Sobel benchmark pipeline and prints both schedule strings
 * side-by-side.  No GPU hardware required; useful for understanding
 * scheduling decisions.
 *
 * Usage:
 *   schedule_compare <sioutas2020-lib> <anderson2021-lib>
 */

#include "Halide.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

using namespace Halide;

// Build a fresh copy of the benchmark pipeline with unique function names
// parameterised by a suffix so two pipelines can coexist in one process.
static Pipeline make_gaussian_sobel(const std::string &suffix) {
    ImageParam input(Float(32), 2, "input" + suffix);
    Var x("x"), y("y");

    Func blur_x("blur_x" + suffix);
    blur_x(x, y) = (    input(x - 2, y)
                   + 4.0f * input(x - 1, y)
                   + 6.0f * input(x,     y)
                   + 4.0f * input(x + 1, y)
                   +        input(x + 2, y)) * (1.0f / 16.0f);

    Func blur_y("blur_y" + suffix);
    blur_y(x, y) = (    blur_x(x, y - 2)
                   + 4.0f * blur_x(x, y - 1)
                   + 6.0f * blur_x(x, y    )
                   + 4.0f * blur_x(x, y + 1)
                   +        blur_x(x, y + 2)) * (1.0f / 16.0f);

    Func gx("gx" + suffix), gy("gy" + suffix);
    gx(x, y) = (-1.0f * blur_y(x - 1, y - 1) + blur_y(x + 1, y - 1)
               - 2.0f * blur_y(x - 1, y    ) + 2.0f * blur_y(x + 1, y    )
               - 1.0f * blur_y(x - 1, y + 1) + blur_y(x + 1, y + 1));

    gy(x, y) = (-1.0f * blur_y(x - 1, y - 1) - 2.0f * blur_y(x, y - 1) - blur_y(x + 1, y - 1)
               +        blur_y(x - 1, y + 1) + 2.0f * blur_y(x, y + 1) + blur_y(x + 1, y + 1));

    Func mag("magnitude" + suffix);
    mag(x, y) = sqrt(gx(x, y) * gx(x, y) + gy(x, y) * gy(x, y));
    mag.set_estimate(x, 2, 1916).set_estimate(y, 2, 1076);

    return Pipeline(mag);
}

static Pipeline make_local_stats(const std::string &suffix) {
    ImageParam input(Float(32), 2, "ls_input" + suffix);
    Var x("x"), y("y");
    RDom r(-2, 5, -2, 5);

    Func mean("mean" + suffix), sq_mean("sq_mean" + suffix);
    mean(x, y)    = sum(input(x + r.x, y + r.y)) * (1.0f / 25.0f);
    sq_mean(x, y) = sum(input(x + r.x, y + r.y) *
                        input(x + r.x, y + r.y)) * (1.0f / 25.0f);

    Func var("variance" + suffix);
    var(x, y) = sq_mean(x, y) - mean(x, y) * mean(x, y);
    var.set_estimate(x, 2, 1916).set_estimate(y, 2, 1076);

    return Pipeline(var);
}

// ---------------------------------------------------------------------------

struct Benchmark {
    const char *name;
    Pipeline (*make)(const std::string &);
};

static const Benchmark kBenchmarks[] = {
    {"Gaussian+Sobel (1920×1080)", make_gaussian_sobel},
    {"Local statistics (1920×1080)", make_local_stats},
};

static const Target kTarget{"x86-64-linux-sse41-avx-avx2-cuda"};

static void run_comparison(const char *name,
                            Pipeline (*make)(const std::string &),
                            const char *s2020_lib,
                            const char *a2021_lib) {
    std::cout << "\n══════════════════════════════════════════════════════════\n"
              << "  Pipeline: " << name << "\n"
              << "══════════════════════════════════════════════════════════\n";

    // Sioutas2020
    {
        Pipeline p = make("_s");
        AutoschedulerParams params{"Sioutas2020"};
        auto r = p.apply_autoscheduler(kTarget, params);
        std::cout << "\n── Sioutas2020 ─────────────────────────────────────────\n"
                  << r.schedule_source << "\n";
    }

    // Anderson2021
    {
        Pipeline p = make("_a");
        AutoschedulerParams params{"Anderson2021",
                                    {{"parallelism", "80"},
                                     {"beam_size",   "32"}}};
        auto r = p.apply_autoscheduler(kTarget, params);
        std::cout << "\n── Anderson2021 ────────────────────────────────────────\n"
                  << r.schedule_source << "\n";
    }
}

int main(int argc, char **argv) {
    if (argc < 3 || !strlen(argv[1]) || !strlen(argv[2])) {
        fprintf(stderr,
                "Usage: %s <sioutas2020-lib> <anderson2021-lib>\n"
                "Prints schedule strings for both autoschedulers on the\n"
                "same pipelines for a direct visual comparison.\n",
                argv[0]);
        return 1;
    }

#ifdef HALIDE_WITH_EXCEPTIONS
    try {
#endif
        load_plugin(argv[1]);   // Sioutas2020
        load_plugin(argv[2]);   // Anderson2021

        for (const auto &bm : kBenchmarks) {
            run_comparison(bm.name, bm.make, argv[1], argv[2]);
        }

        std::cout << "\nDone.\n";

#ifdef HALIDE_WITH_EXCEPTIONS
    } catch (const Halide::Error &e) {
        std::cerr << "Halide error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "std error: " << e.what() << "\n";
        return 1;
    }
#endif
    return 0;
}
