/**
 * demo_common.h
 *
 * Shared utilities for the Sioutas2020 five-way autoscheduler comparison demos.
 * All 8 demo executables include this header.
 *
 * Usage of each demo:
 *   <demo>.exe <sioutas2020.dll> <anderson2021.dll> <adams2019.dll>
 *              <li2018.dll> <mullapudi2016.dll> [report.txt]
 *
 * Schedulers compared:
 *   GPU:  Sioutas2020 (cost-model + compute_at fusion), Anderson2021 (beam search),
 *         Li2018 (GPU), Mullapudi2016 (experimental GPU)
 *   CPU:  Adams2019 (predecessor to Anderson2021), Mullapudi2016
 */

#pragma once

#include "Halide.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// CUDA driver detection
// ---------------------------------------------------------------------------

#ifdef _WIN32
extern "C" {
__declspec(dllimport) void * __stdcall LoadLibraryA(const char *);
__declspec(dllimport) int    __stdcall FreeLibrary(void *);
}
#else
#include <dlfcn.h>
#endif

static inline bool probe_cuda_driver() {
#ifdef _WIN32
    void *h = LoadLibraryA("nvcuda.dll");
    if (h) { FreeLibrary(h); return true; }
    return false;
#else
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (h) { dlclose(h); return true; }
    return false;
#endif
}

using namespace Halide;

// ---------------------------------------------------------------------------
// Report: dual output to stdout and a file
// ---------------------------------------------------------------------------

struct Report {
    std::ofstream file;

    explicit Report(const std::string &path) : file(path) {
        if (!file.is_open()) {
            fprintf(stderr, "Warning: could not open report file '%s'\n", path.c_str());
        }
    }

    Report &operator<<(const std::string &s) {
        fputs(s.c_str(), stdout);
        if (file.is_open()) { file << s; file.flush(); }
        return *this;
    }
    Report &operator<<(const char *s) { return *this << std::string(s); }
    Report &operator<<(char c) {
        fputc(c, stdout);
        if (file.is_open()) { file << c; file.flush(); }
        return *this;
    }
    Report &operator<<(int v)    { return *this << std::to_string(v); }
    Report &operator<<(double v) {
        char buf[32]; snprintf(buf, sizeof(buf), "%.2f", v);
        return *this << buf;
    }

    template<typename... Args>
    void printf(const char *fmt, Args... args) {
        char buf[1024];
        snprintf(buf, sizeof(buf), fmt, args...);
        *this << buf;
    }
};

// ---------------------------------------------------------------------------
// Targets and autoscheduler parameters
// ---------------------------------------------------------------------------

// GPU targets for schedule display -- built from the actual Windows host so
// the OS/arch/ISA fields match this machine.  No GPU hardware is needed;
// apply_autoscheduler only generates schedule source strings.
static inline Target gpu_cuda_target() {
    return get_host_target().with_feature(Target::CUDA);
}
static inline Target gpu_opencl_target() {
    return get_host_target().with_feature(Target::OpenCL);
}
static inline Target gpu_d3d12_target() {
    return get_host_target().with_feature(Target::D3D12Compute);
}

static inline AutoschedulerParams sioutas_params() {
    return {"Sioutas2020"};
}
static inline AutoschedulerParams anderson_params() {
    return {"Anderson2021", {{"parallelism", "80"}, {"beam_size", "32"}}};
}
static inline AutoschedulerParams adams_params() {
    // Adams2019: CPU beam-search scheduler, predecessor to Anderson2021.
    return {"Adams2019", {{"parallelism", "16"}, {"beam_size", "32"}}};
}
static inline AutoschedulerParams li2018_params() {
    return {"Li2018", {{"parallelism", "32"}}};
}
static inline AutoschedulerParams mullapudi_params() {
    // Mullapudi2016: original Halide autoscheduler (CPU).
    return {"Mullapudi2016", {{"parallelism", "16"}}};
}
static inline AutoschedulerParams mullapudi_gpu_params() {
    // Mullapudi2016 with experimental GPU support enabled.
    return {"Mullapudi2016", {{"experimental_gpu_schedule", "true"},
                              {"parallelism", "128"}}};
}

// ---------------------------------------------------------------------------
// Plugin loading
// ---------------------------------------------------------------------------

static inline std::string default_report_path(int argc, char **argv,
                                               const char *demo_name) {
    if (argc >= 7 && strlen(argv[6]) > 0) return argv[6];
    return std::string(demo_name) + "_report.txt";
}

static inline void load_all_plugins(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr,
                "Usage: %s <sioutas2020-dll> <anderson2021-dll> <adams2019-dll>"
                " <li2018-dll> <mullapudi2016-dll> [report.txt]\n",
                argv[0]);
        exit(1);
    }
    load_plugin(argv[1]);  // Sioutas2020
    load_plugin(argv[2]);  // Anderson2021
    load_plugin(argv[3]);  // Adams2019
    load_plugin(argv[4]);  // Li2018
    load_plugin(argv[5]);  // Mullapudi2016
}

// ---------------------------------------------------------------------------
// Schedule statistics
// ---------------------------------------------------------------------------

struct ScheduleStats {
    int total_lines        = 0;
    int gpu_tile_count     = 0;
    int gpu_count          = 0;   // .gpu(outer, inner) calls (Sioutas2020 full path)
    int compute_root_count = 0;
    int compute_at_count   = 0;   // .compute_at( calls (fusion)
    int gpu_single_thread  = 0;
    int parallel_count     = 0;
    int vectorize_count    = 0;
    int rfactor_count      = 0;
};

static inline int count_occurrences(const std::string &src,
                                    const std::string &needle) {
    int n = 0; size_t pos = 0;
    while ((pos = src.find(needle, pos)) != std::string::npos) {
        ++n; pos += needle.size();
    }
    return n;
}

static inline ScheduleStats compute_stats(const std::string &src) {
    ScheduleStats s;
    s.total_lines        = count_occurrences(src, "\n");
    s.gpu_tile_count     = count_occurrences(src, ".gpu_tile(");
    s.gpu_count          = count_occurrences(src, ".gpu(");
    s.compute_root_count = count_occurrences(src, ".compute_root()");
    s.compute_at_count   = count_occurrences(src, ".compute_at(");
    s.gpu_single_thread  = count_occurrences(src, ".gpu_single_thread()");
    s.parallel_count     = count_occurrences(src, ".parallel(");
    s.vectorize_count    = count_occurrences(src, ".vectorize(");
    s.rfactor_count      = count_occurrences(src, ".rfactor(");
    return s;
}

// ---------------------------------------------------------------------------
// Benchmark timing
// ---------------------------------------------------------------------------

static inline double time_ms(std::function<void()> fn,
                              int warmup = 2, int iters = 5) {
    for (int i = 0; i < warmup; i++) fn();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

// Returns ms/call, or -1.0 if not applicable.
using BenchFn = std::function<double(const AutoschedulerParams &,
                                     const Target &)>;

// ---------------------------------------------------------------------------
// ASCII table helpers -- 5 data columns, 10 chars each
//
// Column layout (87 chars total):
//   | label(18) | data(10) | data(10) | data(10) | data(10) | data(10) |
//   +--20-------+-12-------+-12-------+-12-------+-12-------+-12-------+
// ---------------------------------------------------------------------------

static const char *kSep5 =
    "+--------------------+------------+------------+------------+------------+------------+\n";
static const char *kSepWide5 =
    "+====================+============+============+============+============+============+\n";

static inline void table_sep(Report &rpt, bool heavy = false) {
    rpt << (heavy ? kSepWide5 : kSep5);
}

static inline void table_header(Report &rpt,
                                 const char *c0, const char *c1, const char *c2,
                                 const char *c3, const char *c4, const char *c5) {
    rpt.printf("| %-18s | %-10s | %-10s | %-10s | %-10s | %-10s |\n",
               c0, c1, c2, c3, c4, c5);
}

static inline void table_row_iiiii(Report &rpt, const char *label,
                                    int v0, int v1, int v2, int v3, int v4) {
    rpt.printf("| %-18s | %-10d | %-10d | %-10d | %-10d | %-10d |\n",
               label, v0, v1, v2, v3, v4);
}

static inline void table_row_sssss(Report &rpt, const char *label,
                                    const char *s0, const char *s1, const char *s2,
                                    const char *s3, const char *s4) {
    rpt.printf("| %-18s | %-10s | %-10s | %-10s | %-10s | %-10s |\n",
               label, s0, s1, s2, s3, s4);
}

// ---------------------------------------------------------------------------
// Printing sections
// ---------------------------------------------------------------------------

static inline void print_banner(Report &rpt, const char *name) {
    rpt << "\n"
        << "============================================================\n"
        << "  Demo: " << name << "\n"
        << "============================================================\n";
}

static inline void print_schedule_section(Report &rpt,
                                          const char *scheduler_name,
                                          const AutoSchedulerResults &r) {
    rpt << "\n-- " << scheduler_name
        << " " << std::string(50 - strlen(scheduler_name), '-') << "\n"
        << r.schedule_source << "\n";
}

// names[5]: S2020, A2021, A2019, Li2018, M2016
// targets[5]: the target string used for each scheduler's schedule display
// Emit one stats row only if at least one value is non-zero.
#define STATS_ROW(label, field) do { \
    int v0=stats[0].field, v1=stats[1].field, v2=stats[2].field, \
        v3=stats[3].field, v4=stats[4].field; \
    if (v0|v1|v2|v3|v4) table_row_iiiii(rpt, label, v0, v1, v2, v3, v4); \
} while(0)

static inline void print_stats_table(Report &rpt,
                                     const char *const names[5],
                                     const ScheduleStats stats[5]) {
    table_sep(rpt);
    table_header(rpt, "Metric", names[0], names[1], names[2], names[3], names[4]);
    table_sep(rpt);
    STATS_ROW("Lines",            total_lines);
    STATS_ROW("gpu_tile",         gpu_tile_count);
    STATS_ROW("gpu",              gpu_count);
    STATS_ROW("compute_root",     compute_root_count);
    STATS_ROW("compute_at",       compute_at_count);
    STATS_ROW("gpu_single_thread",gpu_single_thread);
    STATS_ROW("parallel",         parallel_count);
    STATS_ROW("vectorize",        vectorize_count);
    STATS_ROW("rfactor",          rfactor_count);
    table_sep(rpt);
}
#undef STATS_ROW

// Format time in microseconds: "NNNus" or "N/A"
// Input is milliseconds; multiply by 1000 to get microseconds.
static inline void fmt_us(char *buf, size_t sz, double ms) {
    if (ms > 0) snprintf(buf, sz, "%.0fus", ms * 1000.0);
    else        snprintf(buf, sz, "N/A");
}

// Format speedup relative to baseline: "N.NNx" or "1.00x(base)" or "N/A"
static inline void fmt_spd(char *buf, size_t sz, double base, double ms,
                            bool is_base) {
    if (is_base && base > 0)        snprintf(buf, sz, "1.00x(base)");
    else if (base > 0 && ms > 0)    snprintf(buf, sz, "%.2fx", base / ms);
    else                            snprintf(buf, sz, "N/A");
}

static inline void print_cpu_perf_table(Report &rpt,
                                        double ms_s,    // Sioutas2020
                                        double ms_a2019,// Adams2019
                                        double ms_m) {  // Mullapudi2016
    const std::string tgt = get_host_target().to_string();
    rpt << "\n--- CPU Performance Benchmark (target: " << tgt << ") ---\n"
        << "    Note: Anderson2021/Li2018 are GPU-only schedulers.\n";

    char t_s[16], t_a[16], t_m[16];
    char sp_s[16], sp_a[16], sp_m[16];
    fmt_us(t_s, sizeof(t_s), ms_s);
    fmt_us(t_a, sizeof(t_a), ms_a2019);
    fmt_us(t_m, sizeof(t_m), ms_m);
    fmt_spd(sp_s, sizeof(sp_s), ms_s, ms_s, true);
    fmt_spd(sp_a, sizeof(sp_a), ms_s, ms_a2019, false);
    fmt_spd(sp_m, sizeof(sp_m), ms_s, ms_m, false);

    table_sep(rpt);
    table_header(rpt, "Scheduler", "Time(us)", "vs S2020", "Scheduler", "Time(us)", "vs S2020");
    table_sep(rpt);
    // Row 1: Sioutas2020 | Adams2019
    table_row_sssss(rpt, "Sioutas2020", t_s, sp_s, "Adams2019", t_a, sp_a);
    // Row 2: Mullapudi2016
    table_row_sssss(rpt, "Mullapudi2016", t_m, sp_m, "", "", "");
    table_sep(rpt);
}

// GPU perf table: GPU-capable schedulers only (no Adams2019 — it's CPU-only)
static inline void print_gpu_perf_table(Report &rpt,
                                        double ms_s,   // Sioutas2020
                                        double ms_a21, // Anderson2021
                                        double ms_l,   // Li2018
                                        double ms_m,   // Mullapudi2016
                                        const std::string &gpu_target_str) {
    rpt << "\n--- GPU Performance Benchmark (target: " << gpu_target_str << ") ---\n";

    char t_s[16], t_a[16], t_l[16], t_m[16];
    char sp_s[16], sp_a[16], sp_l[16], sp_m[16];
    fmt_us(t_s, sizeof(t_s), ms_s);
    fmt_us(t_a, sizeof(t_a), ms_a21);
    fmt_us(t_l, sizeof(t_l), ms_l);
    fmt_us(t_m, sizeof(t_m), ms_m);
    fmt_spd(sp_s, sizeof(sp_s), ms_s, ms_s, true);
    fmt_spd(sp_a, sizeof(sp_a), ms_s, ms_a21, false);
    fmt_spd(sp_l, sizeof(sp_l), ms_s, ms_l, false);
    fmt_spd(sp_m, sizeof(sp_m), ms_s, ms_m, false);

    table_sep(rpt);
    table_header(rpt, "Scheduler", "Time(us)", "vs S2020", "Scheduler", "Time(us)", "vs S2020");
    table_sep(rpt);
    table_row_sssss(rpt, "Sioutas2020",   t_s, sp_s, "Anderson2021", t_a, sp_a);
    table_row_sssss(rpt, "Li2018",        t_l, sp_l, "Mullapudi(GPU)", t_m, sp_m);
    table_sep(rpt);
}

// ---------------------------------------------------------------------------
// Core five-way comparison + benchmark runner
// ---------------------------------------------------------------------------

// Helper: run apply_autoscheduler with error isolation.
// Returns true on success; on failure writes a note and leaves stats zeroed.
static inline bool safe_apply(Pipeline &p, const Target &t,
                              const AutoschedulerParams &params,
                              const char *label,
                              Report &rpt, ScheduleStats &stats) {
    try {
        auto r = p.apply_autoscheduler(t, params);
        stats = compute_stats(r.schedule_source);
        print_schedule_section(rpt, label, r);
        return true;
    } catch (const std::exception &e) {
        rpt << "\n-- " << label
            << " " << std::string(50 - strlen(label), '-') << "\n"
            << "  (scheduler returned an error: " << e.what() << ")\n\n";
        return false;
    } catch (...) {
        rpt << "\n-- " << label
            << " " << std::string(50 - strlen(label), '-') << "\n"
            << "  (scheduler crashed with unknown exception)\n\n";
        return false;
    }
}

static inline void run_five_way_with_bench(
    Report &rpt,
    const char *pipeline_name,
    Pipeline (*make_fn)(const std::string &),
    BenchFn bench_fn)
{
    print_banner(rpt, pipeline_name);

    // Abbreviated column headers for the stats table
    const char *names[5] = {"S2020", "A2021", "A2019", "Li2018", "M2016"};
    ScheduleStats stats[5];

    // ---- Schedule comparison -----------------------------------------------
    // Three Windows GPU backends + host CPU.
    // No GPU hardware needed -- apply_autoscheduler generates source strings only.
    const Target cuda_target   = gpu_cuda_target();
    const Target opencl_target = gpu_opencl_target();
    const Target d3d12_target  = gpu_d3d12_target();
    const Target cpu_target    = get_host_target();

    // ---- CUDA schedules (stats[0..1,3] used for the primary stats table) ----
    rpt << "\n--- Schedules (CUDA: " << cuda_target.to_string() << ") ---\n";
    {
        Pipeline p = make_fn("_sc");
        safe_apply(p, cuda_target, sioutas_params(), "Sioutas2020 (CUDA)", rpt, stats[0]);
    }
    {
        Pipeline p = make_fn("_ac");
        safe_apply(p, cuda_target, anderson_params(), "Anderson2021 (CUDA)", rpt, stats[1]);
    }
    {   // Li2018 stats[3] -- CUDA variant
        Pipeline p = make_fn("_lc");
        safe_apply(p, cuda_target, li2018_params(), "Li2018 (CUDA)", rpt, stats[3]);
    }

    // ---- OpenCL schedules ---------------------------------------------------
    rpt << "\n--- Schedules (OpenCL: " << opencl_target.to_string() << ") ---\n";
    {   // Unused stats -- OpenCL schedules shown for reference only
        ScheduleStats tmp;
        Pipeline p = make_fn("_so");
        safe_apply(p, opencl_target, sioutas_params(), "Sioutas2020 (OpenCL)", rpt, tmp);
    }
    {
        ScheduleStats tmp;
        Pipeline p = make_fn("_ao");
        safe_apply(p, opencl_target, anderson_params(), "Anderson2021 (OpenCL)", rpt, tmp);
    }
    {
        ScheduleStats tmp;
        Pipeline p = make_fn("_lo");
        safe_apply(p, opencl_target, li2018_params(), "Li2018 (OpenCL)", rpt, tmp);
    }

    // ---- D3D12Compute schedules ---------------------------------------------
    rpt << "\n--- Schedules (D3D12Compute: " << d3d12_target.to_string() << ") ---\n";
    {
        ScheduleStats tmp;
        Pipeline p = make_fn("_sd");
        safe_apply(p, d3d12_target, sioutas_params(), "Sioutas2020 (D3D12)", rpt, tmp);
    }
    {
        ScheduleStats tmp;
        Pipeline p = make_fn("_ad");
        safe_apply(p, d3d12_target, anderson_params(), "Anderson2021 (D3D12)", rpt, tmp);
    }
    {
        ScheduleStats tmp;
        Pipeline p = make_fn("_ld");
        safe_apply(p, d3d12_target, li2018_params(), "Li2018 (D3D12)", rpt, tmp);
    }

    // ---- CPU schedulers -----------------------------------------------------
    rpt << "\n--- Schedules (CPU: " << cpu_target.to_string() << ") ---\n";
    {
        Pipeline p = make_fn("_a2");
        safe_apply(p, cpu_target, adams_params(), "Adams2019 (CPU)", rpt, stats[2]);
    }
    {
        Pipeline p = make_fn("_m");
        safe_apply(p, cpu_target, mullapudi_params(), "Mullapudi2016 (CPU)", rpt, stats[4]);
    }

    // Stats table uses the CUDA results as the primary GPU comparison.
    rpt << "\n--- Schedule Statistics (GPU target: CUDA) ---\n";
    print_stats_table(rpt, names, stats);

    // ---- CPU benchmark -----------------------------------------------------
    // Only CPU-capable schedulers: Sioutas2020 (CPU fallback), Adams2019, Mullapudi2016.
    // Li2018 is GPU-only and hard-crashes on some CPU pipelines (e.g. NL-means).
    rpt << "\n    Benchmarking Sioutas2020 on CPU...\n";   fflush(stdout);
    double ms_s = -1.0;
    try { ms_s = bench_fn(sioutas_params(), cpu_target); }
    catch (const std::exception &e) { rpt << "    ERROR: " << e.what() << "\n"; }

    rpt << "    Benchmarking Adams2019 on CPU...\n";       fflush(stdout);
    double ms_a2019 = -1.0;
    try { ms_a2019 = bench_fn(adams_params(), cpu_target); }
    catch (const std::exception &e) { rpt << "    ERROR: " << e.what() << "\n"; }

    rpt << "    Benchmarking Mullapudi2016 on CPU...\n";   fflush(stdout);
    double ms_m = -1.0;
    try { ms_m = bench_fn(mullapudi_params(), cpu_target); }
    catch (const std::exception &e) { rpt << "    ERROR: " << e.what() << "\n"; }

    print_cpu_perf_table(rpt, ms_s, ms_a2019, ms_m);

    // ---- GPU schedule analysis + optional execution benchmark --------------
    rpt << "\n--- GPU Schedule Analysis (CUDA target: " << cuda_target.to_string() << ") ---\n"
        << "    GPU primitives used by each scheduler (from CUDA schedule above):\n";
    rpt.printf("    %-22s  gpu=%d  gpu_tile=%d  compute_root=%d  compute_at=%d  gpu_single_thread=%d\n",
               "Sioutas2020:", stats[0].gpu_count, stats[0].gpu_tile_count,
               stats[0].compute_root_count, stats[0].compute_at_count, stats[0].gpu_single_thread);
    rpt.printf("    %-22s  gpu=%d  gpu_tile=%d  compute_root=%d  compute_at=%d  gpu_single_thread=%d\n",
               "Anderson2021:", stats[1].gpu_count, stats[1].gpu_tile_count,
               stats[1].compute_root_count, stats[1].compute_at_count, stats[1].gpu_single_thread);
    rpt.printf("    %-22s  gpu=%d  gpu_tile=%d  compute_root=%d  compute_at=%d  gpu_single_thread=%d\n",
               "Li2018:", stats[3].gpu_count, stats[3].gpu_tile_count,
               stats[3].compute_root_count, stats[3].compute_at_count, stats[3].gpu_single_thread);
    rpt << "    Mullapudi2016: CPU scheduler shown above; GPU via experimental_gpu_schedule.\n"
        << "    Sioutas2020 (full cost-model path): .gpu(outer,inner) + .compute_at() fusion\n"
        << "    per the TACO 2020 paper.  SimpleAutoSchedule fallback uses gpu_tile().\n"
        << "    Anderson2021/Li2018 use fine-grained gpu_blocks()+gpu_threads() search.\n"
        << "    (GPU execution benchmark: set HL_GPU_BENCH=1 to enable.)\n";

    // GPU execution benchmarks (opt-in: set HL_GPU_BENCH=1)
    // First realize() triggers LLVM NVPTX codegen + CUDA driver PTX JIT,
    // which can take several minutes on Optimus/hybrid-GPU systems.
    const char *gpu_bench_env = std::getenv("HL_GPU_BENCH");
    const bool run_gpu_bench = (gpu_bench_env && std::string(gpu_bench_env) == "1");

    if (run_gpu_bench) {
        if (!probe_cuda_driver()) {
            rpt << "    GPU execution skipped: CUDA driver (nvcuda.dll) not found.\n";
        } else {
            // SM 8.6 = RTX 3060/3070/3080 (Ampere).  Adjust CUDACapability if needed.
            Target gpu_target = get_host_target()
                                    .with_feature(Target::CUDA)
                                    .with_feature(Target::CUDACapability86);
            const std::string gpu_target_str = gpu_target.to_string();

            // ---- GPU driver warmup ----------------------------------------
            // The first realize() on a CUDA target triggers cuInit() (discrete
            // GPU power-up on Optimus) + LLVM NVPTX codegen.  Run a trivial
            // kernel first so that power-up cost doesn't skew benchmark times.
            rpt << "\n    Warming up CUDA driver (may take 1-5 min on Optimus)...\n";
            fflush(stdout);
            bool gpu_ok = false;
            try {
                Var gx("gx"), gxo("gxo"), gxi("gxi");
                Func warmup_fn("gpu_warmup_fn");
                warmup_fn(gx) = cast<float>(gx) * 0.001f;
                warmup_fn.gpu_tile(gx, gxo, gxi, 64).compute_root();
                Buffer<float> warm_buf(1024);
                Pipeline(warmup_fn).realize(warm_buf, gpu_target);
                warm_buf.copy_to_host();
                gpu_ok = true;
                rpt << "    CUDA driver ready.\n";
            } catch (const std::exception &e) {
                rpt << "    CUDA warmup failed: " << e.what() << "\n";
            } catch (...) {
                rpt << "    CUDA warmup failed (unknown error).\n";
            }
            fflush(stdout);

            if (gpu_ok) {
                // Each bench_fn call compiles a new NVPTX kernel (~30-60s each).
                double gms_s = -1.0, gms_a21 = -1.0, gms_l = -1.0, gms_m = -1.0;

                rpt << "    Benchmarking Sioutas2020 on GPU...\n";   fflush(stdout);
                try { gms_s = bench_fn(sioutas_params(), gpu_target); }
                catch (const std::exception &e) { rpt << "    GPU ERROR (S2020): " << e.what() << "\n"; }
                catch (...) { rpt << "    GPU ERROR (S2020): unknown\n"; }

                rpt << "    Benchmarking Anderson2021 on GPU...\n";  fflush(stdout);
                try { gms_a21 = bench_fn(anderson_params(), gpu_target); }
                catch (const std::exception &e) { rpt << "    GPU ERROR (A2021): " << e.what() << "\n"; }
                catch (...) { rpt << "    GPU ERROR (A2021): unknown\n"; }

                rpt << "    Benchmarking Li2018 on GPU...\n";        fflush(stdout);
                try { gms_l = bench_fn(li2018_params(), gpu_target); }
                catch (const std::exception &e) { rpt << "    GPU ERROR (Li2018): " << e.what() << "\n"; }
                catch (...) { rpt << "    GPU ERROR (Li2018): unknown\n"; }

                // Mullapudi2016 GPU (experimental_gpu_schedule=true) crashes the process
                // with STATUS_STACK_BUFFER_OVERRUN inside the DLL -- skip GPU benchmark.
                rpt << "    Mullapudi2016 GPU benchmark skipped (DLL stack overrun with experimental_gpu_schedule).\n";

                print_gpu_perf_table(rpt, gms_s, gms_a21, gms_l, gms_m, gpu_target_str);
            }
        }
    }
    rpt << "\n";
}
