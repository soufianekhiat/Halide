/** \file
 *
 * Sioutas2020 GPU Autoscheduler
 *
 * Based on the SimpleAutoSchedule approach from:
 *   "Scheduling Halide Pipelines on GPUs" — Sioutas, Sousa, Stuijk, Basten,
 *   Corporaal (TUE-EE-ES), TACO 2020.
 *
 * Algorithm:
 *   1. Build the transitive closure of all pipeline functions.
 *   2. Inline functions whose compute cost is comparable to call cost
 *      (trivial functions) and element-wise single-consumer functions.
 *   3. For every remaining function on a GPU target: apply gpu_tile() on the
 *      two outermost pure dimensions; update definitions (reductions) are
 *      tiled when the left-hand side uses the pure variables directly, and
 *      fall back to gpu_single_thread() otherwise.
 *   4. For CPU targets: tile + parallelize + vectorize.
 */

#include "HalidePlugin.h"
#include "ParamParser.h"

// AutoScheduleUtils, FindCalls, RealizationOrder and all other internal
// Halide headers are included transitively via HalidePlugin.h → Halide.h
// (the generated amalgamated header in build/include/).

#include <sstream>
#include <string>
#include <map>
#include <vector>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using std::map;
using std::ostringstream;
using std::string;
using std::vector;

namespace {

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/** Tunable parameters for the Sioutas2020 autoscheduler.
 *
 * Tile sizes follow the TACO benchmark defaults from the paper:
 * 16×16 GPU threads, 4-wide channel tile.  CPU defaults are
 * conservative (64×64 tile, 8-wide SIMD vectorization).
 */
struct Sioutas2020Params {
    /** GPU thread-block tile along the x (innermost) dimension. */
    int gpu_tile_x{16};
    /** GPU thread-block tile along the y dimension. */
    int gpu_tile_y{16};
    /** GPU tile along a third (channel) dimension when present. */
    int gpu_tile_channel{4};
    /** CPU tile along x. */
    int cpu_tile_x{64};
    /** CPU tile along y. */
    int cpu_tile_y{64};
    /** Maximum CPU parallelism (outer parallel loops). */
    int parallelism{16};
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Return true if every left-hand-side argument of update definition
 * \p idx is the same pure Variable as the corresponding element of
 * Function::args().  This identifies simple gather-style reductions
 * of the form  f(x,y) += expr  where the output coordinates are
 * unchanged across update iterations, making them safe to GPU-tile
 * over x and y. */
bool update_lhs_is_pure_identity(const Function &func, int idx) {
    const auto &pure_args = func.args();               // e.g. {"x","y"}
    const auto &lhs_exprs = func.update(idx).args();   // LHS of update
    if (lhs_exprs.size() != pure_args.size()) {
        return false;
    }
    for (size_t i = 0; i < pure_args.size(); ++i) {
        const Variable *v = lhs_exprs[i].as<Variable>();
        if (!v || v->name != pure_args[i]) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Per-function GPU schedulers
// ---------------------------------------------------------------------------

/** Apply GPU tiling to a pure (no update definitions) function and
 * emit the corresponding schedule-source string. */
void schedule_pure_gpu(Func &f,
                       const Sioutas2020Params &p,
                       ostringstream &src) {
    const string &name = f.name();
    const int ndims = f.dimensions();

    if (ndims == 0) {
        f.compute_root();
        src << name << ".compute_root();\n";
        return;
    }

    const auto &args = f.args();

    if (ndims >= 2) {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
        f.gpu_tile(args[0], args[1],
                   xo, yo, xi, yi,
                   p.gpu_tile_x, p.gpu_tile_y,
                   TailStrategy::GuardWithIf, DeviceAPI::Default_GPU)
         .compute_root();
        src << name << ".gpu_tile("
            << args[0].name() << ", " << args[1].name()
            << ", xo, yo, xi, yi, "
            << p.gpu_tile_x << ", " << p.gpu_tile_y
            << ").compute_root();\n";
    } else {
        // 1-D function
        Var xo("xo"), xi("xi");
        f.gpu_tile(args[0], xo, xi,
                   p.gpu_tile_x,
                   TailStrategy::GuardWithIf, DeviceAPI::Default_GPU)
         .compute_root();
        src << name << ".gpu_tile("
            << args[0].name() << ", xo, xi, " << p.gpu_tile_x
            << ").compute_root();\n";
    }
}

/** Apply GPU scheduling to a function that has one or more update
 * (reduction) definitions.
 *
 *  - Pure init stage: gpu_tile on the two outermost pure dims.
 *  - Update stages whose LHS uses the pure vars directly (gather
 *    reductions): gpu_tile the output dimensions, keeping the
 *    reduction variable(s) sequential inside each thread.
 *  - Other update stages (scatter reductions, histograms …): run
 *    entirely on a single GPU thread to stay on-device while
 *    avoiding race conditions.
 */
void schedule_reduction_gpu(Func &f,
                             const Sioutas2020Params &p,
                             ostringstream &src) {
    const string &name = f.name();
    const int ndims = f.dimensions();
    const Function &func = f.function();
    const int num_updates = static_cast<int>(func.updates().size());

    if (ndims == 0) {
        f.compute_root();
        src << name << ".compute_root(); // 0-D reduction\n";
        return;
    }

    const auto &args = f.args();

    // ---- Pure init stage ------------------------------------------------
    if (ndims >= 2) {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
        f.gpu_tile(args[0], args[1],
                   xo, yo, xi, yi,
                   p.gpu_tile_x, p.gpu_tile_y,
                   TailStrategy::GuardWithIf, DeviceAPI::Default_GPU)
         .compute_root();
        src << name << ".gpu_tile("
            << args[0].name() << ", " << args[1].name()
            << ", xo, yo, xi, yi, "
            << p.gpu_tile_x << ", " << p.gpu_tile_y
            << ").compute_root(); // init\n";
    } else {
        Var xo("xo"), xi("xi");
        f.gpu_tile(args[0], xo, xi,
                   p.gpu_tile_x,
                   TailStrategy::GuardWithIf, DeviceAPI::Default_GPU)
         .compute_root();
        src << name << ".gpu_tile("
            << args[0].name() << ", xo, xi, " << p.gpu_tile_x
            << ").compute_root(); // init\n";
    }

    // ---- Update (reduction) stages --------------------------------------
    for (int i = 0; i < num_updates; ++i) {
        Stage upd = f.update(i);
        if (update_lhs_is_pure_identity(func, i)) {
            // Safe to tile over output dimensions; reduction var(s)
            // remain sequential within each thread.
            if (ndims >= 2) {
                Var uxo("uxo"), uyo("uyo"), uxi("uxi"), uyi("uyi");
                upd.gpu_tile(args[0], args[1],
                             uxo, uyo, uxi, uyi,
                             p.gpu_tile_x, p.gpu_tile_y,
                             TailStrategy::GuardWithIf, DeviceAPI::Default_GPU);
                src << name << ".update(" << i << ").gpu_tile("
                    << args[0].name() << ", " << args[1].name()
                    << ", uxo, uyo, uxi, uyi, "
                    << p.gpu_tile_x << ", " << p.gpu_tile_y << ");\n";
            } else {
                Var uxo("uxo"), uxi("uxi");
                upd.gpu_tile(args[0], uxo, uxi,
                             p.gpu_tile_x,
                             TailStrategy::GuardWithIf, DeviceAPI::Default_GPU);
                src << name << ".update(" << i << ").gpu_tile("
                    << args[0].name() << ", uxo, uxi, "
                    << p.gpu_tile_x << ");\n";
            }
        } else {
            // Scatter / complex LHS: run on a single GPU thread to keep
            // the intermediate buffer on-device without races.
            upd.gpu_single_thread();
            src << name << ".update(" << i
                << ").gpu_single_thread(); // scatter/complex reduction\n";
        }
    }
}

// ---------------------------------------------------------------------------
// CPU scheduler
// ---------------------------------------------------------------------------

/** Apply a tiled + parallel + vectorized CPU schedule and emit source.
 *  Reduction funcs (has_update_definition) get vectorize-only (no parallel)
 *  on the pure init to avoid deadlocking Halide's thread pool when the
 *  unscheduled update runs after a parallel init barrier. */
void schedule_for_cpu(Func &f,
                      const Sioutas2020Params &p,
                      ostringstream &src) {
    const string &name = f.name();
    const int ndims = f.dimensions();

    if (ndims == 0) {
        f.compute_root();
        src << name << ".compute_root();\n";
        return;
    }

    const auto &args = f.args();
    constexpr int vec_width = 8;
    // Do not parallelize the pure init of reduction funcs: the unscheduled
    // update runs sequentially after the init barrier, and parallelizing the
    // init can deadlock Halide's thread pool when the update tries to re-enter.
    const bool has_update = f.has_update_definition();

    if (ndims >= 2) {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
        if (has_update) {
            f.tile(args[0], args[1],
                   xo, yo, xi, yi,
                   p.cpu_tile_x, p.cpu_tile_y)
             .vectorize(xi, vec_width)
             .compute_root();
            src << name << ".tile("
                << args[0].name() << ", " << args[1].name()
                << ", xo, yo, xi, yi, "
                << p.cpu_tile_x << ", " << p.cpu_tile_y
                << ").vectorize(xi, " << vec_width
                << ").compute_root(); // reduction: no parallel on init\n";
        } else {
            f.tile(args[0], args[1],
                   xo, yo, xi, yi,
                   p.cpu_tile_x, p.cpu_tile_y)
             .parallel(yo)
             .vectorize(xi, vec_width)
             .compute_root();
            src << name << ".tile("
                << args[0].name() << ", " << args[1].name()
                << ", xo, yo, xi, yi, "
                << p.cpu_tile_x << ", " << p.cpu_tile_y
                << ").parallel(yo).vectorize(xi, " << vec_width
                << ").compute_root();\n";
        }
    } else {
        Var xo("xo"), xi("xi");
        if (has_update) {
            f.split(args[0], xo, xi, p.cpu_tile_x)
             .vectorize(xi, vec_width)
             .compute_root();
            src << name << ".split("
                << args[0].name() << ", xo, xi, " << p.cpu_tile_x
                << ").vectorize(xi, " << vec_width
                << ").compute_root(); // reduction: no parallel on init\n";
        } else {
            f.split(args[0], xo, xi, p.cpu_tile_x)
             .parallel(xo)
             .vectorize(xi, vec_width)
             .compute_root();
            src << name << ".split("
                << args[0].name() << ", xo, xi, " << p.cpu_tile_x
                << ").parallel(xo).vectorize(xi, " << vec_width
                << ").compute_root();\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level schedule generation
// ---------------------------------------------------------------------------

string generate_schedule(const vector<Function> &outputs,
                          const Target &target,
                          const Sioutas2020Params &params) {
    ostringstream src;
    src << "// Sioutas2020 autoscheduler -- target: "
        << target.to_string() << "\n\n";

    // 1. Transitive closure of all pipeline functions.
    map<string, Function> env = build_environment(outputs);

    // 2. Lock loop-level references before any scheduling.
    for (auto &kv : env) {
        kv.second.lock_loop_levels();
    }

    // 3. Topological order is needed for the inlining pre-passes.
    vector<string> top_order = topological_order(outputs, env);

    // 4. Inline trivial functions (cost-model-guided).
    if (inline_all_trivial_functions(outputs, top_order, env)) {
        env = build_environment(outputs);
    }

    // 5. Realization order for the element-wise inlining pass.
    vector<string> order = realization_order(outputs, env).first;

    // 6. Inline single-consumer element-wise functions (repeat until
    //    no more inlining is possible).
    while (inline_all_element_wise_functions(outputs, order, env)) {
        env = build_environment(outputs);
        order = realization_order(outputs, env).first;
    }

    // 7. Schedule every remaining function.
    const bool is_gpu = target.has_gpu_feature();
    src << "// " << order.size() << " function(s) to schedule:\n";

    for (const string &name : order) {
        auto it = env.find(name);
        if (it == env.end()) {
            continue;
        }
        const Function &func = it->second;

        if (func.has_extern_definition()) {
            src << "// " << name << ": extern function — not scheduled\n";
            continue;
        }

        Func f(func);

        if (is_gpu) {
            if (func.has_update_definition()) {
                schedule_reduction_gpu(f, params, src);
            } else {
                schedule_pure_gpu(f, params, src);
            }
        } else {
            schedule_for_cpu(f, params, src);
        }
    }

    return src.str();
}

// ---------------------------------------------------------------------------
// Plugin entry point
// ---------------------------------------------------------------------------

struct Sioutas2020 {
    void operator()(const Pipeline &p,
                    const Target &target,
                    const AutoschedulerParams &params_in,
                    AutoSchedulerResults *results) {
        internal_assert(params_in.name == "Sioutas2020");

        Sioutas2020Params params;
        {
            ParamParser parser(params_in.extra);
            parser.parse("gpu_tile_x",      &params.gpu_tile_x);
            parser.parse("gpu_tile_y",      &params.gpu_tile_y);
            parser.parse("gpu_tile_channel",&params.gpu_tile_channel);
            parser.parse("cpu_tile_x",      &params.cpu_tile_x);
            parser.parse("cpu_tile_y",      &params.cpu_tile_y);
            parser.parse("parallelism",     &params.parallelism);
            parser.finish();
        }

        vector<Function> pipeline_outputs;
        pipeline_outputs.reserve(p.outputs().size());
        for (const Func &f : p.outputs()) {
            pipeline_outputs.push_back(f.function());
        }

        results->target = target;
        results->autoscheduler_params = params_in;
        results->schedule_source =
            generate_schedule(pipeline_outputs, target, params);
    }
};

REGISTER_AUTOSCHEDULER(Sioutas2020)

}  // namespace (anonymous)
}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
