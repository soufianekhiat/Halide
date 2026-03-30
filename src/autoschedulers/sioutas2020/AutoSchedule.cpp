/** \file
 *
 * Sioutas2020 GPU Autoscheduler -- Full Cost-Model Version
 *
 * Based on "Scheduling Halide Pipelines on GPUs" -- Sioutas, Sousa, Stuijk,
 * Basten, Corporaal (TUE-EE-ES), TACO 2020.
 *
 * Reference: https://github.com/TUE-EE-ES/HalideAutoGPU
 *
 * Algorithm (full cost-model path, enable_fusion=true):
 *   1. Build environment; inline trivial and element-wise functions.
 *   2. Infer bounds from output estimates.
 *   3. Build DependenceAnalysis (regions_required via boxes_required).
 *   4. Run Partitioner:
 *        Phase Inline  -- inline pure functions when cost-beneficial.
 *        Phase FastMem -- fuse producers into consumers via compute_at
 *                         when it reduces memory traffic.
 *   5. Generate schedules from groups:
 *        - Group output: split + gpu_threads + gpu_blocks + compute_root.
 *        - Non-output members: compute_at(output, tile_inner_var).
 *
 * Fallback (enable_fusion=false || enable_cost_model=false):
 *   SimpleAutoSchedule -- extent-based 2D/1D gpu_tile on every function.
 */

#include "HalidePlugin.h"
#include "ParamParser.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace {

// ============================================================================
// Parameters
// ============================================================================

struct Sioutas2020Params {
    /** GPU thread-block tile along the innermost tile dimension. */
    int gpu_tile_x{16};
    /** GPU thread-block tile along the second tile dimension. */
    int gpu_tile_y{16};
    /** GPU tile along a third (channel/fused) dimension when present. */
    int gpu_tile_channel{4};
    /** CPU tile along the innermost tile dimension. */
    int cpu_tile_x{64};
    /** CPU tile along the second tile dimension. */
    int cpu_tile_y{64};
    /** Maximum CPU outer parallelism. */
    int parallelism{16};
    /** Unroll reduction variables with estimated extent <= this value. */
    int unroll_rvar_size{16};
    /** Enable producer-consumer fusion (compute_at) via Partitioner. */
    bool enable_fusion{true};
    /** Enable tile-size search via cost model. */
    bool enable_cost_model{true};
};

// ============================================================================
// SimpleAutoSchedule utilities (bounds-inference, tiling helpers)
// ============================================================================

static vector<int> sort_indices(const vector<int> &v) {
    vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&v](int a, int b) { return v[a] < v[b]; });
    return idx;
}

static vector<int> get_int_bounds(const Box &bounds) {
    vector<int> result;
    result.reserve(bounds.size());
    for (int i = 0; i < (int)bounds.size(); i++) {
        const Interval &interval = bounds[i];
        Expr extent = simplify(interval.max - interval.min + 1);
        extent = simplify(substitute_var_estimates(extent));
        auto ext_int = as_const_int(extent);
        if (!ext_int || *ext_int <= 0) {
            return {};
        }
        result.push_back((int)*ext_int);
    }
    return result;
}

static map<string, Box> infer_bounds(const vector<Function> &outputs) {
    vector<Func> funcs;
    vector<Box> output_bounds;
    funcs.reserve(outputs.size());
    output_bounds.reserve(outputs.size());

    for (const Function &f : outputs) {
        const FuncSchedule &sched = f.schedule();
        const vector<Bound> &estimates = sched.estimates();

        Box b;
        bool valid = true;
        for (const string &arg : f.args()) {
            bool found = false;
            for (int i = (int)estimates.size() - 1; i >= 0; --i) {
                if (estimates[i].var == arg &&
                    estimates[i].min.defined() &&
                    estimates[i].extent.defined()) {
                    b.push_back(Interval(
                        estimates[i].min,
                        simplify(estimates[i].min + estimates[i].extent - 1)));
                    found = true;
                    break;
                }
            }
            if (!found) { valid = false; break; }
        }

        if (!valid) {
            debug(1) << "[Sioutas2020] Missing estimates on output '"
                     << f.name() << "' -- using positional tiling fallback.\n";
            return {};
        }

        funcs.emplace_back(f);
        output_bounds.push_back(b);
    }

    return inference_bounds(funcs, output_bounds);
}

static bool update_lhs_is_pure_identity(const Function &func, int idx) {
    const auto &pure_args = func.args();
    const auto &lhs_exprs = func.update(idx).args();
    if (lhs_exprs.size() != pure_args.size()) return false;
    for (size_t i = 0; i < pure_args.size(); ++i) {
        const Variable *v = lhs_exprs[i].as<Variable>();
        if (!v || v->name != pure_args[i]) return false;
    }
    return true;
}

struct TilePlan {
    enum Strategy { TILE_2D, TILE_1D, FUSE_ALL, SINGLE_THREAD };
    Strategy strategy{SINGLE_THREAD};
    int dim0{0};
    int dim1{1};
    int tile_size_0{16};
    int tile_size_1{16};
    vector<int> fuse_dims;
    int fuse_tile_size{4};
};

static TilePlan select_tile_dims(int ndims,
                                  const vector<int> &int_bounds,
                                  const Sioutas2020Params &p) {
    TilePlan plan;
    if (ndims == 0) { plan.strategy = TilePlan::SINGLE_THREAD; return plan; }

    const bool have_bounds = (!int_bounds.empty() && (int)int_bounds.size() >= ndims);

    if (!have_bounds) {
        if (ndims >= 2) {
            plan.strategy = TilePlan::TILE_2D;
            plan.dim0 = 0; plan.dim1 = 1;
            plan.tile_size_0 = p.gpu_tile_x;
            plan.tile_size_1 = p.gpu_tile_y;
        } else {
            plan.strategy = TilePlan::TILE_1D;
            plan.dim0 = 0;
            plan.tile_size_0 = p.gpu_tile_x * p.gpu_tile_y;
        }
        return plan;
    }

    vector<int> sorted = sort_indices(int_bounds);

    if (ndims >= 2) {
        int largest = sorted.back();
        int second  = sorted[(int)sorted.size() - 2];
        int d0 = std::min(largest, second);
        int d1 = std::max(largest, second);
        int ext0 = int_bounds[d0];
        int ext1 = int_bounds[d1];

        if (ext0 >= p.gpu_tile_x && ext1 >= p.gpu_tile_y) {
            plan.strategy    = TilePlan::TILE_2D;
            plan.dim0        = d0;
            plan.dim1        = d1;
            plan.tile_size_0 = p.gpu_tile_x;
            plan.tile_size_1 = p.gpu_tile_y;

            if (ndims >= 3) {
                int extra_product = 1;
                vector<int> extra_dims;
                for (int i = 0; i < ndims; i++) {
                    if (i != d0 && i != d1) {
                        extra_dims.push_back(i);
                        extra_product *= int_bounds[i];
                    }
                }
                if (extra_product >= p.gpu_tile_channel) {
                    plan.fuse_dims      = extra_dims;
                    plan.fuse_tile_size = p.gpu_tile_channel;
                }
            }
            return plan;
        }

        int ext_largest = int_bounds[largest];
        int tile_1d     = p.gpu_tile_x * p.gpu_tile_y;
        if (ext_largest >= tile_1d) {
            plan.strategy    = TilePlan::TILE_1D;
            plan.dim0        = largest;
            plan.tile_size_0 = tile_1d;
            return plan;
        }
    } else {
        int ext_largest = int_bounds[sorted.back()];
        int tile_1d     = p.gpu_tile_x * p.gpu_tile_y;
        if (ext_largest >= tile_1d) {
            plan.strategy    = TilePlan::TILE_1D;
            plan.dim0        = sorted.back();
            plan.tile_size_0 = tile_1d;
            return plan;
        }
    }

    int total = 1;
    for (int i = 0; i < ndims; i++) total *= int_bounds[i];
    if (total >= 2) {
        plan.strategy       = TilePlan::FUSE_ALL;
        plan.fuse_tile_size = std::min(total, 32);
    } else {
        plan.strategy = TilePlan::SINGLE_THREAD;
    }
    return plan;
}

static void apply_gpu_tile_to_func(Func &f,
                                    const TilePlan &plan,
                                    ostringstream &src) {
    const string &name = f.name();
    const auto   &args = f.args();
    const int     ndims = f.dimensions();

    switch (plan.strategy) {

    case TilePlan::SINGLE_THREAD:
        f.compute_root();
        src << name << ".compute_root();\n";
        break;

    case TilePlan::TILE_1D: {
        Var d0 = args[plan.dim0];
        Var xo("xo"), xi("xi");
        f.gpu_tile(d0, xo, xi,
                   plan.tile_size_0,
                   TailStrategy::GuardWithIf, DeviceAPI::Default_GPU)
         .compute_root();
        src << name << ".gpu_tile("
            << d0.name() << ", xo, xi, " << plan.tile_size_0
            << ").compute_root();\n";
        break;
    }

    case TilePlan::FUSE_ALL: {
        Var fused = args[0];
        for (int i = 1; i < ndims; i++) f.fuse(fused, args[i], fused);
        Var xo("xo"), xi("xi");
        f.gpu_tile(fused, xo, xi,
                   plan.fuse_tile_size,
                   TailStrategy::GuardWithIf, DeviceAPI::Default_GPU)
         .compute_root();
        src << name << ".fuse(all_dims).gpu_tile(fused, xo, xi, "
            << plan.fuse_tile_size << ").compute_root();\n";
        break;
    }

    case TilePlan::TILE_2D: {
        Var d0 = args[plan.dim0];
        Var d1 = args[plan.dim1];

        if (plan.fuse_dims.empty()) {
            Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
            f.gpu_tile(d0, d1, xo, yo, xi, yi,
                       plan.tile_size_0, plan.tile_size_1,
                       TailStrategy::GuardWithIf, DeviceAPI::Default_GPU)
             .compute_root();
            src << name << ".gpu_tile("
                << d0.name() << ", " << d1.name()
                << ", xo, yo, xi, yi, "
                << plan.tile_size_0 << ", " << plan.tile_size_1
                << ").compute_root();\n";
        } else {
            Var fused_var = args[plan.fuse_dims[0]];
            for (int i = 1; i < (int)plan.fuse_dims.size(); i++)
                f.fuse(fused_var, args[plan.fuse_dims[i]], fused_var);
            f.reorder(d0, d1, fused_var);
            Var xo("xo"), yo("yo"), zo("zo"), xi("xi"), yi("yi"), zi("zi");
            f.gpu_tile(d0, d1, fused_var,
                       xo, yo, zo, xi, yi, zi,
                       plan.tile_size_0, plan.tile_size_1, plan.fuse_tile_size,
                       TailStrategy::GuardWithIf, DeviceAPI::Default_GPU)
             .compute_root();
            src << name << ".reorder(" << d0.name() << ", " << d1.name()
                << ", fused).gpu_tile("
                << d0.name() << ", " << d1.name() << ", fused"
                << ", xo, yo, zo, xi, yi, zi, "
                << plan.tile_size_0 << ", " << plan.tile_size_1 << ", "
                << plan.fuse_tile_size << ").compute_root();\n";
        }
        break;
    }
    }
}

static void apply_gpu_tile_to_stage(Stage            &upd,
                                     const vector<Var> &args,
                                     const TilePlan    &plan,
                                     const string      &func_name,
                                     int                update_idx,
                                     ostringstream     &src) {
    const int ndims = (int)args.size();

    switch (plan.strategy) {

    case TilePlan::SINGLE_THREAD:
        upd.gpu_single_thread();
        src << func_name << ".update(" << update_idx << ").gpu_single_thread();\n";
        break;

    case TilePlan::TILE_1D: {
        Var d0 = args[plan.dim0];
        Var uxo("uxo"), uxi("uxi");
        upd.gpu_tile(d0, uxo, uxi, plan.tile_size_0,
                     TailStrategy::GuardWithIf, DeviceAPI::Default_GPU);
        src << func_name << ".update(" << update_idx << ").gpu_tile("
            << d0.name() << ", uxo, uxi, " << plan.tile_size_0 << ");\n";
        break;
    }

    case TilePlan::FUSE_ALL: {
        Var fused = args[0];
        for (int i = 1; i < ndims; i++) upd.fuse(fused, args[i], fused);
        Var uxo("uxo"), uxi("uxi");
        upd.gpu_tile(fused, uxo, uxi, plan.fuse_tile_size,
                     TailStrategy::GuardWithIf, DeviceAPI::Default_GPU);
        src << func_name << ".update(" << update_idx
            << ").fuse(all_dims).gpu_tile(fused, uxo, uxi, "
            << plan.fuse_tile_size << ");\n";
        break;
    }

    case TilePlan::TILE_2D: {
        Var d0 = args[plan.dim0];
        Var d1 = args[plan.dim1];

        if (plan.fuse_dims.empty()) {
            Var uxo("uxo"), uyo("uyo"), uxi("uxi"), uyi("uyi");
            upd.gpu_tile(d0, d1, uxo, uyo, uxi, uyi,
                         plan.tile_size_0, plan.tile_size_1,
                         TailStrategy::GuardWithIf, DeviceAPI::Default_GPU);
            src << func_name << ".update(" << update_idx << ").gpu_tile("
                << d0.name() << ", " << d1.name()
                << ", uxo, uyo, uxi, uyi, "
                << plan.tile_size_0 << ", " << plan.tile_size_1 << ");\n";
        } else {
            Var fused_var = args[plan.fuse_dims[0]];
            for (int i = 1; i < (int)plan.fuse_dims.size(); i++)
                upd.fuse(fused_var, args[plan.fuse_dims[i]], fused_var);
            upd.reorder(d0, d1, fused_var);
            Var uxo("uxo"), uyo("uyo"), uzo("uzo"), uxi("uxi"), uyi("uyi"), uzi("uzi");
            upd.gpu_tile(d0, d1, fused_var,
                         uxo, uyo, uzo, uxi, uyi, uzi,
                         plan.tile_size_0, plan.tile_size_1, plan.fuse_tile_size,
                         TailStrategy::GuardWithIf, DeviceAPI::Default_GPU);
            src << func_name << ".update(" << update_idx
                << ").reorder(" << d0.name() << ", " << d1.name() << ", fused)"
                << ".gpu_tile(" << d0.name() << ", " << d1.name() << ", fused"
                << ", uxo, uyo, uzo, uxi, uyi, uzi, "
                << plan.tile_size_0 << ", " << plan.tile_size_1 << ", "
                << plan.fuse_tile_size << ");\n";
        }
        break;
    }
    }
}

static void unroll_small_rvars(Func                    &f,
                                int                      update_idx,
                                const Sioutas2020Params &p,
                                ostringstream           &src) {
    if (p.unroll_rvar_size <= 0) return;

    Stage upd = f.update(update_idx);
    const vector<ReductionVariable> &rvars = upd.get_schedule().rvars();

    for (const auto &rv : rvars) {
        Expr extent = simplify(substitute_var_estimates(rv.extent));
        auto ext_int = as_const_int(extent);
        if (ext_int && *ext_int > 0 && *ext_int <= p.unroll_rvar_size) {
            upd.unroll(RVar(rv.var));
            src << f.name() << ".update(" << update_idx
                << ").unroll(" << rv.var
                << "); // rvar extent=" << *ext_int << "\n";
        }
    }
}

static void schedule_func_gpu(Func                    &f,
                               const Function          &func,
                               const vector<int>       &int_bounds,
                               const Sioutas2020Params &p,
                               ostringstream           &src) {
    const int ndims = f.dimensions();

    if (ndims == 0) {
        f.compute_root();
        src << f.name() << ".compute_root();\n";
        return;
    }

    TilePlan plan = select_tile_dims(ndims, int_bounds, p);
    apply_gpu_tile_to_func(f, plan, src);

    const int         num_updates = static_cast<int>(func.updates().size());
    const vector<Var> &args       = f.args();

    for (int i = 0; i < num_updates; ++i) {
        Stage upd = f.update(i);
        if (update_lhs_is_pure_identity(func, i)) {
            apply_gpu_tile_to_stage(upd, args, plan, f.name(), i, src);
            unroll_small_rvars(f, i, p, src);
        } else {
            upd.gpu_single_thread();
            src << f.name() << ".update(" << i
                << ").gpu_single_thread();\n";
        }
    }
}

static void schedule_for_cpu(Func                    &f,
                              const vector<int>       &int_bounds,
                              const Sioutas2020Params &p,
                              ostringstream           &src) {
    const string &name      = f.name();
    const int     ndims     = f.dimensions();
    const bool    has_update = f.has_update_definition();
    constexpr int vec_width = 8;

    if (ndims == 0) {
        f.compute_root();
        src << name << ".compute_root();\n";
        return;
    }

    const auto &args = f.args();

    int d0 = 0, d1 = 1;
    const bool have_bounds = (!int_bounds.empty() && (int)int_bounds.size() >= ndims);
    if (have_bounds && ndims >= 2) {
        vector<int> sorted = sort_indices(int_bounds);
        int largest = sorted.back();
        int second  = sorted[(int)sorted.size() - 2];
        d0 = std::min(largest, second);
        d1 = std::max(largest, second);
    }

    if (ndims >= 2) {
        Var dim0 = args[d0], dim1 = args[d1];
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

        if (has_update) {
            f.tile(dim0, dim1, xo, yo, xi, yi, p.cpu_tile_x, p.cpu_tile_y)
             .vectorize(xi, vec_width).compute_root();
            src << name << ".tile(" << dim0.name() << ", " << dim1.name()
                << ", xo, yo, xi, yi, " << p.cpu_tile_x << ", " << p.cpu_tile_y
                << ").vectorize(xi, " << vec_width << ").compute_root();\n";
        } else {
            f.tile(dim0, dim1, xo, yo, xi, yi, p.cpu_tile_x, p.cpu_tile_y)
             .parallel(yo).vectorize(xi, vec_width).compute_root();
            src << name << ".tile(" << dim0.name() << ", " << dim1.name()
                << ", xo, yo, xi, yi, " << p.cpu_tile_x << ", " << p.cpu_tile_y
                << ").parallel(yo).vectorize(xi, " << vec_width << ").compute_root();\n";
        }
    } else {
        int d = have_bounds ? sort_indices(int_bounds).back() : 0;
        Var dim0 = args[d];
        Var xo("xo"), xi("xi");

        if (has_update) {
            f.split(dim0, xo, xi, p.cpu_tile_x).vectorize(xi, vec_width).compute_root();
            src << name << ".split(" << dim0.name() << ", xo, xi, " << p.cpu_tile_x
                << ").vectorize(xi, " << vec_width << ").compute_root();\n";
        } else {
            f.split(dim0, xo, xi, p.cpu_tile_x).parallel(xo).vectorize(xi, vec_width).compute_root();
            src << name << ".split(" << dim0.name() << ", xo, xi, " << p.cpu_tile_x
                << ").parallel(xo).vectorize(xi, " << vec_width << ").compute_root();\n";
        }
    }
}

// ============================================================================
// Cost-model helper functions (local implementations)
// ============================================================================

static void substitute_estimates_box(Box &box) {
    box.used = substitute_var_estimates(box.used);
    for (auto &b : box.bounds) {
        b.min = substitute_var_estimates(b.min);
        b.max = substitute_var_estimates(b.max);
    }
}

static void substitute_estimates_region(map<string, Box> &region) {
    for (auto &iter : region) substitute_estimates_box(iter.second);
}

static void local_simplify_box(Box &b) {
    for (size_t i = 0; i < b.size(); i++) {
        b[i].min = simplify(b[i].min);
        b[i].max = simplify(b[i].max);
    }
}

static void merge_regions(map<string, Box> &result, const map<string, Box> &partial) {
    for (const auto &reg : partial) {
        auto iter = result.find(reg.first);
        if (iter == result.end()) {
            result.emplace(reg.first, reg.second);
        } else {
            merge_boxes(iter->second, reg.second);
        }
    }
}

static string get_base_name(const string &name) {
    size_t dot_pos = name.rfind('.');
    if (dot_pos != string::npos) return name.substr(dot_pos + 1);
    return name;
}

static inline bool tile_maps_equal(const map<string, Expr> &m1,
                                    const map<string, Expr> &m2) {
    if (m1.size() != m2.size()) return false;
    for (const auto &it1 : m1) {
        auto it2 = m2.find(it1.first);
        if (it2 == m2.end()) return false;
        if (!equal(it1.second, it2->second)) return false;
    }
    return true;
}

// ============================================================================
// FStage and StageBounds (ported from mullapudi2016)
// ============================================================================

struct FStage {
    Function func;
    uint32_t stage_num;

    FStage(Function func, uint32_t stage_num)
        : func(std::move(func)), stage_num(stage_num) {}

    bool operator==(const FStage &o) const {
        return (func.name() == o.func.name()) && (stage_num == o.stage_num);
    }
    bool operator<(const FStage &o) const {
        return func.name() < o.func.name() ||
               ((func.name() == o.func.name()) && (stage_num < o.stage_num));
    }
    friend std::ostream &operator<<(std::ostream &s, const FStage &fs) {
        if (fs.stage_num == 0) s << fs.func.name();
        else s << fs.func.name() << ".update(" << (fs.stage_num - 1) << ")";
        return s;
    }
};

struct StageBounds {
    FStage f_stage;
    DimBounds bounds;

    StageBounds(const FStage &fs, const DimBounds &b) : f_stage(fs), bounds(b) {}
    StageBounds(Function func, uint32_t stage_num, const DimBounds &b)
        : f_stage(FStage(std::move(func), stage_num)), bounds(b) {}

    bool operator==(const StageBounds &o) const {
        return (f_stage == o.f_stage) && (bounds == o.bounds);
    }
    bool operator<(const StageBounds &o) const {
        return (f_stage < o.f_stage) ||
               ((f_stage == o.f_stage) && (bounds.size() < o.bounds.size()));
    }
};

// ============================================================================
// DependenceAnalysis helpers (ported from mullapudi2016)
// ============================================================================

static void queue_func_regions(map<FStage, DimBounds> &fs_bounds,
                                const Function &prod_func, const Box &region,
                                const set<StageBounds> &visited) {
    DimBounds prod_pure_bounds;
    const vector<string> &args = prod_func.args();
    internal_assert(region.size() == args.size());

    for (size_t v = 0; v < args.size(); v++)
        prod_pure_bounds[args[v]] = region[v];

    vector<DimBounds> prod_bounds = get_stage_bounds(prod_func, prod_pure_bounds);
    size_t num_stages = prod_func.updates().size() + 1;
    internal_assert(prod_bounds.size() == num_stages);

    for (size_t prod_s = 0; prod_s < num_stages; prod_s++) {
        StageBounds sb(prod_func, prod_s, prod_bounds[prod_s]);
        if (visited.find(sb) == visited.end()) {
            auto iter = fs_bounds.find(sb.f_stage);
            if (iter == fs_bounds.end()) {
                fs_bounds.emplace(sb.f_stage, sb.bounds);
            } else {
                for (const auto &b : sb.bounds) {
                    DimBounds &curr_bounds = iter->second;
                    auto b_iter = curr_bounds.find(b.first);
                    if (b_iter == curr_bounds.end()) {
                        curr_bounds.emplace(b.first, b.second);
                    } else {
                        if (b_iter->second.has_lower_bound() && b.second.has_lower_bound())
                            b_iter->second.min = simplify(Interval::make_min(b_iter->second.min, b.second.min));
                        else
                            b_iter->second.min = Interval::neg_inf();
                        if (b_iter->second.has_upper_bound() && b.second.has_upper_bound())
                            b_iter->second.max = simplify(Interval::make_max(b_iter->second.max, b.second.max));
                        else
                            b_iter->second.max = Interval::pos_inf();
                    }
                }
            }
        }
    }
}

static void merge_and_queue_regions(map<FStage, DimBounds> &fs_bounds,
                                     map<string, Box> &regions,
                                     map<string, Box> &curr_regions,
                                     const set<string> &prods,
                                     const map<string, Function> &env,
                                     bool only_regions_computed,
                                     const string &curr_func_name,
                                     const set<StageBounds> &visited) {
    for (const auto &reg : curr_regions) {
        if (!only_regions_computed ||
            (only_regions_computed && (reg.first != curr_func_name))) {
            auto iter = regions.find(reg.first);
            if (iter == regions.end()) regions.emplace(reg.first, reg.second);
            else merge_boxes(iter->second, reg.second);
        }

        if (prods.find(reg.first) == prods.end()) continue;
        const auto &it = env.find(reg.first);
        if ((it != env.end()) && (reg.first != curr_func_name))
            queue_func_regions(fs_bounds, it->second, reg.second, visited);
    }
}

// ============================================================================
// DependenceAnalysis (ported from mullapudi2016)
// ============================================================================

struct DependenceAnalysis {
    map<string, Function> env;
    vector<string> order;
    FuncValueBounds func_val_bounds;

    struct RegionsRequiredQuery {
        string f;
        int stage;
        set<string> prods;
        bool only_regions_computed;

        RegionsRequiredQuery(const string &f, int stage, const set<string> &prods,
                             bool only_regions_computed)
            : f(f), stage(stage), prods(prods),
              only_regions_computed(only_regions_computed) {}

        bool operator==(const RegionsRequiredQuery &o) const {
            return (f == o.f) && (stage == o.stage) && (prods == o.prods) &&
                   (only_regions_computed == o.only_regions_computed);
        }
        bool operator<(const RegionsRequiredQuery &o) const {
            if (f < o.f) return true; if (f > o.f) return false;
            if (stage < o.stage) return true; if (stage > o.stage) return false;
            if (only_regions_computed < o.only_regions_computed) return true;
            if (only_regions_computed > o.only_regions_computed) return false;
            return prods < o.prods;
        }
    };

    struct RegionsRequired {
        DimBounds bounds;
        map<string, Box> regions;
        RegionsRequired(const DimBounds &b, const map<string, Box> &r)
            : bounds(b), regions(r) {}
    };

    map<RegionsRequiredQuery, vector<RegionsRequired>> regions_required_cache;

    DependenceAnalysis(const map<string, Function> &env,
                       const vector<string> &order,
                       const FuncValueBounds &func_val_bounds)
        : env(env), order(order), func_val_bounds(func_val_bounds) {}

    map<string, Box> regions_required(const Function &f, int stage_num,
                                      const DimBounds &bounds,
                                      const set<string> &prods,
                                      bool only_regions_computed,
                                      const Scope<Interval> *input_estimates);

    map<string, Box> regions_required(const Function &f,
                                      const DimBounds &pure_bounds,
                                      const set<string> &prods,
                                      bool only_regions_computed,
                                      const Scope<Interval> *input_estimates);

    map<string, Box> redundant_regions(const Function &f, int stage_num,
                                       const string &var,
                                       const DimBounds &bounds,
                                       const set<string> &prods,
                                       bool only_regions_computed,
                                       const Scope<Interval> *input_estimates);

    vector<map<string, Box>> overlap_regions(const Function &f, int stage_num,
                                              const DimBounds &bounds,
                                              const set<string> &prods,
                                              bool only_regions_computed,
                                              const Scope<Interval> *input_estimates);
};

map<string, Box>
DependenceAnalysis::regions_required(const Function &f,
                                     const DimBounds &pure_bounds,
                                     const set<string> &prods,
                                     bool only_regions_computed,
                                     const Scope<Interval> *input_estimates) {
    map<string, Box> regions;
    int num_stages = f.updates().size() + 1;
    for (int s = 0; s < num_stages; s++) {
        DimBounds bounds = get_stage_bounds(f, s, pure_bounds);
        map<string, Box> stage_regions =
            regions_required(f, s, bounds, prods, only_regions_computed, input_estimates);
        merge_regions(regions, stage_regions);
    }
    return regions;
}

map<string, Box>
DependenceAnalysis::regions_required(const Function &f, int stage_num,
                                     const DimBounds &bounds,
                                     const set<string> &prods,
                                     bool only_regions_computed,
                                     const Scope<Interval> *input_estimates) {
    RegionsRequiredQuery query(f.name(), stage_num, prods, only_regions_computed);
    const auto &iter = regions_required_cache.find(query);
    if (iter != regions_required_cache.end()) {
        const auto &it = std::find_if(iter->second.begin(), iter->second.end(),
                                      [&bounds](const RegionsRequired &r) {
                                          return (r.bounds == bounds);
                                      });
        if (it != iter->second.end()) return it->regions;
    }

    map<string, Box> regions;
    map<FStage, DimBounds> fs_bounds;
    set<StageBounds> visited;

    fs_bounds.emplace(FStage(f, stage_num), bounds);

    while (!fs_bounds.empty()) {
        for (int i = order.size() - 1; i >= 0; --i) {
            const Function &func = env.find(order[i])->second;
            int num_stages = func.updates().size() + 1;
            for (int sn = 0; sn < num_stages; ++sn) {
                FStage s(func, sn);

                const auto &it = fs_bounds.find(s);
                if (it == fs_bounds.end()) continue;

                DimBounds curr_bounds = it->second;
                visited.insert(StageBounds(s, curr_bounds));

                Scope<Interval> curr_scope;
                curr_scope.set_containing_scope(input_estimates);

                if (s.func.has_extern_definition()) {
                    for (const ExternFuncArgument &arg : s.func.extern_arguments()) {
                        if (arg.is_func()) {
                            string prod_name = Function(arg.func).name();
                            const Function &prod_func = get_element(env, prod_name);
                            map<string, Box> prod_reg;
                            const vector<string> &pargs = prod_func.args();
                            for (size_t v = 0; v < pargs.size(); v++)
                                prod_reg[prod_name].push_back(Interval());
                            merge_and_queue_regions(fs_bounds, regions, prod_reg, prods, env,
                                                    only_regions_computed, s.func.name(), visited);
                        } else if (arg.is_expr()) {
                            Expr subs_arg = substitute_var_estimates(arg.expr);
                            map<string, Box> arg_regions = boxes_required(subs_arg, curr_scope, func_val_bounds);
                            substitute_estimates_region(arg_regions);
                            merge_and_queue_regions(fs_bounds, regions, arg_regions, prods, env,
                                                    only_regions_computed, s.func.name(), visited);
                        } else if (arg.is_image_param() || arg.is_buffer()) {
                            Buffer<> buf;
                            if (arg.is_image_param()) buf = arg.image_param.buffer();
                            else buf = arg.buffer;
                            map<string, Box> buf_reg;
                            for (int v = 0; v < buf.dimensions(); v++)
                                buf_reg[buf.name()].push_back(Interval());
                            merge_regions(regions, buf_reg);
                        }
                    }
                } else {
                    Definition def = get_stage_definition(s.func, s.stage_num);
                    const vector<Dim> &dims = def.schedule().dims();

                    for (int d = 0; d < (int)dims.size() - 1; d++) {
                        Interval simple_bounds = get_element(curr_bounds, dims[d].var);
                        simple_bounds.min = substitute_var_estimates(simple_bounds.min);
                        simple_bounds.max = substitute_var_estimates(simple_bounds.max);
                        curr_scope.push(dims[d].var, simple_bounds);
                    }

                    class GetAllExprs : public IRMutator {
                    public:
                        vector<Expr> exprs;
                        using IRMutator::mutate;
                        Expr mutate(const Expr &e) override {
                            exprs.push_back(e);
                            return e;
                        }
                    } get_all_exprs;
                    def.mutate(&get_all_exprs);

                    for (const auto &val : get_all_exprs.exprs) {
                        Expr subs_val = substitute_var_estimates(val);
                        map<string, Box> curr_regions =
                            boxes_required(subs_val, curr_scope, func_val_bounds);
                        substitute_estimates_region(curr_regions);

                        Box left_reg;
                        for (const Expr &arg : def.args()) {
                            Expr subs_arg = substitute_var_estimates(arg);
                            map<string, Box> arg_regions =
                                boxes_required(subs_arg, curr_scope, func_val_bounds);
                            substitute_estimates_region(arg_regions);
                            merge_regions(curr_regions, arg_regions);

                            Interval arg_bounds =
                                bounds_of_expr_in_scope(arg, curr_scope, func_val_bounds);
                            left_reg.push_back(arg_bounds);
                        }

                        auto iter_curr = curr_regions.find(s.func.name());
                        if (iter_curr == curr_regions.end())
                            curr_regions.emplace(s.func.name(), left_reg);
                        else
                            merge_boxes(iter_curr->second, left_reg);

                        merge_and_queue_regions(fs_bounds, regions, curr_regions, prods, env,
                                                only_regions_computed, s.func.name(), visited);
                    }
                }

                fs_bounds.erase(it);
            }
        }
    }

    // Substitute pipeline-level estimates for unbounded regions
    map<string, Box> concrete_regions;
    for (auto &f_reg : regions) {
        local_simplify_box(f_reg.second);

        Box concrete_box;
        for (size_t i = 0; i < f_reg.second.size(); i++) {
            Expr lower = f_reg.second[i].min;
            Expr upper = f_reg.second[i].max;

            auto it = env.find(f_reg.first);
            bool in_env = (it != env.end());

            if (!lower.as<IntImm>() && in_env) {
                const Function &curr_f = it->second;
                for (const auto &b : curr_f.schedule().estimates()) {
                    size_t num_pure_args = curr_f.args().size();
                    if ((i < num_pure_args) && (b.var == curr_f.args()[i]))
                        lower = b.min;
                }
            }
            if (!upper.as<IntImm>() && in_env) {
                const Function &curr_f = it->second;
                for (const auto &b : curr_f.schedule().estimates()) {
                    size_t num_pure_args = curr_f.args().size();
                    if ((i < num_pure_args) && (b.var == curr_f.args()[i])) {
                        const IntImm *bmin = b.min.as<IntImm>();
                        const IntImm *bextent = b.extent.as<IntImm>();
                        if (bmin && bextent)
                            upper = IntImm::make(Int(32), bmin->value + bextent->value - 1);
                    }
                }
            }
            concrete_box.push_back(Interval(lower, upper));
        }
        concrete_regions[f_reg.first] = concrete_box;
    }

    regions_required_cache[query].emplace_back(bounds, concrete_regions);
    return concrete_regions;
}

map<string, Box>
DependenceAnalysis::redundant_regions(const Function &f, int stage_num,
                                      const string &var,
                                      const DimBounds &bounds,
                                      const set<string> &prods,
                                      bool only_regions_computed,
                                      const Scope<Interval> *input_estimates) {
    map<string, Box> regions =
        regions_required(f, stage_num, bounds, prods, only_regions_computed, input_estimates);

    DimBounds shifted_bounds;
    for (const auto &b : bounds) {
        if (b.first == var) {
            Expr len = b.second.max - b.second.min + 1;
            shifted_bounds[b.first] = Interval(b.second.min + len, b.second.max + len);
        } else {
            shifted_bounds[b.first] = b.second;
        }
    }

    map<string, Box> regions_shifted =
        regions_required(f, stage_num, shifted_bounds, prods, only_regions_computed, input_estimates);

    map<string, Box> overlaps;
    for (const auto &reg : regions) {
        auto it = regions_shifted.find(reg.first);
        if (it == regions_shifted.end()) continue;
        const Box &b = reg.second;
        const Box &b_shifted = it->second;
        internal_assert(b.size() == b_shifted.size());

        Box b_intersect;
        for (uint32_t i = 0; i < b.size(); i++)
            b_intersect.push_back(Interval::make_intersection(b[i], b_shifted[i]));
        internal_assert(overlaps.find(reg.first) == overlaps.end());
        overlaps.emplace(reg.first, b_intersect);
    }

    for (auto &ov : overlaps) local_simplify_box(ov.second);
    return overlaps;
}

vector<map<string, Box>>
DependenceAnalysis::overlap_regions(const Function &f, int stage_num,
                                    const DimBounds &bounds,
                                    const set<string> &prods,
                                    bool only_regions_computed,
                                    const Scope<Interval> *input_estimates) {
    vector<map<string, Box>> conc_overlaps;
    const vector<Dim> &dims = get_stage_dims(f, stage_num);

    for (int d = 0; d < (int)dims.size() - 1; d++) {
        map<string, Box> conc_reg =
            redundant_regions(f, stage_num, dims[d].var, bounds, prods,
                              only_regions_computed, input_estimates);
        conc_overlaps.push_back(conc_reg);
    }
    return conc_overlaps;
}

// ============================================================================
// get_pipeline_bounds
// ============================================================================

static map<string, Box> get_pipeline_bounds(DependenceAnalysis &analysis,
                                             const vector<Function> &outputs,
                                             const Scope<Interval> *input_estimates) {
    map<string, Box> pipeline_bounds;

    for (const auto &out : outputs) {
        DimBounds pure_bounds;
        Box out_box;
        const auto &estimates = out.schedule().estimates();
        for (const auto &arg : out.args()) {
            int i;
            for (i = estimates.size() - 1; i >= 0; --i) {
                const auto &est = estimates[i];
                if ((est.var == arg) && est.min.defined() && est.extent.defined()) {
                    Interval in = Interval(est.min, simplify(est.min + est.extent - 1));
                    pure_bounds.emplace(arg, in);
                    out_box.push_back(in);
                    break;
                }
            }
            internal_assert(i >= 0) << "Could not find estimate for " << arg << "\n";
        }

        set<string> prods;
        for (const pair<const string, Function> &fpair : analysis.env)
            prods.insert(fpair.first);

        map<string, Box> regions =
            analysis.regions_required(out, pure_bounds, prods, false, input_estimates);
        regions.emplace(out.name(), out_box);
        merge_regions(pipeline_bounds, regions);
    }

    return pipeline_bounds;
}

// ============================================================================
// Partitioner
// ============================================================================

struct Partitioner {
    struct GroupingChoice {
        string prod;
        FStage cons;

        GroupingChoice(const string &prod, const FStage &cons)
            : prod(prod), cons(cons) {}

        bool operator==(const GroupingChoice &o) const {
            return (prod == o.prod) && (cons == o.cons);
        }
        bool operator<(const GroupingChoice &o) const {
            return (prod < o.prod) || ((prod == o.prod) && (cons < o.cons));
        }
    };

    struct Group {
        FStage output;
        vector<FStage> members;
        set<string> inlined;
        map<string, Expr> tile_sizes;

        Group(const FStage &output, const vector<FStage> &members)
            : output(output), members(members) {}
    };

    struct GroupAnalysis {
        Cost cost;
        Expr parallelism;

        GroupAnalysis() : cost(Cost()), parallelism(Expr()) {}
        GroupAnalysis(const Cost &c, Expr p)
            : cost(c), parallelism(std::move(p)) {}

        bool defined() const {
            return cost.defined() && parallelism.defined();
        }
        void simplify() {
            cost.simplify();
            if (parallelism.defined()) parallelism = Internal::simplify(parallelism);
        }
    };

    struct GroupConfig {
        map<string, Expr> tile_sizes;
        GroupAnalysis analysis;
        GroupConfig(const map<string, Expr> &ts, const GroupAnalysis &a)
            : tile_sizes(ts), analysis(a) {}
        GroupConfig() {}
    };

    enum class Level { Inline, FastMem };

    map<GroupingChoice, GroupConfig> grouping_cache;
    map<FStage, Group> groups;
    map<FStage, set<FStage>> children;
    map<FStage, GroupAnalysis> group_costs;

    const map<string, Box> &pipeline_bounds;
    const Target &target;
    DependenceAnalysis &dep_analysis;
    RegionCosts &costs;
    const vector<Function> &outputs;
    // GPU cost model parameters
    float gpu_balance;
    uint64_t gpu_cache_size;
    int min_parallelism;

    Partitioner(const map<string, Box> &pipeline_bounds,
                const Target &target,
                DependenceAnalysis &dep_analysis,
                RegionCosts &costs,
                const vector<Function> &outputs);

    void initialize_groups();

    vector<map<string, Expr>> generate_tile_configs(const FStage &stg);

    GroupAnalysis analyze_group(const Group &g, bool show_analysis);

    pair<map<string, Expr>, GroupAnalysis> find_best_tile_config(const Group &g);

    void group(Level level);

    vector<pair<GroupingChoice, GroupConfig>>
    choose_candidate_grouping(const vector<pair<string, string>> &cands, Level level);

    GroupConfig evaluate_choice(const GroupingChoice &choice, Level level);

    Group merge_groups(const Group &prod_group, const Group &cons_group);

    void merge_groups(const GroupingChoice &choice, const GroupConfig &eval, Level level);

    Expr estimate_benefit(const GroupAnalysis &old_g, const GroupAnalysis &new_g,
                          bool no_redundant_work, bool ensure_parallelism) const;

    Expr estimate_benefit(const vector<pair<GroupingChoice, GroupConfig>> &new_grouping,
                          bool no_redundant_work, bool ensure_parallelism) const;

    DimBounds get_bounds(const FStage &stg);

    DimBounds get_bounds_from_tile_sizes(const FStage &stg,
                                         const map<string, Expr> &tile_sizes);

    map<string, Expr> bounds_to_estimates(const DimBounds &bounds);

    Cost get_pipeline_cost();

    map<FStage, map<string, Box>> group_storage_bounds();

    map<FStage, map<FStage, DimBounds>> group_loop_bounds();
};

Partitioner::Partitioner(const map<string, Box> &_pipeline_bounds,
                         const Target &_target,
                         DependenceAnalysis &_dep_analysis,
                         RegionCosts &_costs,
                         const vector<Function> &_outputs)
    : pipeline_bounds(_pipeline_bounds), target(_target),
      dep_analysis(_dep_analysis), costs(_costs), outputs(_outputs) {

    // GPU cost model defaults (match mullapudi2016 GPU params)
    gpu_balance    = 20.0f;
    gpu_cache_size = 48 * 1024;  // 48 KB shared memory
    min_parallelism = 32;        // at least one GPU warp

    // Create one group per stage
    for (const auto &f : dep_analysis.env) {
        if (!pipeline_bounds.count(f.first)) continue;
        int num_stages = f.second.updates().size() + 1;
        for (int s = 0; s < num_stages; s++) {
            FStage stg(f.second, s);
            groups.insert(std::make_pair(stg, Group(stg, {stg})));
        }
    }

    // Build children map
    for (const auto &f : dep_analysis.env) {
        int num_stages = f.second.updates().size() + 1;
        for (int s = 0; s < num_stages; s++) {
            set<string> parents = get_parents(f.second, s);
            for (const string &c : parents) {
                auto iter = dep_analysis.env.find(c);
                if ((c != f.first) && (iter != dep_analysis.env.end())) {
                    const Function &prod_func = iter->second;
                    int final_stage = prod_func.updates().size();
                    FStage prod_stage(prod_func, final_stage);
                    FStage cons_stage(f.second, s);
                    children[prod_stage].insert(cons_stage);
                }
            }
            if (s > 0) {
                FStage prod_stage(f.second, s - 1);
                FStage cons_stage(f.second, s);
                children[prod_stage].insert(cons_stage);
            }
        }
    }
}

void Partitioner::initialize_groups() {
    for (pair<const FStage, Group> &g : groups) {
        auto best = find_best_tile_config(g.second);
        g.second.tile_sizes = best.first;
        group_costs.emplace(g.second.output, best.second);
    }
    grouping_cache.clear();
}

vector<map<string, Expr>> Partitioner::generate_tile_configs(const FStage &stg) {
    const vector<Dim> &dims = get_stage_dims(stg.func, stg.stage_num);
    vector<string> tile_vars;
    for (int d = 0; d < (int)dims.size() - 1; d++) {
        if (!dims[d].is_rvar()) tile_vars.push_back(dims[d].var);
    }

    if (tile_vars.empty()) return {};

    // GPU tile sizes: powers-of-2 up to 64 threads per dim
    vector<int> size_variants = {1, 4, 8, 16, 32, 64};

    // For GPU: identify 2 largest dims by pipeline bounds extent
    // to focus search (avoids combinatorial explosion for 4D+)
    vector<pair<int, string>> dim_extents;
    auto pb_it = pipeline_bounds.find(stg.func.name());
    if (pb_it != pipeline_bounds.end()) {
        const vector<string> &args = stg.func.args();
        for (const string &var : tile_vars) {
            for (int i = 0; i < (int)args.size(); i++) {
                if (args[i] == var) {
                    auto ext_int = as_const_int(simplify(substitute_var_estimates(
                        get_extent(pb_it->second[i]))));
                    if (ext_int && *ext_int > 0)
                        dim_extents.push_back({(int)*ext_int, var});
                    else
                        dim_extents.push_back({1, var});
                    break;
                }
            }
        }
    } else {
        for (const string &var : tile_vars)
            dim_extents.push_back({1, var});
    }

    // Sort by extent descending; pick top 2 for tiling
    std::sort(dim_extents.begin(), dim_extents.end(),
              [](const pair<int,string> &a, const pair<int,string> &b) {
                  return a.first > b.first;
              });

    // Primary tiling dimensions (at most 2)
    vector<string> primary;
    for (int i = 0; i < (int)dim_extents.size() && i < 2; i++) {
        if (dim_extents[i].first >= 4) primary.push_back(dim_extents[i].second);
    }
    if (primary.empty()) primary.push_back(tile_vars[0]);

    vector<map<string, Expr>> configs;
    auto is_dup = [&](const map<string, Expr> &t) {
        return std::find_if(configs.begin(), configs.end(),
                            [&t](const map<string, Expr> &m) {
                                return tile_maps_equal(t, m);
                            }) != configs.end();
    };

    if (primary.size() == 1) {
        for (int s0 : size_variants) {
            if (s0 < 32 || s0 > 1024) continue;
            map<string, Expr> tiling;
            for (const string &v : tile_vars)
                tiling[v] = (v == primary[0]) ? s0 : 1;
            if (!is_dup(tiling)) configs.push_back(tiling);
        }
    } else {
        // 2D search: require at least 128 threads/block (4 warps) for good occupancy.
        // Single-warp (32-thread) blocks have no latency hiding; a warp-group minimum
        // matches the practical lower bound on modern NVIDIA GPUs (Volta/Ampere).
        for (int s0 : size_variants) {
            for (int s1 : size_variants) {
                if (s0 * s1 < 128 || s0 * s1 > 1024) continue;
                map<string, Expr> tiling;
                for (const string &v : tile_vars) {
                    if (v == primary[0]) tiling[v] = s0;
                    else if (v == primary[1]) tiling[v] = s1;
                    else tiling[v] = 1;
                }
                if (!is_dup(tiling)) configs.push_back(tiling);
            }
        }
    }

    return configs;
}

Partitioner::GroupAnalysis
Partitioner::analyze_group(const Group &g, bool show_analysis) {
    set<string> group_inputs;
    set<string> group_members;

    for (const auto &stg : g.members) {
        group_members.insert(stg.func.name());
        set<string> parents = get_parents(stg.func, stg.stage_num);
        for (const auto &c : parents) {
            bool is_member = false;
            for (const auto &m : g.members) {
                if (m.func.name() == c) { is_member = true; break; }
            }
            if (!is_member) group_inputs.insert(c);
        }
    }

    // Count tiles and threads per block
    Expr estimate_tiles = make_one(Int(64));
    Expr threads_per_block = make_one(Int(64));

    if (!g.output.func.has_extern_definition()) {
        Definition def = get_stage_definition(g.output.func, g.output.stage_num);
        const vector<Dim> &dims = def.schedule().dims();
        DimBounds stg_bounds = get_bounds(g.output);

        for (int d = 0; d < (int)dims.size() - 1; d++) {
            const string &var = dims[d].var;
            const auto &it = g.tile_sizes.find(var);
            if (it != g.tile_sizes.end()) {
                const Expr &size = it->second;
                Expr extent = get_extent(get_element(stg_bounds, var));
                if (!extent.defined()) return GroupAnalysis();

                Expr dim_tiles = simplify((extent + size - 1) / size);
                estimate_tiles *= dim_tiles;

                // Track threads per block for pure dims (not rvars)
                if (!dims[d].is_rvar()) {
                    threads_per_block *= size;
                }
            }
        }
    }

    // GPU validation: reject invalid thread block sizes
    if (target.has_gpu_feature()) {
        Expr tpb_simplified = simplify(threads_per_block);
        auto tpb_int = as_const_int(tpb_simplified);
        if (tpb_int) {
            if (*tpb_int > 1024 || *tpb_int < 1) return GroupAnalysis();
        }
    }

    DimBounds tile_bounds = get_bounds_from_tile_sizes(g.output, g.tile_sizes);

    map<string, Box> alloc_regions = dep_analysis.regions_required(
        g.output.func, g.output.stage_num, tile_bounds, group_members,
        false, &costs.input_estimates);

    map<string, Box> compute_regions = dep_analysis.regions_required(
        g.output.func, g.output.stage_num, tile_bounds, group_members,
        true, &costs.input_estimates);

    map<string, Box> group_reg, input_reg;

    for (const auto &reg : compute_regions) {
        if ((group_members.find(reg.first) != group_members.end()) &&
            (reg.first != g.output.func.name())) {
            group_reg.emplace(reg.first, reg.second);
        } else if (group_inputs.find(reg.first) != group_inputs.end()) {
            if (dep_analysis.env.find(reg.first) == dep_analysis.env.end())
                input_reg.emplace(reg.first, reg.second);
        }
    }

    Cost tile_cost = costs.region_cost(group_reg, g.inlined);
    if (!tile_cost.defined()) return GroupAnalysis();

    Cost out_cost = costs.stage_region_cost(g.output.func.name(),
                                            g.output.stage_num,
                                            tile_bounds, g.inlined);
    if (!out_cost.defined()) return GroupAnalysis();

    for (const auto &reg : alloc_regions) {
        if (!box_size(reg.second).defined()) return GroupAnalysis();
    }

    // GPU shared memory feasibility check: for fused (non-output) group members,
    // the per-tile allocation lives in shared memory per block. Non-GPU-tiled dims
    // (tile_size <= 1) run as serial loops inside the block and their full pipeline
    // extent contributes to shared memory. Reject if any non-tiled dim exceeds the
    // shared memory budget. Only applies when tile_sizes is non-empty.
    if (target.has_gpu_feature() && g.members.size() > 1 && !g.tile_sizes.empty()) {
        // Estimate shmem bytes for each member: product of per-dim extents.
        // For GPU-tiled dims: use the tile_size (from alloc_regions, +stencil).
        // For non-GPU-tiled dims: use the full pipeline extent.
        // We approximate by multiplying tile_bound extents for tiled dims and
        // pipeline_bound extents for non-tiled dims, taking the max member.
        const auto pb_it = pipeline_bounds.find(g.output.func.name());
        if (pb_it != pipeline_bounds.end()) {
            const vector<string> &out_args = g.output.func.args();
            // Compute "effective block extent" for each output dim
            int64_t block_vol = 1;
            for (size_t ai = 0; ai < out_args.size() && ai < pb_it->second.size(); ai++) {
                const string &arg = out_args[ai];
                auto ts_it = g.tile_sizes.find(arg);
                bool gpu_tiled = false;
                if (ts_it != g.tile_sizes.end()) {
                    auto tsz = as_const_int(ts_it->second);
                    if (tsz && *tsz > 1) { block_vol *= (*tsz + 16); gpu_tiled = true; }
                }
                if (!gpu_tiled) {
                    Expr ext = simplify(substitute_var_estimates(
                        get_extent(pb_it->second[ai])));
                    auto ext_int = as_const_int(ext);
                    if (!ext_int || *ext_int <= 0) { block_vol = (int64_t)gpu_cache_size + 1; break; }
                    block_vol *= (*ext_int + 16);   // +16 conservative stencil
                }
            }
            // Each non-output member occupies ~block_vol floats in shared memory
            int64_t members_count = (int64_t)(g.members.size() - 1);
            if (members_count > 0 &&
                block_vol * members_count * 4 > static_cast<int64_t>(gpu_cache_size))
                return GroupAnalysis();
        }
    }

    Cost group_cost(simplify(tile_cost.arith + out_cost.arith),
                    simplify(tile_cost.memory + out_cost.memory));

    // Load cost model
    map<string, Expr> group_load_costs = costs.detailed_load_costs(group_reg, g.inlined);
    map<string, Expr> out_load_costs =
        costs.stage_detailed_load_costs(g.output.func.name(),
                                        g.output.stage_num, tile_bounds, g.inlined);
    combine_load_costs(group_load_costs, out_load_costs);

    Box out_tile_extent;
    if (g.output.stage_num == 0) {
        const vector<string> &args = g.output.func.args();
        for (const auto &arg : args) {
            auto it = tile_bounds.find(arg);
            out_tile_extent.push_back(it != tile_bounds.end() ? it->second : Interval());
        }
    }

    // GPU: use shared memory (48KB) as effective cache size
    float balance = gpu_balance;
    float cache_size_float = static_cast<float>(gpu_cache_size);
    float load_slope = balance / cache_size_float;

    Cost per_tile_cost(group_cost.arith, make_zero(Int(64)));

    for (const auto &f_load : group_load_costs) {
        internal_assert(g.inlined.find(f_load.first) == g.inlined.end());

        const auto &alloc_reg = get_element(alloc_regions, f_load.first);

        Expr footprint;
        bool is_group_member = (group_members.find(f_load.first) != group_members.end());
        bool is_output = (f_load.first == g.output.func.name());

        if (!is_output && is_group_member) {
            footprint = costs.region_size(f_load.first, alloc_reg);
        } else {
            Expr initial_footprint;
            const auto &f_load_pb = get_element(pipeline_bounds, f_load.first);
            bool is_function = (dep_analysis.env.find(f_load.first) != dep_analysis.env.end());

            if (!is_function) {
                initial_footprint = costs.input_region_size(f_load.first, f_load_pb);
                footprint = costs.input_region_size(f_load.first, alloc_reg);
            } else if (is_output) {
                initial_footprint = costs.region_size(f_load.first, f_load_pb);
                footprint = costs.region_size(f_load.first, out_tile_extent);
            } else {
                initial_footprint = costs.region_size(f_load.first, f_load_pb);
                footprint = costs.region_size(f_load.first, alloc_reg);
            }
            footprint = initial_footprint;
            if (!footprint.defined()) return GroupAnalysis();
        }

        Expr cost_factor =
            cast<int64_t>(min(1 + footprint * load_slope, (float)balance));
        per_tile_cost.memory += cost_factor * f_load.second;
    }

    // GPU parallelism = total threads (blocks × threads_per_block)
    Expr parallelism = simplify(estimate_tiles * threads_per_block);

    GroupAnalysis g_analysis(
        Cost(per_tile_cost.arith * estimate_tiles,
             per_tile_cost.memory * estimate_tiles),
        parallelism);
    g_analysis.simplify();

    return g_analysis;
}

pair<map<string, Expr>, Partitioner::GroupAnalysis>
Partitioner::find_best_tile_config(const Group &g) {
    map<string, Expr> no_tile_config;
    Group no_tile = g;
    no_tile.tile_sizes = no_tile_config;

    GroupAnalysis no_tile_analysis = analyze_group(no_tile, false);
    GroupAnalysis best_analysis = no_tile_analysis;
    map<string, Expr> best_config = no_tile_config;

    if (!best_analysis.cost.defined()) return {best_config, best_analysis};

    vector<map<string, Expr>> configs = generate_tile_configs(g.output);

    for (const auto &config : configs) {
        Group new_group = g;
        new_group.tile_sizes = config;
        GroupAnalysis new_analysis = analyze_group(new_group, false);

        Expr benefit = estimate_benefit(best_analysis, new_analysis, false, true);
        if (benefit.defined() && can_prove(benefit > 0)) {
            best_config  = config;
            best_analysis = new_analysis;
        }
    }

    return {best_config, best_analysis};
}

vector<pair<Partitioner::GroupingChoice, Partitioner::GroupConfig>>
Partitioner::choose_candidate_grouping(const vector<pair<string, string>> &cands,
                                        Level level) {
    vector<pair<GroupingChoice, GroupConfig>> best_grouping;
    Expr best_benefit = make_zero(Int(64));

    for (const auto &p : cands) {
        vector<pair<GroupingChoice, GroupConfig>> grouping;
        const Function &prod_f = get_element(dep_analysis.env, p.first);
        int final_stage = prod_f.updates().size();
        FStage prod(prod_f, final_stage);

        auto children_it = children.find(prod);
        if (children_it == children.end()) continue;

        for (const FStage &c : children_it->second) {
            GroupConfig best_config;
            GroupingChoice cand_choice(prod_f.name(), c);

            auto it = grouping_cache.find(cand_choice);
            if (it != grouping_cache.end()) {
                best_config = it->second;
            } else {
                best_config = evaluate_choice(cand_choice, level);
                grouping_cache.emplace(cand_choice, best_config);
            }
            grouping.emplace_back(cand_choice, best_config);
        }

        Expr overall_benefit = estimate_benefit(grouping, false, true);
        if (overall_benefit.defined() && can_prove(best_benefit < overall_benefit)) {
            best_grouping = grouping;
            best_benefit  = overall_benefit;
        }
    }

    return best_grouping;
}

void Partitioner::group(Level level) {
    bool fixpoint = false;
    while (!fixpoint) {
        fixpoint = true;
        vector<pair<string, string>> candidates;

        for (const pair<const FStage, Group> &g : groups) {
            bool is_output = false;
            for (const Function &f : outputs) {
                if (g.first.func.name() == f.name()) { is_output = true; break; }
            }

            const Function &prod_f = get_element(dep_analysis.env, g.first.func.name());
            bool is_final_stage = (g.first.stage_num == prod_f.updates().size());

            if (is_output || !is_final_stage) continue;

            auto it = children.find(g.first);
            if (it != children.end()) {
                set<string> child_groups;
                for (const FStage &s : it->second) child_groups.insert(s.func.name());
                int num_children = child_groups.size();

                if ((num_children == 1) && (level == Level::FastMem)) {
                    candidates.emplace_back(prod_f.name(), *child_groups.begin());
                } else if ((level == Level::Inline) && prod_f.is_pure()) {
                    candidates.emplace_back(prod_f.name(), "");
                }
            }
        }

        vector<pair<GroupingChoice, GroupConfig>> best =
            choose_candidate_grouping(candidates, level);
        if (best.empty()) continue;
        fixpoint = false;

        const string &prod = best[0].first.prod;
        const Function &prod_f = get_element(dep_analysis.env, prod);
        size_t num_stages = prod_f.updates().size() + 1;

        FStage final_stage(prod_f, num_stages - 1);
        set<FStage> prod_group_children = get_element(children, final_stage);

        // Invalidate cache
        set<GroupingChoice> invalid_keys;
        for (const auto &c : prod_group_children) {
            for (const auto &entry : grouping_cache) {
                if ((entry.first.prod == c.func.name()) || (entry.first.cons == c))
                    invalid_keys.insert(entry.first);
            }
        }
        for (const auto &key : invalid_keys) grouping_cache.erase(key);

        for (const auto &grp : best) {
            internal_assert(grp.first.prod == prod);
            merge_groups(grp.first, grp.second, level);
        }

        for (size_t s = 0; s < num_stages; s++) {
            FStage prod_group(prod_f, s);
            groups.erase(prod_group);
            group_costs.erase(prod_group);
            children.erase(prod_group);
            for (auto &f : children) {
                set<FStage> &cons = f.second;
                auto cit = cons.find(prod_group);
                if (cit != cons.end()) {
                    cons.erase(cit);
                    cons.insert(prod_group_children.begin(), prod_group_children.end());
                }
            }
        }
    }
}

DimBounds Partitioner::get_bounds(const FStage &s) {
    DimBounds bounds;
    const vector<string> &args = s.func.args();
    const auto &pb = get_element(pipeline_bounds, s.func.name());
    for (size_t d = 0; d < args.size(); d++) bounds[args[d]] = pb[d];
    return get_stage_bounds(s.func, s.stage_num, bounds);
}

DimBounds Partitioner::get_bounds_from_tile_sizes(const FStage &s,
                                                   const map<string, Expr> &tile_sizes) {
    map<string, Interval> bounds;
    const map<string, Interval> &def_bounds = get_bounds(s);
    const vector<Dim> &dims = get_stage_dims(s.func, s.stage_num);

    for (int d = 0; d < (int)dims.size() - 1; d++) {
        string var = dims[d].var;
        const Interval &bound = get_element(def_bounds, var);
        auto it = tile_sizes.find(var);
        if (it != tile_sizes.end()) {
            const Expr &size = it->second;
            Expr extent = get_extent(bound);
            internal_assert(extent.defined());
            if (can_prove(extent >= 2 * size))
                bounds[var] = Interval(0, simplify(size - 1));
            else
                bounds[var] = bound;
        } else {
            bounds[var] = bound;
        }
    }
    return bounds;
}

map<string, Expr> Partitioner::bounds_to_estimates(const DimBounds &bounds) {
    map<string, Expr> estimates;
    for (const auto &bound : bounds)
        estimates.emplace(bound.first, get_extent(bound.second));
    return estimates;
}

Cost Partitioner::get_pipeline_cost() {
    Cost total_cost(0, 0);
    for (const pair<const FStage, Group> &g : groups) {
        const GroupAnalysis &analysis = get_element(group_costs, g.first);
        if (!analysis.cost.defined()) return Cost();
        total_cost.arith  += analysis.cost.arith;
        total_cost.memory += analysis.cost.memory;
    }
    total_cost.simplify();
    return total_cost;
}

Partitioner::Group Partitioner::merge_groups(const Group &prod_group,
                                              const Group &cons_group) {
    vector<FStage> group_members;
    group_members.reserve(prod_group.members.size() + cons_group.members.size());
    for (const auto &s : prod_group.members) group_members.push_back(s);
    for (const auto &s : cons_group.members) group_members.push_back(s);

    Group merged(cons_group.output, group_members);
    for (const auto &f : prod_group.inlined) merged.inlined.insert(f);
    for (const auto &f : cons_group.inlined) merged.inlined.insert(f);
    return merged;
}

void Partitioner::merge_groups(const GroupingChoice &choice,
                                const GroupConfig &eval, Level level) {
    const Function &prod_f = get_element(dep_analysis.env, choice.prod);
    size_t num_stages = prod_f.updates().size() + 1;

    const FStage &child = choice.cons;
    Group &child_group  = get_element(groups, child);

    for (size_t s = 0; s < num_stages; s++) {
        FStage cand(prod_f, s);
        Group &cand_group = get_element(groups, cand);
        child_group.members.insert(child_group.members.end(),
                                   cand_group.members.begin(),
                                   cand_group.members.end());
        if (level == Level::Inline) {
            for (const auto &stg : cand_group.members)
                child_group.inlined.insert(stg.func.name());
        } else {
            for (const auto &in : cand_group.inlined)
                child_group.inlined.insert(in);
        }
    }
    child_group.tile_sizes = eval.tile_sizes;
    group_costs[child] = eval.analysis;
}

Partitioner::GroupConfig
Partitioner::evaluate_choice(const GroupingChoice &choice, Level level) {
    const Function &prod_f = get_element(dep_analysis.env, choice.prod);
    int num_prod_stages = prod_f.updates().size() + 1;
    vector<Group> prod_groups;
    for (int s = 0; s < num_prod_stages; s++) {
        FStage prod_s(prod_f, s);
        prod_groups.push_back(get_element(groups, prod_s));
    }

    Group cons = get_element(groups, choice.cons);
    Group merged = cons;
    for (const auto &pg : prod_groups) merged = merge_groups(pg, merged);

    if (level == Level::Inline) {
        map<string, Expr> tile_sizes;
        const Function &cons_f = cons.output.func;
        const vector<Dim> &dims = get_stage_dims(cons_f, cons.output.stage_num);
        for (int d = 0; d < (int)dims.size() - 1; d++)
            tile_sizes[dims[d].var] = 1;

        merged.tile_sizes = tile_sizes;
        for (const auto &pg : prod_groups)
            for (const FStage &s : pg.members)
                merged.inlined.insert(s.func.name());
        for (const string &f : cons.inlined) merged.inlined.insert(f);

        GroupAnalysis analysis = analyze_group(merged, false);
        return GroupConfig(tile_sizes, analysis);
    } else {
        auto config = find_best_tile_config(merged);
        return GroupConfig(config.first, config.second);
    }
}

Expr Partitioner::estimate_benefit(const GroupAnalysis &old_g, const GroupAnalysis &new_g,
                                    bool no_redundant_work, bool ensure_parallelism) const {
    if (ensure_parallelism &&
        (!new_g.parallelism.defined() ||
         !can_prove(new_g.parallelism >= min_parallelism))) {
        return Expr();
    }
    if (!old_g.cost.defined() || !new_g.cost.defined()) return Expr();

    Expr arith_benefit = old_g.cost.arith - new_g.cost.arith;
    if (no_redundant_work && !can_prove(arith_benefit >= 0)) return Expr();
    Expr mem_benefit = old_g.cost.memory - new_g.cost.memory;
    return simplify(mem_benefit + arith_benefit);
}

Expr Partitioner::estimate_benefit(
    const vector<pair<GroupingChoice, GroupConfig>> &new_grouping,
    bool no_redundant_work, bool ensure_parallelism) const {

    set<FStage> old_groups;
    GroupAnalysis new_group_analysis(Cost(0, 0), Int(64).max());

    for (const auto &g : new_grouping) {
        const Function &prod_f = get_element(dep_analysis.env, g.first.prod);
        int num_prod_stages = prod_f.updates().size() + 1;
        for (int s = 0; s < num_prod_stages; s++) old_groups.insert(FStage(prod_f, s));
        old_groups.insert(g.first.cons);

        GroupAnalysis analysisg = g.second.analysis;
        if (analysisg.defined()) {
            new_group_analysis.cost.arith  += analysisg.cost.arith;
            new_group_analysis.cost.memory += analysisg.cost.memory;
            new_group_analysis.parallelism =
                min(new_group_analysis.parallelism, analysisg.parallelism);
        } else {
            new_group_analysis.cost = Cost();
            new_group_analysis.parallelism = Expr();
            break;
        }
    }
    new_group_analysis.simplify();

    GroupAnalysis old_group_analysis(Cost(0, 0), Int(64).max());
    for (const auto &g : old_groups) {
        auto it = group_costs.find(g);
        internal_assert(it != group_costs.end());
        GroupAnalysis analysisg = it->second;
        if (analysisg.defined()) {
            old_group_analysis.cost.arith  += analysisg.cost.arith;
            old_group_analysis.cost.memory += analysisg.cost.memory;
            old_group_analysis.parallelism =
                min(old_group_analysis.parallelism, analysisg.parallelism);
        } else {
            old_group_analysis.cost = Cost();
            old_group_analysis.parallelism = Expr();
            break;
        }
    }
    old_group_analysis.simplify();

    return estimate_benefit(old_group_analysis, new_group_analysis,
                            no_redundant_work, ensure_parallelism);
}

map<FStage, map<string, Box>> Partitioner::group_storage_bounds() {
    map<FStage, map<string, Box>> result;
    for (const pair<const FStage, Group> &gpair : groups) {
        const Group &g = gpair.second;
        DimBounds bounds = get_bounds_from_tile_sizes(g.output, g.tile_sizes);

        set<string> prods;
        for (const FStage &s : g.members) prods.insert(s.func.name());

        map<string, Box> reg_alloc = dep_analysis.regions_required(
            g.output.func, g.output.stage_num, bounds, prods,
            false, &costs.input_estimates);

        map<string, Box> group_alloc;
        for (const FStage &s : g.members) {
            auto it = reg_alloc.find(s.func.name());
            if ((it != reg_alloc.end()) && (s.func.name() != g.output.func.name()))
                group_alloc[s.func.name()] = it->second;
        }
        result[gpair.first] = group_alloc;
    }
    return result;
}

map<FStage, map<FStage, DimBounds>> Partitioner::group_loop_bounds() {
    map<FStage, map<FStage, DimBounds>> result;
    for (const pair<const FStage, Group> &gpair : groups) {
        const Group &g = gpair.second;
        DimBounds bounds = get_bounds_from_tile_sizes(g.output, g.tile_sizes);

        set<string> prods;
        for (const FStage &s : g.members) prods.insert(s.func.name());

        map<string, Box> reg_computed = dep_analysis.regions_required(
            g.output.func, g.output.stage_num, bounds, prods,
            true, &costs.input_estimates);

        map<FStage, DimBounds> mem_bounds;
        for (const FStage &s : g.members) {
            auto it = reg_computed.find(s.func.name());
            if (it != reg_computed.end()) {
                map<string, Expr> tile_sizes;
                const vector<string> &args = s.func.args();
                for (size_t arg = 0; arg < args.size(); arg++)
                    tile_sizes[args[arg]] = get_extent(it->second[arg]);
                mem_bounds[s] = get_bounds_from_tile_sizes(s, tile_sizes);
            }
        }
        result[gpair.first] = mem_bounds;
    }
    return result;
}

// ============================================================================
// GPU schedule generation from Partitioner groups
// ============================================================================

/** Apply a GPU schedule to a single Partitioner group.
 *
 * Group output: split tiled dims → gpu_threads (inner) + gpu_blocks (outer) + compute_root.
 * Non-output members: compute_at(output, tile_inner_var).
 *
 * When tile_sizes is empty (or all 1), falls back to gpu_single_thread().
 */
static void apply_group_gpu_schedule(
    const Partitioner::Group &g,
    const Sioutas2020Params &params,
    const map<string, Box> &func_bounds,
    ostringstream &src)
{
    Function g_out_func = g.output.func;
    const string &out_name = g_out_func.name();
    Func f_out(g_out_func);
    const int ndims = f_out.dimensions();

    // Collect tiled pure dims from g.tile_sizes
    vector<pair<string, int>> tiled;   // (varname, tile_size)

    if (ndims > 0) {
        const vector<Dim> &dims = get_stage_dims(g_out_func, g.output.stage_num);
        for (int d = 0; d < (int)dims.size() - 1; d++) {
            if (dims[d].is_rvar()) continue;
            const string &vname = dims[d].var;
            auto it = g.tile_sizes.find(vname);
            if (it == g.tile_sizes.end()) continue;
            auto ti = as_const_int(it->second);
            if (!ti || *ti <= 1) continue;
            tiled.push_back({vname, (int)*ti});
        }
    }

    // Variable names for inner/outer dims
    vector<Var> inner_vars, outer_vars;
    string tile_inner_var_name;

    if (tiled.empty() || ndims == 0) {
        // No tiling: single thread or compute_root
        if (ndims == 0) {
            f_out.compute_root();
            src << out_name << ".compute_root();\n";
        } else {
            // Fall back to SimpleAutoSchedule with params defaults
            vector<int> int_bounds;
            auto bnd_it = func_bounds.find(out_name);
            if (bnd_it != func_bounds.end())
                int_bounds = get_int_bounds(bnd_it->second);
            schedule_func_gpu(f_out, g_out_func, int_bounds, params, src);
        }
    } else {
        // Apply splits
        for (auto &[vname, tsz] : tiled) {
            Var v(vname);
            Var vi(vname + "_i"), vo(vname + "_o");
            f_out.split(v, vo, vi, tsz, TailStrategy::GuardWithIf);
            inner_vars.push_back(vi);
            outer_vars.push_back(vo);
        }

        // Reorder: inner first, then outer (inner = gpu_threads, outer = gpu_blocks)
        vector<VarOrRVar> ordering;
        for (auto &v : inner_vars) ordering.emplace_back(v);
        for (auto &v : outer_vars) ordering.emplace_back(v);
        f_out.reorder(ordering);

        // Apply gpu() for each tiled dimension (maps outer→block, inner→thread)
        for (int i = 0; i < (int)tiled.size() && i < 3; i++) {
            f_out.gpu(outer_vars[i], inner_vars[i]);
        }

        f_out.compute_root();

        // Build source string
        src << out_name;
        for (auto &[vname, tsz] : tiled) {
            src << ".split(" << vname << ", " << vname << "_o, " << vname << "_i, "
                << tsz << ", TailStrategy::GuardWithIf)";
        }
        {
            src << ".reorder(";
            bool first = true;
            for (auto &v : inner_vars) {
                if (!first) src << ", "; first = false;
                src << v.name();
            }
            for (auto &v : outer_vars) {
                src << ", " << v.name();
            }
            src << ")";
        }
        for (int i = 0; i < (int)tiled.size() && i < 3; i++) {
            src << ".gpu(" << outer_vars[i].name() << ", " << inner_vars[i].name() << ")";
        }
        src << ".compute_root();\n";

        // tile_inner_var = the innermost of the outer dims
        // After reorder([inner...][outer...]), the first outer var
        // (outer_vars[0]) is the innermost outer dim.
        tile_inner_var_name = outer_vars[0].name();

        // Schedule update stages for output
        int num_updates = static_cast<int>(g_out_func.updates().size());
        for (int i = 0; i < num_updates; i++) {
            Stage upd = f_out.update(i);
            if (update_lhs_is_pure_identity(g_out_func, i)) {
                // Apply same tiling structure to update stage
                for (auto &[vname, tsz] : tiled) {
                    Var v(vname), vi(vname + "_i"), vo(vname + "_o");
                    upd.split(v, vo, vi, tsz, TailStrategy::GuardWithIf);
                }
                // Reorder + gpu() for update
                vector<VarOrRVar> upd_ordering;
                for (auto &v : inner_vars) upd_ordering.emplace_back(v);
                for (auto &v : outer_vars) upd_ordering.emplace_back(v);
                upd.reorder(upd_ordering);
                for (int j = 0; j < (int)tiled.size() && j < 3; j++) {
                    upd.gpu(outer_vars[j], inner_vars[j]);
                }
                unroll_small_rvars(f_out, i, params, src);
            } else {
                upd.gpu_single_thread();
                src << out_name << ".update(" << i << ").gpu_single_thread();\n";
            }
        }
    }

    // Schedule non-output, non-inlined members with compute_at
    if (tile_inner_var_name.empty()) return;

    Var tile_inner_var(tile_inner_var_name);

    for (const FStage &mem : g.members) {
        if (g.inlined.count(mem.func.name())) continue;
        if (mem.func.name() == out_name) continue;
        if (mem.stage_num > 0) continue;  // only init stages; updates follow their func

        Func f_mem(mem.func);
        f_mem.compute_at(f_out, tile_inner_var);
        src << mem.func.name()
            << ".compute_at(" << out_name << ", " << tile_inner_var_name << ");\n";

        // Schedule update stages of the member at the same level
        int mem_updates = static_cast<int>(mem.func.updates().size());
        for (int i = 0; i < mem_updates; i++) {
            Stage upd = f_mem.update(i);
            if (update_lhs_is_pure_identity(mem.func, i)) {
                unroll_small_rvars(f_mem, i, params, src);
            } else {
                upd.gpu_single_thread();
                src << mem.func.name() << ".update(" << i << ").gpu_single_thread();\n";
            }
        }
    }
}

// ============================================================================
// Full-scheduler path: generate schedule via Partitioner
// ============================================================================

static string generate_schedule_from_partitioner(
    const vector<Function> &outputs,
    const Target &target,
    const Sioutas2020Params &params,
    const map<string, Function> &env,
    const vector<string> &order,
    const map<string, Box> &func_bounds)
{
    ostringstream src;
    src << "// Sioutas2020 full cost-model schedule -- target: "
        << target.to_string() << "\n\n";

    // Build cost analysis infrastructure
    FuncValueBounds func_val_bounds = compute_function_value_bounds(order, env);
    DependenceAnalysis dep_analysis(env, order, func_val_bounds);
    RegionCosts costs(env, order);

    // Get pipeline bounds (from output estimates)
    map<string, Box> pipeline_bounds =
        get_pipeline_bounds(dep_analysis, outputs, &costs.input_estimates);

    if (pipeline_bounds.empty()) {
        src << "// WARNING: pipeline bounds unavailable; falling back to "
               "SimpleAutoSchedule\n";
        return "";
    }

    // Build and run Partitioner
    Partitioner partitioner(pipeline_bounds, target, dep_analysis, costs, outputs);

    // Phase 1: Inline pure functions where beneficial
    partitioner.initialize_groups();
    partitioner.group(Partitioner::Level::Inline);

    // Phase 2: FastMem grouping (compute_at)
    partitioner.initialize_groups();  // re-initialize after inline
    partitioner.group(Partitioner::Level::FastMem);

    debug(2) << "[Sioutas2020] Groups after partitioning:\n";
    for (const auto &g : partitioner.groups) {
        debug(2) << "  Group output: " << g.first << "\n";
        debug(2) << "  Members:";
        for (const auto &m : g.second.members) debug(2) << " " << m;
        debug(2) << "\n";
        debug(2) << "  Inlined:";
        for (const auto &i : g.second.inlined) debug(2) << " " << i;
        debug(2) << "\n";
    }

    // Generate schedules: iterate in realization order (earlier functions first)
    // Track which functions have been scheduled to avoid double-scheduling
    set<string> scheduled;

    // Iterate order in reverse (innermost consumers last)
    // We need to schedule group outputs, and members are captured by compute_at
    for (int i = order.size() - 1; i >= 0; i--) {
        const string &fname = order[i];

        // Find the group that has this function as its output
        auto it = std::find_if(
            partitioner.groups.begin(), partitioner.groups.end(),
            [&fname](const pair<const FStage, Partitioner::Group> &kv) {
                return kv.second.output.func.name() == fname && kv.second.output.stage_num == 0;
            });

        if (it == partitioner.groups.end()) continue;

        // Skip if already scheduled as a member of another group
        if (scheduled.count(fname)) continue;

        const Partitioner::Group &g = it->second;

        // Mark all members as scheduled
        for (const FStage &mem : g.members) {
            scheduled.insert(mem.func.name());
        }
        for (const string &inl : g.inlined) {
            scheduled.insert(inl);
        }

        auto fn_it = env.find(fname);
        if (fn_it == env.end()) continue;
        if (fn_it->second.has_extern_definition()) {
            Func(fn_it->second).compute_root();
            src << fname << ".compute_root(); // extern\n";
            continue;
        }

        apply_group_gpu_schedule(g, params, func_bounds, src);
    }

    return src.str();
}

// ============================================================================
// Top-level schedule generation
// ============================================================================

static string generate_schedule(const vector<Function>  &outputs,
                                 const Target            &target,
                                 const Sioutas2020Params &params) {
    ostringstream src;
    src << "// Sioutas2020 autoscheduler -- target: "
        << target.to_string() << "\n\n";

    // 1. Transitive closure
    map<string, Function> env = build_environment(outputs);

    // 2. Lock loop-level references
    for (auto &kv : env) kv.second.lock_loop_levels();

    // 3. Topological order
    vector<string> top_order = topological_order(outputs, env);

    // 4. Inline trivial functions
    if (inline_all_trivial_functions(outputs, top_order, env)) {
        env = build_environment(outputs);
    }

    // 5. Realization order for element-wise inlining
    vector<string> order = realization_order(outputs, env).first;

    // 6. Inline single-consumer element-wise functions to fixpoint
    while (inline_all_element_wise_functions(outputs, order, env)) {
        env = build_environment(outputs);
        order = realization_order(outputs, env).first;
    }

    // 7. Bounds inference
    map<string, Box> func_bounds = infer_bounds(outputs);

    const bool is_gpu = target.has_gpu_feature();

    // 8a. Full cost-model path (GPU + fusion enabled)
    if (is_gpu && params.enable_fusion && params.enable_cost_model) {
        string full_src = generate_schedule_from_partitioner(
            outputs, target, params, env, order, func_bounds);

        if (!full_src.empty()) {
            return src.str() + full_src;
        }
        // Fall through if Partitioner fails (empty pipeline bounds)
        src << "// Partitioner failed -- using SimpleAutoSchedule fallback\n\n";
    }

    // 8b. SimpleAutoSchedule path
    src << "// " << order.size() << " function(s) to schedule:\n";

    for (const string &name : order) {
        auto it = env.find(name);
        if (it == env.end()) continue;
        const Function &func = it->second;

        if (func.has_extern_definition()) {
            src << "// " << name << ": extern -- not scheduled\n";
            continue;
        }

        Func f(func);
        vector<int> int_bounds;
        auto bnd_it = func_bounds.find(name);
        if (bnd_it != func_bounds.end())
            int_bounds = get_int_bounds(bnd_it->second);

        if (is_gpu) {
            schedule_func_gpu(f, func, int_bounds, params, src);
        } else {
            schedule_for_cpu(f, int_bounds, params, src);
        }
    }

    return src.str();
}

// ============================================================================
// Plugin entry point
// ============================================================================

struct Sioutas2020 {
    void operator()(const Pipeline            &p,
                    const Target              &target,
                    const AutoschedulerParams &params_in,
                    AutoSchedulerResults      *results) {
        internal_assert(params_in.name == "Sioutas2020");

        Sioutas2020Params params;
        {
            ParamParser parser(params_in.extra);
            parser.parse("gpu_tile_x",          &params.gpu_tile_x);
            parser.parse("gpu_tile_y",          &params.gpu_tile_y);
            parser.parse("gpu_tile_channel",    &params.gpu_tile_channel);
            parser.parse("cpu_tile_x",          &params.cpu_tile_x);
            parser.parse("cpu_tile_y",          &params.cpu_tile_y);
            parser.parse("parallelism",         &params.parallelism);
            parser.parse("unroll_rvar_size",    &params.unroll_rvar_size);
            parser.parse("enable_fusion",       &params.enable_fusion);
            parser.parse("enable_cost_model",   &params.enable_cost_model);
            parser.finish();
        }

        vector<Function> pipeline_outputs;
        pipeline_outputs.reserve(p.outputs().size());
        for (const Func &f : p.outputs())
            pipeline_outputs.push_back(f.function());

        results->target               = target;
        results->autoscheduler_params = params_in;
        results->schedule_source      =
            generate_schedule(pipeline_outputs, target, params);
    }
};

REGISTER_AUTOSCHEDULER(Sioutas2020)

}  // namespace (anonymous)
}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
