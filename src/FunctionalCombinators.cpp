#include "FunctionalCombinators.h"

#include "Func.h"
#include "Util.h"
#include "Var.h"

namespace Halide {

ParallelScanResult parallel_scan(
    std::function<Expr(Expr, Expr)> f,
    const Func &input,
    int scan_dim,
    int log2_n) {

    user_assert(input.defined())
        << "Func passed to parallel_scan must have a definition.\n";
    const int ndims = input.dimensions();
    user_assert(ndims >= 1)
        << "Func passed to parallel_scan must have at least one dimension.\n";
    user_assert(scan_dim >= 0 && scan_dim < ndims)
        << "scan_dim " << scan_dim << " is out of range for a "
        << ndims << "-dimensional Func.\n";
    user_assert(log2_n >= 1)
        << "parallel_scan requires log2_n >= 1.\n";

    // One named Var per dimension of the input.
    std::vector<Var> vars(ndims);
    for (int i = 0; i < ndims; i++) {
        vars[i] = Var("d" + std::to_string(i));
    }

    // Expr view of the vars for calling Funcs (Var is implicitly Expr).
    std::vector<Expr> args(vars.begin(), vars.end());

    std::vector<Func> stages(log2_n + 1);

    // Stage 0: copy of the input so later stages don't embed input
    // as an unnamed dependency that the caller cannot schedule.
    stages[0] = Func("parallel_scan" + Internal::unique_name('_'));
    stages[0](vars) = input(args);
    stages[0].compute_root();

    for (int d = 0; d < log2_n; d++) {
        stages[d + 1] = Func("parallel_scan" + Internal::unique_name('_'));

        const Expr stride = 1 << d;

        // Args with the scan dimension shifted back by stride.
        std::vector<Expr> shifted = args;
        shifted[scan_dim] = vars[scan_dim] - stride;

        // Hillis-Steele step: if we are far enough from the left boundary,
        // fold in the value stride positions back; otherwise pass through.
        stages[d + 1](vars) = select(vars[scan_dim] >= stride,
                                     f(stages[d](args), stages[d](shifted)),
                                     stages[d](args));
        stages[d + 1].compute_root();
    }

    Func result = stages.back();
    return {result, std::move(stages)};
}

}  // namespace Halide
