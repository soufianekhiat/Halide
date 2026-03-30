#ifndef HALIDE_FUNCTIONAL_COMBINATORS_H
#define HALIDE_FUNCTIONAL_COMBINATORS_H

#include "Func.h"

#include <functional>
#include <vector>

/** \file
 * Defines a multi-dimensional parallel prefix scan combinator using the
 * Hillis-Steele algorithm.
 *
 * See test/correctness/functional_combinators.cpp for example usage. */

namespace Halide {

/** Result of parallel_scan: the final output Func plus all intermediate
 *  stage Funcs, exposed for custom scheduling. */
struct ParallelScanResult {
    /** The final scan output. Equal to stages.back(). */
    Func result;

    /** All intermediate stage Funcs. stages[0] is a copy of the input;
     *  stages[d] has absorbed strides up to 2^(d-1); stages.back() == result.
     *
     *  By default every stage is scheduled compute_root(). Override the
     *  schedule here to, for example, fuse the last K stages inside the
     *  consumer's tile for execution in cache or GPU shared memory (LDS):
     *  \code
     *    for (int d = log2_n - K + 1; d <= log2_n; d++)
     *        ps.stages[d].compute_at(ps.result, tile_var)
     *                    .store_in(MemoryType::GPUShared);
     *  \endcode
     */
    std::vector<Func> stages;
};

/** Hillis-Steele parallel inclusive scan along one dimension of a Func.
 *
 * Generates \p log2_n + 1 pure Funcs in a chain:
 * \code
 *   stages[0](vars)   = input(vars)
 *   stages[d+1](vars) = select(vars[scan_dim] >= 2^d,
 *                               f(stages[d](vars), stages[d](vars with scan_dim -= 2^d)),
 *                               stages[d](vars))
 * \endcode
 *
 * The result is an inclusive prefix scan of \p input along dimension
 * \p scan_dim using binary operator \p f.  All other dimensions pass
 * through unchanged and remain fully parallelizable.
 *
 * \p log2_n must be a compile-time constant int.  For an array of n
 * elements along the scan dimension use log2_n = ceil(log2(n)).
 *
 * Example — row-wise prefix sum of a 2-D HDRI for marginal CDF:
 * \code
 *   Func image = ...;  // image(x, y)
 *   auto ps = parallel_scan(
 *       [](Expr a, Expr b) { return a + b; },
 *       image, 0, 10);        // scan along x, supports up to 2^10 = 1024 cols
 *   ps.result.compute_root().parallel(Var("d1"));  // parallel over rows
 * \endcode
 *
 * Each generated stage is a pure function — no RDom, no update
 * definitions — so Halide's scheduler can parallelize, vectorize, and
 * tile each stage independently.  The non-scan dimensions are always
 * parallel; the scan dimension within each stage is also parallel
 * (each output element depends only on the *previous stage*, not on
 * neighbours within the same stage).
 */
ParallelScanResult parallel_scan(
    std::function<Expr(Expr, Expr)> f,
    const Func &input,
    int scan_dim,
    int log2_n);

}  // namespace Halide

#endif  // HALIDE_FUNCTIONAL_COMBINATORS_H
