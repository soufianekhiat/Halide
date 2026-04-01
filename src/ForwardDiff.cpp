/** \file
 *  Forward-mode automatic differentiation (JVP) for Halide.
 *
 *  Implements propagate_tangents(), which computes d(output)/d(param) as a Func
 *  with the same dimensionality and Tuple width as the output. This is efficient
 *  for differential/inverse rendering where the output is a high-dimensional
 *  image and the parameters are low-dimensional scene parameters.
 *
 *  ## Forward-mode vs reverse-mode
 *
 *  Reverse-mode AD (propagate_adjoints) is efficient when there is one scalar
 *  output and many parameters: one backward pass yields all gradients.
 *  Forward-mode AD is efficient when there is one scalar parameter and a
 *  high-dimensional output (e.g., a full image): one forward pass yields
 *  d(output)/d(param) for every pixel simultaneously.
 *
 *  For differential rendering with N scene parameters and W*H pixel output:
 *  - Forward-mode: N passes, each O(W*H)  → best when N << W*H
 *  - Reverse-mode: 1 pass, each O(N*W*H)  → best when scalar loss is available
 *
 *  ## Memory model
 *
 *  Tangent Funcs are ordinary Halide Funcs with no special scheduling.
 *  The default compute_inline scheduling fuses tangent expressions into the
 *  consumer's computation loop -- no extra buffers are allocated, analogous
 *  to Dr.JIT's JIT kernel fusion.  Users may compute_root individual tangent
 *  Funcs to checkpoint them (trading recomputation for memory, analogous to
 *  Dr.JIT's dr.eval()).
 *
 *  ## Algorithm overview
 *
 *  1. Topological sort: get all Funcs in the pipeline in producer-first order.
 *  2. For each Func f in order:
 *     a. Create tangent_f with the same args and Tuple width.
 *     b. Pure definition: tangent_f(args) = diff_expr(f.values(), ...).
 *     c. Register tangent_f in the context *before* processing update
 *        definitions, so that self-references in update RHS resolve correctly.
 *     d. For each update definition: tangent_f(update_args) =
 *        diff_expr(update_values, ...).
 *  3. Return the tangent Func for the requested output.
 */

#include "Derivative.h"
#include "DerivativeUtils.h"
#include "Error.h"
#include "ExprUsesVar.h"
#include "FindCalls.h"
#include "IROperator.h"
#include "RealizationOrder.h"
#include "Simplify.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace Halide {

using namespace Internal;
using std::map;
using std::set;
using std::string;
using std::vector;

namespace {

// Halide lowers floating-point math externals to type-suffixed names at
// code-generation time: sin(x) becomes sin_f32(x), sin_f64(x), etc.
// By the time we see expressions in Func.values(), the names have already
// been mangled, so we must match all three float widths.
bool is_float_extern_fwd(const string &op_name, const string &func_name) {
    return op_name == (func_name + "_f16") ||
           op_name == (func_name + "_f32") ||
           op_name == (func_name + "_f64");
}

// Context threaded through the entire differentiation pass.
//
// Two differentiation "modes" share this struct:
//
//   (A) Parameter tangent mode (normal use): diff_expr computes d(expr)/d(p)
//       where p is either a named scalar Param (target_scalar_param) or a
//       Buffer/ImageParam with an associated tangent direction (buffer_tangents).
//
//   (B) Spatial partial derivative mode (chain rule): diff_expr computes
//       d(g_body)/d(z_i) where z_i is a positional Var of g.  This sub-mode
//       is activated by setting partial_var = z_i.name() and leaving
//       target_scalar_param/buffer_tangents empty.  It is used only inside
//       the chain-rule computation for param-dependent call arguments.
struct ForwardDiffContext {
    // Name of the scalar Param we are differentiating w.r.t.
    // Empty when using the map-based API (buffer_tangents handles everything).
    string target_scalar_param;

    // Maps an input name (Buffer or Param) to its tangent Func.
    //   - N-D Func: tangent direction for a Buffer/ImageParam of the same shape.
    //   - 0-D Func: tangent scalar for a named Param (value is usually 1.0).
    // In the map-based overload, scalar-param tangents are stored here as 0-D
    // Funcs rather than in target_scalar_param, enabling simultaneous
    // differentiation w.r.t. multiple inputs.
    map<string, Func> buffer_tangents;

    // Tangent Func for every Func processed so far in the forward pass.
    // Keyed by Func name (same key as find_transitive_calls uses).
    // A Func is inserted here immediately after its pure definition tangent
    // is computed and before its update definitions are processed (see the
    // comment on the registration step in propagate_tangents_impl).
    map<string, Func> tangent_funcs;

    // Maps a Let-bound variable name to its tangent expression.
    // Updated on entry/exit from each Let node to implement correct scoping
    // for nested Let expressions (the save/restore pattern in diff_expr).
    map<string, Expr> let_tangent;

    // Non-empty only in spatial partial derivative mode (mode B above).
    // Holds the name of the Var being treated as the differentiation target.
    // When diff_expr encounters a Variable whose name matches this string,
    // it returns make_one(type) instead of make_zero(type).
    string partial_var;
};

// Forward declaration so diff_call can call diff_expr for sub-expressions.
Expr diff_expr(const Expr &e, ForwardDiffContext &ctx,
               map<string, map<int, Func>> &partial_cache);

// Compute the tangent of a Call node (extern math, intrinsic, Halide Func, or
// Image/ImageParam load).  Each case applies the chain rule to the call.
Expr diff_call(const Call *op, ForwardDiffContext &ctx,
               map<string, map<int, Func>> &partial_cache) {

    // ── Math externals (sin, exp, sqrt, ...) ─────────────────────────────────
    //
    // These are Call::PureExtern nodes.  Halide names them with a type suffix
    // (_f16/_f32/_f64), so we use is_float_extern_fwd() for matching.
    // All derivatives follow the standard chain rule: d/dp f(a) = f'(a) * da/dp.
    if (op->is_extern()) {
        if (!op->type.is_float()) {
            // Integer or boolean extern: no meaningful derivative.
            return make_zero(op->type);
        }
        const Expr &a = op->args[0];
        // Use a lambda so the derivative of the first arg is only computed
        // for functions that actually need it (avoids redundant work when
        // the compiler doesn't inline across branches).
        auto da = [&]() { return diff_expr(a, ctx, partial_cache); };

        if (is_float_extern_fwd(op->name, "exp")) {
            return exp(a) * da();
        } else if (is_float_extern_fwd(op->name, "log")) {
            return da() / a;
        } else if (is_float_extern_fwd(op->name, "sin")) {
            return cos(a) * da();
        } else if (is_float_extern_fwd(op->name, "asin")) {
            return da() / sqrt(make_one(op->type) - a * a);
        } else if (is_float_extern_fwd(op->name, "cos")) {
            return -sin(a) * da();
        } else if (is_float_extern_fwd(op->name, "acos")) {
            return -da() / sqrt(make_one(op->type) - a * a);
        } else if (is_float_extern_fwd(op->name, "tan")) {
            // d/dp tan(a) = da / cos(a)^2.  Store cos(a) to avoid computing it twice.
            Expr c = cos(a);
            return da() / (c * c);
        } else if (is_float_extern_fwd(op->name, "atan")) {
            return da() / (make_one(op->type) + a * a);
        } else if (is_float_extern_fwd(op->name, "atan2")) {
            // atan2(y, x): args[0]=y, args[1]=x (Halide convention matches libm).
            // d/dp atan2(y,x) = (x*dy - y*dx) / (x^2 + y^2)
            const Expr &y = op->args[0];
            const Expr &x = op->args[1];
            Expr dy = diff_expr(y, ctx, partial_cache);
            Expr dx = diff_expr(x, ctx, partial_cache);
            Expr x2y2 = x * x + y * y;
            return (x * dy - y * dx) / x2y2;
        } else if (is_float_extern_fwd(op->name, "sinh")) {
            return cosh(a) * da();
        } else if (is_float_extern_fwd(op->name, "asinh")) {
            return da() / sqrt(make_one(op->type) + a * a);
        } else if (is_float_extern_fwd(op->name, "cosh")) {
            return sinh(a) * da();
        } else if (is_float_extern_fwd(op->name, "acosh")) {
            // d/dp acosh(a) = da / sqrt(a^2 - 1).
            // Written as sqrt(a-1)*sqrt(a+1) rather than sqrt(a^2-1) to avoid
            // potential catastrophic cancellation when a is very close to 1,
            // where a^2 - 1 can round to a negative value in float.
            return da() / (sqrt(a - make_one(op->type)) * sqrt(a + make_one(op->type)));
        } else if (is_float_extern_fwd(op->name, "tanh")) {
            // d/dp tanh(a) = da / cosh(a)^2.
            Expr c = cosh(a);
            return da() / (c * c);
        } else if (is_float_extern_fwd(op->name, "atanh")) {
            return da() / (make_one(op->type) - a * a);
        } else if (is_float_extern_fwd(op->name, "ceil") ||
                   is_float_extern_fwd(op->name, "floor") ||
                   is_float_extern_fwd(op->name, "round") ||
                   is_float_extern_fwd(op->name, "trunc")) {
            // Rounding functions are piecewise constant: derivative is zero
            // almost everywhere (undefined at integer boundaries, but those
            // are a measure-zero set).
            return make_zero(op->type);
        } else if (is_float_extern_fwd(op->name, "sqrt")) {
            return make_const(op->type, 0.5) * da() / sqrt(a);
        } else if (is_float_extern_fwd(op->name, "pow")) {
            // General power rule: d/dp pow(a,b) = pow(a,b)*(b*da/a + db*log(a))
            // This handles both base-dependent and exponent-dependent cases.
            const Expr &b = op->args[1];
            Expr da_ = diff_expr(a, ctx, partial_cache);
            Expr db_ = diff_expr(b, ctx, partial_cache);
            return pow(a, b) * (b * da_ / a + db_ * log(a));
        } else if (is_float_extern_fwd(op->name, "fast_inverse")) {
            // fast_inverse(a) ≈ 1/a (hardware reciprocal).
            // d/dp (1/a) = -da/a^2 = -da * (1/a)^2.
            // Reuse fast_inverse(a) so the tangent uses the same approximation
            // as the primal (consistent with treating fast_inverse as the exact
            // function for differentiation purposes).
            Expr inv = fast_inverse(a);
            return -da() * inv * inv;
        } else if (is_float_extern_fwd(op->name, "fast_inverse_sqrt")) {
            // fast_inverse_sqrt(a) ≈ 1/sqrt(a).
            // d/dp (a^{-1/2}) = -1/2 * da * a^{-3/2} = -1/2 * da * (1/sqrt(a))^3.
            Expr inv_sqrt = fast_inverse_sqrt(a);
            return make_const(op->type, -0.5) * da() * inv_sqrt * inv_sqrt * inv_sqrt;
        } else if (op->name == "halide_print") {
            // halide_print is a side-effecting debug call; its "return value"
            // carries no gradient.
            return make_zero(op->type);
        } else {
            user_error << "propagate_tangents: derivative of extern function '"
                      << op->name << "' is not implemented.\n";
            return make_zero(op->type);
        }
    }

    // ── Intrinsics ───────────────────────────────────────────────────────────
    //
    // Intrinsics are Call::Intrinsic nodes that the Halide compiler inserts.
    // Most have straightforward chain-rule derivatives; a few (bitwise, undef)
    // carry no gradient information.
    if (op->is_intrinsic()) {
        if (op->is_intrinsic(Call::abs)) {
            // |a|: d/dp = sign(a) * da.  At a=0 the subgradient is zero (we
            // pick the right limit sign(a)>0 path via the select).
            Expr a = op->args[0];
            Expr da = diff_expr(a, ctx, partial_cache);
            return select(a > 0, da, -da);
        } else if (op->is_intrinsic(Call::lerp)) {
            // lerp(a, b, w) = a*(1-w) + b*w.
            // Full product rule over all three potentially-param-dependent args.
            Expr a = op->args[0], b = op->args[1], w = op->args[2];
            Expr da = diff_expr(a, ctx, partial_cache);
            Expr db = diff_expr(b, ctx, partial_cache);
            Expr dw = diff_expr(w, ctx, partial_cache);
            return da * (make_one(op->type) - w) + db * w + (b - a) * dw;
        } else if (Call::as_tag(op) != nullptr) {
            // likely() / likely_if_innermost() are purely hint annotations;
            // they pass their single argument through unchanged.
            return diff_expr(op->args[0], ctx, partial_cache);
        } else if (op->is_intrinsic(Call::return_second)) {
            // return_second(a, b) evaluates both args but returns b.
            // The tangent is the tangent of b (the returned value).
            return diff_expr(op->args[1], ctx, partial_cache);
        } else if (op->is_intrinsic(Call::undef)) {
            // undef() has no defined value, so no defined derivative.
            return make_zero(op->type);
        } else if (op->is_intrinsic(Call::round)) {
            // round() is implemented as a PureIntrinsic (unlike floor/ceil/trunc
            // which are PureExtern).  It is piecewise constant: zero tangent.
            return make_zero(op->type);
        } else if (op->is_intrinsic(Call::bitwise_and) ||
                   op->is_intrinsic(Call::bitwise_not) ||
                   op->is_intrinsic(Call::bitwise_or) ||
                   op->is_intrinsic(Call::bitwise_xor) ||
                   op->is_intrinsic(Call::shift_right) ||
                   op->is_intrinsic(Call::shift_left)) {
            // Bitwise operations are piecewise constant on the integers;
            // they carry no gradient.
            return make_zero(op->type);
        } else {
            user_warning << "propagate_tangents: dropping derivative at intrinsic '"
                        << op->name << "'\n";
            return make_zero(op->type);
        }
    }

    // ── Call to another Halide Func ──────────────────────────────────────────
    //
    // For a call g(e1, e2) the chain rule gives:
    //
    //   d/dp g(e1, e2) = tangent_g(e1, e2)            (base: tangent propagated from g)
    //                  + Σ_i ∂g/∂z_i(e1, e2) * d(e_i)/dp   (when args depend on p)
    //
    // The second term is non-zero only when a call argument directly uses the
    // differentiated parameter -- unusual in practice, since Halide Func indices
    // are always integer-typed, and float params cannot naturally appear in an
    // integer expression with a non-zero derivative.
    if (op->call_type == Call::Halide) {
        if (!op->type.is_float()) {
            // Integer-valued Func calls: no meaningful derivative.
            return make_zero(op->type);
        }

        auto it = ctx.tangent_funcs.find(op->name);
        if (it == ctx.tangent_funcs.end()) {
            // The called Func is outside the pipeline we're differentiating
            // (e.g., a Func defined in a different compilation unit that
            // find_transitive_calls didn't traverse).  Treat its contribution
            // as zero, which is correct when the external Func doesn't depend
            // on the parameter.
            return make_zero(op->type);
        }
        Func &tangent_g = it->second;

        // Evaluate the pre-computed tangent of g at the call's arguments.
        // For Tuple Funcs, each element is a separate Call with value_index set;
        // we extract the corresponding element of the tangent Tuple.
        Expr base_tangent;
        if (tangent_g.outputs() == 1) {
            base_tangent = tangent_g(op->args);
        } else {
            base_tangent = tangent_g(op->args)[op->value_index];
        }

        // Fast path: if no call argument depends on the scalar param,
        // the second chain-rule term is zero and we return immediately.
        // This is the overwhelmingly common case in rendering pipelines,
        // where spatial arguments are pure integer coordinate expressions.
        bool any_arg_depends = false;
        if (!ctx.target_scalar_param.empty()) {
            for (const auto &arg : op->args) {
                if (expr_uses_var(arg, ctx.target_scalar_param)) {
                    any_arg_depends = true;
                    break;
                }
            }
        }
        // Buffer-dependent call args (map overload) are not checked here;
        // they would require a different detection mechanism and are not
        // supported in V1.

        if (!any_arg_depends) {
            return base_tangent;
        }

        // ── Chain rule for param-dependent call arguments ────────────────────
        //
        // d/dp g(e1, e2) = tangent_g(e1, e2)
        //                + ∂g/∂z1(e1, e2) * d(e1)/dp
        //                + ∂g/∂z2(e1, e2) * d(e2)/dp
        //
        // ∂g/∂z_i is computed by re-running diff_expr on g's pure body in
        // "spatial partial derivative mode" (partial_var = z_i.name()).
        //
        // V1 limitation: only pure-definition Funcs are supported here.
        // Funcs with update definitions (scans, reductions) have history-
        // dependent spatial derivatives that require additional analysis.
        Func g_func(Function(op->func));
        if (g_func.num_update_definitions() > 0) {
            user_error << "propagate_tangents: function '" << op->name
                      << "' is called with parameter-dependent arguments but has "
                      << "update definitions. This is not supported in V1. "
                      << "Restructure the pipeline so that parameter-dependent "
                      << "expressions appear in the function body, not in call "
                      << "arguments, for functions with update definitions.\n";
            return make_zero(op->type);
        }

        Expr result = base_tangent;
        const vector<Var> &g_pure_args = g_func.args();

        for (int i = 0; i < (int)op->args.size() && i < (int)g_pure_args.size(); i++) {
            Expr darg = diff_expr(op->args[i], ctx, partial_cache);
            // Skip zero terms early to avoid building useless partial_g Funcs.
            if (is_const_zero(darg)) {
                continue;
            }

            // Lazily compute ∂g/∂z_i and cache it.
            // partial_cache[func_name][arg_index] stores the partial Func so
            // we don't recompute it if g is called multiple times in the pipeline.
            auto &g_partials = partial_cache[op->name];
            if (g_partials.find(i) == g_partials.end()) {
                // Spatial partial derivative mode: set partial_var to z_i's name.
                // This causes diff_expr to return 1 when it encounters Variable z_i
                // and 0 for all other non-param Variables.
                ForwardDiffContext partial_ctx;
                partial_ctx.partial_var = g_pure_args[i].name();
                // Do not propagate tangent_funcs into this sub-computation:
                // we are differentiating g's body w.r.t. one of its own Vars,
                // not w.r.t. the global parameter p.

                vector<Var> partial_args = g_func.args();
                const Tuple &g_values = g_func.values();

                Func partial_g("d_" + op->name + "_darg" + std::to_string(i) + "__");
                if ((int)g_values.as_vector().size() == 1) {
                    partial_g(partial_args) =
                        diff_expr(g_values.as_vector()[0], partial_ctx, partial_cache);
                } else {
                    vector<Expr> partials;
                    for (const Expr &v : g_values.as_vector()) {
                        partials.push_back(diff_expr(v, partial_ctx, partial_cache));
                    }
                    partial_g(partial_args) = Tuple(partials);
                }
                g_partials[i] = partial_g;
            }

            // Evaluate the partial at the actual call arguments, then accumulate.
            Func &partial_g = g_partials[i];
            Expr partial_val;
            if (partial_g.outputs() == 1) {
                partial_val = partial_g(op->args);
            } else {
                partial_val = partial_g(op->args)[op->value_index];
            }
            result = result + partial_val * darg;
        }
        return result;
    }

    // ── Buffer / ImageParam load ─────────────────────────────────────────────
    //
    // Both concrete Buffer<> and ImageParam appear as Call::Image nodes in the
    // Halide IR.  op->name is the buffer/param name.
    //
    // Note on ImageParam's internal structure: ImageParam("inp") creates an
    // internal wrapper Func "inp_im" whose definition is:
    //   inp_im(_0, _1) = Call::Image("inp", [_0, _1])
    // where _0, _1 are Halide's implicit Vars.  find_transitive_calls includes
    // this wrapper in the traversal.  When we process "inp_im", its body
    // contains a Call::Image node for "inp".  The name "inp" is what we look
    // up in buffer_tangents -- this is why the user provides the ImageParam's
    // name (inp.name()), not the wrapper name.
    if (op->call_type == Call::Image) {
        if (!op->type.is_float()) {
            return make_zero(op->type);
        }

        auto it = ctx.buffer_tangents.find(op->name);
        if (it == ctx.buffer_tangents.end()) {
            // This buffer is not a differentiation target; it's a constant
            // w.r.t. the parameter.
            return make_zero(op->type);
        }

        // V1 limitation: we don't support param-dependent buffer indices
        // (that would require computing the spatial gradient of the buffer
        // content, i.e., image gradients, which is a separate operation).
        if (!ctx.target_scalar_param.empty()) {
            for (const auto &arg : op->args) {
                if (expr_uses_var(arg, ctx.target_scalar_param)) {
                    user_error << "propagate_tangents: buffer '" << op->name
                              << "' is accessed with parameter-dependent indices. "
                              << "This is not supported in V1. Restructure the pipeline "
                              << "to avoid parameter-dependent buffer index expressions.\n";
                    return make_zero(op->type);
                }
            }
        }

        // Evaluate the user-provided tangent direction at the load's coordinates.
        // The tangent direction specifies how the buffer "moves" in parameter space:
        // for a unit basis vector (1 at pixel (i,j), 0 elsewhere), this computes
        // the directional derivative d(output)/d(buf(i,j)).
        Func &tangent_input = it->second;
        if (tangent_input.dimensions() == 0) {
            // 0-D tangent Func: the buffer is treated as a scalar parameter.
            // Its tangent is just the scalar value, independent of coordinates.
            return tangent_input();
        }
        return tangent_input(op->args);
    }

    user_error << "propagate_tangents: unknown call type for '" << op->name << "'\n";
    return make_zero(op->type);
}

// Recursively compute the forward-mode tangent of expression e w.r.t. the
// differentiation target stored in ctx.
//
// The result has the same type as e.  For non-float types, returns zero
// (differentiation is only meaningful for floating-point arithmetic).
//
// This function is the core of the forward-mode AD pass.  It traverses the
// Halide IR expression tree and applies standard calculus rules at each node.
Expr diff_expr(const Expr &e, ForwardDiffContext &ctx,
               map<string, map<int, Func>> &partial_cache) {

    // Constants: d(c)/dp = 0 for any constant c.
    if (e.as<IntImm>() || e.as<UIntImm>()) {
        return make_zero(e.type());
    }
    if (e.as<FloatImm>()) {
        return make_zero(e.type());
    }
    if (e.as<StringImm>()) {
        return make_zero(e.type());
    }

    // Reinterpret: bit-level reinterpretation has no smooth derivative.
    if (const Reinterpret *op = e.as<Reinterpret>()) {
        (void)op;
        return make_zero(e.type());
    }

    if (const Cast *op = e.as<Cast>()) {
        if (op->type.is_float()) {
            // Casting to a float type is a smooth linear operation (for widening
            // casts: exact; for narrowing casts: approximate but differentiable).
            // The tangent is cast to the same target type so types stay consistent.
            return cast(op->type, diff_expr(op->value, ctx, partial_cache));
        }
        // Casting to an integer (rounding) is piecewise constant: derivative zero.
        // In particular, cast<int>(float_param) has zero derivative, which means
        // param-dependent Func index expressions always have zero tangent.
        // This makes the chain-rule path for param-dependent call arguments
        // unreachable for normal Halide pipelines (see diff_call above).
        return make_zero(op->type);
    }

    if (const Variable *op = e.as<Variable>()) {
        // A Variable node can be:
        //   (a) A Param reference (op->param.defined() == true)
        //   (b) A Var used as a loop variable or function argument
        //   (c) A let-bound variable (appears in Let body as a free variable)
        //
        // Cases (b) and (c) are distinguishable only by checking let_tangent.

        if (op->param.defined()) {
            // Scalar Param: return 1 if this is the differentiation target,
            // or the 0-D tangent Func value if using the map-based API.
            if (!ctx.target_scalar_param.empty() &&
                op->name == ctx.target_scalar_param) {
                return make_one(op->type);
            }
            // Map-based API: look for a 0-D Func in buffer_tangents.
            // This handles directional derivatives along scalar param axes
            // (e.g., tangent value 1.0 for d/dp1, 0.5 for d/dp in a
            // blended direction).
            auto it = ctx.buffer_tangents.find(op->name);
            if (it != ctx.buffer_tangents.end() && it->second.dimensions() == 0) {
                return it->second();
            }
            // Parameter not being differentiated: treat as constant.
            return make_zero(op->type);
        }

        // Spatial partial derivative mode: the named Var is the "seed" direction.
        if (!ctx.partial_var.empty() && op->name == ctx.partial_var) {
            return make_one(op->type);
        }

        // Let-bound variable: look up the pre-computed tangent expression.
        // The tangent is stored as a Variable node (d_<name>) and will be
        // bound by the Let wrapper generated in the Let case below.
        auto it = ctx.let_tangent.find(op->name);
        if (it != ctx.let_tangent.end()) {
            return it->second;
        }

        // Free Var (loop variable like x, y) not being differentiated: zero.
        return make_zero(op->type);
    }

    // Linearity rules: d(a+b)/dp = da+db, d(a-b)/dp = da-db.
    if (const Add *op = e.as<Add>()) {
        return diff_expr(op->a, ctx, partial_cache) +
               diff_expr(op->b, ctx, partial_cache);
    }
    if (const Sub *op = e.as<Sub>()) {
        return diff_expr(op->a, ctx, partial_cache) -
               diff_expr(op->b, ctx, partial_cache);
    }

    if (const Mul *op = e.as<Mul>()) {
        // Product rule: d(a*b)/dp = a'*b + a*b'.
        // Ordering: primal * tangent puts the primal first, giving Halide's
        // simplifier a better chance to fold zero terms (e.g., const*zero → 0).
        Expr da = diff_expr(op->a, ctx, partial_cache);
        Expr db = diff_expr(op->b, ctx, partial_cache);
        return op->a * db + op->b * da;
    }

    if (const Div *op = e.as<Div>()) {
        // Quotient rule: d(a/b)/dp = (a'*b - a*b') / b^2.
        Expr da = diff_expr(op->a, ctx, partial_cache);
        Expr db = diff_expr(op->b, ctx, partial_cache);
        return (da * op->b - op->a * db) / (op->b * op->b);
    }

    if (const Mod *op = e.as<Mod>()) {
        // Modulo (both integer and float fmod) is piecewise constant:
        // the derivative is zero almost everywhere.
        return make_zero(op->type);
    }

    if (const Min *op = e.as<Min>()) {
        // min(a,b): the derivative follows whichever branch is active.
        // This is a subgradient (Clarke generalized derivative); at a==b
        // it is undefined but has measure zero.
        Expr da = diff_expr(op->a, ctx, partial_cache);
        Expr db = diff_expr(op->b, ctx, partial_cache);
        return select(op->a < op->b, da, db);
    }
    if (const Max *op = e.as<Max>()) {
        Expr da = diff_expr(op->a, ctx, partial_cache);
        Expr db = diff_expr(op->b, ctx, partial_cache);
        return select(op->a > op->b, da, db);
    }

    // Boolean/comparison nodes: their output is integer-typed (bool) and
    // carries no gradient.  They appear as conditions in Select/Min/Max and
    // are not themselves differentiated.
    if (e.as<EQ>() || e.as<NE>() || e.as<LT>() || e.as<LE>() ||
        e.as<GT>() || e.as<GE>() || e.as<And>() || e.as<Or>() || e.as<Not>()) {
        return make_zero(e.type());
    }

    if (const Select *op = e.as<Select>()) {
        // select(cond, t, f): the condition is not differentiated (it's a
        // boolean predicate that switches branches).  This implements the
        // "straight-through estimator": the derivative flows through the
        // active branch only.  At cond boundary points the derivative is
        // undefined, but those are measure zero and practically harmless.
        Expr dt = diff_expr(op->true_value, ctx, partial_cache);
        Expr df = diff_expr(op->false_value, ctx, partial_cache);
        return select(op->condition, dt, df);
    }

    if (const Let *op = e.as<Let>()) {
        // Let(v, val, body): the derivative is Let(v, val, Let(d_v, d_val, d_body)).
        //
        // We introduce a new let-binding d_v = d(val)/dp so that any occurrence
        // of v in d_body can reference d_v instead of recomputing d(val).
        // This preserves sharing: if v appears N times in body, both the primal
        // computation of v and its tangent d_v are computed once.
        //
        // The original binding for v is retained in the outer Let so that the
        // primal value is available both in the body and in computing d_v.
        Expr dval = diff_expr(op->value, ctx, partial_cache);
        string dv_name = "d_" + op->name;

        // Save the existing tangent mapping for op->name (if any) before
        // overwriting it.  This handles rare cases of shadowed let names
        // that can appear after CSE or simplification passes.
        Expr prev_tangent;
        auto prev_it = ctx.let_tangent.find(op->name);
        if (prev_it != ctx.let_tangent.end()) {
            prev_tangent = prev_it->second;
        }
        // Inside the body, Variable(op->name) has tangent Variable(dv_name).
        ctx.let_tangent[op->name] = Variable::make(op->value.type(), dv_name);

        Expr dbody = diff_expr(op->body, ctx, partial_cache);

        // Restore the previous mapping (or remove the entry if there was none).
        if (prev_tangent.defined()) {
            ctx.let_tangent[op->name] = prev_tangent;
        } else {
            ctx.let_tangent.erase(op->name);
        }

        // Build: let v = val in let d_v = dval in dbody.
        // The outer Let makes the primal v available in dval (needed if dval
        // references v, e.g., for d(v^2)/dp = 2*v*dv/dp).
        return Let::make(op->name, op->value,
                         Let::make(dv_name, dval, dbody));
    }

    if (const Call *op = e.as<Call>()) {
        return diff_call(op, ctx, partial_cache);
    }

    user_warning << "propagate_tangents: unhandled expression node type, "
                    "derivative set to zero\n";
    return make_zero(e.type());
}

// Core driver shared by all propagate_tangents() overloads.
//
// Takes the ForwardDiffContext by value (moved in) because the context is
// mutated heavily during the pass (tangent_funcs is built up incrementally).
Func propagate_tangents_impl(const Func &output, ForwardDiffContext ctx) {
    // Discover all Funcs in the pipeline via transitive call graph traversal,
    // then sort them in topological order (producers before consumers).
    // This is the same infrastructure used by propagate_adjoints().
    // For forward-mode we need producers first so that tangent_g is already
    // defined when we process a consumer that calls g.
    map<string, Function> env = find_transitive_calls(output.function());
    vector<string> order = topological_order({output.function()}, env);

    // Cache for spatial partial derivative Funcs used in the chain rule.
    // partial_cache[func_name][arg_index] -> partial Func d(func)/d(z_i).
    // Keyed separately from ctx.tangent_funcs because partials are helper
    // Funcs that don't appear in the final result.
    map<string, map<int, Func>> partial_cache;

    // Forward pass: process each Func in topological order.
    for (const auto &func_name : order) {
        Func f(env[func_name]);

        // The tangent Func mirrors f exactly: same args, same Tuple width.
        // The "__fwd__" suffix avoids name collisions with the pipeline's
        // own Funcs and with tangent Funcs from nested calls.
        vector<Var> args = f.args();
        Func tangent_f("d_" + func_name + "_fwd__");

        // Compute the pure-definition tangent.
        //
        // IMPORTANT: f.values() returns a *temporary* Tuple object.
        // Calling .as_vector() directly on the temporary returns a const
        // reference into the temporary's internal vector.  Once the temporary
        // is destroyed at the semicolon, the reference dangling, leading to
        // a use-after-free that manifests as "Tuples must have at least one
        // element" assertions deep in the Halide internals.
        // The fix is to store the Tuple by value first.
        Tuple pure_tuple = f.values();
        const vector<Expr> &pure_vals = pure_tuple.as_vector();
        if ((int)pure_vals.size() == 1) {
            tangent_f(args) = diff_expr(pure_vals[0], ctx, partial_cache);
        } else {
            vector<Expr> dvals;
            dvals.reserve(pure_vals.size());
            for (const Expr &val : pure_vals) {
                dvals.push_back(diff_expr(val, ctx, partial_cache));
            }
            tangent_f(args) = Tuple(dvals);
        }

        // Register tangent_f in the context BEFORE processing update definitions.
        //
        // Why before updates? An update definition's RHS may contain a
        // self-reference to f, e.g.:
        //   scan(r) = scan(r-1) + p * r      (prefix scan)
        //   d_scan(r) = d_scan(r-1) + r       (tangent)
        //
        // When diff_expr encounters scan(r-1) in the update RHS, it looks up
        // ctx.tangent_funcs["scan"] to find d_scan.  If we hadn't registered
        // yet, it would return tangent=0, silently dropping the recurrence.
        //
        // Halide's sequential update semantics ensure that d_scan(r-1) in the
        // RHS of d_scan(r) = ... reads the value written by the previous
        // iteration -- exactly the correct behavior for prefix scans and
        // accumulators.
        ctx.tangent_funcs[func_name] = tangent_f;

        // Compute update-definition tangents.
        // f.update_args(upd) returns the LHS expressions (possibly containing
        // RVars), which become the LHS of the tangent update verbatim -- the
        // tangent iterates over the same reduction domain.
        for (int upd = 0; upd < f.num_update_definitions(); upd++) {
            vector<Expr> update_args = f.update_args(upd);
            // Same Tuple-lifetime caution as for pure values above.
            Tuple update_tuple = f.update_values(upd);
            const vector<Expr> &update_vals = update_tuple.as_vector();

            if ((int)update_vals.size() == 1) {
                tangent_f(update_args) =
                    diff_expr(update_vals[0], ctx, partial_cache);
            } else {
                vector<Expr> dvals;
                dvals.reserve(update_vals.size());
                for (const Expr &val : update_vals) {
                    dvals.push_back(diff_expr(val, ctx, partial_cache));
                }
                tangent_f(update_args) = Tuple(dvals);
            }
        }
    }

    auto it = ctx.tangent_funcs.find(output.name());
    user_assert(it != ctx.tangent_funcs.end())
        << "propagate_tangents: output function '" << output.name()
        << "' not found in pipeline\n";
    return it->second;
}

}  // anonymous namespace

// ── Public API ───────────────────────────────────────────────────────────────

// Single scalar Param: seed tangent = 1.  The returned Func has the same
// dimensionality and Tuple width as output and represents d(output)/d(param).
Func propagate_tangents(const Func &output, const Param<> &param) {
    ForwardDiffContext ctx;
    ctx.target_scalar_param = param.name();
    return propagate_tangents_impl(output, std::move(ctx));
}

// Single Buffer/ImageParam with an explicit tangent direction.
// tangent_input(x, y, ...) specifies the perturbation direction in the buffer's
// parameter space.  The result is the Jacobian-vector product J * tangent_input.
// For the full Jacobian column corresponding to pixel (i,j), pass a basis vector
// that is 1 at (i,j) and 0 elsewhere.
Func propagate_tangents(const Func &output,
                        const Buffer<> &buffer_param,
                        const Func &tangent_input) {
    ForwardDiffContext ctx;
    // buffer_param.name() matches the op->name seen in Call::Image nodes for
    // this buffer.  Buffer<float> is implicitly convertible to Buffer<> so
    // callers can pass typed buffers directly.
    ctx.buffer_tangents[buffer_param.name()] = tangent_input;
    return propagate_tangents_impl(output, std::move(ctx));
}

// General form: multiple inputs differentiated simultaneously.
// Each entry maps an input name to its tangent Func:
//   - 0-D Func: tangent for a scalar Param (value 1.0 = unit direction).
//   - N-D Func: tangent direction for a Buffer/ImageParam of the same shape.
// By using a 0-D Func with a user-controlled value, this overload also supports
// directional derivatives: e.g., {{"p", t}} where t() = 0.5f differentiates
// along the direction p -> 0.5 (half-step in p's axis).
// target_scalar_param is left empty; the Variable case in diff_expr handles
// 0-D entries in buffer_tangents for scalar Params.
Func propagate_tangents(const Func &output,
                        const std::map<std::string, Func> &tangent_inputs) {
    ForwardDiffContext ctx;
    ctx.buffer_tangents = tangent_inputs;
    return propagate_tangents_impl(output, std::move(ctx));
}

}  // namespace Halide
