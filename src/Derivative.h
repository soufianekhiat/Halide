#ifndef HALIDE_DERIVATIVE_H
#define HALIDE_DERIVATIVE_H

/** \file
 *  Automatic differentiation
 */

#include "Expr.h"
#include "Func.h"

#include <map>
#include <string>
#include <vector>

namespace Halide {

/**
 *  Helper structure storing the adjoints Func.
 *  Use d(func) or d(buffer) to obtain the derivative Func.
 */
class Derivative {
public:
    // function name & update_id, for initialization update_id == -1
    using FuncKey = std::pair<std::string, int>;

    explicit Derivative(const std::map<FuncKey, Func> &adjoints_in)
        : adjoints(adjoints_in) {
    }
    explicit Derivative(std::map<FuncKey, Func> &&adjoints_in)
        : adjoints(std::move(adjoints_in)) {
    }

    // These all return an undefined Func if no derivative is found
    // (typically, if the input Funcs aren't differentiable)
    Func operator()(const Func &func, int update_id = -1) const;
    Func operator()(const Buffer<> &buffer) const;
    Func operator()(const Param<> &param) const;
    Func operator()(const std::string &name) const;

private:
    const std::map<FuncKey, Func> adjoints;
};

/**
 *  Given a Func and a corresponding adjoint, (back)propagate the
 *  adjoint to all dependent Funcs, buffers, and parameters.
 *  The bounds of output and adjoint need to be specified with pair {min, extent}
 *  For each Func the output depends on, and for the pure definition and
 *  each update of that Func, it generates a derivative Func stored in
 *  the Derivative.
 */
Derivative propagate_adjoints(const Func &output,
                              const Func &adjoint,
                              const Region &output_bounds);
/**
 *  Given a Func and a corresponding adjoint buffer, (back)propagate the
 *  adjoint to all dependent Funcs, buffers, and parameters.
 *  For each Func the output depends on, and for the pure definition and
 *  each update of that Func, it generates a derivative Func stored in
 *  the Derivative.
 */
Derivative propagate_adjoints(const Func &output,
                              const Buffer<float> &adjoint);
/**
 *  Given a scalar Func with size 1, (back)propagate the gradient
 *  to all dependent Funcs, buffers, and parameters.
 *  For each Func the output depends on, and for the pure definition and
 *  each update of that Func, it generates a derivative Func stored in
 *  the Derivative.
 */
Derivative propagate_adjoints(const Func &output);

/**
 *  Forward-mode automatic differentiation (JVP): compute d(output)/d(param).
 *  Returns a Func with the same dimensionality and Tuple width as output,
 *  representing the derivative of each output element w.r.t. the given Param.
 *
 *  This is efficient when the output is high-dimensional (e.g., an image)
 *  and the parameter is scalar -- the inverse/differential rendering use case.
 *  For high-dimensional parameter spaces with a scalar loss function, use
 *  propagate_adjoints() (reverse-mode) instead.
 *
 *  The returned Func is an ordinary Halide Func and can be scheduled freely.
 *  Its default scheduling (compute_inline) uses zero extra memory -- tangent
 *  expressions are fused into the consumer's computation.
 */
Func propagate_tangents(const Func &output, const Param<> &param);

/**
 *  Forward-mode JVP w.r.t. a Buffer/ImageParam along a tangent direction.
 *  tangent_input has the same dimensions as the buffer and specifies the
 *  direction to differentiate along (a vector in the parameter space).
 *  Result: d(output) = J * tangent_input  (Jacobian-vector product).
 *
 *  For the full Jacobian of output w.r.t. the buffer, call once per basis
 *  vector (expensive for large buffers). To optimize a buffer via gradient
 *  descent, prefer propagate_adjoints() + scalar loss (reverse-mode).
 */
Func propagate_tangents(const Func &output,
                        const Buffer<> &buffer_param,
                        const Func &tangent_input);

/**
 *  Forward-mode JVP w.r.t. multiple inputs simultaneously.
 *  tangent_inputs maps input names (Param or Buffer names) to their tangent
 *  Funcs. For scalar Params, use a 0-dimensional Func returning 1.0f.
 *  For Buffers/ImageParams, use a Func with the same dimensions returning
 *  the tangent direction.
 */
Func propagate_tangents(const Func &output,
                        const std::map<std::string, Func> &tangent_inputs);

}  // namespace Halide

#endif
