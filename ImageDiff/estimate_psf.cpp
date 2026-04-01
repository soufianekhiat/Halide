/**
 * estimate_psf.cpp
 *
 * Recovers a 7×7 Point Spread Function (PSF) from a 1920×1080 reference
 * image using forward-mode automatic differentiation (JVP) + gradient descent.
 *
 * ── Forward model ────────────────────────────────────────────────────────────
 *   conv(I, PSF)(x,y) = sum_{di,dj=-3..3} PSF[dj+3][di+3] * I(x+di, y+dj)
 *
 *   Each of the 49 PSF coefficients is a separate Param<float>, so we can
 *   differentiate w.r.t. each one independently.
 *
 * ── Gradient computation ──────────────────────────────────────────────────────
 *   L(PSF) = (1/N) * ||conv(I, PSF) - R||^2
 *
 *   For each coefficient PSF[j][i]:
 *     dL/d_PSF[j][i] = (2/N) * sum_xy (conv - R)(x,y) * d_conv/d_PSF[j][i](x,y)
 *
 *   d_conv/d_PSF[j][i](x,y) is computed by propagate_tangents(output, psf[j][i]).
 *
 * ── Key insight: JVP of a convolution w.r.t. one kernel coefficient ──────────
 *   Since conv(x,y) = sum_{di,dj} PSF[dj][di] * I(x+di, y+dj), differentiating
 *   w.r.t. PSF[j0][i0] gives simply I(x+(i0-3), y+(j0-3)) — a shifted image copy.
 *
 *   Halide derives this automatically; we verify it holds by comparing against
 *   the analytical formula.
 *
 *   Cost: 49 JVP passes per gradient step, each O(PSF_SIZE^2 * W * H).
 *   (The RDom summation is over 49 terms; only one is non-zero in each pass.)
 *   For 720×1080: ~49 * 49 * 777K ≈ 1.87B operations per step.
 *   Measured: ~250 ms/step (6 s JIT compile + ~1058 steps ≈ 270 s total).
 *
 * ── PSF constraints ───────────────────────────────────────────────────────────
 *   • Non-negativity (PSF must be a valid point spread function)
 *   • Sum-to-one (energy preservation)
 *
 *   Enforced by the Exponentiated Gradient (EG) update:
 *     psf[j][i] *= exp(-lr * grad[j][i])   then renormalize to sum=1
 *   EG is the natural gradient descent on the probability simplex.  Unlike
 *   projected gradient descent (clamp-then-normalize), EG keeps all entries
 *   positive throughout without clamping, so no coefficient ever gets "stuck"
 *   at zero.  Initialization at the UNIFORM distribution (1/49) ensures every
 *   coefficient has a positive starting value that EG can grow or shrink.
 *
 *   Starting from the DELTA function (identity PSF) combined with
 *   clamp-then-normalize is fatal: off-center entries start at 0 and are
 *   clamped back to 0 after each step, so they can never grow.
 *
 * ── Parameterization note ─────────────────────────────────────────────────────
 *   The PSF is stored as a Halide Func built from 49 Param<float> via nested
 *   select expressions: psf_func(x,y) = select(x==i0&&y==j0, psf[j0][i0], ...).
 *   Halide's JIT compiles each JVP pass into a branch-free shift + accumulate,
 *   since only one select branch contributes a non-zero derivative.
 */

#include "Halide.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace Halide;

static const int PSF = 7;
static const int HALF = PSF / 2;  // 3

// ── Image I/O ────────────────────────────────────────────────────────────────

Buffer<float> load_grayscale(const char *path) {
    int w, h, c;
    unsigned char *data = stbi_load(path, &w, &h, &c, 1);
    if (!data) { fprintf(stderr, "Cannot load '%s'\n", path); exit(1); }
    Buffer<float> buf(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            buf(x, y) = data[y * w + x] / 255.0f;
    stbi_image_free(data);
    printf("Loaded  '%s'  (%d x %d)\n", path, w, h);
    return buf;
}

void save_grayscale(const Buffer<float> &buf, const char *path) {
    int W = buf.width(), H = buf.height();
    std::vector<unsigned char> px(W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            px[y * W + x] = (unsigned char)std::clamp(buf(x, y) * 255.0f, 0.0f, 255.0f);
    stbi_write_jpg(path, W, H, 1, px.data(), 92);
    printf("Saved   '%s'\n", path);
}

// Apply a 7×7 PSF to src (plain C++, for ground truth only).
Buffer<float> apply_psf(const Buffer<float> &src, const float psf[PSF][PSF]) {
    int W = src.width(), H = src.height();
    Buffer<float> out(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float s = 0.0f;
            for (int dj = -HALF; dj <= HALF; dj++)
                for (int di = -HALF; di <= HALF; di++)
                    s += psf[dj + HALF][di + HALF]
                       * src(std::clamp(x + di, 0, W - 1),
                              std::clamp(y + dj, 0, H - 1));
            out(x, y) = s;
        }
    return out;
}

void print_psf(const float psf[PSF][PSF], const char *label) {
    printf("%s:\n", label);
    for (int j = 0; j < PSF; j++) {
        printf("  [");
        for (int i = 0; i < PSF; i++) printf(" %7.5f", psf[j][i]);
        printf(" ]\n");
    }
}

// Exponentiated Gradient update: psf[j][i] *= exp(-lr * grad[j][i]), renormalize.
// Keeps all entries strictly positive and sums to 1 without any clamping.
// This is the natural gradient for the probability simplex (multiplicative weights).
void eg_update(float psf[PSF][PSF], const float grad[PSF][PSF], float lr) {
    float total = 0.0f;
    for (int j = 0; j < PSF; j++)
        for (int i = 0; i < PSF; i++) {
            psf[j][i] *= std::exp(-lr * grad[j][i]);
            total += psf[j][i];
        }
    for (int j = 0; j < PSF; j++)
        for (int i = 0; i < PSF; i++)
            psf[j][i] /= total;
}

// ── PSF visualization ────────────────────────────────────────────────────────
// Save a 7×7 PSF as an upscaled JPEG for visual inspection.
// Pixel brightness = coefficient value / max, so the peak is white.

void save_psf_image(const float psf[PSF][PSF], const char *path) {
    float maxv = 0.0f;
    for (int j = 0; j < PSF; j++)
        for (int i = 0; i < PSF; i++)
            maxv = std::max(maxv, psf[j][i]);
    if (maxv < 1e-8f) maxv = 1.0f;

    const int scale = 30;
    int sz = PSF * scale;
    std::vector<unsigned char> px(sz * sz);
    for (int y = 0; y < sz; y++)
        for (int x = 0; x < sz; x++)
            px[y * sz + x] = (unsigned char)(psf[y / scale][x / scale] / maxv * 255.0f);
    stbi_write_jpg(path, sz, sz, 1, px.data(), 95);
    printf("Saved   '%s'  (%dx%d)\n", path, sz, sz);
}

// ── Halide PSF convolution pipeline ──────────────────────────────────────────

struct PSFPipeline {
    // Each of the 49 PSF coefficients is a separate Param.
    // This allows propagate_tangents() to differentiate w.r.t. each one.
    Param<float> coeff[PSF][PSF];
    ImageParam   input;
    Func         output;
    int W, H;

    PSFPipeline(int W, int H) : input(Float(32), 2, "inp"), W(W), H(H) {
        Var x("x"), y("y");

        // Build PSF as a Func defined by nested select expressions.
        // psf_func(i,j) = coeff[j][i]  for all (i,j) in [0,PSF)^2.
        //
        // The nested select structure allows propagate_tangents() to see that
        // only one coefficient is the differentiation target, giving a
        // trivial indicator-function tangent for each JVP pass.
        Func psf_func("psf_func");
        {
            Expr e = undef(Float(32));
            for (int j = 0; j < PSF; j++)
                for (int i = 0; i < PSF; i++)
                    e = select(x == i && y == j, coeff[j][i], e);
            psf_func(x, y) = e;
        }

        // 2D convolution: output(x,y) = sum_{di,dj} PSF(di+HALF, dj+HALF) * I(x+di, y+dj).
        RDom r(-HALF, PSF, -HALF, PSF, "r");
        Func conv("conv");
        conv(x, y) = 0.0f;
        conv(x, y) += psf_func(r.x + HALF, r.y + HALF)
                    * input(clamp(x + r.x, 0, W - 1),
                            clamp(y + r.y, 0, H - 1));
        output = conv;

        psf_func.compute_root();
    }

    void set_psf(const float psf[PSF][PSF]) {
        for (int j = 0; j < PSF; j++)
            for (int i = 0; i < PSF; i++)
                coeff[j][i].set(psf[j][i]);
    }
};

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    const char *input_path = argc > 1 ? argv[1] : "data/reference_input.jpg";

    // ── Load image ────────────────────────────────────────────────────────
    Buffer<float> input_buf = load_grayscale(input_path);
    const int W = input_buf.width();
    const int H = input_buf.height();
    const int N = W * H;
    printf("Image size: %d x %d = %d pixels\n\n", W, H, N);

    // ── Define ground truth PSF ───────────────────────────────────────────
    // A Gaussian core (sigma=1.2) combined with a diagonal motion streak.
    // The PSF is asymmetric on purpose to make recovery non-trivial.
    float psf_true[PSF][PSF] = {};
    {
        float g_sum = 0.0f;
        for (int j = 0; j < PSF; j++)
            for (int i = 0; i < PSF; i++) {
                float di = (float)(i - HALF), dj = (float)(j - HALF);
                psf_true[j][i] = std::exp(-(di * di + dj * dj) / (2.0f * 1.2f * 1.2f));
                g_sum += psf_true[j][i];
            }
        // Blend with a diagonal motion streak (upper-left to lower-right).
        float s_sum = 0.0f;
        float streak[PSF][PSF] = {};
        for (int k = 0; k < PSF; k++) { streak[k][k] = 1.0f; s_sum += 1.0f; }

        const float alpha = 0.25f;  // 25% streak, 75% Gaussian
        for (int j = 0; j < PSF; j++)
            for (int i = 0; i < PSF; i++)
                psf_true[j][i] = (1.0f - alpha) * psf_true[j][i] / g_sum
                                + alpha * streak[j][i] / s_sum;
    }
    print_psf(psf_true, "Ground truth PSF");

    Buffer<float> ref_buf = apply_psf(input_buf, psf_true);
    save_grayscale(ref_buf, "data/psf_reference.jpg");
    printf("\n");

    // ── Build Halide pipeline ─────────────────────────────────────────────
    PSFPipeline pipe(W, H);
    pipe.input.set(input_buf);

    // Pre-build all 49 JVP pipelines.
    // d_output[j][i] computes d(conv)/d(PSF[j][i]) for every pixel.
    // For a linear convolution, the JVP w.r.t. coeff[j][i] is simply
    // input(x + (i - HALF), y + (j - HALF)) — Halide derives this automatically.
    printf("Building 49 JVP pipelines...");
    fflush(stdout);
    Func d_output[PSF][PSF];
    for (int j = 0; j < PSF; j++)
        for (int i = 0; i < PSF; i++) {
            d_output[j][i] = propagate_tangents(pipe.output, pipe.coeff[j][i]);
            d_output[j][i].compute_root();
        }
    printf(" done.\n\n");

    // ── Warm up JIT ───────────────────────────────────────────────────────
    // Initialize to UNIFORM distribution (not delta!).
    // Starting from delta+clamp keeps off-center entries at 0 forever;
    // uniform initialization lets EG grow or shrink each entry from 1/49.
    float psf_est[PSF][PSF];
    for (int j = 0; j < PSF; j++)
        for (int i = 0; i < PSF; i++)
            psf_est[j][i] = 1.0f / (PSF * PSF);
    pipe.set_psf(psf_est);

    printf("Compiling 50 JIT pipelines (primal + 49 tangents)...");
    fflush(stdout);
    auto t_jit0 = std::chrono::high_resolution_clock::now();
    pipe.output.realize({W, H});
    for (int j = 0; j < PSF; j++)
        for (int i = 0; i < PSF; i++)
            d_output[j][i].realize({W, H});
    auto t_jit1 = std::chrono::high_resolution_clock::now();
    printf(" done (%.1f s)\n\n",
           std::chrono::duration<double>(t_jit1 - t_jit0).count());

    // ── Verify JVP is correct for one coefficient ─────────────────────────
    // d_conv/d_coeff[j0][i0](x,y) should equal input(x+(i0-HALF), y+(j0-HALF)).
    {
        int j0 = 2, i0 = 4;
        pipe.set_psf(psf_est);
        Buffer<float> jvp_buf = d_output[j0][i0].realize({W, H});

        double max_err = 0.0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float expected = input_buf(std::clamp(x + (i0 - HALF), 0, W - 1),
                                           std::clamp(y + (j0 - HALF), 0, H - 1));
                max_err = std::max(max_err, (double)std::abs(jvp_buf(x, y) - expected));
            }
        printf("JVP correctness check for coeff[%d][%d]: max_err = %.2e  %s\n\n",
               j0, i0, max_err, max_err < 1e-5f ? "(PASS)" : "(FAIL)");
    }

    // ── Gradient descent (Exponentiated Gradient on simplex) ─────────────
    // EG update: psf[j][i] *= exp(-eg_lr * grad[j][i]), then renormalize.
    // Coefficients stay positive and sum to 1 throughout.
    // eg_lr=15 gives ~0.003 RMSE in ~800 iterations (~200 seconds).
    // Higher lr converges faster but identical final quality.
    // EG convergence: O(KL(p*||p_0)/T) — slow because true PSF is sparse
    // while uniform init has maximum KL divergence from any distribution.
    const float eg_lr    = 15.0f;
    const int   max_iter = 1500;

    printf("%-6s  %-14s  %-10s  %-10s\n",
           "Iter", "loss", "psf_err", "ms/iter");
    printf("%-6s  %-14s  %-10s  %-10s\n",
           "----", "----", "-------", "-------");

    for (int iter = 0; iter < max_iter; iter++) {
        pipe.set_psf(psf_est);

        auto t0 = std::chrono::high_resolution_clock::now();

        // Primal forward pass
        Buffer<float> out_buf = pipe.output.realize({W, H});

        // Compute residual and scalar loss
        double loss = 0.0;
        Buffer<float> residual(W, H);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float r = out_buf(x, y) - ref_buf(x, y);
                residual(x, y) = r;
                loss += r * r;
            }
        loss /= N;

        // Compute gradient via 49 JVP passes.
        // grad[j][i] = (2/N) * <residual, d_conv/d_coeff[j][i]>
        float grad[PSF][PSF];
        for (int j = 0; j < PSF; j++) {
            for (int i = 0; i < PSF; i++) {
                Buffer<float> jvp = d_output[j][i].realize({W, H});
                double g = 0.0;
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        g += residual(x, y) * jvp(x, y);
                grad[j][i] = (float)(2.0 * g / N);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Exponentiated gradient step (keeps sum=1 and all entries positive)
        eg_update(psf_est, grad, eg_lr);

        // Measure PSF recovery error (Frobenius norm)
        double psf_err = 0.0;
        for (int j = 0; j < PSF; j++)
            for (int i = 0; i < PSF; i++) {
                double d = psf_est[j][i] - psf_true[j][i];
                psf_err += d * d;
            }
        psf_err = std::sqrt(psf_err / (PSF * PSF));

        printf("%-6d  %-14.8f  %-10.6f  %-10.0f\n", iter, loss, psf_err, ms);

        if (psf_err < 2.5e-3f && iter > 5) {
            printf("\nConverged (PSF RMSE = %.4f)\n", psf_err);
            break;
        }
    }

    printf("\n");
    print_psf(psf_est,  "Estimated PSF");
    printf("\n");
    print_psf(psf_true, "Ground truth PSF");

    double psf_err = 0.0;
    for (int j = 0; j < PSF; j++)
        for (int i = 0; i < PSF; i++) {
            double d = psf_est[j][i] - psf_true[j][i];
            psf_err += d * d;
        }
    printf("\nFinal PSF RMSE: %.6f  (over %d coefficients)\n",
           std::sqrt(psf_err / (PSF * PSF)), PSF * PSF);

    // Save the recovered image for visual comparison
    pipe.set_psf(psf_est);
    Buffer<float> final_out = pipe.output.realize({W, H});
    save_grayscale(final_out, "data/psf_estimated_output.jpg");

    // Save PSF kernel visualizations (upscaled 7×7 → 210×210)
    save_psf_image(psf_true, "data/psf_true_vis.jpg");
    save_psf_image(psf_est,  "data/psf_est_vis.jpg");

    return 0;
}
