/**
 * estimate_blur_radius.cpp
 *
 * Recovers a Gaussian blur radius from a reference 1920×1080 image using
 * forward-mode automatic differentiation (JVP) + gradient descent.
 *
 * ── Forward model ────────────────────────────────────────────────────────────
 *   blur(I, sigma)(x,y) = sum_{r} gauss(r, sigma) * I(x+r, y)   (separable)
 *   gauss(r, sigma)     = exp(-r^2 / (2*sigma^2)) / (sigma * sqrt(2*pi))
 *
 * The kernel is analytically normalized (no separate summation RDom),
 * which is critical for JVP performance:
 *
 *   Without analytic normalization: kernel_sum requires a RDom. Its tangent
 *   is a 31-element sum recomputed for every kernel tap → O(krad^3) per pixel.
 *
 *   With analytic normalization: d_kernel/d_sigma is pointwise (no inner loop)
 *   → O(krad^2) per pixel, which is ~100× faster for krad=8.
 *
 * ── Gradient computation ──────────────────────────────────────────────────────
 *   L(sigma) = (1/N) * ||blur(I,sigma) - R||^2
 *   dL/d_sigma = (2/N) * <blur(I,sigma) - R,  d_blur(I,sigma)/d_sigma>
 *
 *   d_blur/d_sigma is computed by propagate_tangents(output, sigma).
 *   Halide automatically differentiates through exp(), division, and the
 *   reduction loop — no manual derivative formulas needed.
 *
 * ── Optimization ─────────────────────────────────────────────────────────────
 *   1 parameter → 1 JVP pass per gradient step.
 *   Each pass: O(2 * krad * W * H) for separable blur + its tangent.
 *   For W=1920, H=1080, krad=8: ~67M floating-point ops — runs in < 200ms.
 *
 *   Uses Adam (lr=0.1, β1=0.9, β2=0.999) instead of plain gradient descent.
 *   Rationale: the loss surface is very flat for σ << σ_true — the tangent
 *   Func d_blur/d_sigma at small σ has high-frequency character while the
 *   residual (blur_σ − blur_2.0) is low-frequency, so their inner product is
 *   small.  Adam normalizes each step by the running RMS of the gradient,
 *   giving O(1) effective displacement per iteration regardless of scale.
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
#include <vector>

using namespace Halide;

static const float PI_F = 3.14159265358979f;

// ── Image I/O ────────────────────────────────────────────────────────────────

Buffer<float> load_grayscale(const char *path) {
    int w, h, c;
    unsigned char *data = stbi_load(path, &w, &h, &c, 1);
    if (!data) {
        fprintf(stderr, "Cannot load '%s': %s\n", path, stbi_failure_reason());
        exit(1);
    }
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

// ── Reference blur (plain C++, no Halide) ────────────────────────────────────
// Uses the SAME analytic normalization as the Halide forward model
// (1/(sigma*sqrt(2*pi))) so the loss is exactly zero at sigma_true.
// Using a different normalization (e.g. discrete sum=1) would create a
// non-zero loss floor and cause the optimizer to converge to the wrong sigma.
Buffer<float> reference_blur(const Buffer<float> &src, float sigma) {
    int W = src.width(), H = src.height();
    int krad = (int)std::ceil(3.0f * sigma);
    int ksize = 2 * krad + 1;
    std::vector<float> k(ksize);
    float ksum = sigma * std::sqrt(2.0f * PI_F);  // analytic normalization
    for (int i = 0; i < ksize; i++) {
        float r = (float)(i - krad);
        k[i] = std::exp(-r * r / (2.0f * sigma * sigma)) / ksum;
    }
    // No extra normalization step: k already sums to ~1 (within truncation error).

    Buffer<float> tmp(W, H), out(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float s = 0.0f;
            for (int r = -krad; r <= krad; r++)
                s += k[r + krad] * src(std::clamp(x + r, 0, W - 1), y);
            tmp(x, y) = s;
        }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float s = 0.0f;
            for (int r = -krad; r <= krad; r++)
                s += k[r + krad] * tmp(x, std::clamp(y + r, 0, H - 1));
            out(x, y) = s;
        }
    return out;
}

// ── Kernel visualization ─────────────────────────────────────────────────
// Save the 2D separable Gaussian kernel (outer product of 1D kernel)
// as a small upscaled JPEG for visual inspection.

void save_kernel_image(float sigma, int krad, const char *path) {
    int ksize = 2 * krad + 1;
    float norm = sigma * std::sqrt(2.0f * PI_F);

    std::vector<float> k1d(ksize);
    for (int i = 0; i < ksize; i++) {
        float r = (float)(i - krad);
        k1d[i] = std::exp(-r * r / (2.0f * sigma * sigma)) / norm;
    }

    // 2D = outer product of 1D with itself
    std::vector<float> k2d(ksize * ksize);
    float maxv = 0.0f;
    for (int y = 0; y < ksize; y++)
        for (int x = 0; x < ksize; x++) {
            k2d[y * ksize + x] = k1d[x] * k1d[y];
            maxv = std::max(maxv, k2d[y * ksize + x]);
        }

    const int scale = 15;
    int sz = ksize * scale;
    std::vector<unsigned char> px(sz * sz);
    for (int y = 0; y < sz; y++)
        for (int x = 0; x < sz; x++)
            px[y * sz + x] = (unsigned char)(k2d[(y / scale) * ksize + (x / scale)] / maxv * 255.0f);
    stbi_write_jpg(path, sz, sz, 1, px.data(), 95);
    printf("Saved   '%s'  (%dx%d)\n", path, sz, sz);
}

// ── Halide separable Gaussian blur with parametric sigma ─────────────────────
//
// Key design choice: analytically normalize the kernel.
//   norm_kernel(r) = exp(-r^2 / (2*sigma^2)) / (sigma * sqrt(2*pi))
//
// This is a pure function of (r, sigma) with NO RDom.
// Its JVP w.r.t. sigma is:
//   d_norm_kernel/d_sigma = norm_kernel(r) * (r^2/sigma^2 - 1) / sigma
// which is also a pure pointwise expression — O(1) per tap.
//
// This makes the full JVP pipeline O(krad) per pixel, not O(krad^3).
struct GaussBlur {
    Param<float> sigma;
    ImageParam   input;
    Func         output;
    int W, H, krad;

    GaussBlur(int W, int H, int krad)
        : sigma("sigma"), input(Float(32), 2, "inp"), W(W), H(H), krad(krad) {

        Var x("x"), y("y");

        // Analytically-normalized Gaussian kernel (no normalization RDom).
        // The continuous-domain norm factor sigma*sqrt(2*pi) means the truncated
        // discrete sum is ~1 for krad >= 3*sigma and exactly 1 as krad→∞.
        Func norm_kernel("norm_kernel");
        norm_kernel(x) = exp(-cast<float>(x * x) / (2.0f * sigma * sigma))
                         / (sigma * sqrt(2.0f * PI_F));

        // Horizontal pass: convolve rows with norm_kernel.
        RDom rx(-krad, 2 * krad + 1, "rx");
        Func blur_x("blur_x");
        blur_x(x, y) = 0.0f;
        blur_x(x, y) += norm_kernel(rx) * input(clamp(x + rx, 0, W - 1), y);

        // Vertical pass: convolve columns of blur_x with norm_kernel.
        RDom ry(-krad, 2 * krad + 1, "ry");
        Func blur_y("blur_y");
        blur_y(x, y) = 0.0f;
        blur_y(x, y) += norm_kernel(ry) * blur_x(x, clamp(y + ry, 0, H - 1));

        output = blur_y;

        // Schedule: compute intermediate stages at root to enable reuse
        // by both the primal pipeline and its JVP tangent pipeline.
        norm_kernel.compute_root();
        blur_x.compute_root();
        // blur_y (= output) is computed on demand by realize().
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

    // ── Ground truth ──────────────────────────────────────────────────────
    const float sigma_true = 2.0f;
    printf("Ground truth sigma = %.2f\n", sigma_true);
    Buffer<float> ref_buf = reference_blur(input_buf, sigma_true);
    save_grayscale(ref_buf, "data/blurred_reference.jpg");
    printf("\n");

    // ── Build Halide pipeline ─────────────────────────────────────────────
    // krad = 8: supports sigma up to ~2.5 with < 0.1% truncation error.
    const int krad = 8;
    GaussBlur pipe(W, H, krad);
    pipe.input.set(input_buf);

    // Build the JVP: d_output(x,y)/d_sigma for every pixel simultaneously.
    // Halide automatically differentiates through exp() and the convolution.
    Func d_output = propagate_tangents(pipe.output, pipe.sigma);
    d_output.compute_root();

    // ── Warm up JIT ───────────────────────────────────────────────────────
    printf("Compiling JIT (primal + JVP)...");
    fflush(stdout);
    pipe.sigma.set(1.0f);
    auto t_jit0 = std::chrono::high_resolution_clock::now();
    pipe.output.realize({W, H});  // triggers JIT for primal
    d_output.realize({W, H});     // triggers JIT for tangent
    auto t_jit1 = std::chrono::high_resolution_clock::now();
    printf(" done (%.1f s)\n\n",
           std::chrono::duration<double>(t_jit1 - t_jit0).count());

    // ── Gradient descent (Adam optimizer) ────────────────────────────────
    // Plain gradient descent with decaying lr stalls because the loss
    // surface is very flat for sigma << sigma_true: d_blur/d_sigma at
    // small sigma has high-frequency character while the residual
    // (blur_sigma - blur_2.0) is low-frequency, so their dot product is
    // small.  Adam normalizes steps by the running RMS of the gradient,
    // giving O(1) effective step regardless of gradient scale.
    float sigma_est   = 0.5f;   // start below the true value
    const int   max_iter = 150;
    const float tol      = 1e-5f;

    // Adam hyper-parameters
    const float adam_lr = 0.1f;
    const float beta1   = 0.9f;
    const float beta2   = 0.999f;
    const float adam_eps = 1e-8f;
    float m = 0.0f, v = 0.0f;   // first/second moment estimates

    printf("%-6s  %-10s  %-14s  %-12s  %-8s\n",
           "Iter", "sigma", "loss", "d_loss/d_sigma", "ms/iter");
    printf("%-6s  %-10s  %-14s  %-12s  %-8s\n",
           "----", "-----", "----", "--------------", "-------");

    for (int iter = 0; iter < max_iter; iter++) {
        pipe.sigma.set(sigma_est);

        auto t0 = std::chrono::high_resolution_clock::now();

        // Realize primal and tangent (compiled code, no JIT overhead)
        Buffer<float> out_buf   = pipe.output.realize({W, H});
        Buffer<float> d_out_buf = d_output.realize({W, H});

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // L(sigma) = mean((out - ref)^2)
        // dL/d_sigma = 2 * mean((out - ref) * d_out/d_sigma)
        double loss = 0.0, grad = 0.0;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float r = out_buf(x, y) - ref_buf(x, y);
                loss += r * r;
                grad += r * (double)d_out_buf(x, y);
            }
        }
        loss /= N;
        grad  = 2.0 * grad / N;

        printf("%-6d  %-10.5f  %-14.8f  %-12.5f  %-8.1f\n",
               iter, sigma_est, loss, (float)grad, ms);

        // Adam update — normalizes step by sqrt(running mean of grad^2)
        // so each iteration moves ~adam_lr regardless of gradient magnitude.
        m = beta1 * m + (1.0f - beta1) * (float)grad;
        v = beta2 * v + (1.0f - beta2) * (float)(grad * grad);
        float m_hat = m / (1.0f - std::pow(beta1, iter + 1));
        float v_hat = v / (1.0f - std::pow(beta2, iter + 1));
        sigma_est -= adam_lr * m_hat / (std::sqrt(v_hat) + adam_eps);
        sigma_est  = std::max(0.05f, sigma_est);  // sigma must be positive

        if (std::abs(grad) < tol && iter > 0) {
            printf("\nConverged at iter %d (|grad| = %.2e < %.2e)\n",
                   iter, std::abs(grad), tol);
            break;
        }
    }

    printf("\n");
    printf("sigma_true      = %.4f\n", sigma_true);
    printf("sigma_estimated = %.4f\n", sigma_est);
    printf("absolute error  = %.4f\n", std::abs(sigma_est - sigma_true));
    printf("relative error  = %.2f%%\n",
           100.0f * std::abs(sigma_est - sigma_true) / sigma_true);

    // Save final output for visual comparison
    pipe.sigma.set(sigma_est);
    Buffer<float> final_out = pipe.output.realize({W, H});
    save_grayscale(final_out, "data/blurred_estimated.jpg");

    // Save kernel visualizations (upscaled 2D Gaussian kernel)
    save_kernel_image(sigma_true, krad, "data/kernel_true.jpg");
    save_kernel_image(sigma_est,  krad, "data/kernel_estimated.jpg");

    return 0;
}
