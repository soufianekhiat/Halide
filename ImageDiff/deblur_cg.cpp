/**
 * deblur_cg.cpp
 *
 * Non-blind image deblurring via Conjugate Gradient, using forward-mode AD
 * (JVP) to apply the blur operator.
 *
 * ── Showcase: ImageParam JVP ────────────────────────────────────────────────
 *   propagate_tangents(output, {{input.name(), tangent_v}}) differentiates
 *   the blur pipeline w.r.t. the INPUT IMAGE — producing J * v = blur(v)
 *   for any direction v in a single forward pass.
 *
 *   This is the Buffer/ImageParam overload of forward-mode AD.  Unlike the
 *   scalar-Param demos (estimate_blur_radius, estimate_psf, estimate_tone_curve),
 *   which differentiate w.r.t. a scalar Param<float>, this demo differentiates
 *   w.r.t. an entire image (ImageParam of size W x H).
 *
 * ── Key insight: symmetric blur → J = J^T ──────────────────────────────────
 *   For a symmetric Gaussian kernel, the blur matrix is symmetric:
 *     J = J^T, so A*v = J^T*(J*v) + lambda*v = blur(blur(v)) + lambda*v
 *   Each CG iteration needs just TWO JVP calls (no reverse-mode AD needed).
 *
 * ── Problem ──────────────────────────────────────────────────────────────────
 *   Given: b = blur(x_true) + noise   (blurred noisy observation)
 *   Solve: min_x ||blur(x) - b||^2 + lambda ||x||^2   (Tikhonov)
 *   Via CG on normal equations: (J^T J + lambda I) x = J^T b
 *
 * ── Performance ──────────────────────────────────────────────────────────────
 *   ~30-40 CG iterations × 2 JVP calls = ~60-80 pipeline realizations.
 *   At ~100-200 ms per realization (1920×1080, krad=6): ~10-15 s total.
 *   Typical PSNR improvement: ~4-8 dB over the blurred input.
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
#include <random>
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

// ── Reference blur (plain C++) ───────────────────────────────────────────────
// Uses the SAME analytic normalization as the Halide pipeline.

Buffer<float> reference_blur(const Buffer<float> &src, float sigma) {
    int W = src.width(), H = src.height();
    int krad = (int)std::ceil(3.0f * sigma);
    int ksize = 2 * krad + 1;
    std::vector<float> k(ksize);
    float ksum = sigma * std::sqrt(2.0f * PI_F);
    for (int i = 0; i < ksize; i++) {
        float r = (float)(i - krad);
        k[i] = std::exp(-r * r / (2.0f * sigma * sigma)) / ksum;
    }
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

// ── PSNR ─────────────────────────────────────────────────────────────────────

double compute_psnr(const Buffer<float> &a, const Buffer<float> &b) {
    int W = a.width(), H = a.height();
    double mse = 0.0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            double d = a(x, y) - b(x, y);
            mse += d * d;
        }
    mse /= (double)(W * H);
    if (mse < 1e-16) return 99.0;
    return 10.0 * std::log10(1.0 / mse);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    const char *input_path = argc > 1 ? argv[1] : "data/reference_input.jpg";

    Buffer<float> sharp_buf = load_grayscale(input_path);
    const int W = sharp_buf.width();
    const int H = sharp_buf.height();
    printf("Image size: %d x %d\n\n", W, H);

    // ── Generate blurred + noisy observation ─────────────────────────────
    const float sigma = 1.5f;
    const float noise_sigma = 0.002f;
    printf("Blur sigma:  %.1f\n", sigma);
    printf("Noise sigma: %.3f\n", noise_sigma);

    Buffer<float> observed = reference_blur(sharp_buf, sigma);
    std::mt19937 rng(42);
    std::normal_distribution<float> ndist(0.0f, noise_sigma);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            observed(x, y) = std::clamp(observed(x, y) + ndist(rng), 0.0f, 1.0f);
    save_grayscale(observed, "data/deblur_observed.jpg");
    double input_psnr = compute_psnr(sharp_buf, observed);
    printf("Input PSNR:  %.2f dB  (blurred+noisy vs sharp)\n\n", input_psnr);

    // ── Build Halide blur pipeline ───────────────────────────────────────
    // The JVP of this pipeline w.r.t. the input image computes blur(v)
    // for any tangent direction v — the Jacobian-vector product of the
    // blur operator.  For a linear pipeline, J*v = blur(v) regardless of
    // the point at which the derivative is evaluated.
    const int krad = (int)std::ceil(3.0f * sigma);

    ImageParam inp(Float(32), 2, "inp");
    inp.set(observed);

    Param<float> sigma_p("sigma_p");
    sigma_p.set(sigma);

    Var x("x"), y("y");
    Func norm_kernel("nk");
    norm_kernel(x) = exp(-cast<float>(x * x) / (2.0f * sigma_p * sigma_p))
                     / (sigma_p * sqrt(2.0f * PI_F));

    RDom rx(-krad, 2 * krad + 1, "rx");
    Func blur_x("bx");
    blur_x(x, y) = 0.0f;
    blur_x(x, y) += norm_kernel(rx) * inp(clamp(x + rx, 0, W - 1), y);

    RDom ry(-krad, 2 * krad + 1, "ry");
    Func blur_y("by");
    blur_y(x, y) = 0.0f;
    blur_y(x, y) += norm_kernel(ry) * blur_x(x, clamp(y + ry, 0, H - 1));

    norm_kernel.compute_root();
    blur_x.compute_root();

    // ── JVP w.r.t. input image ───────────────────────────────────────────
    // tangent_buf holds the "direction" vector v.  tangent_v wraps it as a
    // Halide Func so propagate_tangents can use it as the tangent input.
    // Between JVP realizations, we update tangent_buf contents in C++;
    // the JVP pipeline reads the new values automatically.
    Buffer<float> tangent_buf(W, H);
    Func tangent_v("tv");
    tangent_v(x, y) = tangent_buf(x, y);

    printf("Building JVP for blur w.r.t. input image...\n");
    Func jvp = propagate_tangents(blur_y, {{inp.name(), tangent_v}});
    jvp.compute_root();

    // ── JIT compile ──────────────────────────────────────────────────────
    printf("Compiling JIT...");
    fflush(stdout);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            tangent_buf(xx, yy) = 0.0f;
    auto t_jit0 = std::chrono::high_resolution_clock::now();
    jvp.realize({W, H});
    auto t_jit1 = std::chrono::high_resolution_clock::now();
    printf(" done (%.1f s)\n\n",
           std::chrono::duration<double>(t_jit1 - t_jit0).count());

    // ── Conjugate Gradient ───────────────────────────────────────────────
    // Solve (J^T J + lambda I) x = J^T b
    // where J = blur, J^T = J (symmetric Gaussian), b = observed image.
    //
    // Each CG iteration computes A*p = blur(blur(p)) + lambda*p via:
    //   1. Copy p into tangent_buf → realize JVP → tmp = blur(p)
    //   2. Copy tmp into tangent_buf → realize JVP → Ap = blur(blur(p))
    //   3. Ap += lambda * p
    const float lambda = 0.0005f;
    const int max_cg = 50;
    printf("CG solve: (blur^2 + %.3f*I) x = blur(b),  max %d iters\n\n",
           lambda, max_cg);

    Buffer<float> x_buf(W, H);
    Buffer<float> r_buf(W, H);
    Buffer<float> p_buf(W, H);
    Buffer<float> Ap_buf(W, H);
    Buffer<float> tmp_buf(W, H);

    // rhs = J^T * b = blur(b)  [since J^T = J for symmetric kernel]
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            tangent_buf(xx, yy) = observed(xx, yy);
    Buffer<float> rhs(W, H);
    jvp.realize(rhs);

    // x = 0, r = rhs, p = r
    double rr = 0.0;
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++) {
            x_buf(xx, yy) = 0.0f;
            r_buf(xx, yy) = rhs(xx, yy);
            p_buf(xx, yy) = rhs(xx, yy);
            rr += (double)rhs(xx, yy) * rhs(xx, yy);
        }
    double rr0 = rr;

    printf("%-4s  %-12s  %-10s  %-8s\n",
           "Iter", "||r||/||r0||", "PSNR", "ms/iter");
    printf("%-4s  %-12s  %-10s  %-8s\n",
           "----", "------------", "----", "-------");

    auto t_cg0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_cg; iter++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Ap = blur(blur(p)) + lambda * p
        // Step 1: tmp = blur(p)
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++)
                tangent_buf(xx, yy) = p_buf(xx, yy);
        jvp.realize(tmp_buf);

        // Step 2: Ap = blur(tmp) = blur(blur(p))
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++)
                tangent_buf(xx, yy) = tmp_buf(xx, yy);
        jvp.realize(Ap_buf);

        // Step 3: Ap += lambda * p
        double pAp = 0.0;
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++) {
                Ap_buf(xx, yy) += lambda * p_buf(xx, yy);
                pAp += (double)p_buf(xx, yy) * Ap_buf(xx, yy);
            }

        float alpha = (float)(rr / pAp);

        double rr_new = 0.0;
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++) {
                x_buf(xx, yy) += alpha * p_buf(xx, yy);
                r_buf(xx, yy) -= alpha * Ap_buf(xx, yy);
                rr_new += (double)r_buf(xx, yy) * r_buf(xx, yy);
            }

        float beta = (float)(rr_new / rr);
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++)
                p_buf(xx, yy) = r_buf(xx, yy) + beta * p_buf(xx, yy);

        rr = rr_new;

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double psnr = compute_psnr(sharp_buf, x_buf);
        printf("%-4d  %-12.4e  %-10.2f  %-8.0f\n",
               iter, std::sqrt(rr / rr0), psnr, ms);

        if (std::sqrt(rr / rr0) < 1e-4) {
            printf("\nConverged at CG iteration %d\n", iter);
            break;
        }
    }

    auto t_cg1 = std::chrono::high_resolution_clock::now();
    printf("\nCG wall time: %.1f s\n",
           std::chrono::duration<double>(t_cg1 - t_cg0).count());

    // ── Results ──────────────────────────────────────────────────────────
    double recovered_psnr = compute_psnr(sharp_buf, x_buf);
    printf("\nInput PSNR:     %.2f dB\n", input_psnr);
    printf("Recovered PSNR: %.2f dB\n", recovered_psnr);
    printf("PSNR gain:      %.2f dB\n", recovered_psnr - input_psnr);

    save_grayscale(x_buf, "data/deblur_recovered.jpg");

    return 0;
}
