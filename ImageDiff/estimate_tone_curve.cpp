/**
 * estimate_tone_curve.cpp
 *
 * Recovers a parametric tone curve (gamma + gain) from a reference image
 * using forward-mode AD (JVP) + gradient descent.
 *
 * ── Showcase: per-pixel sensitivity maps ─────────────────────────────────
 *   The JVP produces a full-resolution TANGENT IMAGE d(output)/d(param)
 *   for every pixel, in a single forward pass.  These tangent images are
 *   saved as "sensitivity maps" showing how each pixel responds to a
 *   parameter change — exactly what a photographer's slider preview does.
 *
 *   d/d_gamma(x,y) = gain * pixel^gamma * log(pixel)
 *     "Gamma sensitivity" — peaks at mid-tones (pixel ≈ 1/e), zero at
 *     pure black/white.  Shows which pixels change most when the tone
 *     curve shape is adjusted.
 *
 *   d/d_gain(x,y) = pixel^gamma
 *     "Gain sensitivity" — the gamma-corrected image itself.  Bright
 *     pixels are most affected by a brightness change.
 *
 * ── Forward model ────────────────────────────────────────────────────────
 *   output(x,y) = gain * pow(max(input(x,y), eps), gamma)
 *
 * ── Performance ──────────────────────────────────────────────────────────
 *   Pointwise operation → O(W*H) per JVP pass (no convolution).
 *   2 params → 2 JVP passes per gradient step → ~2 ms/iter for 720×1080.
 *   Converges in ~30 iterations (<1 s total optimization, excluding JIT).
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

// Save a tangent image as a normalized grayscale sensitivity map.
// abs(tangent) is mapped to [0, 255] so peak sensitivity = white.
void save_tangent_image(const Buffer<float> &buf, const char *path) {
    int W = buf.width(), H = buf.height();
    float maxabs = 0.0f;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            maxabs = std::max(maxabs, std::abs(buf(x, y)));
    if (maxabs < 1e-8f) maxabs = 1.0f;

    std::vector<unsigned char> px(W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            px[y * W + x] = (unsigned char)(std::abs(buf(x, y)) / maxabs * 255.0f);
    stbi_write_jpg(path, W, H, 1, px.data(), 92);
    printf("Saved   '%s'  (tangent range: [%.4f, %.4f])\n", path,
           -maxabs, maxabs);
}

// ── Halide parametric tone curve ─────────────────────────────────────────────

struct ToneCurvePipeline {
    Param<float> gamma, gain;
    ImageParam   input;
    Func         output;

    ToneCurvePipeline()
        : gamma("gamma"), gain("gain"), input(Float(32), 2, "inp") {
        Var x("x"), y("y");
        // Clamp input away from zero so log(pixel) is finite.
        Expr pixel = max(input(x, y), 0.001f);
        output(x, y) = gain * pow(pixel, gamma);
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
    // Simulate a "cinematic lift" look: gamma < 1 brightens shadows,
    // gain < 1 pulls back overall brightness to avoid clipping.
    const float gamma_true = 0.6f;
    const float gain_true  = 0.9f;

    printf("Ground truth:  gamma = %.2f,  gain = %.2f\n", gamma_true, gain_true);

    // Compute reference in float (no JPEG clipping for loss computation)
    Buffer<float> ref_buf(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            ref_buf(x, y) = gain_true
                * std::pow(std::max(input_buf(x, y), 0.001f), gamma_true);
    save_grayscale(ref_buf, "data/tone_reference.jpg");
    printf("\n");

    // ── Build Halide pipeline ─────────────────────────────────────────────
    ToneCurvePipeline pipe;
    pipe.input.set(input_buf);

    // Build JVP for each parameter: one forward pass per parameter
    // produces a full-resolution tangent image.
    Func d_gamma  = propagate_tangents(pipe.output, pipe.gamma);
    Func d_gain   = propagate_tangents(pipe.output, pipe.gain);
    d_gamma.compute_root();
    d_gain.compute_root();

    // ── JIT compile ───────────────────────────────────────────────────────
    printf("Compiling JIT (primal + 2 JVP tangents)...");
    fflush(stdout);
    pipe.gamma.set(1.0f);
    pipe.gain.set(1.0f);
    auto t_jit0 = std::chrono::high_resolution_clock::now();
    pipe.output.realize({W, H});
    d_gamma.realize({W, H});
    d_gain.realize({W, H});
    auto t_jit1 = std::chrono::high_resolution_clock::now();
    printf(" done (%.1f s)\n\n",
           std::chrono::duration<double>(t_jit1 - t_jit0).count());

    // ── Save tangent images at the initial estimate ───────────────────────
    // These are the "sensitivity maps": each pixel shows how much the
    // output would change if gamma (or gain) were nudged by a tiny amount.
    printf("=== Sensitivity maps at initial estimate (gamma=1, gain=1) ===\n");
    {
        Buffer<float> t_gamma = d_gamma.realize({W, H});
        Buffer<float> t_gain  = d_gain.realize({W, H});
        save_tangent_image(t_gamma, "data/tangent_gamma_init.jpg");
        save_tangent_image(t_gain,  "data/tangent_gain_init.jpg");
    }
    printf("\n");

    // ── Gradient descent (Adam) ───────────────────────────────────────────
    float gamma_est = 1.0f;   // start at identity (no correction)
    float gain_est  = 1.0f;

    const int   max_iter = 60;
    const float tol      = 1e-6f;

    // Adam per-parameter state
    const float adam_lr = 0.05f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float m_gamma = 0, v_gamma = 0;
    float m_gain  = 0, v_gain  = 0;

    printf("%-6s  %-8s  %-8s  %-14s  %-8s\n",
           "Iter", "gamma", "gain", "loss", "ms/iter");
    printf("%-6s  %-8s  %-8s  %-14s  %-8s\n",
           "----", "-----", "----", "----", "-------");

    for (int iter = 0; iter < max_iter; iter++) {
        pipe.gamma.set(gamma_est);
        pipe.gain.set(gain_est);

        auto t0 = std::chrono::high_resolution_clock::now();

        Buffer<float> out_buf     = pipe.output.realize({W, H});
        Buffer<float> d_gamma_buf = d_gamma.realize({W, H});
        Buffer<float> d_gain_buf  = d_gain.realize({W, H});

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Compute loss and gradients
        double loss = 0.0, g_gamma = 0.0, g_gain = 0.0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float r = out_buf(x, y) - ref_buf(x, y);
                loss    += r * r;
                g_gamma += r * (double)d_gamma_buf(x, y);
                g_gain  += r * (double)d_gain_buf(x, y);
            }
        loss    /= N;
        g_gamma  = 2.0 * g_gamma / N;
        g_gain   = 2.0 * g_gain  / N;

        printf("%-6d  %-8.4f  %-8.4f  %-14.8f  %-8.1f\n",
               iter, gamma_est, gain_est, loss, ms);

        // Adam update for gamma
        m_gamma = beta1 * m_gamma + (1 - beta1) * (float)g_gamma;
        v_gamma = beta2 * v_gamma + (1 - beta2) * (float)(g_gamma * g_gamma);
        float mh_g = m_gamma / (1 - std::pow(beta1, iter + 1));
        float vh_g = v_gamma / (1 - std::pow(beta2, iter + 1));
        gamma_est -= adam_lr * mh_g / (std::sqrt(vh_g) + eps);
        gamma_est  = std::max(0.1f, gamma_est);

        // Adam update for gain
        m_gain = beta1 * m_gain + (1 - beta1) * (float)g_gain;
        v_gain = beta2 * v_gain + (1 - beta2) * (float)(g_gain * g_gain);
        float mh_b = m_gain / (1 - std::pow(beta1, iter + 1));
        float vh_b = v_gain / (1 - std::pow(beta2, iter + 1));
        gain_est -= adam_lr * mh_b / (std::sqrt(vh_b) + eps);
        gain_est  = std::max(0.1f, gain_est);

        if (std::abs(g_gamma) < tol && std::abs(g_gain) < tol && iter > 0) {
            printf("\nConverged at iter %d\n", iter);
            break;
        }
    }

    // ── Results ───────────────────────────────────────────────────────────
    printf("\n");
    printf("gamma_true      = %.4f    gain_true      = %.4f\n",
           gamma_true, gain_true);
    printf("gamma_estimated = %.4f    gain_estimated = %.4f\n",
           gamma_est, gain_est);
    printf("gamma error     = %.4f    gain error     = %.4f\n",
           std::abs(gamma_est - gamma_true), std::abs(gain_est - gain_true));
    printf("gamma rel error = %.2f%%   gain rel error = %.2f%%\n",
           100.0f * std::abs(gamma_est - gamma_true) / gamma_true,
           100.0f * std::abs(gain_est  - gain_true)  / gain_true);

    // ── Save final outputs and tangent images ─────────────────────────────
    pipe.gamma.set(gamma_est);
    pipe.gain.set(gain_est);
    save_grayscale(pipe.output.realize({W, H}), "data/tone_estimated.jpg");

    // Tangent images at the recovered estimate — compare with initial
    printf("\n=== Sensitivity maps at recovered estimate "
           "(gamma=%.2f, gain=%.2f) ===\n", gamma_est, gain_est);
    {
        Buffer<float> t_gamma = d_gamma.realize({W, H});
        Buffer<float> t_gain  = d_gain.realize({W, H});
        save_tangent_image(t_gamma, "data/tangent_gamma_final.jpg");
        save_tangent_image(t_gain,  "data/tangent_gain_final.jpg");
    }

    return 0;
}
