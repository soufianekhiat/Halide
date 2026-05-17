/**
 * optimize_texture.cpp
 *
 * Recovers a texture from a shaded, blurred rendering using Conjugate Gradient
 * with forward-mode AD (JVP) for the forward operator and host-side blur for
 * the adjoint.
 *
 * ── Forward model ────────────────────────────────────────────────────────────
 *   rendered(x,y) = shading(x,y) * blur(texture)(x,y)
 *
 *   The texture is the unknown image to recover.  The shading is a known
 *   procedural pattern (smooth hemisphere falloff).  The blur is a known
 *   Gaussian (sigma = 1.5).
 *
 * ── Showcase: asymmetric operator with JVP + host-side adjoint ──────────────
 *   J * v   = shading * blur(v)     -- one JVP call (forward-mode AD)
 *   J^T * w = blur(shading * w)     -- host-side separable blur
 *
 *   Unlike the symmetric deblur case (deblur_cg.cpp), J != J^T here because
 *   the per-pixel shading weighting breaks symmetry.  The JVP efficiently
 *   computes J * v; the adjoint J^T * w is computed analytically host-side
 *   using the known blur kernel.
 *
 * ── CG on normal equations ───────────────────────────────────────────────────
 *   (J^T J + lambda I) x = J^T * target
 *   A * v = J^T(J v) + lambda v = blur(shading * (shading * blur(v))) + lambda v
 *   Each iteration: 1 JVP call (J*v) + 1 host-side blur (J^T*w)
 *
 * ── Performance ──────────────────────────────────────────────────────────────
 *   ~30-40 CG iterations.  Each: one JVP realization (~100ms) plus one
 *   host-side separable blur (~200ms for 1920x1080).  Total ~10-15s.
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

// ── Reference blur (plain C++) ───────────────────────────────────────────────

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

    Buffer<float> texture_true = load_grayscale(input_path);
    const int W = texture_true.width();
    const int H = texture_true.height();
    printf("Image size: %d x %d\n\n", W, H);

    // ── Generate procedural shading ──────────────────────────────────────
    // Smooth hemisphere falloff: bright center, darker edges.
    // The 0.15 floor prevents near-zero shading (which would make texture
    // recovery ill-conditioned in those regions).
    Buffer<float> shading_buf(W, H);
    {
        float cx = W / 2.0f, cy = H / 2.0f;
        float rad = (float)std::min(W, H) / 2.0f;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float dx = (x - cx) / rad;
                float dy = (y - cy) / rad;
                float r2 = dx * dx + dy * dy;
                shading_buf(x, y) = 0.15f + 0.85f * std::max(0.0f, 1.0f - r2);
            }
    }
    save_grayscale(shading_buf, "data/texture_shading.jpg");
    printf("Shading: hemisphere falloff, floor=0.15\n");

    // ── Compute target rendering: shading * blur(texture_true) ───────────
    const float sigma = 1.5f;
    printf("Blur sigma: %.1f\n", sigma);

    Buffer<float> blurred_texture = reference_blur(texture_true, sigma);
    Buffer<float> target(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            target(x, y) = shading_buf(x, y) * blurred_texture(x, y);
    save_grayscale(target, "data/texture_target.jpg");
    printf("\n");

    // ── Build Halide pipeline: rendered = shading * blur(texture) ────────
    // The pipeline's Jacobian w.r.t. texture is:
    //   J * v = shading * blur(v)
    // which propagate_tangents computes automatically by differentiating
    // through the multiplication and convolution.
    const int krad = (int)std::ceil(3.0f * sigma);

    ImageParam texture_ip(Float(32), 2, "texture");
    texture_ip.set(texture_true);

    Param<float> sigma_p("sigma_p");
    sigma_p.set(sigma);

    Var x("x"), y("y");
    Func norm_kernel("nk");
    norm_kernel(x) = exp(-cast<float>(x * x) / (2.0f * sigma_p * sigma_p))
                     / (sigma_p * sqrt(2.0f * PI_F));

    RDom rx(-krad, 2 * krad + 1, "rx");
    Func blur_x("bx");
    blur_x(x, y) = 0.0f;
    blur_x(x, y) += norm_kernel(rx)
                   * texture_ip(clamp(x + rx, 0, W - 1), y);

    RDom ry(-krad, 2 * krad + 1, "ry");
    Func blur_y("by");
    blur_y(x, y) = 0.0f;
    blur_y(x, y) += norm_kernel(ry) * blur_x(x, clamp(y + ry, 0, H - 1));

    // rendered(x,y) = shading(x,y) * blur(texture)(x,y)
    // shading_buf is a concrete Buffer — its tangent w.r.t. texture is 0,
    // so the JVP correctly differentiates only through the blur(texture) term.
    Func rendered("rendered");
    rendered(x, y) = shading_buf(x, y) * blur_y(x, y);

    norm_kernel.compute_root();
    blur_x.compute_root();
    blur_y.compute_root();

    // ── JVP: J * v = shading * blur(v) ──────────────────────────────────
    Buffer<float> tangent_buf(W, H);
    Func tangent_v("tv");
    tangent_v(x, y) = tangent_buf(x, y);

    printf("Building JVP for rendered w.r.t. texture...\n");
    Func jvp = propagate_tangents(rendered, {{texture_ip.name(), tangent_v}});
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

    // ── CG: (J^T J + lambda I) x = J^T * target ─────────────────────────
    // Each CG iteration:
    //   1. q = J * p = shading * blur(p)        [one JVP realization]
    //   2. Jtq = J^T * q = blur(shading * q)    [host-side blur]
    //   3. Ap = Jtq + lambda * p                 [host-side add]
    const float lambda = 0.005f;
    const int max_cg = 40;
    printf("CG solve: lambda=%.4f, max %d iters\n\n", lambda, max_cg);

    Buffer<float> x_buf(W, H);
    Buffer<float> r_buf(W, H);
    Buffer<float> p_buf(W, H);
    Buffer<float> Ap_buf(W, H);
    Buffer<float> jvp_out(W, H);
    Buffer<float> sw_buf(W, H);

    // rhs = J^T * target = blur(shading * target)
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            sw_buf(xx, yy) = shading_buf(xx, yy) * target(xx, yy);
    Buffer<float> rhs = reference_blur(sw_buf, sigma);

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

        // Step 1: q = J * p = shading * blur(p)   [JVP call]
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++)
                tangent_buf(xx, yy) = p_buf(xx, yy);
        jvp.realize(jvp_out);

        // Step 2: J^T * q = blur(shading * q)   [host-side blur]
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++)
                sw_buf(xx, yy) = shading_buf(xx, yy) * jvp_out(xx, yy);
        Buffer<float> Jtq = reference_blur(sw_buf, sigma);

        // Step 3: Ap = J^T * q + lambda * p
        double pAp = 0.0;
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++) {
                Ap_buf(xx, yy) = Jtq(xx, yy) + lambda * p_buf(xx, yy);
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

        double psnr = compute_psnr(texture_true, x_buf);
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
    double tex_psnr = compute_psnr(texture_true, x_buf);
    printf("\nRecovered texture PSNR: %.2f dB\n", tex_psnr);

    save_grayscale(x_buf, "data/texture_recovered.jpg");

    // Render with the recovered texture for comparison
    Buffer<float> rec_blurred = reference_blur(x_buf, sigma);
    Buffer<float> rec_rendered(W, H);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            rec_rendered(xx, yy) = shading_buf(xx, yy) * rec_blurred(xx, yy);
    save_grayscale(rec_rendered, "data/texture_rendered.jpg");

    double render_psnr = compute_psnr(target, rec_rendered);
    printf("Rendering PSNR:         %.2f dB\n", render_psnr);

    return 0;
}
