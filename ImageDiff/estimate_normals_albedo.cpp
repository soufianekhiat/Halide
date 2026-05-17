/**
 * estimate_normals_albedo.cpp
 *
 * Joint recovery of a normal map and diffuse albedo texture from RGB
 * renderings under 3 known directional lights (photometric stereo).
 *
 * ── Showcase: 3D ImageParam JVP ─────────────────────────────────────────────
 *   propagate_tangents(rendered, {{normals_ip.name(), tangent_v}})
 *   differentiates a nonlinear Lambertian shading pipeline w.r.t. a
 *   3-channel normal-map ImageParam (W x H x 3).
 *
 *   3 JVP calls with basis-vector tangents (one per normal component) give
 *   the full per-pixel 3x3 Jacobian columns.  This is efficient because
 *   the forward model is pointwise in (x,y) -- each pixel's normal only
 *   affects that pixel's rendering.
 *
 *   The light direction is a Param<float>, so one JVP pipeline handles
 *   all 3 lights by changing the Param value between realize() calls.
 *   9 JVP calls per iteration (3 lights x 3 basis vectors).
 *
 * ── Forward model ───────────────────────────────────────────────────────────
 *   rendered_l(x,y,c) = albedo(x,y,c) * max(dot(normalize(normals(x,y,:)), L_l), 0)
 *
 *   The albedo is a concrete Buffer (tangent = 0), so the JVP only
 *   differentiates through the normals path.
 *
 * ── Optimization ────────────────────────────────────────────────────────────
 *   Adam for both normals and albedo.  Normal gradient via JVP (3 basis
 *   vectors x 3 lights = 9 JVP calls).  Albedo gradient is analytical
 *   (rendering is linear in albedo).
 *
 * ── Ground truth ────────────────────────────────────────────────────────────
 *   512 x 512, four quadrants:
 *     Top-left:     flat normal (0,0,1),              red albedo
 *     Top-right:    wavy sin/cos ripples,             green albedo
 *     Bottom-left:  oblique plane (0.3, -0.2, 1),     blue albedo
 *     Bottom-right: hemisphere normal map,            yellow albedo
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

// ── Image I/O (RGB) ─────────────────────────────────────────────────────────

void save_rgb(const Buffer<float> &buf, const char *path) {
    int W = buf.width(), H = buf.height();
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                px[(y * W + x) * 3 + c] =
                    (unsigned char)std::clamp(buf(x, y, c) * 255.0f, 0.0f, 255.0f);
    stbi_write_jpg(path, W, H, 3, px.data(), 92);
    printf("Saved   '%s'\n", path);
}

void save_normal_map(const Buffer<float> &normals, const char *path) {
    int W = normals.width(), H = normals.height();
    std::vector<unsigned char> px(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float nx = normals(x, y, 0);
            float ny = normals(x, y, 1);
            float nz = normals(x, y, 2);
            float len = std::sqrt(nx * nx + ny * ny + nz * nz + 1e-8f);
            px[(y * W + x) * 3 + 0] = (unsigned char)std::clamp((nx / len * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
            px[(y * W + x) * 3 + 1] = (unsigned char)std::clamp((ny / len * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
            px[(y * W + x) * 3 + 2] = (unsigned char)std::clamp((nz / len * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
        }
    stbi_write_jpg(path, W, H, 3, px.data(), 92);
    printf("Saved   '%s'\n", path);
}

// ── PSNR (RGB) ──────────────────────────────────────────────────────────────

double compute_psnr_rgb(const Buffer<float> &a, const Buffer<float> &b) {
    int W = a.width(), H = a.height();
    double mse = 0.0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++) {
                double d = a(x, y, c) - b(x, y, c);
                mse += d * d;
            }
    mse /= (double)(W * H * 3);
    if (mse < 1e-16) return 99.0;
    return 10.0 * std::log10(1.0 / mse);
}

// ── Ground truth generation ─────────────────────────────────────────────────

void generate_ground_truth(int W, int H,
                           Buffer<float> &normals_out,
                           Buffer<float> &albedo_out) {
    normals_out = Buffer<float>(W, H, 3);
    albedo_out = Buffer<float>(W, H, 3);

    int hw = W / 2, hh = H / 2;

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float nx, ny, nz;
            float ar, ag, ab;

            if (x < hw && y < hh) {
                // Top-left: flat surface, red
                nx = 0.0f; ny = 0.0f; nz = 1.0f;
                ar = 0.9f; ag = 0.2f; ab = 0.1f;
            } else if (x >= hw && y < hh) {
                // Top-right: wavy surface, green
                float fx = (float)(x - hw);
                float fy = (float)y;
                nx = 0.3f * std::sin(2.0f * PI_F * fx / 64.0f);
                ny = 0.3f * std::cos(2.0f * PI_F * fy / 64.0f);
                nz = 1.0f;
                ar = 0.2f; ag = 0.8f; ab = 0.2f;
            } else if (x < hw && y >= hh) {
                // Bottom-left: oblique plane, blue
                nx = 0.3f; ny = -0.2f; nz = 1.0f;
                ar = 0.2f; ag = 0.3f; ab = 0.9f;
            } else {
                // Bottom-right: hemisphere, yellow
                float cx = hw / 2.0f;
                float cy = hh / 2.0f;
                float rad = (float)std::min(hw, hh) / 2.0f;
                float dx = ((float)(x - hw) - cx) / rad;
                float dy = ((float)(y - hh) - cy) / rad;
                float r2 = dx * dx + dy * dy;
                if (r2 < 1.0f) {
                    nx = dx;
                    ny = dy;
                    nz = std::sqrt(1.0f - r2);
                } else {
                    nx = 0.0f; ny = 0.0f; nz = 1.0f;
                }
                ar = 0.9f; ag = 0.8f; ab = 0.2f;
            }

            // Normalize
            float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            normals_out(x, y, 0) = nx / len;
            normals_out(x, y, 1) = ny / len;
            normals_out(x, y, 2) = nz / len;

            albedo_out(x, y, 0) = ar;
            albedo_out(x, y, 1) = ag;
            albedo_out(x, y, 2) = ab;
        }
}

// ── Host-side rendering ─────────────────────────────────────────────────────

Buffer<float> render_host(const Buffer<float> &normals,
                          const Buffer<float> &albedo,
                          float Lx, float Ly, float Lz) {
    int W = normals.width(), H = normals.height();
    Buffer<float> out(W, H, 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float ndotl = normals(x, y, 0) * Lx +
                          normals(x, y, 1) * Ly +
                          normals(x, y, 2) * Lz;
            ndotl = std::max(ndotl, 0.0f);
            for (int c = 0; c < 3; c++)
                out(x, y, c) = albedo(x, y, c) * ndotl;
        }
    return out;
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    const int W = 512, H = 512;
    printf("Image size: %d x %d (RGB)\n\n", W, H);

    // ── Generate ground truth ────────────────────────────────────────────
    Buffer<float> normals_true, albedo_true;
    generate_ground_truth(W, H, normals_true, albedo_true);
    save_normal_map(normals_true, "data/normals_true.jpg");
    save_rgb(albedo_true, "data/albedo_true.jpg");

    // ── 3 light directions (photometric stereo) ──────────────────────────
    const int NUM_LIGHTS = 3;
    float lights[NUM_LIGHTS][3] = {
        { 1.0f,  1.0f,  2.0f},   // front-right
        {-1.0f,  0.5f,  2.0f},   // front-left
        { 0.0f, -1.0f,  2.0f},   // back-center
    };
    // Normalize
    for (int l = 0; l < NUM_LIGHTS; l++) {
        float len = std::sqrt(lights[l][0]*lights[l][0] +
                              lights[l][1]*lights[l][1] +
                              lights[l][2]*lights[l][2]);
        lights[l][0] /= len; lights[l][1] /= len; lights[l][2] /= len;
    }
    printf("Light directions (3-light photometric stereo):\n");
    for (int l = 0; l < NUM_LIGHTS; l++)
        printf("  L%d = (%.4f, %.4f, %.4f)\n", l, lights[l][0], lights[l][1], lights[l][2]);

    // ── Render target images ─────────────────────────────────────────────
    Buffer<float> targets[NUM_LIGHTS];
    const char *target_paths[NUM_LIGHTS] = {
        "data/normals_target_L0.jpg",
        "data/normals_target_L1.jpg",
        "data/normals_target_L2.jpg",
    };
    for (int l = 0; l < NUM_LIGHTS; l++) {
        targets[l] = render_host(normals_true, albedo_true,
                                 lights[l][0], lights[l][1], lights[l][2]);
        save_rgb(targets[l], target_paths[l]);
    }
    printf("\n");

    // ── Build Halide pipeline ────────────────────────────────────────────
    // Light direction as Params -- change between realize() calls to
    // handle different lights with one compiled pipeline.
    ImageParam normals_ip(Float(32), 3, "normals");
    Param<float> light_x("lx"), light_y("ly"), light_z("lz");
    Var x("x"), y("y"), c("c");

    // Normalize the normal vector
    Expr nx = normals_ip(x, y, 0);
    Expr ny = normals_ip(x, y, 1);
    Expr nz = normals_ip(x, y, 2);
    Expr len = sqrt(nx * nx + ny * ny + nz * nz + 1e-8f);

    Func nn("nn");
    nn(x, y, c) = select(c == 0, nx / len,
                         c == 1, ny / len,
                                 nz / len);

    // NdotL
    Func ndotl("ndotl");
    ndotl(x, y) = max(nn(x, y, 0) * light_x + nn(x, y, 1) * light_y
                    + nn(x, y, 2) * light_z, 0.0f);

    // Albedo buffer (concrete -- not differentiated by JVP)
    Buffer<float> albedo_est(W, H, 3);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            for (int cc = 0; cc < 3; cc++)
                albedo_est(xx, yy, cc) = 0.5f;

    // Rendered image
    Func rendered("rendered");
    rendered(x, y, c) = albedo_est(x, y, c) * ndotl(x, y);

    nn.compute_root();
    ndotl.compute_root();

    // ── JVP w.r.t. normals ───────────────────────────────────────────────
    Buffer<float> tangent_buf(W, H, 3);
    Func tangent_v("tv");
    tangent_v(x, y, c) = tangent_buf(x, y, c);

    printf("Building JVP for rendered w.r.t. normals...\n");
    Func jvp = propagate_tangents(rendered, {{normals_ip.name(), tangent_v}});
    jvp.compute_root();

    // ── JIT compile ──────────────────────────────────────────────────────
    Buffer<float> normals_est(W, H, 3);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++) {
            normals_est(xx, yy, 0) = 0.0f;
            normals_est(xx, yy, 1) = 0.0f;
            normals_est(xx, yy, 2) = 1.0f;
        }
    normals_ip.set(normals_est);

    // Set initial light for JIT compile
    light_x.set(lights[0][0]);
    light_y.set(lights[0][1]);
    light_z.set(lights[0][2]);

    printf("Compiling JIT...");
    fflush(stdout);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            for (int cc = 0; cc < 3; cc++)
                tangent_buf(xx, yy, cc) = 0.0f;
    auto t_jit0 = std::chrono::high_resolution_clock::now();
    jvp.realize({W, H, 3});
    auto t_jit1 = std::chrono::high_resolution_clock::now();
    printf(" done (%.1f s)\n\n",
           std::chrono::duration<double>(t_jit1 - t_jit0).count());

    // ── Optimization: Adam for normals + albedo ──────────────────────────
    const float normals_lr = 0.03f;
    const float albedo_lr = 0.01f;
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    const float lambda_n = 0.0005f;  // weak Tikhonov toward (0,0,1)
    const int max_iters = 150;

    printf("Optimization: Adam, %d iters, 3 lights x 3 basis = 9 JVP/iter\n",
           max_iters);
    printf("  normals_lr=%.3f  albedo_lr=%.3f  lambda_n=%.4f\n\n",
           normals_lr, albedo_lr, lambda_n);

    // Adam state
    Buffer<float> mn_buf(W, H, 3), vn_buf(W, H, 3);
    Buffer<float> ma_buf(W, H, 3), va_buf(W, H, 3);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            for (int cc = 0; cc < 3; cc++) {
                mn_buf(xx, yy, cc) = 0.0f; vn_buf(xx, yy, cc) = 0.0f;
                ma_buf(xx, yy, cc) = 0.0f; va_buf(xx, yy, cc) = 0.0f;
            }

    Buffer<float> grad_n(W, H, 3), grad_a(W, H, 3);
    Buffer<float> jvp_out(W, H, 3);
    Buffer<float> rendered_buf(W, H, 3);

    printf("%-4s  %-12s  %-10s  %-10s  %-10s  %-8s\n",
           "Iter", "Loss", "Normal PSNR", "Albedo PSNR", "Render PSNR", "ms/iter");
    printf("%-4s  %-12s  %-10s  %-10s  %-10s  %-8s\n",
           "----", "----", "-----------", "-----------", "-----------", "-------");

    auto t_opt0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iters; iter++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Zero gradients
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++)
                for (int cc = 0; cc < 3; cc++) {
                    grad_n(xx, yy, cc) = 0.0f;
                    grad_a(xx, yy, cc) = 0.0f;
                }

        // ── Accumulate gradients over 3 lights ──────────────────────
        for (int l = 0; l < NUM_LIGHTS; l++) {
            float lx = lights[l][0], ly = lights[l][1], lz = lights[l][2];

            // Set light Params for this light
            light_x.set(lx); light_y.set(ly); light_z.set(lz);

            // Compute rendered host-side for this light
            for (int yy = 0; yy < H; yy++)
                for (int xx = 0; xx < W; xx++) {
                    float n0 = normals_est(xx, yy, 0);
                    float n1 = normals_est(xx, yy, 1);
                    float n2 = normals_est(xx, yy, 2);
                    float nl = std::sqrt(n0*n0 + n1*n1 + n2*n2 + 1e-8f);
                    float ndl = (n0/nl)*lx + (n1/nl)*ly + (n2/nl)*lz;
                    ndl = std::max(ndl, 0.0f);
                    for (int cc = 0; cc < 3; cc++)
                        rendered_buf(xx, yy, cc) = albedo_est(xx, yy, cc) * ndl;
                }

            // Normal gradient: 3 JVP calls (basis vectors)
            for (int k = 0; k < 3; k++) {
                for (int yy = 0; yy < H; yy++)
                    for (int xx = 0; xx < W; xx++)
                        for (int cc = 0; cc < 3; cc++)
                            tangent_buf(xx, yy, cc) = (cc == k) ? 1.0f : 0.0f;

                jvp.realize(jvp_out);

                for (int yy = 0; yy < H; yy++)
                    for (int xx = 0; xx < W; xx++) {
                        float g = 0.0f;
                        for (int cc = 0; cc < 3; cc++) {
                            float res = rendered_buf(xx, yy, cc) - targets[l](xx, yy, cc);
                            g += 2.0f * res * jvp_out(xx, yy, cc);
                        }
                        grad_n(xx, yy, k) += g;
                    }
            }

            // Albedo gradient (analytical): d_loss/d_albedo = 2*(rendered-target)*NdotL
            for (int yy = 0; yy < H; yy++)
                for (int xx = 0; xx < W; xx++) {
                    float n0 = normals_est(xx, yy, 0);
                    float n1 = normals_est(xx, yy, 1);
                    float n2 = normals_est(xx, yy, 2);
                    float nl = std::sqrt(n0*n0 + n1*n1 + n2*n2 + 1e-8f);
                    float ndl = (n0/nl)*lx + (n1/nl)*ly + (n2/nl)*lz;
                    ndl = std::max(ndl, 0.0f);
                    for (int cc = 0; cc < 3; cc++)
                        grad_a(xx, yy, cc) += 2.0f *
                            (rendered_buf(xx, yy, cc) - targets[l](xx, yy, cc)) * ndl;
                }
        }

        // Tikhonov on normals: lambda_n * 2 * (n - n0)
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++) {
                grad_n(xx, yy, 0) += lambda_n * 2.0f * normals_est(xx, yy, 0);
                grad_n(xx, yy, 1) += lambda_n * 2.0f * normals_est(xx, yy, 1);
                grad_n(xx, yy, 2) += lambda_n * 2.0f * (normals_est(xx, yy, 2) - 1.0f);
            }

        // ── Adam update for normals ──────────────────────────────────
        float bc1 = 1.0f - std::pow(beta1, (float)(iter + 1));
        float bc2 = 1.0f - std::pow(beta2, (float)(iter + 1));

        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++)
                for (int cc = 0; cc < 3; cc++) {
                    float g = grad_n(xx, yy, cc);
                    mn_buf(xx, yy, cc) = beta1 * mn_buf(xx, yy, cc) + (1-beta1) * g;
                    vn_buf(xx, yy, cc) = beta2 * vn_buf(xx, yy, cc) + (1-beta2) * g*g;
                    float mh = mn_buf(xx, yy, cc) / bc1;
                    float vh = vn_buf(xx, yy, cc) / bc2;
                    normals_est(xx, yy, cc) -= normals_lr * mh / (std::sqrt(vh) + eps);
                }

        // Renormalize normals
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++) {
                float n0 = normals_est(xx, yy, 0);
                float n1 = normals_est(xx, yy, 1);
                float n2 = normals_est(xx, yy, 2);
                float nl = std::sqrt(n0*n0 + n1*n1 + n2*n2 + 1e-8f);
                normals_est(xx, yy, 0) = n0 / nl;
                normals_est(xx, yy, 1) = n1 / nl;
                normals_est(xx, yy, 2) = n2 / nl;
            }

        // ── Adam update for albedo ───────────────────────────────────
        for (int yy = 0; yy < H; yy++)
            for (int xx = 0; xx < W; xx++)
                for (int cc = 0; cc < 3; cc++) {
                    float g = grad_a(xx, yy, cc);
                    ma_buf(xx, yy, cc) = beta1 * ma_buf(xx, yy, cc) + (1-beta1) * g;
                    va_buf(xx, yy, cc) = beta2 * va_buf(xx, yy, cc) + (1-beta2) * g*g;
                    float mh = ma_buf(xx, yy, cc) / bc1;
                    float vh = va_buf(xx, yy, cc) / bc2;
                    albedo_est(xx, yy, cc) -= albedo_lr * mh / (std::sqrt(vh) + eps);
                    albedo_est(xx, yy, cc) = std::clamp(albedo_est(xx, yy, cc), 0.0f, 1.0f);
                }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ── Print progress ───────────────────────────────────────────
        if (iter % 10 == 0 || iter == max_iters - 1) {
            double loss = 0.0;
            for (int l = 0; l < NUM_LIGHTS; l++) {
                Buffer<float> ren = render_host(normals_est, albedo_est,
                                                lights[l][0], lights[l][1], lights[l][2]);
                for (int yy = 0; yy < H; yy++)
                    for (int xx = 0; xx < W; xx++)
                        for (int cc = 0; cc < 3; cc++) {
                            double d = ren(xx, yy, cc) - targets[l](xx, yy, cc);
                            loss += d * d;
                        }
            }
            loss /= (double)(W * H * NUM_LIGHTS);

            double normal_psnr = compute_psnr_rgb(normals_true, normals_est);
            double albedo_psnr = compute_psnr_rgb(albedo_true, albedo_est);
            Buffer<float> ren0 = render_host(normals_est, albedo_est,
                                             lights[0][0], lights[0][1], lights[0][2]);
            double render_psnr = compute_psnr_rgb(targets[0], ren0);

            printf("%-4d  %-12.6f  %-10.2f  %-10.2f  %-10.2f  %-8.0f\n",
                   iter, loss, normal_psnr, albedo_psnr, render_psnr, ms);
        }
    }

    auto t_opt1 = std::chrono::high_resolution_clock::now();
    printf("\nOptimization wall time: %.1f s\n",
           std::chrono::duration<double>(t_opt1 - t_opt0).count());

    // ── Results ──────────────────────────────────────────────────────────
    double normal_psnr = compute_psnr_rgb(normals_true, normals_est);
    double albedo_psnr = compute_psnr_rgb(albedo_true, albedo_est);
    Buffer<float> ren0 = render_host(normals_est, albedo_est,
                                     lights[0][0], lights[0][1], lights[0][2]);
    double render_psnr = compute_psnr_rgb(targets[0], ren0);

    printf("\nNormal PSNR:   %.2f dB\n", normal_psnr);
    printf("Albedo PSNR:   %.2f dB\n", albedo_psnr);
    printf("Render PSNR:   %.2f dB\n", render_psnr);

    save_normal_map(normals_est, "data/normals_est.jpg");
    save_rgb(albedo_est, "data/albedo_est.jpg");
    save_rgb(ren0, "data/normals_rerendered.jpg");

    return 0;
}
