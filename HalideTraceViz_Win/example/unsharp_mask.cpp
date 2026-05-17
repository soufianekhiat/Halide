// Multi-stage RGB image enhancement: unsharp masking with luminance-only blur.
//
// 8 stages, each `compute_root()` so they animate one at a time in
// HalideTraceViz instead of getting interleaved across tiles:
//
//   1. clamped     - input cast to float, boundary-clamped (the ORIGINAL)
//   2. luma        - Rec.601 luminance (0.299R + 0.587G + 0.114B)
//   3. blur_x      - horizontal 5-tap Gaussian on luminance
//   4. blur_y      - vertical 5-tap Gaussian (the fully blurred luminance)
//   5. high_pass   - luma - blur_y (the detail/edge signal)
//   6. sharp_luma  - luma + amount * high_pass (sharpened luminance)
//   7. sharp_rgb   - per-channel reweighting: input * (sharp_luma / luma)
//   8. output      - cast back to uint8
//
// Each stage produces a meaningful intermediate image. The first stage
// (clamped) is materialized so the original input is visible as the very
// first panel. With --auto_layout HalideTraceViz arranges all 8 panels in
// a grid and you see them fill in sequentially, one stage after another.
//
// Note: this schedule is intentionally pedagogical, not fast. If you want
// to benchmark speed, change to per-tile compute_at + parallel.
//
// Usage:
//   unsharp_mask [input.jpg] [output.png]

#include "Halide.h"
#include "halide_image_io.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    const char *in_path  = (argc > 1) ? argv[1] : "input.jpg";
    const char *out_path = (argc > 2) ? argv[2] : "unsharp_out.png";

    Buffer<uint8_t> input;
    try {
        input = load_image(in_path);
    } catch (const std::exception &e) {
        fprintf(stderr,
                "Failed to load image '%s': %s\n\n"
                "Run download_image.ps1 to fetch a sample, or pass an RGB\n"
                "image path (any size) as the first argument.\n",
                in_path, e.what());
        return 1;
    }

    if (input.dimensions() < 3 || input.channels() < 3) {
        fprintf(stderr, "Expected an RGB image (>=3 channels)\n");
        return 1;
    }

    const int w = input.width();
    const int h = input.height();

    printf("Input  : %s  (%dx%d x %d channels)\n",
           in_path, w, h, input.channels());

    Var x("x"), y("y"), c("c");

    // 1. Cast input to float, NORMALIZED to 0..1, with boundary clamping.
    //    HalideTraceViz auto_layout normalizes floats roughly into 0..1
    //    when drawing them, so keeping the whole pipeline in that range
    //    means the visualization shows the actual image content (instead
    //    of saturating to white when values > 1).
    Func clamped("clamped");
    clamped(x, y, c) = cast<float>(
        input(clamp(x, 0, w - 1), clamp(y, 0, h - 1), c)) * (1.0f / 255.0f);

    // 2. Compute Rec.601 luminance (also in 0..1).
    Func luma("luma");
    luma(x, y) = 0.299f * clamped(x, y, 0)
               + 0.587f * clamped(x, y, 1)
               + 0.114f * clamped(x, y, 2);

    // 3. Horizontal 5-tap Gaussian on luminance.
    Func blur_x("blur_x");
    blur_x(x, y) = 0.0625f * luma(x - 2, y)
                 + 0.2500f * luma(x - 1, y)
                 + 0.3750f * luma(x,     y)
                 + 0.2500f * luma(x + 1, y)
                 + 0.0625f * luma(x + 2, y);

    // 4. Vertical 5-tap Gaussian.
    Func blur_y("blur_y");
    blur_y(x, y) = 0.0625f * blur_x(x, y - 2)
                 + 0.2500f * blur_x(x, y - 1)
                 + 0.3750f * blur_x(x, y)
                 + 0.2500f * blur_x(x, y + 1)
                 + 0.0625f * blur_x(x, y + 2);

    // 5. High-pass = original luma - blurred luma.
    Func high_pass("high_pass");
    high_pass(x, y) = luma(x, y) - blur_y(x, y);

    // 6. Sharpened luminance = luma + amount * high_pass.
    const float sharpen_amount = 1.5f;
    Func sharp_luma("sharp_luma");
    sharp_luma(x, y) = luma(x, y) + sharpen_amount * high_pass(x, y);

    // 7. Apply the luminance ratio to each color channel to preserve hue.
    //    Use a small epsilon to avoid division by zero in dark areas.
    Func sharp_rgb("sharp_rgb");
    Expr safe_luma = max(luma(x, y), 1.0f / 255.0f);
    sharp_rgb(x, y, c) = clamped(x, y, c) * (sharp_luma(x, y) / safe_luma);

    // 8. Scale back to 0..255 and cast to uint8 with clamp.
    Func output("output");
    output(x, y, c) = cast<uint8_t>(
        clamp(sharp_rgb(x, y, c) * 255.0f, 0.0f, 255.0f));

    // ---- Schedule ---------------------------------------------------------
    // Sequential pipeline: every stage compute_root() so each one is fully
    // materialized before the next starts. The trace then shows the
    // pipeline animate stage-by-stage with no interleaving across tiles.
    clamped   .compute_root();
    luma      .compute_root();
    blur_x    .compute_root();
    blur_y    .compute_root();
    high_pass .compute_root();
    sharp_luma.compute_root();
    sharp_rgb .compute_root();
    // output is the root consumer - left at default scheduling.

    // ---- Tracing ----------------------------------------------------------
    // trace_stores on every stage gives one panel per func in --auto_layout.
    // clamped is included so the ORIGINAL input image is visible as the
    // first panel.
    clamped   .trace_stores();
    luma      .trace_stores();
    blur_x    .trace_stores();
    blur_y    .trace_stores();
    high_pass .trace_stores();
    sharp_luma.trace_stores();
    sharp_rgb .trace_stores();
    output    .trace_stores();

    // ---- Run --------------------------------------------------------------
    Buffer<uint8_t> result(w, h, 3);
    output.realize(result);

    save_image(result, out_path);
    printf("Saved  : %s\n", out_path);
    return 0;
}
