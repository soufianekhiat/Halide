// Tiled separated Gaussian blur on an RGB image.
//
// This is a substantial example for HalideTraceViz that shows:
//   - A real image pipeline (load, blur, save)
//   - A separated blur (two 1D passes instead of one 2D pass)
//   - Tiled parallelism with per-tile recomputation of the intermediate
//
// Usage:
//   tiled_blur [input.png] [output.png]
//
// Any RGB image works. Suggested source: pexels.com (search for e.g.
// "landscape", "portrait", or "fruit still life"). Save the file next to
// this executable as "input.png" (or pass the path as the first arg).

#include "Halide.h"
#include "halide_image_io.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    const char *in_path  = (argc > 1) ? argv[1] : "input.png";
    const char *out_path = (argc > 2) ? argv[2] : "blur_out.png";

    Buffer<uint8_t> input;
    try {
        input = load_image(in_path);
    } catch (const std::exception &e) {
        fprintf(stderr,
                "Failed to load image '%s': %s\n\n"
                "Download any RGB image (e.g. from pexels.com) and save it\n"
                "as '%s' next to this executable, or pass its path as the\n"
                "first argument.\n",
                in_path, e.what(), in_path);
        return 1;
    }

    if (input.dimensions() < 3 || input.channels() < 3) {
        fprintf(stderr, "Expected an RGB image (>=3 channels); got %d\n",
                input.dimensions() >= 3 ? input.channels() : 0);
        return 1;
    }

    // Crop to a fixed region from the image center. We keep this small so
    // the trace visualization stays a manageable length.
    const int CROP = 256;
    const int w = std::min(input.width(),  CROP);
    const int h = std::min(input.height(), CROP);
    const int x_off = (input.width()  - w) / 2;
    const int y_off = (input.height() - h) / 2;

    printf("Input : %s  (%dx%d x %d channels)\n",
           in_path, input.width(), input.height(), input.channels());
    printf("Crop  : %dx%d at offset (%d, %d)\n", w, h, x_off, y_off);

    Var x("x"), y("y"), c("c");
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");

    // Cropped + boundary-clamped source, promoted to float.
    Func clamped("clamped");
    clamped(x, y, c) = cast<float>(
        input(clamp(x + x_off, 0, input.width()  - 1),
              clamp(y + y_off, 0, input.height() - 1),
              c));

    // Horizontal 5-tap Gaussian.
    Func blur_x("blur_x");
    blur_x(x, y, c) = 0.0625f * clamped(x - 2, y, c)
                    + 0.2500f * clamped(x - 1, y, c)
                    + 0.3750f * clamped(x,     y, c)
                    + 0.2500f * clamped(x + 1, y, c)
                    + 0.0625f * clamped(x + 2, y, c);

    // Vertical 5-tap Gaussian.
    Func blur_y("blur_y");
    blur_y(x, y, c) = 0.0625f * blur_x(x, y - 2, c)
                    + 0.2500f * blur_x(x, y - 1, c)
                    + 0.3750f * blur_x(x, y,     c)
                    + 0.2500f * blur_x(x, y + 1, c)
                    + 0.0625f * blur_x(x, y + 2, c);

    // Cast back to uint8 for the final image.
    Func output("output");
    output(x, y, c) = cast<uint8_t>(clamp(blur_y(x, y, c), 0.0f, 255.0f));

    // ---- Schedule ---------------------------------------------------------
    // Output is tiled 64x64. Tile rows are computed in parallel. The
    // horizontal blur (blur_x) is computed once per tile, interleaved with
    // the vertical blur's consumption, showing the classic separated-blur
    // locality tradeoff.
    const int TILE_W = 64, TILE_H = 64;

    output.tile(x, y, xo, yo, xi, yi, TILE_W, TILE_H)
          .parallel(yo)
          .vectorize(xi, 8);

    blur_x.compute_at(output, xo)
          .vectorize(x, 8);

    // ---- Tracing ----------------------------------------------------------
    // These annotations cause HL_TRACE_FILE to receive events we can
    // visualize with HalideTraceViz.
    clamped.trace_loads();
    blur_x.trace_stores();
    blur_y.trace_stores();
    output.trace_stores();

    // ---- Run --------------------------------------------------------------
    Buffer<uint8_t> result(w, h, 3);
    output.realize(result);

    save_image(result, out_path);
    printf("Saved : %s\n", out_path);
    return 0;
}
