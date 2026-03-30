#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// ---- helpers ----------------------------------------------------------------

static bool check(const char *label, int got, int expected, int a, int b = -1, int c = -1) {
    if (got != expected) {
        if (c >= 0)
            printf("%s: out(%d,%d,%d) = %d, expected %d\n", label, a, b, c, got, expected);
        else if (b >= 0)
            printf("%s: out(%d,%d) = %d, expected %d\n", label, a, b, got, expected);
        else
            printf("%s: out(%d) = %d, expected %d\n", label, a, got, expected);
        return false;
    }
    return true;
}

// ---- 1-D inclusive prefix sum -----------------------------------------------
// input(x) = x + 1  →  {1, 2, 3, 4, 5, 6, 7, 8}
// expected  =           {1, 3, 6,10,15,21,28,36}
static bool test_1d_sum() {
    const int N = 8;
    const int LOG2_N = 3;  // 2^3 = 8

    Func input("input");
    Var x("x");
    input(x) = x + 1;

    auto ps = parallel_scan([](Expr a, Expr b) { return a + b; },
                             input, 0, LOG2_N);
    ps.result.compute_root();

    Buffer<int> out = ps.result.realize({N});

    const int expected[N] = {1, 3, 6, 10, 15, 21, 28, 36};
    for (int i = 0; i < N; i++) {
        if (!check("1d_sum", out(i), expected[i], i)) return false;
    }
    return true;
}

// ---- 1-D inclusive prefix product -------------------------------------------
// input(x) = x + 1  →  {1, 2, 3, 4, 5}
// expected  =           {1, 2, 6,24,120}
static bool test_1d_product() {
    const int N = 5;
    const int LOG2_N = 3;  // 2^3 = 8 >= 5

    Func input("input");
    Var x("x");
    input(x) = x + 1;

    auto ps = parallel_scan([](Expr a, Expr b) { return a * b; },
                             input, 0, LOG2_N);
    ps.result.compute_root();

    Buffer<int> out = ps.result.realize({N});

    const int expected[N] = {1, 2, 6, 24, 120};
    for (int i = 0; i < N; i++) {
        if (!check("1d_product", out(i), expected[i], i)) return false;
    }
    return true;
}

// ---- 1-D edge: log2_n = 1 ---------------------------------------------------
// input = {10, 20}
// stage1(0) = 10, stage1(1) = 10 + 20 = 30
static bool test_1d_log2n_1() {
    Func input("input");
    Var x("x");
    input(x) = (x + 1) * 10;

    auto ps = parallel_scan([](Expr a, Expr b) { return a + b; },
                             input, 0, 1);
    ps.result.compute_root();

    Buffer<int> out = ps.result.realize({2});

    if (!check("1d_log2n_1 [0]", out(0), 10, 0)) return false;
    if (!check("1d_log2n_1 [1]", out(1), 30, 1)) return false;
    return true;
}

// ---- 2-D row-wise prefix sum (scan_dim = 0) ---------------------------------
// image(x, y) = y * 4 + x + 1
//   row 0: [1, 2, 3, 4]   → [1, 3,  6, 10]
//   row 1: [5, 6, 7, 8]   → [5,11, 18, 26]
//   row 2: [9,10,11,12]   → [9,19, 30, 42]
//   row 3: [13,14,15,16]  → [13,27,42, 58]
static bool test_2d_row_scan() {
    const int W = 4, H = 4;
    const int LOG2_W = 2;  // 2^2 = 4

    Func image("image");
    Var x("x"), y("y");
    image(x, y) = y * W + x + 1;

    auto ps = parallel_scan([](Expr a, Expr b) { return a + b; },
                             image, 0, LOG2_W);
    ps.result.compute_root();

    Buffer<int> out = ps.result.realize({W, H});

    // Recompute expected: prefix sum of each row.
    for (int row = 0; row < H; row++) {
        int acc = 0;
        for (int col = 0; col < W; col++) {
            acc += row * W + col + 1;
            if (!check("2d_row_scan", out(col, row), acc, col, row)) return false;
        }
    }
    return true;
}

// ---- 2-D column-wise prefix sum (scan_dim = 1) ------------------------------
// image(x, y) = y * 4 + x + 1 (same as above)
// For column x=0: values {1,5,9,13}  → { 1, 6,15,28}
// For column x=1: values {2,6,10,14} → { 2, 8,18,32}
static bool test_2d_col_scan() {
    const int W = 4, H = 4;
    const int LOG2_H = 2;  // 2^2 = 4

    Func image("image");
    Var x("x"), y("y");
    image(x, y) = y * W + x + 1;

    auto ps = parallel_scan([](Expr a, Expr b) { return a + b; },
                             image, 1, LOG2_H);
    ps.result.compute_root();

    Buffer<int> out = ps.result.realize({W, H});

    // Expected: prefix sum of each column.
    for (int col = 0; col < W; col++) {
        int acc = 0;
        for (int row = 0; row < H; row++) {
            acc += row * W + col + 1;
            if (!check("2d_col_scan", out(col, row), acc, col, row)) return false;
        }
    }
    return true;
}

// ---- 2-D stages exposed: verify stage counts and compute_root default -------
static bool test_stages_count() {
    const int LOG2_N = 4;

    Func input("input");
    Var x("x"), y("y");
    input(x, y) = x + y;

    auto ps = parallel_scan([](Expr a, Expr b) { return a + b; },
                             input, 0, LOG2_N);

    if ((int)ps.stages.size() != LOG2_N + 1) {
        printf("stages_count: got %d stages, expected %d\n",
               (int)ps.stages.size(), LOG2_N + 1);
        return false;
    }
    // result must equal last stage
    if (ps.result.name() != ps.stages.back().name()) {
        printf("stages_count: result.name() != stages.back().name()\n");
        return false;
    }
    return true;
}

// ---- 3-D scan along middle dimension (scan_dim = 1) -------------------------
// volume(x, y, z) = x + y + z
// Scan along y for fixed (x,z): prefix sum of {x+0+z, x+1+z, x+2+z, x+3+z}
static bool test_3d_scan() {
    const int NX = 3, NY = 4, NZ = 2;
    const int LOG2_NY = 2;

    Func vol("vol");
    Var x("x"), y("y"), z("z");
    vol(x, y, z) = x + y + z;

    auto ps = parallel_scan([](Expr a, Expr b) { return a + b; },
                             vol, 1, LOG2_NY);
    ps.result.compute_root();

    Buffer<int> out = ps.result.realize({NX, NY, NZ});

    for (int zi = 0; zi < NZ; zi++) {
        for (int xi = 0; xi < NX; xi++) {
            int acc = 0;
            for (int yi = 0; yi < NY; yi++) {
                acc += xi + yi + zi;
                if (!check("3d_scan", out(xi, yi, zi), acc, xi, yi, zi)) return false;
            }
        }
    }
    return true;
}

// ---- max parallel scan (prefix maximum) ------------------------------------
static bool test_prefix_max() {
    const int N = 8;
    const int LOG2_N = 3;

    // input: 3,1,4,1,5,9,2,6
    const int vals[N] = {3, 1, 4, 1, 5, 9, 2, 6};
    Buffer<int> buf(N);
    for (int i = 0; i < N; i++) buf(i) = vals[i];

    Func input("input");
    Var x("x");
    input(x) = buf(x);

    auto ps = parallel_scan([](Expr a, Expr b) { return max(a, b); },
                             input, 0, LOG2_N);
    ps.result.compute_root();

    Buffer<int> out = ps.result.realize({N});

    int running = vals[0];
    for (int i = 0; i < N; i++) {
        running = std::max(running, vals[i]);
        if (!check("prefix_max", out(i), running, i)) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (!test_1d_sum())       { return 1; }
    if (!test_1d_product())   { return 1; }
    if (!test_1d_log2n_1())   { return 1; }
    if (!test_2d_row_scan())  { return 1; }
    if (!test_2d_col_scan())  { return 1; }
    if (!test_stages_count()) { return 1; }
    if (!test_3d_scan())      { return 1; }
    if (!test_prefix_max())   { return 1; }

    printf("Success!\n");
    return 0;
}
