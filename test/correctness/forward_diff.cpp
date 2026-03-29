#include "Halide.h"

#include <cmath>
#include <cstdio>

using namespace Halide;

static bool check_close(const Buffer<float> &got, const Buffer<float> &expected,
                        float tol, const char *name) {
    for (int y = 0; y < got.height(); y++) {
        for (int x = 0; x < got.width(); x++) {
            float diff = std::abs(got(x, y) - expected(x, y));
            if (diff > tol) {
                printf("FAIL %s at (%d,%d): got %g expected %g (diff %g)\n",
                       name, x, y, got(x, y), expected(x, y), diff);
                return false;
            }
        }
    }
    return true;
}

static bool check_close_1d(const Buffer<float> &got, const Buffer<float> &expected,
                            float tol, const char *name) {
    for (int i = 0; i < got.width(); i++) {
        float diff = std::abs(got(i) - expected(i));
        if (diff > tol) {
            printf("FAIL %s at [%d]: got %g expected %g (diff %g)\n",
                   name, i, got(i), expected(i), diff);
            return false;
        }
    }
    return true;
}

// Test 1: pointwise scalar param
// f(x,y) = p * sin(cast<float>(x)) + cast<float>(y)
// d/dp f = sin(cast<float>(x))  [exact, no FD needed]
static bool test_pointwise() {
    Param<float> p;
    p.set(1.5f);
    Var x, y;
    Func f("f");
    f(x, y) = p * sin(cast<float>(x)) + cast<float>(y);

    Func df = propagate_tangents(f, p);
    df.compute_root();

    const int W = 8, H = 8;
    Buffer<float> got = df.realize({W, H});

    // Exact derivative: sin(cast<float>(x))
    Func exact("exact_pw");
    exact(x, y) = sin(cast<float>(x));
    Buffer<float> expected = exact.realize({W, H});

    if (!check_close(got, expected, 1e-5f, "test_pointwise")) return false;
    printf("test_pointwise: OK\n");
    return true;
}

// Test 2: multi-stage pipeline
// stage1(x,y) = p * cast<float>(x) + cast<float>(y)
// stage2(x,y) = stage1(x,y) * stage1(x,y)
// d/dp stage2 = 2 * stage1 * d(stage1)/dp = 2 * stage1(x,y) * x
static bool test_multistage() {
    Param<float> p;
    p.set(2.0f);
    Var x, y;

    Func stage1("stage1"), stage2("stage2");
    stage1(x, y) = p * cast<float>(x) + cast<float>(y);
    stage2(x, y) = stage1(x, y) * stage1(x, y);
    stage1.compute_root();

    Func df = propagate_tangents(stage2, p);
    df.compute_root();

    const int W = 6, H = 6;
    Buffer<float> got = df.realize({W, H});

    // Exact derivative: 2 * stage1(x,y) * x = 2 * (p*x+y) * x
    float pval = 2.0f;
    Buffer<float> expected(W, H);
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float s1 = pval * (float)xx + (float)yy;
            expected(xx, yy) = 2.0f * s1 * (float)xx;
        }
    }
    if (!check_close(got, expected, 1e-4f, "test_multistage")) return false;
    printf("test_multistage: OK\n");
    return true;
}

// Test 3: Tuple output
// color(x,y) = Tuple(p*x, p*p*y, x+y)
// d/dp color[0] = x, d/dp color[1] = 2*p*y, d/dp color[2] = 0
static bool test_tuple() {
    Param<float> p;
    p.set(3.0f);
    Var x, y;

    Func color("color");
    color(x, y) = Tuple(p * cast<float>(x),
                        p * p * cast<float>(y),
                        cast<float>(x + y));

    Func dc = propagate_tangents(color, p);
    dc.compute_root();

    const int W = 4, H = 4;
    float pval = 3.0f;

    {
        Func e0;
        Var vx, vy;
        e0(vx, vy) = dc(vx, vy)[0];
        Buffer<float> got = e0.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float exp_val = (float)xx;
                if (std::abs(got(xx, yy) - exp_val) > 1e-5f) {
                    printf("FAIL test_tuple[0] at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_val);
                    return false;
                }
            }
        }
    }
    {
        Func e1;
        Var vx, vy;
        e1(vx, vy) = dc(vx, vy)[1];
        Buffer<float> got = e1.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float exp_val = 2.0f * pval * (float)yy;
                if (std::abs(got(xx, yy) - exp_val) > 1e-4f) {
                    printf("FAIL test_tuple[1] at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_val);
                    return false;
                }
            }
        }
    }
    {
        Func e2;
        Var vx, vy;
        e2(vx, vy) = dc(vx, vy)[2];
        Buffer<float> got = e2.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                if (std::abs(got(xx, yy)) > 1e-6f) {
                    printf("FAIL test_tuple[2] at (%d,%d): got %g expected 0\n",
                           xx, yy, got(xx, yy));
                    return false;
                }
            }
        }
    }
    printf("test_tuple: OK\n");
    return true;
}

// Test 4: RDom / accumulation update
// acc(x) = 0; acc(r) = acc(r) + p * cast<float>(r)
// After all updates: acc(i) = p * i
// d(acc(i))/dp = i  [exact]
static bool test_rdom() {
    Param<float> p;
    p.set(1.0f);
    Var x;
    const int N = 8;

    Func acc("acc");
    RDom r(0, N);
    acc(x) = cast<float>(0);
    acc(r) = acc(r) + p * cast<float>(r);
    acc.compute_root();

    Func dacc = propagate_tangents(acc, p);
    dacc.compute_root();

    Buffer<float> got = dacc.realize({N});

    // Exact: d(acc(i))/dp = i
    Buffer<float> expected(N);
    for (int i = 0; i < N; i++) expected(i) = (float)i;

    if (!check_close_1d(got, expected, 1e-5f, "test_rdom")) return false;
    printf("test_rdom: OK\n");
    return true;
}

// Test 5: Buffer / ImageParam JVP (directional derivative)
// f(x,y) = inp(x,y) * inp(x,y)
// Tangent direction: all-ones -> d/dt f = 2*inp(x,y)
static bool test_buffer_jvp() {
    const int W = 4, H = 4;
    ImageParam inp(Float(32), 2, "inp");
    Var x, y;

    Buffer<float> inp_buf(W, H);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            inp_buf(xx, yy) = (float)(xx + yy + 1);
    inp.set(inp_buf);

    Func f("f");
    f(x, y) = inp(x, y) * inp(x, y);

    Func tangent_dir("tangent_dir");
    tangent_dir(x, y) = 1.0f;

    Func df = propagate_tangents(f, {{inp.name(), tangent_dir}});
    df.compute_root();

    Buffer<float> got = df.realize({W, H});
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float expected_val = 2.0f * inp_buf(xx, yy);
            if (std::abs(got(xx, yy) - expected_val) > 1e-4f) {
                printf("FAIL test_buffer_jvp at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), expected_val);
                return false;
            }
        }
    }
    printf("test_buffer_jvp: OK\n");
    return true;
}

// Test 6: math functions (exp)
// f(x,y) = exp(p * cast<float>(x) / W)
// d/dp f = (x/W) * exp(p * x/W)  [exact]
static bool test_math_funcs() {
    Param<float> p;
    p.set(1.0f);
    Var x, y;
    const int W = 8, H = 4;

    Func f("f_math");
    f(x, y) = exp(p * cast<float>(x) / cast<float>(W));

    Func df = propagate_tangents(f, p);
    df.compute_root();

    Buffer<float> got = df.realize({W, H});

    // Exact derivative: (x/W) * exp(p * x/W)
    float pval = 1.0f;
    Buffer<float> expected(W, H);
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float frac = (float)xx / (float)W;
            expected(xx, yy) = frac * std::exp(pval * frac);
        }
    }
    if (!check_close(got, expected, 1e-5f, "test_math_funcs")) return false;
    printf("test_math_funcs: OK\n");
    return true;
}

// Test 7: inverse rendering demo
// rendered(x,y) = p * cast<float>(x) / cast<float>(W)   (linear in p)
// Target: cast<float>(x) / cast<float>(W)  (rendered at p=1)
// GD with forward-mode Jacobian should converge to p=1, loss=0
static bool test_inverse_rendering() {
    Param<float> p;
    Var x, y;
    const int W = 8, H = 1;

    Func rendered("rendered");
    rendered(x, y) = p * cast<float>(x) / cast<float>(W);

    Func d_rendered = propagate_tangents(rendered, p);

    // target = x/W (achievable at p=1)
    Func target_f("target_f");
    target_f(x, y) = cast<float>(x) / cast<float>(W);

    // Loss gradient: sum 2*(rendered - target)*d_rendered
    RDom rd(0, W, 0, H);
    Func grad_p("grad_p");
    grad_p() = sum(2.0f * (rendered(rd.x, rd.y) - target_f(rd.x, rd.y)) *
                   d_rendered(rd.x, rd.y));

    rendered.compute_root();
    d_rendered.compute_root();
    target_f.compute_root();
    grad_p.compute_root();

    p.set(0.0f);
    float lr = 0.1f;
    for (int step = 0; step < 100; step++) {
        Buffer<float> g = grad_p.realize();
        float pval = p.get();
        p.set(pval - lr * g());
    }

    // Compute final loss
    Func loss("loss");
    loss() = sum(pow(rendered(rd.x, rd.y) - target_f(rd.x, rd.y), 2));
    loss.compute_root();
    Buffer<float> final_loss = loss.realize();
    if (final_loss() > 1e-5f) {
        printf("FAIL test_inverse_rendering: final loss %g, p=%g\n",
               final_loss(), p.get());
        return false;
    }
    printf("test_inverse_rendering: OK (final loss = %g, p = %g)\n",
           final_loss(), p.get());
    return true;
}

// Test 8: cross-validate against known exact formula
// f(x,y) = p * p * cast<float>(x) + p * cast<float>(y)
// d/dp f = 2*p*x + y  [exact]
static bool test_cross_validate() {
    Param<float> p;
    p.set(2.0f);
    Var x, y;
    const int W = 4, H = 4;

    Func f("f_xv");
    f(x, y) = p * p * cast<float>(x) + p * cast<float>(y);

    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    float pval = 2.0f;
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float expected_val = 2.0f * pval * (float)xx + (float)yy;
            if (std::abs(got(xx, yy) - expected_val) > 1e-4f) {
                printf("FAIL test_cross_validate at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), expected_val);
                return false;
            }
        }
    }
    printf("test_cross_validate: OK\n");
    return true;
}

// ── New tests for additional coverage ────────────────────────────────────────

// Test 9: division — quotient rule with param in numerator and denominator
// f(x,y) = p * (x+1) / (y+2)           -> d/dp = (x+1)/(y+2)
// g(x,y) = cast<float>(x+1) / (p+1)    -> d/dp = -(x+1)/(p+1)^2
static bool test_division() {
    Param<float> p;
    p.set(2.0f);
    Var x, y;
    const int W = 5, H = 5;
    float pval = 2.0f;

    {
        Func f("f_div_num");
        f(x, y) = p * cast<float>(x + 1) / cast<float>(y + 2);
        Func df = propagate_tangents(f, p);
        df.compute_root();
        Buffer<float> got = df.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float exp_v = (float)(xx + 1) / (float)(yy + 2);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_division(num) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    {
        // Param in denominator: g = (x+1) / (p+1)
        // d/dp = -(x+1) / (p+1)^2
        Func g("g_div_den");
        g(x, y) = cast<float>(x + 1) / (p + 1.0f);
        Func dg = propagate_tangents(g, p);
        dg.compute_root();
        Buffer<float> got = dg.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float denom = pval + 1.0f;
                float exp_v = -(float)(xx + 1) / (denom * denom);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_division(den) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    printf("test_division: OK\n");
    return true;
}

// Test 10: min and max — conditional derivative
// f(x,y) = min(p * cast<float>(x), cast<float>(y+1))
// d/dp = select(p*x < y+1, x, 0)   i.e. x when p*x is the smaller value
// g(x,y) = max(p * cast<float>(x), cast<float>(y+1))
// d/dp = select(p*x > y+1, x, 0)
static bool test_min_max() {
    Param<float> p;
    p.set(0.5f);
    Var x, y;
    const int W = 6, H = 4;
    float pval = 0.5f;

    {
        Func f("f_min");
        f(x, y) = min(p * cast<float>(x), cast<float>(y + 1));
        Func df = propagate_tangents(f, p);
        df.compute_root();
        Buffer<float> got = df.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float px = pval * (float)xx;
                float yp1 = (float)(yy + 1);
                float exp_v = (px < yp1) ? (float)xx : 0.0f;
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_min at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    {
        Func g("g_max");
        g(x, y) = max(p * cast<float>(x), cast<float>(y + 1));
        Func dg = propagate_tangents(g, p);
        dg.compute_root();
        Buffer<float> got = dg.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float px = pval * (float)xx;
                float yp1 = (float)(yy + 1);
                float exp_v = (px > yp1) ? (float)xx : 0.0f;
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_max at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    printf("test_min_max: OK\n");
    return true;
}

// Test 11: select with param in both branches
// f(x,y) = select(x > 3, p * cast<float>(x), p*p * cast<float>(y+1))
// d/dp = select(x > 3, x, 2*p*(y+1))
static bool test_select_branches() {
    Param<float> p;
    p.set(2.0f);
    Var x, y;
    const int W = 8, H = 4;
    float pval = 2.0f;

    Func f("f_select");
    f(x, y) = select(x > 3,
                     p * cast<float>(x),
                     p * p * cast<float>(y + 1));

    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float exp_v = (xx > 3) ? (float)xx : (2.0f * pval * (float)(yy + 1));
            if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                printf("FAIL test_select_branches at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_select_branches: OK\n");
    return true;
}

// Test 12: cosine function
// f(x,y) = cos(p * cast<float>(x) / W)
// d/dp = -sin(p * x/W) * x/W
static bool test_cos() {
    Param<float> p;
    p.set(1.0f);
    Var x, y;
    const int W = 8, H = 4;
    float pval = 1.0f;

    Func f("f_cos");
    f(x, y) = cos(p * cast<float>(x) / cast<float>(W));

    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float frac = (float)xx / (float)W;
            float exp_v = -std::sin(pval * frac) * frac;
            if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                printf("FAIL test_cos at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_cos: OK\n");
    return true;
}

// Test 13: log and sqrt derivatives
// f_log(x,y) = log(p * cast<float>(x+1) + 1.0f)   d/dp = (x+1)/(p*(x+1)+1)
// f_sqrt(x,y) = sqrt(p * cast<float>(x+1))          d/dp = 0.5*(x+1)/sqrt(p*(x+1))
static bool test_log_sqrt() {
    Param<float> p;
    p.set(1.0f);
    Var x, y;
    const int W = 6, H = 4;
    float pval = 1.0f;

    {
        Func f("f_log");
        f(x, y) = log(p * cast<float>(x + 1) + 1.0f);
        Func df = propagate_tangents(f, p);
        df.compute_root();
        Buffer<float> got = df.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float xp1 = (float)(xx + 1);
                float exp_v = xp1 / (pval * xp1 + 1.0f);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_log at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    {
        Func g("g_sqrt");
        g(x, y) = sqrt(p * cast<float>(x + 1));
        Func dg = propagate_tangents(g, p);
        dg.compute_root();
        Buffer<float> got = dg.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float xp1 = (float)(xx + 1);
                float exp_v = 0.5f * xp1 / std::sqrt(pval * xp1);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_sqrt at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    printf("test_log_sqrt: OK\n");
    return true;
}

// Test 14: pow with param as exponent
// f(x,y) = pow(cast<float>(x+2), p)
// d/dp = pow(x+2, p) * log(x+2)
static bool test_pow_exponent() {
    Param<float> p;
    p.set(2.0f);
    Var x, y;
    const int W = 5, H = 4;
    float pval = 2.0f;

    Func f("f_pow");
    f(x, y) = pow(cast<float>(x + 2), p);

    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float base = (float)(xx + 2);
            float exp_v = std::pow(base, pval) * std::log(base);
            if (std::abs(got(xx, yy) - exp_v) > 1e-4f) {
                printf("FAIL test_pow_exponent at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_pow_exponent: OK\n");
    return true;
}

// Test 15: atan2 with param in first argument
// f(x,y) = atan2(p * cast<float>(y+1), cast<float>(x+1))
// d/dp = (x+1)*(y+1) / ((x+1)^2 + p^2*(y+1)^2)
static bool test_atan2() {
    Param<float> p;
    p.set(1.0f);
    Var x, y;
    const int W = 5, H = 5;
    float pval = 1.0f;

    Func f("f_atan2");
    f(x, y) = atan2(p * cast<float>(y + 1), cast<float>(x + 1));

    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float xp1 = (float)(xx + 1);
            float yp1 = (float)(yy + 1);
            float x2y2 = xp1 * xp1 + pval * pval * yp1 * yp1;
            float exp_v = xp1 * yp1 / x2y2;
            if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                printf("FAIL test_atan2 at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_atan2: OK\n");
    return true;
}

// Test 16: abs intrinsic — |p * (x - N/2)|
// d/dp |p*(x-cx)| = |x-cx|  (= (x-cx)*sign(p*(x-cx)) but since p>0: sign = sign(x-cx))
static bool test_abs_intrinsic() {
    Param<float> p;
    p.set(1.5f);
    Var x, y;
    const int W = 8, H = 4;
    const int cx = 4;

    Func f("f_abs");
    f(x, y) = abs(p * cast<float>(x - cx));

    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    // d/dp |p*(x-cx)| where p=1.5 > 0:
    // = sign(p*(x-cx)) * (x-cx) = sign(x-cx) * (x-cx) = |x-cx|
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float exp_v = std::abs((float)(xx - cx));
            if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                printf("FAIL test_abs_intrinsic at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_abs_intrinsic: OK\n");
    return true;
}

// Test 17: lerp intrinsic — lerp(a, b, p) = a*(1-p) + b*p
// f(x,y) = lerp(cast<float>(x), cast<float>(y+1), p)
// d/dp = (y+1) - x
static bool test_lerp_intrinsic() {
    Param<float> p;
    p.set(0.4f);
    Var x, y;
    const int W = 6, H = 4;

    Func f("f_lerp");
    f(x, y) = lerp(cast<float>(x), cast<float>(y + 1), p);

    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    // d/dp lerp(a,b,p) = b - a = (y+1) - x
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float exp_v = (float)(yy + 1) - (float)xx;
            if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                printf("FAIL test_lerp_intrinsic at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_lerp_intrinsic: OK\n");
    return true;
}

// Test 18: three-stage pure pipeline
// g1(x,y) = p * cast<float>(x) + cast<float>(y)   d/dp = x
// g2(x,y) = g1(x,y) * g1(x,y)                      d/dp = 2*g1*x
// g3(x,y) = g2(x,y) + g1(x,y)                      d/dp = 2*g1*x + x = (2*g1+1)*x
static bool test_three_stage_chain() {
    Param<float> p;
    p.set(1.0f);
    Var x, y;
    const int W = 5, H = 5;
    float pval = 1.0f;

    Func g1("g1"), g2("g2"), g3("g3");
    g1(x, y) = p * cast<float>(x) + cast<float>(y);
    g2(x, y) = g1(x, y) * g1(x, y);
    g3(x, y) = g2(x, y) + g1(x, y);
    g1.compute_root();
    g2.compute_root();

    Func dg3 = propagate_tangents(g3, p);
    dg3.compute_root();
    Buffer<float> got = dg3.realize({W, H});

    // d/dp g3 = d/dp g2 + d/dp g1 = 2*g1*(d/dp g1) + (d/dp g1) = (2*g1+1)*x
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float g1_val = pval * (float)xx + (float)yy;
            float exp_v = (2.0f * g1_val + 1.0f) * (float)xx;
            if (std::abs(got(xx, yy) - exp_v) > 1e-4f) {
                printf("FAIL test_three_stage_chain at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_three_stage_chain: OK\n");
    return true;
}

// Test 19: prefix sum scan — d/dp f(i) = i*(i+1)/2
// scan(x) = 0; scan(r) = scan(r-1) + p * cast<float>(r)  for r in [1, N-1]
// After scan: scan(i) = p * i*(i+1)/2
// Tangent: dtangent(x) = 0; dtangent(r) = dtangent(r-1) + cast<float>(r)
static bool test_prefix_sum() {
    Param<float> p;
    p.set(1.0f);
    Var x;
    const int N = 7;

    Func scan("scan");
    RDom r(1, N - 1);  // r from 1 to N-1
    scan(x) = cast<float>(0);
    scan(r) = scan(r - 1) + p * cast<float>(r);
    scan.compute_root();

    Func dscan = propagate_tangents(scan, p);
    dscan.compute_root();
    Buffer<float> got = dscan.realize({N});

    // Exact: d(scan(i))/dp = i*(i+1)/2
    for (int i = 0; i < N; i++) {
        float exp_v = (float)(i * (i + 1)) / 2.0f;
        if (std::abs(got(i) - exp_v) > 1e-4f) {
            printf("FAIL test_prefix_sum at [%d]: got %g expected %g\n",
                   i, got(i), exp_v);
            return false;
        }
    }
    printf("test_prefix_sum: OK\n");
    return true;
}

// Test 20: independent param — output does not depend on p → tangent = 0
// f(x,y) = cast<float>(x*x) + cast<float>(y)  (no p)
static bool test_independent_param() {
    Param<float> p;
    p.set(3.0f);
    Var x, y;
    const int W = 6, H = 6;

    Func f("f_indep");
    f(x, y) = cast<float>(x * x) + cast<float>(y);

    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            if (std::abs(got(xx, yy)) > 1e-6f) {
                printf("FAIL test_independent_param at (%d,%d): got %g, expected 0\n",
                       xx, yy, got(xx, yy));
                return false;
            }
        }
    }
    printf("test_independent_param: OK\n");
    return true;
}

// Test 21: 2-D reduction producing a scalar (0-D) tangent
// f(x,y) = p * (x+1) * (y+1)
// loss() = sum_{x,y}(f(x,y))
// d(loss)/dp = sum_{x,y}((x+1)*(y+1)) = (sum_x(x+1)) * (sum_y(y+1))
static bool test_2d_reduction() {
    Param<float> p;
    p.set(1.0f);
    Var x, y;
    const int W = 4, H = 4;

    Func f("f_2dred");
    f(x, y) = p * cast<float>(x + 1) * cast<float>(y + 1);
    f.compute_root();

    RDom rd(0, W, 0, H);
    Func loss("loss_2d");
    loss() = sum(f(rd.x, rd.y));
    loss.compute_root();

    Func dloss = propagate_tangents(loss, p);
    dloss.compute_root();
    Buffer<float> got = dloss.realize();

    // sum_{x=0}^{3} (x+1) = 10;  sum^2 = 100
    float exp_v = 100.0f;
    if (std::abs(got() - exp_v) > 1e-3f) {
        printf("FAIL test_2d_reduction: got %g expected %g\n", got(), exp_v);
        return false;
    }
    printf("test_2d_reduction: OK (d_loss/dp = %g)\n", got());
    return true;
}

// Test 22: basis-vector JVP for Buffer — one-pixel perturbation
// f(x,y) = inp(x,y) * (cast<float>(x) + cast<float>(y) + 1.0f)
// tangent = 1 at (px, py), 0 elsewhere
// d/d(inp(px,py)) f(x,y) = select(x==px && y==py, px+py+1, 0)
static bool test_buffer_basis_vector() {
    const int W = 5, H = 5;
    const int px = 2, py = 1;
    ImageParam inp(Float(32), 2, "inp_bv");
    Var x, y;

    Buffer<float> inp_buf(W, H);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            inp_buf(xx, yy) = (float)(xx * 2 + yy + 1);
    inp.set(inp_buf);

    Func f("f_bv");
    f(x, y) = inp(x, y) * (cast<float>(x) + cast<float>(y) + 1.0f);

    // Basis vector: 1 at (px,py), 0 elsewhere
    Func basis("basis");
    basis(x, y) = select(x == px && y == py, 1.0f, 0.0f);

    Func df = propagate_tangents(f, {{inp.name(), basis}});
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    float exp_at_pixel = (float)(px + py + 1);
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float exp_v = (xx == px && yy == py) ? exp_at_pixel : 0.0f;
            if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                printf("FAIL test_buffer_basis_vector at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_buffer_basis_vector: OK\n");
    return true;
}

// Test 23: multi-param map — differentiate w.r.t. each of two scalar params
// f(x,y) = p1 * cast<float>(x) + p2 * cast<float>(y)
// Directional derivative (1,0): d/dp1 = x
// Directional derivative (0,1): d/dp2 = y
static bool test_multi_param_map() {
    Param<float> p1("p1_mpm"), p2("p2_mpm");
    p1.set(2.0f);
    p2.set(3.0f);
    Var x, y;
    const int W = 5, H = 5;

    Func f("f_mpm");
    f(x, y) = p1 * cast<float>(x) + p2 * cast<float>(y);

    // Direction (1,0): tangent w.r.t. p1
    Func t1_a("t1_a"), t2_a("t2_a");
    t1_a() = 1.0f;
    t2_a() = 0.0f;
    Func df1 = propagate_tangents(f, {{p1.name(), t1_a}, {p2.name(), t2_a}});
    df1.compute_root();
    Buffer<float> got1 = df1.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            if (std::abs(got1(xx, yy) - (float)xx) > 1e-5f) {
                printf("FAIL test_multi_param_map(p1) at (%d,%d): got %g expected %g\n",
                       xx, yy, got1(xx, yy), (float)xx);
                return false;
            }
        }
    }

    // Direction (0,1): tangent w.r.t. p2
    Func t1_b("t1_b"), t2_b("t2_b");
    t1_b() = 0.0f;
    t2_b() = 1.0f;
    Func df2 = propagate_tangents(f, {{p1.name(), t1_b}, {p2.name(), t2_b}});
    df2.compute_root();
    Buffer<float> got2 = df2.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            if (std::abs(got2(xx, yy) - (float)yy) > 1e-5f) {
                printf("FAIL test_multi_param_map(p2) at (%d,%d): got %g expected %g\n",
                       xx, yy, got2(xx, yy), (float)yy);
                return false;
            }
        }
    }
    printf("test_multi_param_map: OK\n");
    return true;
}

// Test 24: propagate_tangents(Func, Buffer<>, Func) — second API overload with concrete Buffer<>
// f(x,y) = buf(x,y) * 3.0f
// Tangent direction: all-ones -> d_f = 3 everywhere
static bool test_concrete_buffer_overload() {
    const int W = 4, H = 4;
    Buffer<float> buf(W, H, "concrete_buf");
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            buf(xx, yy) = (float)(xx + yy * 2 + 1);

    Var x, y;
    Func f("f_cbuf");
    f(x, y) = buf(x, y) * 3.0f;

    Func tangent_dir("td_conc");
    tangent_dir(x, y) = 1.0f;

    // Use the second overload: propagate_tangents(Func, Buffer<>, Func)
    Func df = propagate_tangents(f, buf, tangent_dir);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            if (std::abs(got(xx, yy) - 3.0f) > 1e-5f) {
                printf("FAIL test_concrete_buffer_overload at (%d,%d): got %g expected 3.0\n",
                       xx, yy, got(xx, yy));
                return false;
            }
        }
    }
    printf("test_concrete_buffer_overload: OK\n");
    return true;
}

// Test 25: hyperbolic functions — sinh, cosh, tanh
// f(x,y) = sinh(p * x/W)  d/dp = cosh(p*x/W) * x/W
// g(x,y) = cosh(p * x/W)  d/dp = sinh(p*x/W) * x/W
// h(x,y) = tanh(p * x/W)  d/dp = (x/W) / cosh(p*x/W)^2
static bool test_hyperbolic() {
    Param<float> p;
    p.set(0.5f);
    Var x, y;
    const int W = 5, H = 3;
    float pval = 0.5f;

    {
        Func f("f_sinh");
        f(x, y) = sinh(p * cast<float>(x) / cast<float>(W));
        Func df = propagate_tangents(f, p);
        df.compute_root();
        Buffer<float> got = df.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float frac = (float)xx / (float)W;
                float exp_v = std::cosh(pval * frac) * frac;
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_hyperbolic(sinh) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    {
        Func g("g_cosh");
        g(x, y) = cosh(p * cast<float>(x) / cast<float>(W));
        Func dg = propagate_tangents(g, p);
        dg.compute_root();
        Buffer<float> got = dg.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float frac = (float)xx / (float)W;
                float exp_v = std::sinh(pval * frac) * frac;
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_hyperbolic(cosh) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    {
        Func h("h_tanh");
        h(x, y) = tanh(p * cast<float>(x) / cast<float>(W));
        Func dh = propagate_tangents(h, p);
        dh.compute_root();
        Buffer<float> got = dh.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float frac = (float)xx / (float)W;
                float c = std::cosh(pval * frac);
                float exp_v = frac / (c * c);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_hyperbolic(tanh) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    printf("test_hyperbolic: OK\n");
    return true;
}

// Test 26: zero-tangent cases — floor, ceil, mod
// These are piecewise constant (discontinuous), so their tangent is zero.
static bool test_zero_tangent_cases() {
    Param<float> p;
    p.set(1.7f);
    Var x, y;
    const int W = 5, H = 3;

    // floor(p * x) -> d/dp = 0
    {
        Func f("f_floor");
        f(x, y) = floor(p * cast<float>(x));
        Func df = propagate_tangents(f, p);
        df.compute_root();
        Buffer<float> got = df.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                if (std::abs(got(xx, yy)) > 1e-6f) {
                    printf("FAIL test_zero_tangent(floor) at (%d,%d): got %g expected 0\n",
                           xx, yy, got(xx, yy));
                    return false;
                }
            }
        }
    }
    // ceil(p * x) -> d/dp = 0
    {
        Func g("g_ceil");
        g(x, y) = ceil(p * cast<float>(x));
        Func dg = propagate_tangents(g, p);
        dg.compute_root();
        Buffer<float> got = dg.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                if (std::abs(got(xx, yy)) > 1e-6f) {
                    printf("FAIL test_zero_tangent(ceil) at (%d,%d): got %g expected 0\n",
                           xx, yy, got(xx, yy));
                    return false;
                }
            }
        }
    }
    // (p * (x+1)) % 3 -> d/dp = 0 (Mod is piecewise constant)
    {
        Func h("h_mod");
        h(x, y) = (p * cast<float>(x + 1)) % 3.0f;
        Func dh = propagate_tangents(h, p);
        dh.compute_root();
        Buffer<float> got = dh.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                if (std::abs(got(xx, yy)) > 1e-6f) {
                    printf("FAIL test_zero_tangent(mod) at (%d,%d): got %g expected 0\n",
                           xx, yy, got(xx, yy));
                    return false;
                }
            }
        }
    }
    printf("test_zero_tangent_cases: OK\n");
    return true;
}

// Test 27: Buffer JVP through a multi-stage pipeline
// stage1(x,y) = inp(x,y) * 3.0f
// stage2(x,y) = stage1(x,y) * 2.0f
// Tangent direction: all-ones on inp
// d_stage1 = 3;  d_stage2 = 2 * d_stage1 = 6  everywhere
static bool test_buffer_multistage() {
    const int W = 4, H = 4;
    ImageParam inp(Float(32), 2, "inp_ms");
    Var x, y;

    Buffer<float> inp_buf(W, H);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            inp_buf(xx, yy) = (float)(xx + yy + 1);
    inp.set(inp_buf);

    Func stage1("stage1_ms"), stage2("stage2_ms");
    stage1(x, y) = inp(x, y) * 3.0f;
    stage2(x, y) = stage1(x, y) * 2.0f;
    stage1.compute_root();

    Func tangent_dir("td_ms");
    tangent_dir(x, y) = 1.0f;

    Func df = propagate_tangents(stage2, {{inp.name(), tangent_dir}});
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            if (std::abs(got(xx, yy) - 6.0f) > 1e-5f) {
                printf("FAIL test_buffer_multistage at (%d,%d): got %g expected 6.0\n",
                       xx, yy, got(xx, yy));
                return false;
            }
        }
    }
    printf("test_buffer_multistage: OK\n");
    return true;
}

// Test 28: mixed Param + Buffer — differentiate w.r.t. buffer while param is also present
// f(x,y) = p * inp(x,y) + inp(x,y) * inp(x,y)
// Tangent w.r.t. inp (all-ones direction), p = 2:
// d_f = p * 1 + 2 * inp(x,y) * 1 = p + 2 * inp(x,y)
static bool test_mixed_param_buffer() {
    const int W = 4, H = 4;
    Param<float> p;
    p.set(2.0f);
    ImageParam inp(Float(32), 2, "inp_mix");
    Var x, y;

    Buffer<float> inp_buf(W, H);
    for (int yy = 0; yy < H; yy++)
        for (int xx = 0; xx < W; xx++)
            inp_buf(xx, yy) = (float)(xx + 1);
    inp.set(inp_buf);

    Func f("f_mix");
    f(x, y) = p * inp(x, y) + inp(x, y) * inp(x, y);

    // Differentiate w.r.t. inp (all-ones direction), not w.r.t. p
    Func tangent_dir("td_mix");
    tangent_dir(x, y) = 1.0f;

    Func df = propagate_tangents(f, {{inp.name(), tangent_dir}});
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    float pval = 2.0f;
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float inp_val = inp_buf(xx, yy);
            float exp_v = pval + 2.0f * inp_val;
            if (std::abs(got(xx, yy) - exp_v) > 1e-4f) {
                printf("FAIL test_mixed_param_buffer at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_mixed_param_buffer: OK\n");
    return true;
}

// Test 29: inverse trig functions — asin, acos, tan, atan
// All differentiated w.r.t. p, using inputs in their valid domains.
static bool test_inverse_trig() {
    Param<float> p;
    p.set(0.5f);
    Var x, y;
    const int W = 5, H = 3;
    float pval = 0.5f;

    // asin(p*x/W):  d/dp = (x/W) / sqrt(1 - (p*x/W)^2)
    // Domain: p*x/W in [0, 0.4], well inside (-1, 1).
    {
        Func f("f_asin");
        f(x, y) = asin(p * cast<float>(x) / cast<float>(W));
        Func df = propagate_tangents(f, p);
        df.compute_root();
        Buffer<float> got = df.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float frac = (float)xx / (float)W;
                float a = pval * frac;
                float exp_v = frac / std::sqrt(1.0f - a * a);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_inverse_trig(asin) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    // acos(p*x/W):  d/dp = -(x/W) / sqrt(1 - (p*x/W)^2)
    {
        Func g("g_acos");
        g(x, y) = acos(p * cast<float>(x) / cast<float>(W));
        Func dg = propagate_tangents(g, p);
        dg.compute_root();
        Buffer<float> got = dg.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float frac = (float)xx / (float)W;
                float a = pval * frac;
                float exp_v = -frac / std::sqrt(1.0f - a * a);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_inverse_trig(acos) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    // tan(p*x/W):  d/dp = (x/W) / cos(p*x/W)^2
    // Domain: p*x/W in [0, 0.4], well away from pi/2.
    {
        Func h("h_tan");
        h(x, y) = tan(p * cast<float>(x) / cast<float>(W));
        Func dh = propagate_tangents(h, p);
        dh.compute_root();
        Buffer<float> got = dh.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float frac = (float)xx / (float)W;
                float c = std::cos(pval * frac);
                float exp_v = frac / (c * c);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_inverse_trig(tan) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    // atan(p*x/W):  d/dp = (x/W) / (1 + (p*x/W)^2)
    {
        Func k("k_atan");
        k(x, y) = atan(p * cast<float>(x) / cast<float>(W));
        Func dk = propagate_tangents(k, p);
        dk.compute_root();
        Buffer<float> got = dk.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float frac = (float)xx / (float)W;
                float a = pval * frac;
                float exp_v = frac / (1.0f + a * a);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_inverse_trig(atan) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    printf("test_inverse_trig: OK\n");
    return true;
}

// Test 30: inverse hyperbolic functions — asinh, acosh, atanh
static bool test_inverse_hyperbolic() {
    Param<float> p;
    p.set(0.5f);
    Var x, y;
    const int W = 5, H = 3;
    float pval = 0.5f;

    // asinh(p*x/W):  d/dp = (x/W) / sqrt(1 + (p*x/W)^2)
    // Domain: all reals.
    {
        Func f("f_asinh");
        f(x, y) = asinh(p * cast<float>(x) / cast<float>(W));
        Func df = propagate_tangents(f, p);
        df.compute_root();
        Buffer<float> got = df.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float frac = (float)xx / (float)W;
                float a = pval * frac;
                float exp_v = frac / std::sqrt(1.0f + a * a);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_inverse_hyperbolic(asinh) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    // acosh(1 + p*(x+1)/W):  domain requires input > 1 — the +1 offset ensures this.
    // a = 1 + p*(x+1)/W,  da/dp = (x+1)/W
    // d/dp = da/dp / (sqrt(a-1) * sqrt(a+1))
    {
        Func g("g_acosh");
        g(x, y) = acosh(1.0f + p * cast<float>(x + 1) / cast<float>(W));
        Func dg = propagate_tangents(g, p);
        dg.compute_root();
        Buffer<float> got = dg.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float da = (float)(xx + 1) / (float)W;
                float a = 1.0f + pval * da;
                float exp_v = da / (std::sqrt(a - 1.0f) * std::sqrt(a + 1.0f));
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_inverse_hyperbolic(acosh) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    // atanh(p*x/W):  d/dp = (x/W) / (1 - (p*x/W)^2)
    // Domain: |input| < 1 — p*x/W in [0, 0.4], safe.
    {
        Func h("h_atanh");
        h(x, y) = atanh(p * cast<float>(x) / cast<float>(W));
        Func dh = propagate_tangents(h, p);
        dh.compute_root();
        Buffer<float> got = dh.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float frac = (float)xx / (float)W;
                float a = pval * frac;
                float exp_v = frac / (1.0f - a * a);
                if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                    printf("FAIL test_inverse_hyperbolic(atanh) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    printf("test_inverse_hyperbolic: OK\n");
    return true;
}

// Test 31: fast_inverse and fast_inverse_sqrt
// fast_inverse(a) ≈ 1/a;  d/dp fast_inverse(p*(x+1)) ≈ -(x+1)/(p*(x+1))^2 = -1/(p^2*(x+1))
// fast_inverse_sqrt(a) ≈ 1/sqrt(a);  d/dp ≈ -0.5*(x+1)/(p*(x+1))^(3/2)
// Both use hardware approximations (~12-bit mantissa accuracy on x86 rcpps/rsqrtps),
// so we use a loose tolerance of 1e-2 (1%) to accommodate the approximation error.
static bool test_fast_math() {
    Param<float> p;
    p.set(1.0f);
    Var x, y;
    const int W = 4, H = 3;
    float pval = 1.0f;

    {
        Func f("f_finv");
        f(x, y) = fast_inverse(p * cast<float>(x + 1));
        Func df = propagate_tangents(f, p);
        df.compute_root();
        Buffer<float> got = df.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float a = pval * (float)(xx + 1);
                // Exact: d/dp [1/(p*(x+1))] = -(x+1)/(p*(x+1))^2 = -1/(p^2*(x+1))
                float exp_v = -1.0f / (pval * pval * (float)(xx + 1));
                float rel_tol = 1e-2f * std::abs(exp_v);
                if (std::abs(got(xx, yy) - exp_v) > std::max(rel_tol, 1e-6f)) {
                    printf("FAIL test_fast_math(fast_inverse) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
                (void)a;
            }
        }
    }
    {
        Func g("g_finvsqrt");
        g(x, y) = fast_inverse_sqrt(p * cast<float>(x + 1));
        Func dg = propagate_tangents(g, p);
        dg.compute_root();
        Buffer<float> got = dg.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                float a = pval * (float)(xx + 1);
                // Exact: d/dp [1/sqrt(p*(x+1))] = -(x+1) / (2*(p*(x+1))^(3/2))
                float exp_v = -(float)(xx + 1) / (2.0f * std::pow(a, 1.5f));
                float rel_tol = 1e-2f * std::abs(exp_v);
                if (std::abs(got(xx, yy) - exp_v) > std::max(rel_tol, 1e-6f)) {
                    printf("FAIL test_fast_math(fast_inverse_sqrt) at (%d,%d): got %g expected %g\n",
                           xx, yy, got(xx, yy), exp_v);
                    return false;
                }
            }
        }
    }
    printf("test_fast_math: OK\n");
    return true;
}

// Test 32: likely() intrinsic pass-through
// likely() wraps a value with a branch-prediction hint but doesn't change its value.
// The tangent should pass through unchanged.
// f(x,y) = likely(p * cast<float>(x));  d/dp = cast<float>(x)
static bool test_likely_passthrough() {
    Param<float> p;
    p.set(2.0f);
    Var x, y;
    const int W = 6, H = 3;

    Func f("f_likely");
    f(x, y) = likely(p * cast<float>(x));
    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float exp_v = (float)xx;
            if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                printf("FAIL test_likely_passthrough at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_likely_passthrough: OK\n");
    return true;
}

// Test 33: bitwise operations — zero tangent
// Bitwise ops are piecewise constant on integers; their tangent is zero.
// f(x,y) = p * cast<float>(x & 3)
// d/dp = cast<float>(x & 3)   (the & has zero tangent; p's tangent is 1)
static bool test_bitwise_zero_tangent() {
    Param<float> p;
    p.set(2.0f);
    Var x, y;
    const int W = 8, H = 3;

    Func f("f_bitwise");
    f(x, y) = p * cast<float>(x & 3);
    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float exp_v = (float)(xx & 3);
            if (std::abs(got(xx, yy) - exp_v) > 1e-5f) {
                printf("FAIL test_bitwise_zero_tangent at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_bitwise_zero_tangent: OK\n");
    return true;
}

// Test 34: round and trunc — zero tangent (completes rounding coverage alongside test 26)
// round()/trunc() are piecewise constant like floor()/ceil().
static bool test_round_trunc_zero_tangent() {
    Param<float> p;
    p.set(1.7f);
    Var x, y;
    const int W = 5, H = 3;

    {
        Func f("f_round");
        f(x, y) = round(p * cast<float>(x));
        Func df = propagate_tangents(f, p);
        df.compute_root();
        Buffer<float> got = df.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                if (std::abs(got(xx, yy)) > 1e-6f) {
                    printf("FAIL test_round_trunc(round) at (%d,%d): got %g expected 0\n",
                           xx, yy, got(xx, yy));
                    return false;
                }
            }
        }
    }
    {
        Func g("g_trunc");
        g(x, y) = trunc(p * cast<float>(x));
        Func dg = propagate_tangents(g, p);
        dg.compute_root();
        Buffer<float> got = dg.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                if (std::abs(got(xx, yy)) > 1e-6f) {
                    printf("FAIL test_round_trunc(trunc) at (%d,%d): got %g expected 0\n",
                           xx, yy, got(xx, yy));
                    return false;
                }
            }
        }
    }
    printf("test_round_trunc_zero_tangent: OK\n");
    return true;
}

// Test 35: Let expression differentiation
// Let nodes appear when Halide's IR contains shared subexpressions.
// We construct one explicitly to verify the nested Let output structure:
//   diff(let v = val in body) = let v = val in let d_v = d(val) in d(body)
//
// f(x,y) = let v = p*x in v*v  = (p*x)^2
// d/dp = let v = p*x in let d_v = x in 2*v*d_v  = 2*(p*x)*x
static bool test_let_expression() {
    Param<float> p;
    p.set(2.0f);
    Var x, y;
    const int W = 4, H = 3;
    float pval = 2.0f;

    // Build the Let expression directly via the internal IR API.
    // Let::make("v", val, body) creates: let v = val in body.
    Expr v_val = p * cast<float>(x);
    Expr v_ref = Internal::Variable::make(Float(32), "v_let_test");
    Func f("f_let");
    f(x, y) = Internal::Let::make("v_let_test", v_val, v_ref * v_ref);

    Func df = propagate_tangents(f, p);
    df.compute_root();
    Buffer<float> got = df.realize({W, H});

    // d/dp (p*x)^2 = 2*(p*x)*x
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            float exp_v = 2.0f * pval * (float)xx * (float)xx;
            if (std::abs(got(xx, yy) - exp_v) > 1e-4f) {
                printf("FAIL test_let_expression at (%d,%d): got %g expected %g\n",
                       xx, yy, got(xx, yy), exp_v);
                return false;
            }
        }
    }
    printf("test_let_expression: OK\n");
    return true;
}

int main(int argc, char **argv) {
    bool ok = true;
#define RUN(t) do { printf("--- " #t " ---\n"); fflush(stdout); ok &= t(); } while(0)
    RUN(test_pointwise);
    RUN(test_multistage);
    RUN(test_tuple);
    RUN(test_rdom);
    RUN(test_buffer_jvp);
    RUN(test_math_funcs);
    RUN(test_inverse_rendering);
    RUN(test_cross_validate);
    RUN(test_division);
    RUN(test_min_max);
    RUN(test_select_branches);
    RUN(test_cos);
    RUN(test_log_sqrt);
    RUN(test_pow_exponent);
    RUN(test_atan2);
    RUN(test_abs_intrinsic);
    RUN(test_lerp_intrinsic);
    RUN(test_three_stage_chain);
    RUN(test_prefix_sum);
    RUN(test_independent_param);
    RUN(test_2d_reduction);
    RUN(test_buffer_basis_vector);
    RUN(test_multi_param_map);
    RUN(test_concrete_buffer_overload);
    RUN(test_hyperbolic);
    RUN(test_zero_tangent_cases);
    RUN(test_buffer_multistage);
    RUN(test_mixed_param_buffer);
    RUN(test_inverse_trig);
    RUN(test_inverse_hyperbolic);
    RUN(test_fast_math);
    RUN(test_likely_passthrough);
    RUN(test_bitwise_zero_tangent);
    RUN(test_round_trunc_zero_tangent);
    RUN(test_let_expression);
#undef RUN

    if (ok) {
        printf("[forward_diff] Success!\n");
        return 0;
    }
    printf("[forward_diff] FAILED\n");
    return 1;
}
