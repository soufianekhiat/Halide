cmake_minimum_required(VERSION 3.28)
project(Sioutas2020Demo CXX)

# ---------------------------------------------------------------------------
# Locate Halide from the install tree
# ---------------------------------------------------------------------------
set(Halide_DIR        "C:/git/Halide/install_455b34b/lib/cmake/Halide")
set(HalideHelpers_DIR "C:/git/Halide/install_455b34b/lib/cmake/HalideHelpers")
find_package(Halide REQUIRED)

# ---------------------------------------------------------------------------
# Build all 6 demo executables
# ---------------------------------------------------------------------------
set(DEMOS
    demo_gaussian_blur
    demo_harris_corner
    demo_matmul
    demo_histogram_eq
    demo_unsharp_mask
    demo_conv2d_relu
)

foreach(demo IN LISTS DEMOS)
    add_executable(${demo} ${demo}.cpp)
    target_link_libraries(${demo} PRIVATE Halide::Halide)
    # demo_common.h is a header-only include from the source directory
    target_include_directories(${demo} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
