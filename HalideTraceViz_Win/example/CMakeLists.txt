cmake_minimum_required(VERSION 3.28)
project(halide_blur_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Halide_DIR / HalideHelpers_DIR are passed from build_Example.ps1.
find_package(Halide REQUIRED)

add_executable(tiled_blur tiled_blur.cpp)
target_link_libraries(tiled_blur PRIVATE
    Halide::Halide
    Halide::ImageIO
    Halide::Tools)

add_executable(unsharp_mask unsharp_mask.cpp)
target_link_libraries(unsharp_mask PRIVATE
    Halide::Halide
    Halide::ImageIO
    Halide::Tools)

# Keep the outputs next to the binary so the named-pipe workflow stays
# straightforward.
set_target_properties(tiled_blur unsharp_mask PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
