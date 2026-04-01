cmake_minimum_required(VERSION 3.16)
project(HalideImageDiff CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# ── Locate the Halide build tree ──────────────────────────────────────────────
# This project links against the sk/Image_Diff build of Halide, which contains
# propagate_tangents() in Halide.h. Adjust HALIDE_BUILD_DIR if the build is
# in a different location.
set(HALIDE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build_455b34b"
    CACHE PATH "Halide build directory (sk/Image_Diff branch)")

set(HALIDE_INCLUDE_DIR  "${HALIDE_BUILD_DIR}/include")        # Halide.h
set(HALIDE_RUNTIME_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime")
set(HALIDE_LIB          "${HALIDE_BUILD_DIR}/src/Release/Halide.lib")
set(HALIDE_DLL          "${HALIDE_BUILD_DIR}/bin/Release/Halide.dll")
set(HALIDE_TOOLS_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/../tools")

if (NOT EXISTS "${HALIDE_INCLUDE_DIR}/Halide.h")
    message(FATAL_ERROR "Halide.h not found at ${HALIDE_INCLUDE_DIR}. "
            "Set HALIDE_BUILD_DIR to the build directory of the sk/Image_Diff branch.")
endif ()
if (NOT EXISTS "${HALIDE_LIB}")
    message(FATAL_ERROR "Halide.lib not found at ${HALIDE_LIB}. "
            "Build the Halide library first: cmake --build ${HALIDE_BUILD_DIR} --config Release")
endif ()

# Shared include dirs and libs for all targets
set(IMAGEDIFF_INCS
    "${HALIDE_INCLUDE_DIR}"        # Halide.h  (contains propagate_tangents)
    "${HALIDE_RUNTIME_DIR}"        # HalideRuntime.h (needed by some internal headers)
    "${HALIDE_TOOLS_DIR}"          # halide_benchmark.h etc.
    "${CMAKE_CURRENT_SOURCE_DIR}"  # stb_image.h / stb_image_write.h
)

# ── estimate_blur_radius ──────────────────────────────────────────────────────
# 1 parameter (sigma), 1 JVP pass per gradient step.
# Recovers Gaussian blur radius from a 1920×1080 image.
add_executable(estimate_blur_radius estimate_blur_radius.cpp)
target_include_directories(estimate_blur_radius PRIVATE ${IMAGEDIFF_INCS})
target_link_libraries(estimate_blur_radius PRIVATE "${HALIDE_LIB}")
target_compile_options(estimate_blur_radius PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wall>
)

# Copy Halide.dll next to the executable so it can be run without PATH changes
add_custom_command(TARGET estimate_blur_radius POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HALIDE_DLL}" "$<TARGET_FILE_DIR:estimate_blur_radius>"
    COMMENT "Copying Halide.dll")

# ── estimate_psf ──────────────────────────────────────────────────────────────
# 49 parameters (7×7 PSF), 49 JVP passes per gradient step.
# Recovers an unknown PSF from a 1920×1080 image.
add_executable(estimate_psf estimate_psf.cpp)
target_include_directories(estimate_psf PRIVATE ${IMAGEDIFF_INCS})
target_link_libraries(estimate_psf PRIVATE "${HALIDE_LIB}")
target_compile_options(estimate_psf PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wall>
)

add_custom_command(TARGET estimate_psf POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HALIDE_DLL}" "$<TARGET_FILE_DIR:estimate_psf>"
    COMMENT "Copying Halide.dll")

# ── estimate_tone_curve ──────────────────────────────────────────────────────
# 2 parameters (gamma, gain), 2 JVP passes per gradient step.
# Recovers a tone curve from a 720×1080 image.
# Showcase: saves per-pixel sensitivity maps (tangent images).
add_executable(estimate_tone_curve estimate_tone_curve.cpp)
target_include_directories(estimate_tone_curve PRIVATE ${IMAGEDIFF_INCS})
target_link_libraries(estimate_tone_curve PRIVATE "${HALIDE_LIB}")
target_compile_options(estimate_tone_curve PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wall>
)

add_custom_command(TARGET estimate_tone_curve POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HALIDE_DLL}" "$<TARGET_FILE_DIR:estimate_tone_curve>"
    COMMENT "Copying Halide.dll")

# ── deblur_cg ────────────────────────────────────────────────────────────────
# Image deblurring via CG using ImageParam JVP.
# Symmetric blur → J = J^T → A*v = blur(blur(v)) + lambda*v (two JVP calls).
add_executable(deblur_cg deblur_cg.cpp)
target_include_directories(deblur_cg PRIVATE ${IMAGEDIFF_INCS})
target_link_libraries(deblur_cg PRIVATE "${HALIDE_LIB}")
target_compile_options(deblur_cg PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wall>
)

add_custom_command(TARGET deblur_cg POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HALIDE_DLL}" "$<TARGET_FILE_DIR:deblur_cg>"
    COMMENT "Copying Halide.dll")

# ── optimize_texture ─────────────────────────────────────────────────────────
# Texture recovery from shaded rendering via CG.
# Asymmetric operator: J*v via JVP, J^T*w via host-side blur.
add_executable(optimize_texture optimize_texture.cpp)
target_include_directories(optimize_texture PRIVATE ${IMAGEDIFF_INCS})
target_link_libraries(optimize_texture PRIVATE "${HALIDE_LIB}")
target_compile_options(optimize_texture PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wall>
)

add_custom_command(TARGET optimize_texture POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HALIDE_DLL}" "$<TARGET_FILE_DIR:optimize_texture>"
    COMMENT "Copying Halide.dll")

# ── estimate_normals_albedo ──────────────────────────────────────────────────
# Joint normal map + albedo recovery from directional lighting (RGB).
# 3D ImageParam JVP, 3 basis-vector JVP calls per iteration + closed-form albedo.
add_executable(estimate_normals_albedo estimate_normals_albedo.cpp)
target_include_directories(estimate_normals_albedo PRIVATE ${IMAGEDIFF_INCS})
target_link_libraries(estimate_normals_albedo PRIVATE "${HALIDE_LIB}")
target_compile_options(estimate_normals_albedo PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2 -Wall>
)

add_custom_command(TARGET estimate_normals_albedo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${HALIDE_DLL}" "$<TARGET_FILE_DIR:estimate_normals_albedo>"
    COMMENT "Copying Halide.dll")
