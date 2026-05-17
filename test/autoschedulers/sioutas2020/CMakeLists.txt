##
# Tests for the Sioutas2020 autoscheduler
#
# Test groups used:
#   sioutas2020           — all Sioutas2020 tests
#   autoschedulers_cpu    — runs without GPU hardware
#   autoschedulers_cuda   — requires NVPTX in LLVM + CUDA hardware for exec
#   multithreaded         — may use multiple CPU threads
##

# =================================================================
# test.cpp — correctness test
#
# Uses apply_autoscheduler() only (no JIT compilation, no GPU HW).
# Both GPU and CPU targets are exercised by checking schedule strings.
# Placed in autoschedulers_cpu so it always runs in CI.
# =================================================================

add_executable(sioutas2020_test_apps_autoscheduler test.cpp)
target_link_libraries(sioutas2020_test_apps_autoscheduler
                      PRIVATE Halide::Halide Halide::Tools ${CMAKE_DL_LIBS})
add_dependencies(sioutas2020_test_apps_autoscheduler Halide::Sioutas2020)

add_halide_test(sioutas2020_test_apps_autoscheduler
                COMMAND sioutas2020_test_apps_autoscheduler
                        $<TARGET_FILE:Halide_Sioutas2020>
                GROUPS sioutas2020 autoschedulers_cpu multithreaded)

set_tests_properties(sioutas2020_test_apps_autoscheduler PROPERTIES
                     ENVIRONMENT_MODIFICATION
                     "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:Halide_Sioutas2020>")

# =================================================================
# schedule_compare — diagnostic: print both schedules side-by-side
# =================================================================

add_executable(sioutas2020_schedule_compare schedule_compare.cpp)
target_link_libraries(sioutas2020_schedule_compare
                      PRIVATE Halide::Halide Halide::Tools ${CMAKE_DL_LIBS})
add_dependencies(sioutas2020_schedule_compare
                 Halide::Sioutas2020
                 Halide::Anderson2021)

add_halide_test(sioutas2020_schedule_compare
                COMMAND sioutas2020_schedule_compare
                        $<TARGET_FILE:Halide_Sioutas2020>
                        $<TARGET_FILE:Halide_Anderson2021>
                GROUPS sioutas2020 autoschedulers_cpu multithreaded)

set_tests_properties(sioutas2020_schedule_compare PROPERTIES
                     ENVIRONMENT_MODIFICATION
                     "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:Halide_Sioutas2020>;$<TARGET_FILE_DIR:Halide_Anderson2021>")

# =================================================================
# bench_generator.cpp — GPU execution benchmarks
#
# The same generator is compiled twice (different autoschedulers) so
# RunGenMain can be used to compare wall-clock throughput on real GPU HW.
# These targets are grouped under autoschedulers_cuda and only built when
# NVPTX is available.
# =================================================================

if ("NVPTX" IN_LIST Halide_LLVM_COMPONENTS)

    add_halide_generator(sioutas2020_bench.generator
                         SOURCES bench_generator.cpp)

    # ---- GaussianSobel compiled with Sioutas2020 ----
    add_halide_library(sioutas2020_bench_gs_s FROM sioutas2020_bench.generator
                       GENERATOR gaussian_sobel
                       TARGETS cmake
                       FEATURES cuda cuda_capability_70
                       AUTOSCHEDULER Halide::Sioutas2020
                       REGISTRATION SIOUTAS_GS_S_REG)

    add_executable(sioutas2020_bench_gs_sioutas ${SIOUTAS_GS_S_REG})
    target_link_libraries(sioutas2020_bench_gs_sioutas
                          PRIVATE sioutas2020_bench_gs_s Halide::RunGenMain)
    add_dependencies(sioutas2020_bench_gs_sioutas Halide::Sioutas2020)

    add_halide_test(sioutas2020_bench_gs_sioutas
                    COMMAND sioutas2020_bench_gs_sioutas
                            --benchmarks=all --benchmark_min_time=1
                            --estimate_all --success --skip_bad_environment
                    GROUPS sioutas2020 autoschedulers_cuda multithreaded)

    # ---- GaussianSobel compiled with Anderson2021 ----
    add_halide_library(sioutas2020_bench_gs_a FROM sioutas2020_bench.generator
                       GENERATOR gaussian_sobel
                       TARGETS cmake
                       FEATURES cuda cuda_capability_70
                       AUTOSCHEDULER Halide::Anderson2021
                       REGISTRATION SIOUTAS_GS_A_REG)

    add_executable(sioutas2020_bench_gs_anderson ${SIOUTAS_GS_A_REG})
    target_link_libraries(sioutas2020_bench_gs_anderson
                          PRIVATE sioutas2020_bench_gs_a Halide::RunGenMain)
    add_dependencies(sioutas2020_bench_gs_anderson Halide::Anderson2021)

    add_halide_test(sioutas2020_bench_gs_anderson
                    COMMAND sioutas2020_bench_gs_anderson
                            --benchmarks=all --benchmark_min_time=1
                            --estimate_all --success --skip_bad_environment
                    GROUPS sioutas2020 autoschedulers_cuda multithreaded)

    # ---- LocalStats compiled with Sioutas2020 ----
    add_halide_library(sioutas2020_bench_ls_s FROM sioutas2020_bench.generator
                       GENERATOR local_stats
                       TARGETS cmake
                       FEATURES cuda cuda_capability_70
                       AUTOSCHEDULER Halide::Sioutas2020
                       REGISTRATION SIOUTAS_LS_S_REG)

    add_executable(sioutas2020_bench_ls_sioutas ${SIOUTAS_LS_S_REG})
    target_link_libraries(sioutas2020_bench_ls_sioutas
                          PRIVATE sioutas2020_bench_ls_s Halide::RunGenMain)
    add_dependencies(sioutas2020_bench_ls_sioutas Halide::Sioutas2020)

    add_halide_test(sioutas2020_bench_ls_sioutas
                    COMMAND sioutas2020_bench_ls_sioutas
                            --benchmarks=all --benchmark_min_time=1
                            --estimate_all --success --skip_bad_environment
                    GROUPS sioutas2020 autoschedulers_cuda multithreaded)

    # ---- LocalStats compiled with Anderson2021 ----
    add_halide_library(sioutas2020_bench_ls_a FROM sioutas2020_bench.generator
                       GENERATOR local_stats
                       TARGETS cmake
                       FEATURES cuda cuda_capability_70
                       AUTOSCHEDULER Halide::Anderson2021
                       REGISTRATION SIOUTAS_LS_A_REG)

    add_executable(sioutas2020_bench_ls_anderson ${SIOUTAS_LS_A_REG})
    target_link_libraries(sioutas2020_bench_ls_anderson
                          PRIVATE sioutas2020_bench_ls_a Halide::RunGenMain)
    add_dependencies(sioutas2020_bench_ls_anderson Halide::Anderson2021)

    add_halide_test(sioutas2020_bench_ls_anderson
                    COMMAND sioutas2020_bench_ls_anderson
                            --benchmarks=all --benchmark_min_time=1
                            --estimate_all --success --skip_bad_environment
                    GROUPS sioutas2020 autoschedulers_cuda multithreaded)

endif ()  # NVPTX
