##
# Sioutas2020 GPU Autoscheduler
#
# Heuristic-based GPU autoscheduler based on:
#   "Scheduling Halide Pipelines on GPUs" — Sioutas et al., TACO 2020.
#   https://github.com/TUE-EE-ES/HalideAutoGPU
#
# The scheduler inlines trivial and element-wise functions then applies
# gpu_tile() to all remaining pure functions.  Reduction update stages
# are tiled when safe or fall back to gpu_single_thread().
##

add_autoscheduler(
    NAME Sioutas2020
    SOURCES
    AutoSchedule.cpp
)
