#Requires -Version 5.1
<#
.SYNOPSIS
    Build all tutorial lesson targets from an existing configured Halide build.
.DESCRIPTION
    Builds the standard JIT lessons (01-09, 11-14, 17-24) and the multi-step
    generator-based lessons (10, 15, 16, 21) in the correct dependency order.
.PARAMETER Config
    Build configuration: Release (default), Debug, or RelWithDebInfo.
.PARAMETER Jobs
    Parallel job count. Defaults to the number of logical CPU cores.
.EXAMPLE
    .\build_Lessons.ps1
    .\build_Lessons.ps1 -Config Debug
#>
param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string] $Config = 'Release',
    [int]    $Jobs   = [Environment]::ProcessorCount
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --- Paths -------------------------------------------------------------------
$HalideRoot = Split-Path $PSScriptRoot -Parent
$BuildDir   = Join-Path $HalideRoot 'build\win64'
$CacheFile  = Join-Path $BuildDir   'CMakeCache.txt'

Write-Host ""
Write-Host "  Build dir : $BuildDir" -ForegroundColor Cyan
Write-Host "  Config    : $Config"   -ForegroundColor Cyan
Write-Host "  Jobs      : $Jobs"     -ForegroundColor Cyan
Write-Host ""

# --- Guard -------------------------------------------------------------------
if (-not (Test-Path $CacheFile)) {
    Write-Error "Build directory not configured: $BuildDir`nRun build_Halide.ps1 first."
}

# --- Helper ------------------------------------------------------------------
function Invoke-Build {
    param([string[]] $Targets)
    # Use ';' (not ',') so each iteration emits two separate pipeline items,
    # producing a flat array: '--target', 't1', '--target', 't2', ...
    $targetArgs = @($Targets | ForEach-Object { '--target'; $_ })
    cmake --build $BuildDir --config $Config --parallel $Jobs @targetArgs
    if ($LASTEXITCODE -ne 0) { throw "Build failed for targets: $($Targets -join ', ')" }
}

# --- Step 1: JIT lessons (no special build order required) -------------------
$jitLessons = @(
    'lesson_01_basics',
    'lesson_02_input_image',
    'lesson_03_debugging_1',
    'lesson_04_debugging_2',
    'lesson_05_scheduling_1',
    'lesson_06_realizing_over_shifted_domains',
    'lesson_07_multi_stage_pipelines',
    'lesson_08_scheduling_2',
    'lesson_09_update_definitions',
    'lesson_11_cross_compilation',
    'lesson_12_using_the_gpu',
    'lesson_13_tuples',
    'lesson_14_types',
    'lesson_17_predicated_rdom',
    'lesson_18_parallel_associative_reductions',
    'lesson_19_wrapper_funcs',
    'lesson_20_cloning_funcs',
    'lesson_22_jit_performance',
    'lesson_23_serialization',
    'lesson_24_async'
)

Write-Host "Building JIT lessons..." -ForegroundColor Yellow
Invoke-Build $jitLessons
Write-Host "  JIT lessons done." -ForegroundColor Green

# --- Step 2: Lesson 10 - AOT (generate first, then the runner) ---------------
Write-Host ""
Write-Host "Building lesson 10 (AOT)..." -ForegroundColor Yellow
Invoke-Build @('lesson_10_aot_compilation_generate')
Invoke-Build @('exec_lesson_10_aot_compilation_generate')
Invoke-Build @('lesson_10_aot_compilation_run')
Write-Host "  Lesson 10 done." -ForegroundColor Green

# --- Step 3: Lesson 15 - Generator variants ----------------------------------
Write-Host ""
Write-Host "Building lesson 15 (generator variants)..." -ForegroundColor Yellow
Invoke-Build @('lesson_15_generate')
Invoke-Build @('lesson_15_targets')
Write-Host "  Lesson 15 done." -ForegroundColor Green

# --- Step 4: Lesson 16 - RGB generator ---------------------------------------
Write-Host ""
Write-Host "Building lesson 16 (RGB generator)..." -ForegroundColor Yellow
Invoke-Build @('lesson_16_rgb_generate')
Invoke-Build @(
    'brighten_planar', 'brighten_interleaved',
    'brighten_either', 'brighten_specialized',
    'lesson_16_rgb_run'
)
Write-Host "  Lesson 16 done." -ForegroundColor Green

# --- Step 5: Lesson 21 - Auto-scheduler --------------------------------------
Write-Host ""
Write-Host "Building lesson 21 (auto-scheduler)..." -ForegroundColor Yellow
Invoke-Build @('lesson_21_auto_scheduler_generate')
Invoke-Build @('auto_schedule_false', 'auto_schedule_true')
Invoke-Build @('lesson_21_auto_scheduler_run')
Write-Host "  Lesson 21 done." -ForegroundColor Green

# --- Summary -----------------------------------------------------------------
Write-Host ""
$BinDir = Join-Path $BuildDir "bin\$Config"
Write-Host "All lessons built. Executables in: $BinDir" -ForegroundColor Green
