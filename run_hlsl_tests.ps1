# Run HLSL correctness and generator tests
# Usage: .\run_hlsl_tests.ps1 [-Config Release|Debug] [-SM <60|62|64|66>]

param(
    [string]$Config = "Release",
    [string]$SM     = "60"
)

$ErrorActionPreference = "Stop"
$root   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$binDir = Join-Path $root "build_hlsl_tests\bin\$Config"

if (-not (Test-Path $binDir)) {
    Write-Error "No build found at $binDir. Run .\build_hlsl_tests.ps1 first."
    exit 1
}

# Always prefer install_halide_only (built with SM6 support) for Halide.dll,
# falling back to the build dir, then older installs.
$halideInstallBin = ""
foreach ($candidate in @(
    "install_halide_only\bin",
    "build_halide_only\bin\Release",
    "build_halide_only\bin\Debug",
    "install_455b34b_HLSL6\bin",
    "install_455b34b\bin"
)) {
    $p = Join-Path $root $candidate
    if (Test-Path (Join-Path $p "Halide.dll")) { $halideInstallBin = $p; break }
}
Write-Host "Halide.dll dir = $halideInstallBin" -ForegroundColor Cyan

# dxcompiler.dll is not installed - it lives in the build tree
$dxcDir = ""
foreach ($c in @(
    (Join-Path $root "build_halide_only\bin\Release"),
    (Join-Path $root "build_halide_only\bin\Debug"),
    (Join-Path $root "build_455b34b\bin\Release"),
    (Join-Path $root "build_455b34b\bin\Debug")
)) {
    if (Test-Path (Join-Path $c "dxcompiler.dll")) { $dxcDir = $c; break }
}
if (-not $dxcDir) {
    Write-Warning "dxcompiler.dll not found - hlsl_sm6x JIT tests will fail."
} else {
    Write-Host "dxcompiler.dll = $dxcDir" -ForegroundColor Cyan
}

$env:PATH = "$halideInstallBin;$dxcDir;$env:PATH"

$gpuTarget = "x86-64-windows-d3d12compute-hlsl_sm$SM"
Write-Host "GPU JIT target = $gpuTarget" -ForegroundColor Cyan

$pass = 0
$fail = 0

function Run-Test {
    param([string]$exe, [string]$label, [string]$jitTarget = "")
    Write-Host "`n-- $label" -ForegroundColor Yellow
    if (-not (Test-Path $exe)) {
        Write-Host "  [MISSING] $exe" -ForegroundColor DarkYellow
        return
    }
    $env:HL_JIT_TARGET = $jitTarget
    & $exe
    $env:HL_JIT_TARGET = ""
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [PASS]" -ForegroundColor Green
        $script:pass++
    } else {
        Write-Host "  [FAIL] exit $LASTEXITCODE" -ForegroundColor Red
        $script:fail++
    }
}

# CPU-only correctness tests - no HL_JIT_TARGET override
Run-Test (Join-Path $binDir "correctness_cross_compilation.exe") "correctness/cross_compilation"
Run-Test (Join-Path $binDir "correctness_math.exe")              "correctness/math"
Run-Test (Join-Path $binDir "correctness_newtons_method.exe")    "correctness/newtons_method"

# D3D12 / HLSL correctness tests - need the GPU target
Run-Test (Join-Path $binDir "correctness_d3d12compute_sm6x.exe")          "correctness/d3d12compute_sm6x"          $gpuTarget
Run-Test (Join-Path $binDir "correctness_d3d12compute_strict_float.exe")  "correctness/d3d12compute_strict_float"  $gpuTarget
Run-Test (Join-Path $binDir "correctness_gpu_mixed_shared_mem_types.exe") "correctness/gpu_mixed_shared_mem_types" $gpuTarget
Run-Test (Join-Path $binDir "correctness_gpu_texture.exe")                "correctness/gpu_texture"                $gpuTarget
Run-Test (Join-Path $binDir "correctness_math.exe")                       "correctness/math (GPU)"                 $gpuTarget
Run-Test (Join-Path $binDir "correctness_newtons_method.exe")             "correctness/newtons_method (GPU)"       $gpuTarget

# 64-bit JIT test (double, uint64, int64 on D3D12)
Run-Test (Join-Path $binDir "test_64bit_jit.exe")                         "test_64bit_jit"                         $gpuTarget

# Generator AOT test (no HL_JIT_TARGET - AOT binary, runs natively)
Run-Test (Join-Path $binDir "generator_gpu_texture_aottest.exe") "generator/gpu_texture_aottest"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Results: $pass passed, $fail failed" -ForegroundColor $(if ($fail -gt 0) { "Red" } else { "Green" })
if ($fail -gt 0) { exit 1 }
