# Build HLSL correctness and generator tests
# Usage: .\build_hlsl_tests.ps1 [-Config Release|Debug] [-HalideDir <path>]

param(
    [string]$Config    = "Release",
    [string]$HalideDir = ""
)

$ErrorActionPreference = "Stop"
$root     = Split-Path -Parent $MyInvocation.MyCommand.Definition
$buildDir = Join-Path $root "build_hlsl_tests"

# Auto-detect Halide install — prefer install_halide_only (built with SM6 support)
if (-not $HalideDir) {
    $candidates = @(
        (Join-Path $root "install_halide_only\lib\cmake\Halide"),
        (Join-Path $root "install_455b34b_HLSL6\lib\cmake\Halide"),
        (Join-Path $root "install_455b34b\lib\cmake\Halide")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $HalideDir = (Resolve-Path $c).Path
            break
        }
    }
}

if (-not $HalideDir) {
    Write-Error "Could not find Halide. Pass -HalideDir <path>/lib/cmake/Halide"
    exit 1
}

Write-Host "Config     = $Config"    -ForegroundColor Cyan
Write-Host "HalideDir  = $HalideDir" -ForegroundColor Cyan
Write-Host "BuildDir   = $buildDir"  -ForegroundColor Cyan

# Wipe stale cache so Halide_DIR changes take effect
$cacheFile = Join-Path $buildDir "CMakeCache.txt"
if (Test-Path $cacheFile) {
    Write-Host "Removing stale CMakeCache.txt" -ForegroundColor Yellow
    Remove-Item $cacheFile -Force
}

cmake -B $buildDir -S "$root\test_hlsl" `
    -DHalide_DIR="$HalideDir"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $buildDir --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Build succeeded: $buildDir\bin\$Config\" -ForegroundColor Green
