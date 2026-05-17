# build.ps1 -- Configure (once) and build the ImageDiff demos.
#
# Usage:
#   .\build.ps1              # build both targets
#   .\build.ps1 -Clean       # wipe build dir and rebuild
#
param([switch]$Clean)

$ErrorActionPreference = "Stop"
$Root     = $PSScriptRoot
$BuildDir = "$Root\build"

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "Cleaning $BuildDir ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

# ── Configure ─────────────────────────────────────────────────────────────
if (-not (Test-Path "$BuildDir\CMakeCache.txt")) {
    Write-Host "`n=== CMake Configure ===" -ForegroundColor Cyan
    cmake -S $Root -B $BuildDir
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
}

# ── Build ─────────────────────────────────────────────────────────────────
Write-Host "`n=== CMake Build (Release) ===" -ForegroundColor Cyan
cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }

Write-Host "`nBuild successful." -ForegroundColor Green
Write-Host "  estimate_blur_radius: $BuildDir\Release\estimate_blur_radius.exe"
Write-Host "  estimate_psf:         $BuildDir\Release\estimate_psf.exe"
Write-Host "  estimate_tone_curve:  $BuildDir\Release\estimate_tone_curve.exe"
Write-Host "  deblur_cg:            $BuildDir\Release\deblur_cg.exe"
Write-Host "  optimize_texture:     $BuildDir\Release\optimize_texture.exe"
Write-Host "  estimate_normals_albedo: $BuildDir\Release\estimate_normals_albedo.exe"
