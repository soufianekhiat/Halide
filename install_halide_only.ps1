# Install Halide from an existing build into install_halide_only
# Usage: .\install_halide_only.ps1 [-Config Release|Debug] [-BuildDir <path>] [-InstallDir <path>]
#
# Run .\build_Halide_Only.ps1 first if build_halide_only does not exist.

param(
    [string]$Config     = "Release",
    [string]$BuildDir   = "",
    [string]$InstallDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition

if (-not $BuildDir)   { $BuildDir   = Join-Path $root "build_halide_only" }
if (-not $InstallDir) { $InstallDir = Join-Path $root "install_halide_only" }

if (-not (Test-Path $BuildDir)) {
    Write-Error "Build directory '$BuildDir' not found. Run .\build_Halide_Only.ps1 first."
    exit 1
}

Write-Host "Config     = $Config"     -ForegroundColor Cyan
Write-Host "BuildDir   = $BuildDir"   -ForegroundColor Cyan
Write-Host "InstallDir = $InstallDir" -ForegroundColor Cyan

cmake --install $BuildDir --config $Config --prefix $InstallDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Install succeeded. Prefix: $InstallDir" -ForegroundColor Green
