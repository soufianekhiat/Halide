#Requires -Version 5.1
<#
.SYNOPSIS
    Build only the HalideTraceViz target from an existing configured Halide build.
.PARAMETER Config
    Build configuration: Release (default), Debug, or RelWithDebInfo.
.EXAMPLE
    .\build_HalideTraceViz.ps1
    .\build_HalideTraceViz.ps1 -Config Debug
#>
param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string] $Config = 'Release'
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
Write-Host ""

# --- Guard -------------------------------------------------------------------
if (-not (Test-Path $CacheFile)) {
    Write-Error "Build directory not configured: $BuildDir`nRun build_Halide.ps1 first."
}

# --- Build -------------------------------------------------------------------
Write-Host "Building HalideTraceViz ($Config)..." -ForegroundColor Yellow
cmake --build $BuildDir --config $Config --target HalideTraceViz --parallel
if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)." }

$Exe = Join-Path $BuildDir "bin\$Config\HalideTraceViz.exe"
if (Test-Path $Exe) {
    Write-Host ""
    Write-Host "HalideTraceViz built successfully." -ForegroundColor Green
    Write-Host "  $Exe" -ForegroundColor Green
} else {
    Write-Warning "Build reported success but executable not found at:`n  $Exe"
}
