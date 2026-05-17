#Requires -Version 5.1
<#
.SYNOPSIS
    Configure and build the tiled_blur example against the Halide build tree.
.PARAMETER Config
    Build configuration. Default: Release.
.PARAMETER Reconfigure
    Wipe the example's CMake cache before building.
#>
param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string] $Config = 'Release',
    [switch] $Reconfigure
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$HalideRoot  = Split-Path $PSScriptRoot -Parent
$SourceDir   = Join-Path $PSScriptRoot 'example'
$BuildDir    = Join-Path $SourceDir 'build'
$HalideBuild = Join-Path $HalideRoot 'build\win64'
$HalideInstall = Join-Path $HalideRoot 'install\win64'

# --- Install Halide locally if needed --------------------------------------
# The build tree doesn't expose a directly-usable find_package() layout; the
# install tree does. Install once to a local prefix and cache it.
$HalideConfigDir = Join-Path $HalideInstall 'lib\cmake\Halide'
$HalideHelpersConfigDir = Join-Path $HalideInstall 'lib\cmake\HalideHelpers'

if (-not (Test-Path (Join-Path $HalideConfigDir 'HalideConfig.cmake'))) {
    if (-not (Test-Path (Join-Path $HalideBuild 'CMakeCache.txt'))) {
        Write-Error "Halide is not built. Run build_Halide.ps1 first."
    }
    Write-Host "Installing Halide to $HalideInstall (one-time)..." -ForegroundColor Yellow
    cmake --install $HalideBuild --config $Config
    if ($LASTEXITCODE -ne 0) { throw "cmake --install failed." }
}

if (-not (Test-Path (Join-Path $HalideConfigDir 'HalideConfig.cmake'))) {
    Write-Error "HalideConfig.cmake still not found after install attempt in $HalideConfigDir."
}

Write-Host ""
Write-Host "  Halide install : $HalideInstall"  -ForegroundColor Cyan
Write-Host "  Halide config  : $HalideConfigDir" -ForegroundColor Cyan
Write-Host "  Source         : $SourceDir"      -ForegroundColor Cyan
Write-Host "  Build dir      : $BuildDir"       -ForegroundColor Cyan
Write-Host "  Config         : $Config"         -ForegroundColor Cyan
Write-Host ""

# --- (Re)configure ---------------------------------------------------------
if ($Reconfigure -and (Test-Path $BuildDir)) {
    Remove-Item $BuildDir -Recurse -Force
}

# Auto-detect VCPKG_ROOT (matches build_Halide.ps1 logic).
if (-not $env:VCPKG_ROOT) {
    $vcpkgCandidates = @(
        'C:\git\vcpkg', 'C:\vcpkg', 'C:\src\vcpkg',
        (Join-Path $env:USERPROFILE 'vcpkg')
    )
    foreach ($c in $vcpkgCandidates) {
        if (Test-Path (Join-Path $c 'scripts\buildsystems\vcpkg.cmake')) {
            $env:VCPKG_ROOT = $c
            Write-Host "  Auto-detected VCPKG_ROOT: $c" -ForegroundColor DarkGray
            break
        }
    }
}
if (-not $env:VCPKG_ROOT) {
    Write-Error "VCPKG_ROOT not set and vcpkg not found. Halide::ImageIO needs libpng/libjpeg from vcpkg."
}
$VcpkgToolchain = Join-Path $env:VCPKG_ROOT 'scripts\buildsystems\vcpkg.cmake'

# Check that the generator actually produced project files. A stale
# CMakeCache.txt from a failed previous configure would otherwise cause us
# to skip reconfigure even though ALL_BUILD.vcxproj is missing.
$AllBuildProj = Join-Path $BuildDir 'ALL_BUILD.vcxproj'
if (-not (Test-Path $AllBuildProj)) {
    if (Test-Path $BuildDir) {
        Remove-Item (Join-Path $BuildDir 'CMakeCache.txt') -Force -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $BuildDir 'CMakeFiles')     -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Configuring tiled_blur..." -ForegroundColor Yellow
    cmake -B $BuildDir -S $SourceDir `
          -G "Visual Studio 17 2022" -A x64 `
          -DCMAKE_TOOLCHAIN_FILE="$VcpkgToolchain" `
          -DVCPKG_MANIFEST_MODE=OFF `
          -DCMAKE_PREFIX_PATH="$HalideInstall" `
          -DHalide_DIR="$HalideConfigDir" `
          -DHalideHelpers_DIR="$HalideHelpersConfigDir" `
          -DCMAKE_BUILD_TYPE="$Config"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed." }
}

# --- Build -----------------------------------------------------------------
Write-Host "Building tiled_blur ($Config)..." -ForegroundColor Yellow
cmake --build $BuildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed." }

$Exe = Join-Path $BuildDir "bin\$Config\tiled_blur.exe"
if (Test-Path $Exe) {
    Write-Host ""
    Write-Host "Built: $Exe" -ForegroundColor Green
} else {
    Write-Warning "Build reported success but executable not found at:`n  $Exe"
}
