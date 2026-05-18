# Build Halide core + autoschedulers (incl. sioutas2020) + utils (HalideTraceViz).
# No tests, docs, tutorials, apps, python bindings.
# Usage: .\build_hlsl6_jvp_sioutas_trace.ps1 [-Config Release|Debug] [-BuildDir <path>]

param(
    [string]$Config     = "Release",
    [string]$BuildDir   = "",
    [string]$InstallDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition

if (-not $BuildDir)   { $BuildDir   = Join-Path $root "build_hlsl6_jvp_sioutas_trace" }
if (-not $InstallDir) { $InstallDir = Join-Path $root "install_hlsl6_jvp_sioutas_trace" }

$LLVMDir  = "C:/git/Halide/LLVM_22_1_1/lib/cmake/llvm"
$ClangDir = "C:/git/Halide/LLVM_22_1_1/lib/cmake/clang"

# vcpkg Find-module hints (FindJPEG/FindPNG/FindZLIB use _INCLUDE_DIR / _LIBRARY vars).
# Use the generic *_LIBRARY (release) so the imported targets get a config-agnostic
# IMPORTED_LOCATION — required once WITH_UTILS=ON links Halide::ImageIO under the
# multi-config VS generator (Debug/Release/MinSizeRel/RelWithDebInfo).
# All deps (ffmpeg, libjpeg-turbo, libpng, zlib) installed release-only via classic vcpkg
$VcpkgPrefix = "C:/git/vcpkg/installed/x64-windows-release"
$VcpkgInc    = "$VcpkgPrefix/include"
$VcpkgLib    = "$VcpkgPrefix/lib"
$FFmpegPrefix = $VcpkgPrefix

Write-Host "Config     = $Config"     -ForegroundColor Cyan
Write-Host "BuildDir   = $BuildDir"   -ForegroundColor Cyan
Write-Host "InstallDir = $InstallDir" -ForegroundColor Cyan
Write-Host "LLVM_DIR   = $LLVMDir"    -ForegroundColor Cyan

# Wipe stale cache so path changes take effect
$cacheFile = Join-Path $BuildDir "CMakeCache.txt"
if (Test-Path $cacheFile) {
    Write-Host "Removing stale CMakeCache.txt" -ForegroundColor Yellow
    Remove-Item $cacheFile -Force
}

cmake -B $BuildDir -S $root `
    -DCMAKE_BUILD_TYPE="$Config" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -DCMAKE_PREFIX_PATH="$FFmpegPrefix" `
    -DLLVM_DIR="$LLVMDir" `
    -DClang_DIR="$ClangDir" `
    -DJPEG_INCLUDE_DIR="$VcpkgInc" `
    -DJPEG_LIBRARY="$VcpkgLib/jpeg.lib" `
    -DPNG_PNG_INCLUDE_DIR="$VcpkgInc" `
    -DPNG_LIBRARY="$VcpkgLib/libpng16.lib" `
    -DZLIB_INCLUDE_DIR="$VcpkgInc" `
    -DZLIB_LIBRARY="$VcpkgLib/zlib.lib" `
    -DWITH_TESTS=OFF `
    -DWITH_DOCS=OFF `
    -DWITH_TUTORIALS=OFF `
    -DWITH_PYTHON_BINDINGS=OFF `
    -DWITH_PACKAGING=ON `
    -DWITH_SERIALIZATION=OFF `
    -DWITH_AUTOSCHEDULERS=ON `
    -DWITH_UTILS=ON `
    -DHalide_ENABLE_RTTI=ON `
    -DHalide_ENABLE_EXCEPTIONS=ON

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $BuildDir --config $Config --parallel 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --install $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Copy runtime DLLs (ffmpeg + image libs) next to the built/installed binaries so
# HalideTraceViz/HalideTraceDump (and any installed tools) resolve them at runtime.
$dllSrc  = "$VcpkgPrefix/bin"
$dllDest = @( (Join-Path $BuildDir "bin/$Config") )
$instBin = Join-Path $InstallDir "bin"
if (Test-Path $instBin) { $dllDest += $instBin }
foreach ($d in $dllDest) {
    if (Test-Path $d) {
        Copy-Item "$dllSrc/*.dll" -Destination $d -Force
        Write-Host "Copied runtime DLLs -> $d" -ForegroundColor Green
    }
}

Write-Host "Build + install succeeded:" -ForegroundColor Green
Write-Host "  Build   : $BuildDir"   -ForegroundColor Green
Write-Host "  Install : $InstallDir" -ForegroundColor Green
