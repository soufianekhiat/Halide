# Build Halide library only (no tests, docs, tutorials, apps)
# Usage: .\build_Halide_Only.ps1 [-Config Release|Debug] [-BuildDir <path>] [-InstallDir <path>]

param(
    [string]$Config      = "Release",
    [string]$BuildDir    = "",
    [string]$InstallDir  = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition

if (-not $BuildDir)   { $BuildDir   = Join-Path $root "build_halide_only" }
if (-not $InstallDir) { $InstallDir = Join-Path $root "install_halide_only" }

$LLVMDir  = "C:/git/Halide/LLVM_22_1_1/lib/cmake/llvm"
$ClangDir = "C:/git/Halide/LLVM_22_1_1/lib/cmake/clang"

# vcpkg Find-module hints (FindJPEG/FindPNG/FindZLIB use _INCLUDE_DIR / _LIBRARY vars,
# not config-mode _DIR vars)
$VcpkgInc  = "C:/git/vcpkg/installed/x64-windows/include"
$VcpkgLib  = "C:/git/vcpkg/installed/x64-windows/lib"
$VcpkgLibD = "C:/git/vcpkg/installed/x64-windows/debug/lib"

Write-Host "Config     = $Config"     -ForegroundColor Cyan
Write-Host "BuildDir   = $BuildDir"   -ForegroundColor Cyan
Write-Host "InstallDir = $InstallDir" -ForegroundColor Cyan
Write-Host "LLVM_DIR   = $LLVMDir"   -ForegroundColor Cyan

# Wipe stale cache so path changes take effect
$cacheFile = Join-Path $BuildDir "CMakeCache.txt"
if (Test-Path $cacheFile) {
    Write-Host "Removing stale CMakeCache.txt" -ForegroundColor Yellow
    Remove-Item $cacheFile -Force
}

cmake -B $BuildDir -S $root `
    -DCMAKE_BUILD_TYPE="$Config" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir" `
    -DLLVM_DIR="$LLVMDir" `
    -DClang_DIR="$ClangDir" `
    -DJPEG_INCLUDE_DIR="$VcpkgInc" `
    -DJPEG_LIBRARY_RELEASE="$VcpkgLib/jpeg.lib" `
    -DJPEG_LIBRARY_DEBUG="$VcpkgLibD/jpeg.lib" `
    -DPNG_PNG_INCLUDE_DIR="$VcpkgInc" `
    -DPNG_LIBRARY_RELEASE="$VcpkgLib/libpng16.lib" `
    -DPNG_LIBRARY_DEBUG="$VcpkgLibD/libpng16d.lib" `
    -DZLIB_INCLUDE_DIR="$VcpkgInc" `
    -DZLIB_LIBRARY_RELEASE="$VcpkgLib/zlib.lib" `
    -DZLIB_LIBRARY_DEBUG="$VcpkgLibD/zlibd.lib" `
    -DWITH_TESTS=OFF `
    -DWITH_DOCS=OFF `
    -DWITH_TUTORIALS=OFF `
    -DWITH_UTILS=OFF `
    -DWITH_AUTOSCHEDULERS=OFF `
    -DWITH_PYTHON_BINDINGS=OFF `
    -DWITH_PACKAGING=ON `
    -DWITH_SERIALIZATION=OFF `
    -DHalide_ENABLE_RTTI=ON `
    -DHalide_ENABLE_EXCEPTIONS=ON

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $BuildDir --config $Config --parallel 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --install $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Build succeeded. Install prefix: $InstallDir" -ForegroundColor Green
