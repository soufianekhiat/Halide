#Requires -Version 5.1
<#
.SYNOPSIS
    Configure and build the complete Halide project on Windows (tutorials + utils).
.PARAMETER Config
    Build configuration: Release (default), Debug, or RelWithDebInfo.
.PARAMETER Jobs
    Parallel job count. Defaults to the number of logical CPU cores.
.PARAMETER Reconfigure
    Wipe the CMake cache before configuring (forces a clean configure).
.EXAMPLE
    .\build_Halide.ps1
    .\build_Halide.ps1 -Config Debug -Reconfigure
#>
param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string] $Config   = 'Release',
    [int]    $Jobs     = [Environment]::ProcessorCount,
    [string] $LLVMRoot = '',
    [switch] $Reconfigure
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --- Paths -------------------------------------------------------------------
$HalideRoot = Split-Path $PSScriptRoot -Parent
$BuildDir   = Join-Path $HalideRoot 'build\win64'
$CacheFile  = Join-Path $BuildDir   'CMakeCache.txt'

# --- Prerequisites -----------------------------------------------------------
foreach ($tool in @('cmake')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error "'$tool' not found in PATH. Install CMake 3.28+ and add it to PATH."
    }
}

if (-not $env:VCPKG_ROOT) {
    $vcpkgCandidates = @(
        'C:\git\vcpkg',
        'C:\vcpkg',
        'C:\src\vcpkg',
        (Join-Path $env:USERPROFILE 'vcpkg'),
        (Join-Path $env:USERPROFILE 'source\repos\vcpkg')
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
    Write-Error "VCPKG_ROOT is not set and vcpkg was not found in common locations.`nSet it with: `$env:VCPKG_ROOT = 'C:\path\to\vcpkg'"
}

if (-not (Test-Path (Join-Path $env:VCPKG_ROOT 'scripts\buildsystems\vcpkg.cmake'))) {
    Write-Error "vcpkg toolchain not found under VCPKG_ROOT=$($env:VCPKG_ROOT)"
}

# --- LLVM auto-detection -----------------------------------------------------
if (-not $LLVMRoot) {
    # Look for LLVM_* directories next to the Halide source tree
    $llvmCandidates = @(Get-ChildItem -Path $HalideRoot -Directory -Filter 'LLVM_*' |
                        Sort-Object Name -Descending |
                        Select-Object -ExpandProperty FullName)
    foreach ($c in $llvmCandidates) {
        if (Test-Path (Join-Path $c 'lib\cmake\llvm\LLVMConfig.cmake')) {
            $LLVMRoot = $c
            Write-Host "  Auto-detected LLVM: $c" -ForegroundColor DarkGray
            break
        }
    }
}

$LLVMDir  = ''
$ClangDir = ''
if ($LLVMRoot) {
    $LLVMDir  = (Join-Path $LLVMRoot 'lib\cmake\llvm')  -replace '\\', '/'
    $ClangDir = (Join-Path $LLVMRoot 'lib\cmake\clang') -replace '\\', '/'
    if (-not (Test-Path (Join-Path $LLVMDir 'LLVMConfig.cmake'))) {
        Write-Error "LLVMConfig.cmake not found in $LLVMDir"
    }
}

Write-Host ""
Write-Host "  Halide root : $HalideRoot" -ForegroundColor Cyan
Write-Host "  Build dir   : $BuildDir"   -ForegroundColor Cyan
Write-Host "  Config      : $Config"     -ForegroundColor Cyan
Write-Host "  Jobs        : $Jobs"       -ForegroundColor Cyan
Write-Host "  VCPKG_ROOT  : $env:VCPKG_ROOT" -ForegroundColor Cyan
if ($LLVMRoot) {
    Write-Host "  LLVM root   : $LLVMRoot" -ForegroundColor Cyan
}
Write-Host ""

# --- Configure ---------------------------------------------------------------
if ($Reconfigure -and (Test-Path $CacheFile)) {
    Write-Host "Removing CMake cache for reconfiguration..." -ForegroundColor Yellow
    Remove-Item $CacheFile -Force
    $cmakeFilesDir = Join-Path $BuildDir 'CMakeFiles'
    if (Test-Path $cmakeFilesDir) {
        Remove-Item $cmakeFilesDir -Recurse -Force
    }
}

if (-not (Test-Path $CacheFile)) {
    Write-Host "Configuring with preset 'win64'..." -ForegroundColor Yellow
    Push-Location $HalideRoot
    try {
        $cmakeArgs = @(
            '--preset', 'win64',
            '-DWITH_TUTORIALS=ON',
            '-DWITH_UTILS=ON',
            '-DWITH_TESTS=OFF',
            '-DWITH_DOCS=OFF',
            '-DWITH_PYTHON_BINDINGS=OFF'
        )
        if ($LLVMDir) {
            $cmakeArgs += "-DLLVM_DIR=$LLVMDir"
            $cmakeArgs += "-DClang_DIR=$ClangDir"
        }
        cmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE)." }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "CMakeCache.txt already exists - skipping configure." -ForegroundColor DarkGray
    Write-Host "  Use -Reconfigure to force a fresh configure." -ForegroundColor DarkGray
}

# --- Build -------------------------------------------------------------------
Write-Host ""
Write-Host "Building all targets  Config=$Config  Jobs=$Jobs..." -ForegroundColor Yellow
cmake --build $BuildDir --config $Config --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "CMake build failed (exit $LASTEXITCODE)." }

$BinDir = Join-Path $BuildDir "bin\$Config"
Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "  Binaries: $BinDir" -ForegroundColor Green
