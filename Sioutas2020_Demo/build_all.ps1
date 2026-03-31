# build_all.ps1
#
# Builds the Halide Sioutas2020 autoscheduler DLL, the test executable,
# and all 8 demo executables in one shot.
#
# Steps:
#   1. Build Halide_Sioutas2020 DLL (from the main Halide build tree)
#   2. Build the test executable (sioutas2020_test_apps_autoscheduler)
#   3. Configure + build the demo suite (Sioutas2020_Demo)
#
# Prerequisites:
#   - Halide must already have a CMake build tree at $BuildDir
#   - Halide must already be installed at $InstallDir (for the demo suite)
#   - Visual Studio 2022 with C++ workload

param(
    [string]$BuildDir   = "C:\git\Halide\build_455b34b",
    [string]$InstallDir = "C:\git\Halide\install_455b34b",
    [string]$SrcDir     = "C:\git\Halide",
    [string]$DemoSrc    = "C:\git\Halide\Sioutas2020_Demo",
    [string]$Config     = "Release",
    [switch]$NoDemos,     # Skip demo build
    [switch]$NoTests,     # Skip test build
    [switch]$RunTests     # Run tests after building
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 1. Build the Sioutas2020 autoscheduler DLL
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Step 1: Build Sioutas2020 autoscheduler DLL" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$dll = "$BuildDir\src\autoschedulers\sioutas2020\$Config\autoschedule_sioutas2020.dll"

cmake --build $BuildDir --target Halide_Sioutas2020 --config $Config
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: Halide_Sioutas2020 build" -ForegroundColor Red
    exit 1
}
Write-Host "  OK: $dll" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 2. Build the test executable
# ---------------------------------------------------------------------------
if (-not $NoTests) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Step 2: Build test executable" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $testProj = "$BuildDir\test\autoschedulers\sioutas2020\sioutas2020_test_apps_autoscheduler.vcxproj"
    if (-not (Test-Path $testProj)) {
        Write-Host "  SKIP: test project not found at $testProj" -ForegroundColor Yellow
    } else {
        # Use the Visual Studio MSBuild that ships with VS 2022
        $msbuild = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
            -latest -requires Microsoft.Component.MSBuild `
            -find "MSBuild\**\Bin\MSBuild.exe" 2>$null | Select-Object -First 1

        if (-not $msbuild) {
            # Fallback: try common path
            $msbuild = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
        }

        if (Test-Path $msbuild) {
            & $msbuild $testProj /p:Configuration=$Config /p:Platform=x64 /v:minimal
            if ($LASTEXITCODE -ne 0) {
                Write-Host "FAILED: test build" -ForegroundColor Red
                exit 1
            }
            Write-Host "  OK: sioutas2020_test_apps_autoscheduler.exe" -ForegroundColor Green
        } else {
            Write-Host "  SKIP: MSBuild not found" -ForegroundColor Yellow
        }
    }
}

# ---------------------------------------------------------------------------
# 3. Build the demo suite
# ---------------------------------------------------------------------------
if (-not $NoDemos) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Step 3: Build demo suite (8 demos)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $demoBuild = "$DemoSrc\build"

    # Configure if needed
    if (-not (Test-Path "$demoBuild\CMakeCache.txt")) {
        Write-Host "  Configuring CMake..." -ForegroundColor DarkGray
        cmake -S $DemoSrc -B $demoBuild -G "Visual Studio 17 2022" -A x64
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: demo CMake configure" -ForegroundColor Red
            exit 1
        }
    }

    cmake --build $demoBuild --config $Config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: demo build" -ForegroundColor Red
        exit 1
    }

    $demoExes = Get-ChildItem "$demoBuild\$Config\demo_*.exe" -ErrorAction SilentlyContinue
    Write-Host "  OK: $($demoExes.Count) demo executables built" -ForegroundColor Green
    foreach ($exe in $demoExes) {
        Write-Host "    $($exe.Name)" -ForegroundColor DarkGray
    }
}

# ---------------------------------------------------------------------------
# 4. Optionally run tests
# ---------------------------------------------------------------------------
if ($RunTests) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Step 4: Run Sioutas2020 tests" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $testExe = "$BuildDir\bin\$Config\sioutas2020_test_apps_autoscheduler.exe"
    if (-not (Test-Path $testExe)) {
        Write-Host "  SKIP: test exe not found" -ForegroundColor Yellow
    } else {
        $env:PATH = "$BuildDir\bin\$Config;$env:PATH"
        & $testExe $dll
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ALL TESTS PASSED" -ForegroundColor Green
        } else {
            Write-Host "  TESTS FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
            exit 1
        }
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Build complete" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  DLL:   $dll"
Write-Host "  Demos: $DemoSrc\build\$Config\"
Write-Host ""
Write-Host "  To run demos:  .\run_all_demos.ps1"
Write-Host "  To run tests:  .\build_all.ps1 -RunTests"
