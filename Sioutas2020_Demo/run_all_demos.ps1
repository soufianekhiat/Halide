# run_all_demos.ps1
#
# Runs all 6 Sioutas2020 comparison demos.
# Each demo writes its own <name>_report.txt; all are then merged into
# combined_report.txt in the current directory.
#
# Adjust $BuildDir / $InstallDir if your layout differs.

param(
    [string]$BuildDir   = "C:\git\Halide\build_455b34b",
    [string]$InstallDir = "C:\git\Halide\install_455b34b",
    [string]$DemoDir    = "C:\git\Halide\Sioutas2020_Demo\build\Release",
    [switch]$NoGPU      # Pass -NoGPU to skip GPU execution benchmarks
)

# Autoscheduler DLLs — all from build dir to avoid Halide.dll version conflicts
$Sioutas     = "$BuildDir\src\autoschedulers\sioutas2020\Release\autoschedule_sioutas2020.dll"
$Anderson    = "$BuildDir\src\autoschedulers\anderson2021\Release\autoschedule_anderson2021.dll"
$Adams2019   = "$BuildDir\src\autoschedulers\adams2019\Release\autoschedule_adams2019.dll"
$Li2018      = "$BuildDir\src\autoschedulers\li2018\Release\autoschedule_li2018.dll"
$Mullapudi   = "$BuildDir\src\autoschedulers\mullapudi2016\Release\autoschedule_mullapudi2016.dll"

# Make Halide.dll visible (build dir only — mixing build+install DLLs causes crashes)
$env:PATH = "$BuildDir\bin\Release;$env:PATH"

# Enable GPU execution benchmarks unless -NoGPU is passed.
# First run compiles CUDA kernels via LLVM NVPTX (30-60s each; first call
# on Optimus laptops may also trigger discrete GPU power-up, adding minutes).
if (-not $NoGPU) {
    $env:HL_GPU_BENCH = "1"
    Write-Host "GPU benchmarks ENABLED  (pass -NoGPU to skip)" -ForegroundColor Yellow
} else {
    $env:HL_GPU_BENCH = ""
    Write-Host "GPU benchmarks DISABLED" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Autoscheduler DLLs:"
Write-Host "  Sioutas2020   : $Sioutas"
Write-Host "  Anderson2021  : $Anderson"
Write-Host "  Adams2019     : $Adams2019"
Write-Host "  Li2018        : $Li2018"
Write-Host "  Mullapudi2016 : $Mullapudi"
Write-Host ""

foreach ($dll in @($Sioutas, $Anderson, $Adams2019, $Li2018, $Mullapudi)) {
    if (-not (Test-Path $dll)) {
        Write-Host "ERROR: DLL not found: $dll" -ForegroundColor Red
        exit 1
    }
}

$Demos = @(
    @{ Exe = "demo_gaussian_blur";  Report = "gaussian_blur_report.txt"  },
    @{ Exe = "demo_harris_corner";  Report = "harris_corner_report.txt"  },
    @{ Exe = "demo_matmul";         Report = "matmul_report.txt"         },
    @{ Exe = "demo_histogram_eq";   Report = "histogram_eq_report.txt"   },
    @{ Exe = "demo_unsharp_mask";   Report = "unsharp_mask_report.txt"   },
    @{ Exe = "demo_conv2d_relu";    Report = "conv2d_relu_report.txt"    }
)

$Passed = 0
$Failed = 0

foreach ($d in $Demos) {
    $exe    = "$DemoDir\$($d.Exe).exe"
    $report = "$DemoDir\$($d.Report)"

    if (-not (Test-Path $exe)) {
        Write-Host "SKIP (not built): $($d.Exe)" -ForegroundColor Yellow
        continue
    }

    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  $($d.Exe)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    & $exe $Sioutas $Anderson $Adams2019 $Li2018 $Mullapudi $report
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [PASS]  report -> $report" -ForegroundColor Green
        $Passed++
    } else {
        Write-Host "  [FAIL]  exit code $LASTEXITCODE" -ForegroundColor Red
        $Failed++
    }
}

# Merge individual reports into one combined file
$combined = "$DemoDir\combined_report.txt"
$header   = "Sioutas2020 Autoscheduler Demo Report`n" +
            "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm')`n" +
            ("=" * 60) + "`n"
Set-Content -Path $combined -Value $header -Encoding UTF8

foreach ($d in $Demos) {
    $report = "$DemoDir\$($d.Report)"
    if (Test-Path $report) {
        Add-Content -Path $combined -Value (Get-Content $report -Raw) -Encoding UTF8
    }
}

Write-Host ""
Write-Host "Results : $Passed passed, $Failed failed" `
    -ForegroundColor $(if ($Failed -eq 0) { "Green" } else { "Red" })
Write-Host "Combined: $combined"
