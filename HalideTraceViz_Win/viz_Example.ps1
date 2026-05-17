#Requires -Version 5.1
<#
.SYNOPSIS
    Run the tiled_blur example and capture a HalideTraceViz video.
.DESCRIPTION
    Launches HalideTraceViz on a Windows named pipe, runs tiled_blur.exe with
    HL_TRACE_FILE pointing to that pipe, and waits for the encoded MP4.
.PARAMETER Image
    Path to an RGB input image. If omitted, tutorial\images\rgb.png is used.
    Any image from pexels.com or similar works; place it anywhere and pass
    the path.
.PARAMETER Config
    Build configuration (Release / Debug). Default: Release.
.PARAMETER OutputVideo
    Output .mp4 path. Default: videos\tiled_blur.mp4 next to this script.
.PARAMETER OutputImage
    Blur result image path. Default: tiled_blur_out.png next to the exe.
.PARAMETER Width
    Visualization frame width. Default: 1920.
.PARAMETER Height
    Visualization frame height. Default: 1080.
.PARAMETER Timestep
    Halide events per video frame. Default: 100.
.PARAMETER Fps
    Frames per second. Default: 60.
.PARAMETER Hold
    Frames of final state to hold at end. Default: 30 (0.5s at 60fps).
#>
param(
    [string] $Image,
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string] $Config      = 'Release',
    [string] $OutputVideo = (Join-Path $PSScriptRoot 'videos\tiled_blur.mp4'),
    [string] $OutputImage = 'tiled_blur_out.png',
    [int]    $Width       = 1920,
    [int]    $Height      = 1080,
    [int]    $Timestep    = 100,
    [int]    $Fps         = 60,
    [int]    $Hold        = 30,
    [int]    $PipeConnectTimeoutSec = 10,
    [int]    $LessonTimeoutSec      = 300,
    [int]    $VizTimeoutSec         = 600
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --- Paths -----------------------------------------------------------------
$HalideRoot   = Split-Path $PSScriptRoot -Parent
$BuildDir     = Join-Path $HalideRoot 'build\win64'
$BinDir       = Join-Path $BuildDir "bin\$Config"
$VizExe       = Join-Path $BinDir 'HalideTraceViz.exe'
$ExampleBin   = Join-Path $PSScriptRoot "example\build\bin\$Config\tiled_blur.exe"
# tiled_blur.exe links dynamically against Halide.dll from the install tree.
$HalideDllDir = Join-Path $HalideRoot 'install\win64\bin'

# Default image: the tutorial's rgb.png (known to work with libpng via vcpkg).
if (-not $Image) {
    $Image = Join-Path $HalideRoot 'tutorial\images\rgb.png'
}

# --- Guards ----------------------------------------------------------------
if (-not (Test-Path $VizExe)) {
    Write-Error "HalideTraceViz.exe not found: $VizExe`nRun build_HalideTraceViz.ps1 first."
}
if (-not (Test-Path $ExampleBin)) {
    Write-Error "tiled_blur.exe not found: $ExampleBin`nRun build_Example.ps1 first."
}
if (-not (Test-Path $Image)) {
    Write-Error "Input image not found: $Image"
}

# Ensure output directory exists
$VideoDir = Split-Path $OutputVideo -Parent
if ($VideoDir -and -not (Test-Path $VideoDir)) {
    New-Item -ItemType Directory -Path $VideoDir | Out-Null
}

Write-Host ""
Write-Host "  Image          : $Image"       -ForegroundColor Cyan
Write-Host "  tiled_blur     : $ExampleBin"  -ForegroundColor Cyan
Write-Host "  HalideTraceViz : $VizExe"      -ForegroundColor Cyan
Write-Host "  Output video   : $OutputVideo" -ForegroundColor Cyan
Write-Host "  Frame size     : ${Width}x${Height}" -ForegroundColor Cyan
Write-Host "  Timestep       : $Timestep"    -ForegroundColor Cyan
Write-Host "  FPS            : $Fps"         -ForegroundColor Cyan
Write-Host ""

# --- Per-run log files -----------------------------------------------------
$LogDir    = Join-Path $PSScriptRoot 'videos'
$VizOut    = Join-Path $LogDir 'tiled_blur.viz.stdout.log'
$VizErr    = Join-Path $LogDir 'tiled_blur.viz.stderr.log'
$LessonOut = Join-Path $LogDir 'tiled_blur.lesson.stdout.log'
$LessonErr = Join-Path $LogDir 'tiled_blur.lesson.stderr.log'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

Get-Process HalideTraceViz -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $PID } | Stop-Process -Force -ErrorAction SilentlyContinue
if (Test-Path $OutputVideo) { Remove-Item $OutputVideo -Force }

# --- Named pipe setup ------------------------------------------------------
$PipeName = 'halide_tiled_blur'
$PipePath = "\\.\pipe\$PipeName"

$VizArgs = @(
    '--input',    $PipePath,
    '-o',         $OutputVideo,
    '--size',     $Width, $Height,
    '--timestep', $Timestep,
    '--fps',      $Fps,
    '--hold',     $Hold,
    '--auto_layout'
)

Write-Host "Starting HalideTraceViz..." -ForegroundColor Yellow
$VizProc = Start-Process `
    -FilePath               $VizExe `
    -ArgumentList           $VizArgs `
    -PassThru `
    -WindowStyle            Hidden `
    -RedirectStandardOutput $VizOut `
    -RedirectStandardError  $VizErr

# Wait for pipe to appear, then settle.
$deadline  = [DateTime]::Now.AddSeconds($PipeConnectTimeoutSec)
$pipeReady = $false
while ([DateTime]::Now -lt $deadline) {
    if ($VizProc.HasExited) { break }
    if ([System.IO.Directory]::GetFiles('\\.\pipe\') -contains $PipePath) {
        $pipeReady = $true
        break
    }
    Start-Sleep -Milliseconds 100
}
if ($pipeReady) { Start-Sleep -Milliseconds 500 }

if (-not $pipeReady) {
    Write-Host "[FAIL] Named pipe did not appear in ${PipeConnectTimeoutSec}s." -ForegroundColor Red
    if (-not $VizProc.HasExited) { $VizProc.Kill() }
    if (Test-Path $VizErr) { Get-Content $VizErr | Write-Host -ForegroundColor DarkRed }
    exit 1
}

# --- Run tiled_blur --------------------------------------------------------
Write-Host "Running tiled_blur..." -ForegroundColor Yellow
$savedTrace  = $env:HL_TRACE_FILE
$savedTarget = $env:HL_TARGET
$savedPath   = $env:PATH
$env:HL_TRACE_FILE = $PipePath
$env:HL_TARGET     = 'host'
# Prepend the Halide install bin dir so tiled_blur.exe can resolve Halide.dll.
$env:PATH = "$HalideDllDir;$env:PATH"

try {
    $LessonProc = Start-Process `
        -FilePath               $ExampleBin `
        -ArgumentList           @($Image, $OutputImage) `
        -PassThru `
        -WindowStyle            Hidden `
        -WorkingDirectory       (Split-Path $ExampleBin -Parent) `
        -RedirectStandardOutput $LessonOut `
        -RedirectStandardError  $LessonErr

    if (-not $LessonProc.WaitForExit($LessonTimeoutSec * 1000)) {
        Write-Host "[WARN] tiled_blur timed out - killing." -ForegroundColor Yellow
        $LessonProc.Kill()
    }
} finally {
    if ($null -eq $savedTrace)  { Remove-Item Env:HL_TRACE_FILE -ErrorAction SilentlyContinue }
    else                        { $env:HL_TRACE_FILE = $savedTrace }
    if ($null -eq $savedTarget) { Remove-Item Env:HL_TARGET -ErrorAction SilentlyContinue }
    else                        { $env:HL_TARGET     = $savedTarget }
    $env:PATH = $savedPath
}

# --- Wait for HalideTraceViz to finish encoding ----------------------------
if (-not $VizProc.WaitForExit($VizTimeoutSec * 1000)) {
    Write-Host "[WARN] HalideTraceViz timed out - killing." -ForegroundColor Yellow
    $VizProc.Kill()
}

# --- Report ----------------------------------------------------------------
Write-Host ""
if (Test-Path $OutputVideo) {
    $sizeMB = [Math]::Round((Get-Item $OutputVideo).Length / 1MB, 2)
    Write-Host "[OK] Video: $OutputVideo ($sizeMB MB)" -ForegroundColor Green
} else {
    Write-Host "[FAIL] No video produced." -ForegroundColor Red
    if (Test-Path $VizErr)    { Write-Host "--- viz stderr ---" -ForegroundColor DarkRed; Get-Content $VizErr }
    if (Test-Path $LessonErr) { Write-Host "--- lesson stderr ---" -ForegroundColor DarkRed; Get-Content $LessonErr }
    if (Test-Path $LessonOut) { Write-Host "--- lesson stdout ---" -ForegroundColor DarkYellow; Get-Content $LessonOut }
    exit 1
}

# --- Show the blurred image path if it was written -------------------------
$resultImage = Join-Path (Split-Path $ExampleBin -Parent) $OutputImage
if (Test-Path $resultImage) {
    Write-Host "[OK] Blurred image: $resultImage" -ForegroundColor Green
}
