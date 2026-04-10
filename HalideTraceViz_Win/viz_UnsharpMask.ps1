#Requires -Version 5.1
<#
.SYNOPSIS
    Build (if needed) and run the multi-stage unsharp_mask example through
    HalideTraceViz, producing an MP4 visualization of the pipeline.
.DESCRIPTION
    The unsharp_mask example traces 7 stages: luminance extraction,
    horizontal Gaussian blur, vertical Gaussian blur, high-pass extraction,
    sharpening, per-channel reweighting, and output cast. With --auto_layout
    HalideTraceViz arranges these in a grid so you can watch the whole
    pipeline animate at once.
.PARAMETER Image
    Input RGB image. Defaults to example\build\bin\Release\input.jpg
    (downloaded by download_image.ps1). Any RGB JPEG/PNG/PPM works.
.PARAMETER Config
    Build configuration. Default: Release.
.PARAMETER OutputVideo
    Output MP4 path. Default: videos\unsharp_mask.mp4.
.PARAMETER OutputImage
    Sharpened image filename (saved next to the binary).
.PARAMETER Width / Height
    Visualization frame size. Default 1920x1080.
.PARAMETER Timestep
    Halide events per video frame. Higher = shorter video / coarser detail.
    For an 800x600 image with 8 compute_root() stages (~9M store events)
    the default 2000 yields ~4500 frames (~75 seconds at 60fps).
.PARAMETER Fps
    Frames per second. Default 60.
.PARAMETER Hold
    Frames of final state to hold at end. Default Fps/2.
#>
param(
    [string] $Image,
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string] $Config      = 'Release',
    [string] $OutputVideo = (Join-Path $PSScriptRoot 'videos\unsharp_mask.mp4'),
    [string] $OutputImage = 'unsharp_out.png',
    [int]    $Width       = 1920,
    [int]    $Height      = 1080,
    [int]    $Timestep    = 2000,
    [int]    $Fps         = 60,
    [int]    $Hold        = -1,
    [int]    $PipeConnectTimeoutSec = 10,
    [int]    $LessonTimeoutSec      = 600,
    [int]    $VizTimeoutSec         = 1200
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($Hold -lt 0) { $Hold = [int]($Fps / 2) }

# --- Paths -----------------------------------------------------------------
$HalideRoot   = Split-Path $PSScriptRoot -Parent
$BuildDir     = Join-Path $HalideRoot 'build\win64'
$BinDir       = Join-Path $BuildDir "bin\$Config"
$VizExe       = Join-Path $BinDir 'HalideTraceViz.exe'
$ExampleBin   = Join-Path $PSScriptRoot "example\build\bin\$Config\unsharp_mask.exe"
$HalideDllDir = Join-Path $HalideRoot 'install\win64\bin'
$VcpkgBinDir  = if ($env:VCPKG_ROOT) {
    Join-Path $env:VCPKG_ROOT 'installed\x64-windows\bin'
} else { '' }

# Default image: input.jpg next to the example binary (downloaded by download_image.ps1)
if (-not $Image) {
    $Image = Join-Path (Split-Path $ExampleBin -Parent) 'input.jpg'
}

# --- Guards ----------------------------------------------------------------
if (-not (Test-Path $VizExe))     { Write-Error "HalideTraceViz.exe not found: $VizExe`nRun build_HalideTraceViz.ps1 first." }
if (-not (Test-Path $ExampleBin)) { Write-Error "unsharp_mask.exe not found: $ExampleBin`nRun build_Example.ps1 first." }
if (-not (Test-Path $Image))      { Write-Error "Input image not found: $Image`nRun download_image.ps1 first, or pass -Image <path>." }

$VideoDir = Split-Path $OutputVideo -Parent
if ($VideoDir -and -not (Test-Path $VideoDir)) {
    New-Item -ItemType Directory -Path $VideoDir | Out-Null
}

Write-Host ""
Write-Host "  Image          : $Image"       -ForegroundColor Cyan
Write-Host "  unsharp_mask   : $ExampleBin"  -ForegroundColor Cyan
Write-Host "  HalideTraceViz : $VizExe"      -ForegroundColor Cyan
Write-Host "  Output video   : $OutputVideo" -ForegroundColor Cyan
Write-Host "  Frame size     : ${Width}x${Height}" -ForegroundColor Cyan
Write-Host "  Timestep       : $Timestep   FPS: $Fps   Hold: $Hold" -ForegroundColor Cyan
Write-Host ""

# --- Per-run log files -----------------------------------------------------
$LogDir    = Join-Path $PSScriptRoot 'videos'
$VizOut    = Join-Path $LogDir 'unsharp_mask.viz.stdout.log'
$VizErr    = Join-Path $LogDir 'unsharp_mask.viz.stderr.log'
$LessonOut = Join-Path $LogDir 'unsharp_mask.lesson.stdout.log'
$LessonErr = Join-Path $LogDir 'unsharp_mask.lesson.stderr.log'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

Get-Process HalideTraceViz -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $PID } | Stop-Process -Force -ErrorAction SilentlyContinue
if (Test-Path $OutputVideo) { Remove-Item $OutputVideo -Force }

# --- Named pipe setup ------------------------------------------------------
$PipeName = 'halide_unsharp_mask'
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

# --- Run unsharp_mask ------------------------------------------------------
Write-Host "Running unsharp_mask..." -ForegroundColor Yellow
$savedTrace  = $env:HL_TRACE_FILE
$savedTarget = $env:HL_TARGET
$savedPath   = $env:PATH
$env:HL_TRACE_FILE = $PipePath
$env:HL_TARGET     = 'host'
# Make Halide.dll and vcpkg DLLs (libpng, jpeg, zlib) discoverable.
$pathPrefix = $HalideDllDir
if ($VcpkgBinDir) { $pathPrefix = "$pathPrefix;$VcpkgBinDir" }
$env:PATH = "$pathPrefix;$env:PATH"

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
        Write-Host "[WARN] unsharp_mask timed out - killing." -ForegroundColor Yellow
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
    Write-Host "[OK] Video: $OutputVideo  ($sizeMB MB)" -ForegroundColor Green
} else {
    Write-Host "[FAIL] No video produced." -ForegroundColor Red
    if (Test-Path $VizErr)    { Write-Host "--- viz stderr ---" -ForegroundColor DarkRed; Get-Content $VizErr }
    if (Test-Path $LessonErr) { Write-Host "--- lesson stderr ---" -ForegroundColor DarkRed; Get-Content $LessonErr }
    if (Test-Path $LessonOut) { Write-Host "--- lesson stdout ---" -ForegroundColor DarkYellow; Get-Content $LessonOut }
    exit 1
}

$resultImage = Join-Path (Split-Path $ExampleBin -Parent) $OutputImage
if (Test-Path $resultImage) {
    Write-Host "[OK] Sharpened image: $resultImage" -ForegroundColor Green
}
