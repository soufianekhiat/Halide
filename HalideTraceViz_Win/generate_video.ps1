#Requires -Version 5.1
<#
.SYNOPSIS
    Generate trace visualization videos for Halide tutorial lessons on Windows.
.DESCRIPTION
    For each lesson, this script:
      1. Starts HalideTraceViz as a background process listening on a Windows
         named pipe (\\.\pipe\halide_trace_<lesson>).
      2. Runs the lesson executable with HL_TRACE_FILE pointing to that pipe,
         enabling live streaming - no intermediate trace files are written.
      3. Waits for HalideTraceViz to finish encoding the output .mp4.

    Lessons known to contain active trace_stores()/trace_loads() calls:
      04  lesson_04_debugging_2
      05  lesson_05_scheduling_1        (many scheduling variants)
      06  lesson_06_realizing_over_shifted_domains
      08  lesson_08_scheduling_2        (producer/consumer pairs)
      09  lesson_09_update_definitions  (loads + stores)

    Other lessons are attempted as well; they will produce a video that
    may contain only begin/end pipeline events (essentially blank frames)
    if no tracing was enabled in that lesson source.

.PARAMETER Config
    Build configuration to use: Release (default), Debug, or RelWithDebInfo.
.PARAMETER OutputDir
    Directory where .mp4 files are written. Default: .\videos\ next to this script.
.PARAMETER Width
    Output frame width in pixels. Default: 1920.
.PARAMETER Height
    Output frame height in pixels. Default: 1080.
.PARAMETER Timestep
    Halide events per video frame. Default: 10000.
.PARAMETER Fps
    Frames per second for the encoded video. Default: 30.
.PARAMETER PipeConnectTimeoutSec
    Seconds to wait for the named pipe to appear before giving up. Default: 10.
.PARAMETER LessonTimeoutSec
    Seconds to wait for a lesson executable to finish. Default: 120.
.PARAMETER VizTimeoutSec
    Seconds to wait for HalideTraceViz to finish encoding after the lesson exits.
    Default: 60.
.PARAMETER OnlyKnownLessons
    When set, only process lessons 04, 05, 06, 08, 09 (those with active tracing).
.EXAMPLE
    .\generate_video.ps1
    .\generate_video.ps1 -OnlyKnownLessons
    .\generate_video.ps1 -Timestep 1000 -OutputDir C:\videos
#>
param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo')]
    [string] $Config                = 'Release',
    [string] $OutputDir             = (Join-Path $PSScriptRoot 'videos'),
    [int]    $Width                 = 1920,
    [int]    $Height                = 1080,
    # Halide events per video frame. Small lessons do only a handful of stores,
    # so the default 10000 in HalideTraceViz's own docs produces a blank flash.
    # 1 gives one frame per store/load event — most dramatic visualization.
    [int]    $Timestep              = 1,
    [int]    $Fps                   = 30,
    # How many frames to hold after the trace ends (so the final image stays
    # visible briefly at the end of the video). Default: 0.5 seconds' worth.
    # Pass 0 to disable the hold entirely.
    [int]    $Hold                  = -1,
    [int]    $PipeConnectTimeoutSec = 10,
    [int]    $LessonTimeoutSec      = 120,
    [int]    $VizTimeoutSec         = 300,
    [switch] $OnlyKnownLessons
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --- Paths -------------------------------------------------------------------
$HalideRoot = Split-Path $PSScriptRoot -Parent
$BuildDir   = Join-Path $HalideRoot 'build\win64'
$BinDir     = Join-Path $BuildDir   "bin\$Config"
$VizExe     = Join-Path $BinDir     'HalideTraceViz.exe'

# --- Lesson table ------------------------------------------------------------
# hasTrace = $true  -> lesson calls trace_stores()/trace_loads() in source
# hasTrace = $false -> lesson runs but produces no/minimal trace events
$AllLessons = @(
    [PSCustomObject]@{ exe = 'lesson_01_basics';                          hasTrace = $false; desc = 'Basics' }
    [PSCustomObject]@{ exe = 'lesson_02_input_image';                     hasTrace = $false; desc = 'Input image' }
    [PSCustomObject]@{ exe = 'lesson_03_debugging_1';                     hasTrace = $false; desc = 'Debugging 1 (print)' }
    [PSCustomObject]@{ exe = 'lesson_04_debugging_2';                     hasTrace = $true;  desc = 'Debugging 2 (trace_stores)' }
    [PSCustomObject]@{ exe = 'lesson_05_scheduling_1';                    hasTrace = $true;  desc = 'Scheduling 1' }
    [PSCustomObject]@{ exe = 'lesson_06_realizing_over_shifted_domains';  hasTrace = $true;  desc = 'Shifted domains' }
    [PSCustomObject]@{ exe = 'lesson_07_multi_stage_pipelines';           hasTrace = $false; desc = 'Multi-stage pipelines' }
    [PSCustomObject]@{ exe = 'lesson_08_scheduling_2';                    hasTrace = $true;  desc = 'Scheduling 2 (producer/consumer)' }
    [PSCustomObject]@{ exe = 'lesson_09_update_definitions';              hasTrace = $true;  desc = 'Update definitions' }
    [PSCustomObject]@{ exe = 'lesson_10_aot_compilation_run';             hasTrace = $false; desc = 'AOT compilation (run)' }
    [PSCustomObject]@{ exe = 'lesson_11_cross_compilation';               hasTrace = $false; desc = 'Cross compilation' }
    [PSCustomObject]@{ exe = 'lesson_12_using_the_gpu';                   hasTrace = $false; desc = 'GPU' }
    [PSCustomObject]@{ exe = 'lesson_13_tuples';                          hasTrace = $false; desc = 'Tuples' }
    [PSCustomObject]@{ exe = 'lesson_14_types';                           hasTrace = $false; desc = 'Types' }
    [PSCustomObject]@{ exe = 'lesson_16_rgb_run';                         hasTrace = $false; desc = 'RGB (run)' }
    [PSCustomObject]@{ exe = 'lesson_17_predicated_rdom';                 hasTrace = $false; desc = 'Predicated RDom' }
    [PSCustomObject]@{ exe = 'lesson_18_parallel_associative_reductions'; hasTrace = $false; desc = 'Parallel associative reductions' }
    [PSCustomObject]@{ exe = 'lesson_19_wrapper_funcs';                   hasTrace = $false; desc = 'Wrapper Funcs' }
    [PSCustomObject]@{ exe = 'lesson_20_cloning_funcs';                   hasTrace = $false; desc = 'Cloning Funcs' }
    [PSCustomObject]@{ exe = 'lesson_21_auto_scheduler_run';              hasTrace = $false; desc = 'Auto-scheduler (run)' }
    [PSCustomObject]@{ exe = 'lesson_22_jit_performance';                 hasTrace = $false; desc = 'JIT performance' }
    [PSCustomObject]@{ exe = 'lesson_23_serialization';                   hasTrace = $false; desc = 'Serialization' }
    [PSCustomObject]@{ exe = 'lesson_24_async';                           hasTrace = $false; desc = 'Async' }
)

$Lessons = if ($OnlyKnownLessons) {
    $AllLessons | Where-Object { $_.hasTrace }
} else {
    $AllLessons
}

# --- Guards ------------------------------------------------------------------
if (-not (Test-Path $VizExe)) {
    Write-Error "HalideTraceViz.exe not found: $VizExe`nRun build_HalideTraceViz.ps1 first."
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Default $Hold to 0.5 seconds of frames if user didn't override.
if ($Hold -lt 0) { $Hold = [int]($Fps / 2) }

Write-Host ""
Write-Host "  HalideTraceViz : $VizExe"     -ForegroundColor Cyan
Write-Host "  Output dir     : $OutputDir"  -ForegroundColor Cyan
Write-Host "  Frame size     : ${Width}x${Height}" -ForegroundColor Cyan
Write-Host "  Timestep       : $Timestep"   -ForegroundColor Cyan
Write-Host "  FPS            : $Fps"         -ForegroundColor Cyan
Write-Host "  Lessons        : $($Lessons.Count)" -ForegroundColor Cyan
Write-Host ""

# --- Result tracking ---------------------------------------------------------
$Results = [System.Collections.Generic.List[PSObject]]::new()

# --- Per-lesson processing ---------------------------------------------------
foreach ($lesson in $Lessons) {
    $name     = $lesson.exe
    $exePath  = Join-Path $BinDir "$name.exe"
    $outFile  = Join-Path $OutputDir "$name.mp4"
    $pipeName = "halide_trace_$name"
    $pipePath = "\\.\pipe\$pipeName"

    Write-Host ("-" * 70)
    Write-Host "  $name" -ForegroundColor White
    Write-Host "  $($lesson.desc)" -ForegroundColor DarkGray

    # Check executable exists
    if (-not (Test-Path $exePath)) {
        Write-Host "  [SKIP] Executable not found: $exePath" -ForegroundColor Yellow
        $Results.Add([PSCustomObject]@{ Lesson = $name; Status = 'SKIPPED (no exe)'; SizeMB = ''; Output = '' })
        continue
    }

    # Build HalideTraceViz argument list.
    # Note: process_args only accepts --size / --timestep (not -s/-t).
    $vizArgs = @(
        '--input',    $pipePath,
        '-o',         $outFile,
        '--size',     $Width, $Height,
        '--timestep', $Timestep,
        '--fps',      $Fps,
        '--hold',     $Hold,
        '--auto_layout'
    )

    # Remove stale output if present
    if (Test-Path $outFile) { Remove-Item $outFile -Force }

    # Per-lesson log files for diagnostics
    $vizStdout    = Join-Path $OutputDir "$name.viz.stdout.log"
    $vizStderr    = Join-Path $OutputDir "$name.viz.stderr.log"
    $lessonStdout = Join-Path $OutputDir "$name.lesson.stdout.log"
    $lessonStderr = Join-Path $OutputDir "$name.lesson.stderr.log"

    # 1. Start HalideTraceViz in the background --------------------------------
    $vizProc = Start-Process `
        -FilePath               $VizExe `
        -ArgumentList           $vizArgs `
        -PassThru `
        -WindowStyle            Hidden `
        -RedirectStandardOutput $vizStdout `
        -RedirectStandardError  $vizStderr

    # 2. Wait for the named pipe to be ready ----------------------------------
    # Give HalideTraceViz time to reach CreateNamedPipe + start servicing
    # connections. Poll with Test-Path but then sleep an extra settle time:
    # Test-Path can see the path immediately after the process image starts,
    # before the pipe server is actually ready to accept a client.
    $deadline  = [DateTime]::Now.AddSeconds($PipeConnectTimeoutSec)
    $pipeReady = $false
    while ([DateTime]::Now -lt $deadline) {
        if ($vizProc.HasExited) { break }
        if ([System.IO.Directory]::GetFiles('\\.\pipe\') -contains $pipePath) {
            $pipeReady = $true
            break
        }
        Start-Sleep -Milliseconds 100
    }
    # Settle: give HalideTraceViz a moment to reach ConnectNamedPipe even
    # after the pipe is visible in the namespace.
    if ($pipeReady) { Start-Sleep -Milliseconds 500 }

    if (-not $pipeReady) {
        if ($vizProc.HasExited) {
            Write-Host "  [FAIL] HalideTraceViz exited (code $($vizProc.ExitCode)) before creating pipe." -ForegroundColor Red
        } else {
            Write-Host "  [FAIL] Named pipe did not appear within ${PipeConnectTimeoutSec}s." -ForegroundColor Red
            $vizProc.Kill()
        }
        if (Test-Path $vizStderr) {
            $err = Get-Content $vizStderr -Raw
            if ($err) { Write-Host "  --- HalideTraceViz stderr ---`n$err" -ForegroundColor DarkRed }
        }
        $Results.Add([PSCustomObject]@{ Lesson = $name; Status = 'FAILED (pipe timeout)'; SizeMB = ''; Output = '' })
        continue
    }

    # 3. Run the lesson with HL_TRACE_FILE pointing to the pipe ---------------
    $savedTrace  = $env:HL_TRACE_FILE
    $savedTarget = $env:HL_TARGET

    $env:HL_TRACE_FILE = $pipePath
    $env:HL_TARGET     = 'host'

    $lessonOk = $false
    try {
        $lessonProc = Start-Process `
            -FilePath               $exePath `
            -PassThru `
            -WindowStyle            Hidden `
            -RedirectStandardOutput $lessonStdout `
            -RedirectStandardError  $lessonStderr

        if (-not $lessonProc.WaitForExit($LessonTimeoutSec * 1000)) {
            Write-Host "  [WARN] Lesson timed out after ${LessonTimeoutSec}s - killing." -ForegroundColor Yellow
            $lessonProc.Kill()
        } else {
            $lessonOk = ($lessonProc.ExitCode -eq 0)
            if (-not $lessonOk) {
                Write-Host "  [WARN] Lesson exited with code $($lessonProc.ExitCode)" -ForegroundColor Yellow
            }
        }
    } finally {
        if ($null -eq $savedTrace)  { Remove-Item Env:HL_TRACE_FILE -ErrorAction SilentlyContinue }
        else                        { $env:HL_TRACE_FILE = $savedTrace }
        if ($null -eq $savedTarget) { Remove-Item Env:HL_TARGET     -ErrorAction SilentlyContinue }
        else                        { $env:HL_TARGET     = $savedTarget }
    }

    # 4. Wait for HalideTraceViz to finish encoding ---------------------------
    # The lesson closing HL_TRACE_FILE signals EOF to the pipe; HalideTraceViz
    # flushes the encoder and exits on its own.
    if (-not $vizProc.WaitForExit($VizTimeoutSec * 1000)) {
        Write-Host "  [WARN] HalideTraceViz timed out - killing." -ForegroundColor Yellow
        $vizProc.Kill()
    }

    # 5. Report ---------------------------------------------------------------
    $vizExitStr = if ($vizProc.HasExited) { "$($vizProc.ExitCode)" } else { 'still-running' }
    if (Test-Path $outFile) {
        $sizeMB = [Math]::Round((Get-Item $outFile).Length / 1MB, 2)
        $status = if ($lessonOk) { 'OK' } else { 'LESSON_ERROR' }
        $color  = if ($lessonOk) { 'Green' } else { 'Yellow' }
        Write-Host "  [$status] $outFile  ($sizeMB MB)" -ForegroundColor $color
        $Results.Add([PSCustomObject]@{ Lesson = $name; Status = $status; SizeMB = $sizeMB; Output = $outFile })
    } else {
        Write-Host "  [FAIL] No output file  viz_exit=$vizExitStr  lesson_exit=$(if($lessonProc){$lessonProc.ExitCode}else{'?'})" -ForegroundColor Red
        if (Test-Path $vizStderr) {
            $err = Get-Content $vizStderr -Raw
            if ($err) { Write-Host "  --- HalideTraceViz stderr ---`n$err" -ForegroundColor DarkRed }
        }
        if (Test-Path $lessonStdout) {
            $out = Get-Content $lessonStdout -Raw
            if ($out) { Write-Host "  --- Lesson stdout ---`n$out" -ForegroundColor DarkYellow }
        }
        if (Test-Path $lessonStderr) {
            $err = Get-Content $lessonStderr -Raw
            if ($err) { Write-Host "  --- Lesson stderr ---`n$err" -ForegroundColor DarkRed }
        }
        $Results.Add([PSCustomObject]@{ Lesson = $name; Status = 'FAILED (no output)'; SizeMB = ''; Output = '' })
    }
}

# --- Summary -----------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 70)
Write-Host "  Summary" -ForegroundColor White
Write-Host ("=" * 70)
$Results | Format-Table -AutoSize -Property Lesson, Status, SizeMB, Output

$ok      = @($Results | Where-Object { $_.Status -eq 'OK' }).Count
$skipped = @($Results | Where-Object { $_.Status -like 'SKIPPED*' }).Count
$failed  = @($Results | Where-Object { $_.Status -notlike 'OK*' -and $_.Status -notlike 'SKIPPED*' }).Count

Write-Host "  OK: $ok   Skipped: $skipped   Failed: $failed" -ForegroundColor $(
    if ($failed -gt 0) { 'Yellow' } else { 'Green' })
Write-Host ""
Write-Host "  Videos in: $OutputDir" -ForegroundColor Cyan
