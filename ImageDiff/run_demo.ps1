# run_demo.ps1 -- Run the ImageDiff JVP demos and print a report.
#
# Usage:
#   .\run_demo.ps1                      # run all 5 demos
#   .\run_demo.ps1 -BlurOnly            # run only the blur-radius demo
#   .\run_demo.ps1 -PSFOnly             # run only the PSF demo
#   .\run_demo.ps1 -ToneOnly            # run only the tone-curve demo
#   .\run_demo.ps1 -DeblurOnly          # run only the CG deblur demo
#   .\run_demo.ps1 -TextureOnly         # run only the texture optimization demo
#   .\run_demo.ps1 -NormalsOnly         # run only the normals+albedo demo
#   .\run_demo.ps1 -Open                # open output images after the report
#
param(
    [switch]$BlurOnly,
    [switch]$PSFOnly,
    [switch]$ToneOnly,
    [switch]$DeblurOnly,
    [switch]$TextureOnly,
    [switch]$NormalsOnly,
    [switch]$Open
)

$ErrorActionPreference = "Stop"
$Root    = $PSScriptRoot
$ExeDir  = "$Root\build\Release"
$DataDir = "$Root\data"

$BlurExe    = "$ExeDir\estimate_blur_radius.exe"
$PSFExe     = "$ExeDir\estimate_psf.exe"
$ToneExe    = "$ExeDir\estimate_tone_curve.exe"
$DeblurExe  = "$ExeDir\deblur_cg.exe"
$TextureExe = "$ExeDir\optimize_texture.exe"
$NormalsExe = "$ExeDir\estimate_normals_albedo.exe"
$InputImage = "$DataDir\reference_input.jpg"

if (-not (Test-Path $InputImage)) {
    throw "Input image not found: $InputImage"
}

# Decide which demos to run
$anyOnly = $BlurOnly -or $PSFOnly -or $ToneOnly -or $DeblurOnly -or $TextureOnly -or $NormalsOnly
$runBlur    = (-not $anyOnly) -or $BlurOnly
$runPSF     = (-not $anyOnly) -or $PSFOnly
$runTone    = (-not $anyOnly) -or $ToneOnly
$runDeblur  = (-not $anyOnly) -or $DeblurOnly
$runTexture = (-not $anyOnly) -or $TextureOnly
$runNormals = (-not $anyOnly) -or $NormalsOnly

# ═══════════════════════════════════════════════════════════════════════════
# Helper: run an exe, show key lines live, return full output
# ═══════════════════════════════════════════════════════════════════════════
function Run-Demo {
    param([string]$Exe, [string]$Label)
    if (-not (Test-Path $Exe)) {
        throw "$Label not found: $Exe`nRun .\build.ps1 first."
    }
    Write-Host "`n$('=' * 72)" -ForegroundColor DarkGray
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "$('=' * 72)" -ForegroundColor DarkGray

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & $Exe $InputImage 2>&1 | ForEach-Object { $_.ToString() }
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    foreach ($line in $output) {
        if ($line -match "^(Loaded|Image size|Ground truth|Compiling|Building|Converged|sigma_|gamma_|gain_|absolute|relative|Final PSF|Saved|JVP correct|=== Sens)") {
            Write-Host "  $line" -ForegroundColor Gray
        } elseif ($line -match "^(Input PSNR|Recovered|PSNR gain|Rendering PSNR|CG solve|CG wall|Blur sigma|Noise sigma|Shading|Normal PSNR|Albedo PSNR|Render PSNR|Optimization|Regularization|Light direction)") {
            Write-Host "  $line" -ForegroundColor Gray
        }
    }

    if ($exitCode -ne 0) {
        Write-Host "  FAILED (exit code $exitCode)" -ForegroundColor Red
        foreach ($line in $output) { Write-Host "  $line" }
        return $null
    }

    Write-Host "  Wall time: $([math]::Round($sw.Elapsed.TotalSeconds, 1)) s" -ForegroundColor DarkYellow
    return $output
}

# ═══════════════════════════════════════════════════════════════════════════
# Run demos
# ═══════════════════════════════════════════════════════════════════════════
$blurOutput = $null; $psfOutput = $null; $toneOutput = $null
$deblurOutput = $null; $textureOutput = $null; $normalsOutput = $null

if ($runTone) {
    $toneOutput = Run-Demo $ToneExe "Tone Curve Recovery (2 params, JVP + Adam) + Sensitivity Maps"
}
if ($runBlur) {
    $blurOutput = Run-Demo $BlurExe "Gaussian Blur Radius Recovery (1 param, JVP + Adam)"
}
if ($runPSF) {
    $psfOutput = Run-Demo $PSFExe "7x7 PSF Recovery (49 params, JVP + Exponentiated Gradient)"
}
if ($runDeblur) {
    $deblurOutput = Run-Demo $DeblurExe "Image Deblurring (ImageParam JVP + CG, symmetric operator)"
}
if ($runTexture) {
    $textureOutput = Run-Demo $TextureExe "Texture Recovery (ImageParam JVP + CG, asymmetric operator)"
}
if ($runNormals) {
    $normalsOutput = Run-Demo $NormalsExe "Normal Map + Albedo Recovery (3D ImageParam JVP + Adam, RGB)"
}

# ═══════════════════════════════════════════════════════════════════════════
# Parse results
# ═══════════════════════════════════════════════════════════════════════════

# Helper to find last iteration number before "Converged" or end
function Find-LastIter($output) {
    if (-not $output) { return $null }
    for ($i = $output.Count - 1; $i -ge 0; $i--) {
        if ($output[$i] -match '^\s*(\d+)\s+[\d.]+') {
            return $Matches[1]
        }
    }
    return $null
}

# ── Tone curve ────────────────────────────────────────────────────────────
$toneGammaTrue = $null; $toneGammaEst = $null; $toneGammaRelErr = $null
$toneGainTrue  = $null; $toneGainEst  = $null; $toneGainRelErr  = $null
$toneIter = $null
if ($toneOutput) {
    foreach ($line in $toneOutput) {
        if ($line -match 'gamma_true\s*=\s*([\d.]+)')      { $toneGammaTrue = $Matches[1] }
        if ($line -match 'gamma_estimated\s*=\s*([\d.]+)')  { $toneGammaEst  = $Matches[1] }
        if ($line -match 'gamma rel error\s*=\s*([\d.]+)%') { $toneGammaRelErr = $Matches[1] }
        if ($line -match 'gain_true\s*=\s*([\d.]+)')        { $toneGainTrue  = $Matches[1] }
        if ($line -match 'gain_estimated\s*=\s*([\d.]+)')   { $toneGainEst   = $Matches[1] }
        if ($line -match 'gain rel error\s*=\s*([\d.]+)%')  { $toneGainRelErr = $Matches[1] }
        if ($line -match 'Converged at iter (\d+)')          { $toneIter      = $Matches[1] }
    }
    if (-not $toneIter) { $toneIter = Find-LastIter $toneOutput }
}

# ── Blur radius ───────────────────────────────────────────────────────────
$blurSigmaTrue = $null; $blurSigmaEst = $null; $blurRelErr = $null; $blurIter = $null
if ($blurOutput) {
    foreach ($line in $blurOutput) {
        if ($line -match 'sigma_true\s*=\s*([\d.]+)')      { $blurSigmaTrue = $Matches[1] }
        if ($line -match 'sigma_estimated\s*=\s*([\d.]+)')  { $blurSigmaEst  = $Matches[1] }
        if ($line -match 'relative error\s*=\s*([\d.]+)%')  { $blurRelErr    = $Matches[1] }
        if ($line -match 'Converged at iter (\d+)')          { $blurIter      = $Matches[1] }
    }
}

# ── PSF ───────────────────────────────────────────────────────────────────
$psfRMSE = $null; $psfIter = $null
if ($psfOutput) {
    foreach ($line in $psfOutput) {
        if ($line -match 'Final PSF RMSE:\s*([\d.]+)')  { $psfRMSE = $Matches[1] }
    }
    $psfIter = Find-LastIter $psfOutput
}

# ── Deblur CG ────────────────────────────────────────────────────────────
$deblurInputPSNR = $null; $deblurRecovPSNR = $null; $deblurGain = $null
$deblurIter = $null
if ($deblurOutput) {
    foreach ($line in $deblurOutput) {
        if ($line -match 'Input PSNR:\s+([\d.]+)\s*dB')        { $deblurInputPSNR = $Matches[1] }
        if ($line -match 'Recovered PSNR:\s+([\d.]+)\s*dB')    { $deblurRecovPSNR = $Matches[1] }
        if ($line -match 'PSNR gain:\s+([\d.]+)\s*dB')         { $deblurGain      = $Matches[1] }
        if ($line -match 'Converged at CG iteration (\d+)')    { $deblurIter      = $Matches[1] }
    }
    if (-not $deblurIter) { $deblurIter = Find-LastIter $deblurOutput }
}

# ── Texture optimization ─────────────────────────────────────────────────
$texturePSNR = $null; $textureRenderPSNR = $null; $textureIter = $null
if ($textureOutput) {
    foreach ($line in $textureOutput) {
        if ($line -match 'Recovered texture PSNR:\s+([\d.]+)\s*dB') { $texturePSNR       = $Matches[1] }
        if ($line -match 'Rendering PSNR:\s+([\d.]+)\s*dB')         { $textureRenderPSNR = $Matches[1] }
        if ($line -match 'Converged at CG iteration (\d+)')         { $textureIter       = $Matches[1] }
    }
    if (-not $textureIter) { $textureIter = Find-LastIter $textureOutput }
}

# ── Normals + Albedo ─────────────────────────────────────────────────────
$normalsNormalPSNR = $null; $normalsAlbedoPSNR = $null; $normalsRenderPSNR = $null
if ($normalsOutput) {
    foreach ($line in $normalsOutput) {
        if ($line -match 'Normal PSNR:\s+([\d.]+)\s*dB')  { $normalsNormalPSNR = $Matches[1] }
        if ($line -match 'Albedo PSNR:\s+([\d.]+)\s*dB')  { $normalsAlbedoPSNR = $Matches[1] }
        if ($line -match 'Render PSNR:\s+([\d.]+)\s*dB')  { $normalsRenderPSNR = $Matches[1] }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

function Color-Error([float]$val, [float]$good, [float]$ok) {
    if ($val -lt $good) { return "Green" }
    elseif ($val -lt $ok) { return "Yellow" }
    else { return "Red" }
}

function Show-Images([array]$files) {
    Write-Host "    Output images:" -ForegroundColor DarkYellow
    foreach ($f in $files) {
        $full = "$Root\$($f.Path)"
        if (Test-Path $full) {
            $sz = [math]::Round((Get-Item $full).Length / 1KB, 1)
            Write-Host "      $($f.Path)  ($($sz) KB)  -- $($f.Desc)"
        }
    }
}

Write-Host ""
Write-Host "$('=' * 72)" -ForegroundColor White
Write-Host "  HALIDE FORWARD-MODE AD (JVP) -- DEMO REPORT" -ForegroundColor White
Write-Host "$('=' * 72)" -ForegroundColor White

# ── Tone curve section ────────────────────────────────────────────────────
if ($toneOutput) {
    Write-Host ""
    Write-Host "  TONE CURVE RECOVERY  (per-pixel sensitivity maps)" -ForegroundColor Cyan
    Write-Host "  $('-' * 48)"
    Write-Host "    Parameters:       2 (gamma, gain)"
    Write-Host "    Optimizer:        Adam (lr=0.05)"

    if ($toneGammaTrue) {
        Write-Host "    gamma_true:       $toneGammaTrue    gain_true:       $toneGainTrue"
        Write-Host "    gamma_estimated:  $toneGammaEst    gain_estimated:  $toneGainEst"
    }
    if ($toneGammaRelErr) {
        $c = Color-Error ([float]$toneGammaRelErr) 1.0 5.0
        Write-Host "    gamma rel error:  $toneGammaRelErr%  gain rel error:  $toneGainRelErr%" -ForegroundColor $c
    }
    if ($toneIter) { Write-Host "    Converged at:     iter $toneIter" }

    Write-Host ""
    Show-Images @(
        @{ Path="data\tangent_gamma_init.jpg";  Desc="Gamma sensitivity at initial estimate" },
        @{ Path="data\tangent_gain_init.jpg";   Desc="Gain sensitivity at initial estimate" },
        @{ Path="data\tangent_gamma_final.jpg"; Desc="Gamma sensitivity at recovered estimate" },
        @{ Path="data\tangent_gain_final.jpg";  Desc="Gain sensitivity at recovered estimate" },
        @{ Path="data\tone_reference.jpg";      Desc="Reference tone-mapped image" },
        @{ Path="data\tone_estimated.jpg";      Desc="Estimated tone-mapped image" }
    )
}

# ── Blur radius section ──────────────────────────────────────────────────
if ($blurOutput) {
    Write-Host ""
    Write-Host "  GAUSSIAN BLUR RADIUS RECOVERY" -ForegroundColor Cyan
    Write-Host "  $('-' * 48)"
    Write-Host "    Parameters:       1 (sigma)"
    Write-Host "    Optimizer:        Adam (lr=0.1)"
    Write-Host "    sigma_true:       $blurSigmaTrue"
    Write-Host "    sigma_estimated:  $blurSigmaEst"

    if ($blurRelErr) {
        $c = Color-Error ([float]$blurRelErr) 1.0 5.0
        Write-Host "    Relative error:   $blurRelErr%" -ForegroundColor $c
    }
    if ($blurIter) { Write-Host "    Converged at:     iter $blurIter" }

    Write-Host ""
    Show-Images @(
        @{ Path="data\kernel_true.jpg";       Desc="True kernel (sigma=$blurSigmaTrue)" },
        @{ Path="data\kernel_estimated.jpg";  Desc="Estimated kernel (sigma=$blurSigmaEst)" },
        @{ Path="data\blurred_reference.jpg"; Desc="Reference blurred image" },
        @{ Path="data\blurred_estimated.jpg"; Desc="Estimated blurred image" }
    )
}

# ── PSF section ──────────────────────────────────────────────────────────
if ($psfOutput) {
    Write-Host ""
    Write-Host "  7x7 PSF RECOVERY" -ForegroundColor Cyan
    Write-Host "  $('-' * 48)"
    Write-Host "    Parameters:       49 (7x7 kernel)"
    Write-Host "    Optimizer:        Exponentiated Gradient (lr=15)"

    if ($psfRMSE) {
        $c = Color-Error ([float]$psfRMSE) 0.003 0.01
        Write-Host "    PSF RMSE:         $psfRMSE" -ForegroundColor $c
    }
    if ($psfIter) { Write-Host "    Converged at:     iter $psfIter" }

    # Print the estimated PSF grid
    $inEstimated = $false
    Write-Host ""
    Write-Host "    Estimated PSF:" -ForegroundColor DarkYellow
    foreach ($line in $psfOutput) {
        if ($line -match '^Estimated PSF') { $inEstimated = $true; continue }
        if ($inEstimated -and $line -match '^\s+\[') {
            Write-Host "      $($line.Trim())"
        }
        if ($inEstimated -and $line -notmatch '^\s+\[' -and $line.Trim().Length -gt 0) {
            $inEstimated = $false
        }
    }

    Write-Host ""
    Show-Images @(
        @{ Path="data\psf_true_vis.jpg";         Desc="True PSF kernel (210x210)" },
        @{ Path="data\psf_est_vis.jpg";          Desc="Estimated PSF kernel (210x210)" },
        @{ Path="data\psf_reference.jpg";        Desc="Reference convolved image" },
        @{ Path="data\psf_estimated_output.jpg"; Desc="Estimated convolved image" }
    )
}

# ── Deblur CG section ───────────────────────────────────────────────────
if ($deblurOutput) {
    Write-Host ""
    Write-Host "  IMAGE DEBLURRING  (ImageParam JVP + CG)" -ForegroundColor Cyan
    Write-Host "  $('-' * 48)"
    Write-Host "    Method:           CG on normal equations (symmetric blur)"
    Write-Host "    JVP target:       ImageParam (input image, W x H)"
    Write-Host "    Operator:         A*v = blur(blur(v)) + lambda*v  (2 JVP calls/iter)"

    if ($deblurInputPSNR) {
        Write-Host "    Input PSNR:       $deblurInputPSNR dB"
    }
    if ($deblurRecovPSNR) {
        $c = Color-Error ([float](40.0 - [float]$deblurRecovPSNR)) 10.0 20.0
        Write-Host "    Recovered PSNR:   $deblurRecovPSNR dB" -ForegroundColor $c
    }
    if ($deblurGain) {
        Write-Host "    PSNR gain:        $deblurGain dB"
    }
    if ($deblurIter) { Write-Host "    CG iterations:    $deblurIter" }

    Write-Host ""
    Show-Images @(
        @{ Path="data\deblur_observed.jpg";  Desc="Blurred + noisy input" },
        @{ Path="data\deblur_recovered.jpg"; Desc="CG-recovered sharp image" }
    )
}

# ── Texture optimization section ─────────────────────────────────────────
if ($textureOutput) {
    Write-Host ""
    Write-Host "  TEXTURE RECOVERY  (ImageParam JVP + CG, asymmetric)" -ForegroundColor Cyan
    Write-Host "  $('-' * 48)"
    Write-Host "    Method:           CG on normal equations (shading * blur)"
    Write-Host "    JVP target:       ImageParam (texture, W x H)"
    Write-Host "    Operator:         J*v via JVP, J^T*w via host-side blur"

    if ($texturePSNR) {
        $c = Color-Error ([float](40.0 - [float]$texturePSNR)) 10.0 20.0
        Write-Host "    Texture PSNR:     $texturePSNR dB" -ForegroundColor $c
    }
    if ($textureRenderPSNR) {
        Write-Host "    Rendering PSNR:   $textureRenderPSNR dB"
    }
    if ($textureIter) { Write-Host "    CG iterations:    $textureIter" }

    Write-Host ""
    Show-Images @(
        @{ Path="data\texture_shading.jpg";   Desc="Procedural shading pattern" },
        @{ Path="data\texture_target.jpg";    Desc="Target rendering (shading * blur(texture))" },
        @{ Path="data\texture_recovered.jpg"; Desc="Recovered texture" },
        @{ Path="data\texture_rendered.jpg";  Desc="Rendering of recovered texture" }
    )
}

# ── Normals + Albedo section ────────────────────────────────────────────
if ($normalsOutput) {
    Write-Host ""
    Write-Host "  NORMAL MAP + ALBEDO RECOVERY  (3D ImageParam JVP + Adam, RGB)" -ForegroundColor Cyan
    Write-Host "  $('-' * 48)"
    Write-Host "    Method:           Adam (normals + albedo), 3-light photometric stereo"
    Write-Host "    JVP target:       ImageParam (normals, W x H x 3)"
    Write-Host "    JVP calls/iter:   9 (3 lights x 3 basis vectors)"

    if ($normalsNormalPSNR) {
        $c = Color-Error ([float](40.0 - [float]$normalsNormalPSNR)) 10.0 20.0
        Write-Host "    Normal PSNR:      $normalsNormalPSNR dB" -ForegroundColor $c
    }
    if ($normalsAlbedoPSNR) {
        $c = Color-Error ([float](40.0 - [float]$normalsAlbedoPSNR)) 10.0 20.0
        Write-Host "    Albedo PSNR:      $normalsAlbedoPSNR dB" -ForegroundColor $c
    }
    if ($normalsRenderPSNR) {
        Write-Host "    Render PSNR:      $normalsRenderPSNR dB"
    }

    Write-Host ""
    Show-Images @(
        @{ Path="data\normals_true.jpg";       Desc="Ground truth normal map (color-coded)" },
        @{ Path="data\albedo_true.jpg";        Desc="Ground truth albedo (RGB)" },
        @{ Path="data\normals_target_L0.jpg";  Desc="Target rendering (light 0: front-right)" },
        @{ Path="data\normals_target_L1.jpg";  Desc="Target rendering (light 1: front-left)" },
        @{ Path="data\normals_target_L2.jpg";  Desc="Target rendering (light 2: back-center)" },
        @{ Path="data\normals_est.jpg";        Desc="Estimated normal map" },
        @{ Path="data\albedo_est.jpg";         Desc="Estimated albedo" },
        @{ Path="data\normals_rerendered.jpg"; Desc="Re-rendered with estimated normals+albedo" }
    )
}

Write-Host ""
Write-Host "$('=' * 72)" -ForegroundColor White

# ── Open images if requested ──────────────────────────────────────────────
if ($Open) {
    Write-Host ""
    Write-Host "Opening output images..." -ForegroundColor DarkYellow
    $allImages = @(
        "data\tangent_gamma_init.jpg", "data\tangent_gain_init.jpg",
        "data\tangent_gamma_final.jpg", "data\tangent_gain_final.jpg",
        "data\kernel_true.jpg", "data\kernel_estimated.jpg",
        "data\psf_true_vis.jpg", "data\psf_est_vis.jpg",
        "data\deblur_observed.jpg", "data\deblur_recovered.jpg",
        "data\texture_shading.jpg", "data\texture_target.jpg",
        "data\texture_recovered.jpg", "data\texture_rendered.jpg",
        "data\normals_true.jpg", "data\albedo_true.jpg",
        "data\normals_target_L0.jpg", "data\normals_est.jpg",
        "data\albedo_est.jpg", "data\normals_rerendered.jpg"
    )
    foreach ($img in $allImages) {
        $full = "$Root\$img"
        if (Test-Path $full) {
            Start-Process $full
        }
    }
}
