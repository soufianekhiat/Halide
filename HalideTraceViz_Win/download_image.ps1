#Requires -Version 5.1
<#
.SYNOPSIS
    Download a sample RGB image from pexels.com (or any URL) for the examples.
.DESCRIPTION
    Pexels supports a ?w=N query parameter that returns the image already
    resized to the requested width. Default fetches a ~800-wide landscape
    photo. The downloaded file is saved to the example bin directory next
    to the example executables, where they look for input.jpg by default.
.PARAMETER Url
    Image URL. Pexels works well; the ?w=800 query gives a ~800px-wide JPEG.
.PARAMETER OutFile
    Where to save it. Defaults to example\build\bin\Release\input.jpg.
.PARAMETER Force
    Re-download even if the file already exists.
.EXAMPLE
    .\download_image.ps1
    .\download_image.ps1 -Url 'https://images.pexels.com/photos/417074/pexels-photo-417074.jpeg?w=800'
#>
param(
    [string] $Url     = 'https://images.pexels.com/photos/417074/pexels-photo-417074.jpeg?cs=srgb&dl=eberhard-grossgasteiger-417074.jpg&w=800',
    [string] $OutFile = (Join-Path $PSScriptRoot 'example\build\bin\Release\input.jpg'),
    [switch] $Force
)

$ErrorActionPreference = 'Stop'

if ((Test-Path $OutFile) -and -not $Force) {
    $sz = (Get-Item $OutFile).Length
    Write-Host "Already exists: $OutFile ($([Math]::Round($sz/1KB, 1)) KB)" -ForegroundColor DarkGray
    Write-Host "Use -Force to re-download." -ForegroundColor DarkGray
    return
}

$dir = Split-Path $OutFile -Parent
if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir | Out-Null
}

Write-Host "Downloading: $Url" -ForegroundColor Cyan
Write-Host "  -> $OutFile" -ForegroundColor Cyan

try {
    # User-Agent header is needed by some CDNs.
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing `
        -Headers @{ 'User-Agent' = 'Mozilla/5.0 (HalideTraceViz example downloader)' }
} catch {
    Write-Error @"
Download failed: $($_.Exception.Message)

Manual download:
  1. Visit https://www.pexels.com and pick any photo (free for commercial use)
  2. Click "Free Download" -> select "Medium" (~640-1280 wide)
  3. Save as: $OutFile
  Or pass a different -Url to this script.
"@
}

if (Test-Path $OutFile) {
    $sz = (Get-Item $OutFile).Length
    Write-Host ""
    Write-Host "Saved: $OutFile  ($([Math]::Round($sz/1KB, 1)) KB)" -ForegroundColor Green
}
