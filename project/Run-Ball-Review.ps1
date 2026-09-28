# Run the five-frame GridTrackNet ball model over an existing run and write a NEW
# runs/gridtracknet-<timestamp>/ folder with review.mp4. Before the review it prepares
# models/gridtracknet/ (setup_temporal: downloads the pinned public checkpoint once if it
# is missing, verifies its SHA-256, exports ONNX). The source run is never modified.
# This is inference and can take a long time: run it in your own terminal.
#
#   .\Run-Ball-Review.ps1
#   .\Run-Ball-Review.ps1 -SourceRun runs/rally-confirmed -Scene sebbie-scene.json
#
# Python: project\.venv (see CLAUDE-OVERNIGHT.md "Environment"); falls back to the
# legacy .venv-ball if .venv is missing. Never creates environments or installs packages:
# if a dependency is missing it prints the install command for you to run yourself.
param(
    [string]$SourceRun = 'runs/rally-confirmed',
    [string]$Scene = 'sebbie-scene.json'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $ballPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    $usingLegacy = $false
    if (-not (Test-Path -LiteralPath $ballPython)) {
        $legacyPython = Join-Path $PSScriptRoot '.venv-ball/Scripts/python.exe'
        if (-not (Test-Path -LiteralPath $legacyPython)) {
            throw "No Python environment found. Expected $ballPython (or the legacy .venv-ball). Set up project\.venv as described in CLAUDE-OVERNIGHT.md ""Environment""; this script does not create environments or install packages."
        }
        Write-Warning "project\.venv not found; using the legacy environment $legacyPython"
        $ballPython = $legacyPython
        $usingLegacy = $true
    }
    & $ballPython -c 'import cv2, onnxruntime, onnx, h5py, tqdm'
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Missing temporal dependencies (cv2, onnxruntime, onnx, h5py, tqdm) in $ballPython."
        Write-Host 'This script does not install packages. Install them yourself, then rerun it:'
        if ($usingLegacy) {
            Write-Host "  & '$ballPython' -m pip install -r requirements-temporal.txt"
        } else {
            Write-Host '  project\.venv is pinned by requirements-lock-windows.txt; see CLAUDE-OVERNIGHT.md "Environment".'
            Write-Host '  It uses onnxruntime-directml, so do not install requirements-temporal.txt into it.'
        }
        throw 'Temporal dependencies are missing; nothing was installed or run.'
    }
    & $ballPython -m tennis_vision.setup_temporal
    if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed.' }
    $reviewName = 'runs/gridtracknet-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
    & $ballPython -m tennis_vision.temporal_review --run $SourceRun --output $reviewName --scene $Scene
    if ($LASTEXITCODE -ne 0) { throw 'Temporal review failed; inspect the message above.' }
    Write-Host "Finished. Open $PSScriptRoot/$reviewName/review.mp4"
} finally {
    Pop-Location
}
