# Rebuild the sample rally's shot table and 3D replay from existing detections and the
# verified event review (runs\rally-neural-ball, runs\rally-event-review-verified).
# No models are rerun. Writes a NEW runs\shot-replay-<timestamp>\replay.html; the input
# runs are never modified.
#
#   .\Run-Shot-Replay.ps1
#   .\Run-Shot-Replay.ps1 -Labels "C:\Users\John\Downloads\tennis-review-labels.json"
#   .\Run-Shot-Replay.ps1 -Corrections runs\court-bounce-20260925-110335-545\calibration-corrections.json -NearHand right
#
# Python: project\.venv (see CLAUDE-OVERNIGHT.md "Environment"); falls back to the
# legacy .venv-ball if .venv is missing. Never creates environments or installs packages.
param(
    [string]$Labels,
    [string]$Corrections,
    [ValidateSet('auto','left','right','unknown')][string]$NearHand = 'auto',
    [ValidateSet('auto','left','right','unknown')][string]$FarHand = 'auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $legacyPython = Join-Path $projectRoot '.venv-ball\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $legacyPython)) {
        throw "No Python environment found. Expected $pythonPath (or the legacy .venv-ball). Set up project\.venv as described in CLAUDE-OVERNIGHT.md ""Environment""; this script does not create environments or install packages."
    }
    Write-Warning "project\.venv not found; using the legacy environment $legacyPython"
    $pythonPath = $legacyPython
}
if ($Labels -and -not (Test-Path -LiteralPath $Labels -PathType Leaf)) { throw "Labels file not found: $Labels" }
$outputPath = Join-Path $projectRoot ('runs\shot-replay-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
$arguments = @('-m','tennis_vision.shot_replay','--run',(Join-Path $projectRoot 'runs\rally-neural-ball'),
    '--review',(Join-Path $projectRoot 'runs\rally-event-review-verified'),'--court',(Join-Path $projectRoot 'court.yaml'),
    '--output',$outputPath,'--near-hand',$NearHand,'--far-hand',$FarHand)
if ($Labels) { $arguments += @('--labels',(Resolve-Path -LiteralPath $Labels).Path) }
if ($Corrections) { $arguments += @('--corrections',(Resolve-Path -LiteralPath $Corrections).Path) }
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Shot/replay build failed. See the error above.' }
    Write-Host "Ready: $outputPath\replay.html"
    Write-Host 'Open replay.html in your browser. No models were rerun.'
} finally {
    Pop-Location
}
