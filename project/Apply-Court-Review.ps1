# Apply a court-calibration review file (exported from the calibration desk) to the
# sample rally. Writes a NEW runs\court-bounce-<timestamp>\ folder with calibration.html
# (calibration + bounce desk) and replay.html. The review file and input runs are
# never modified.
#
#   .\Apply-Court-Review.ps1 -ReviewFile "C:\Users\John\Downloads\calibration-corrections.json"
#
# Python: project\.venv (see CLAUDE-OVERNIGHT.md "Environment"); falls back to the
# legacy .venv-ball if .venv is missing. Never creates environments or installs packages.
param([Parameter(Mandatory=$true)][string]$ReviewFile)
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
if (-not (Test-Path -LiteralPath $ReviewFile -PathType Leaf)) { throw "Review file not found: $ReviewFile" }
$reviewPath = (Resolve-Path -LiteralPath $ReviewFile).Path
$outputPath = Join-Path $projectRoot ('runs\court-bounce-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m tennis_vision.calibration_review --run runs\rally-neural-ball --review runs\rally-event-review-verified --court court.yaml --corrections $reviewPath --output $outputPath
    if ($LASTEXITCODE -ne 0) { throw 'Calibration build failed; original runs are unchanged.' }
    Write-Host "Calibration + bounce desk: $outputPath\calibration.html"
    Write-Host "3D replay: $outputPath\replay.html"
} finally { Pop-Location }
