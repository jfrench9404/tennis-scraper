# Rebuild the sample rally's 3D replay with focused crop poses, contact candidates and
# the reviewed court calibration. Reads runs\rally-neural-ball and
# runs\rally-event-review-verified, reuses runs\pose-cache-<hash>.json when it matches
# (pose inference runs otherwise), writes a NEW runs\player-shots-<timestamp>\ folder
# and opens its replay.html. The input runs are never modified.
#
#   .\Run-Player-Shots.ps1
#   .\Run-Player-Shots.ps1 -CalibrationRun runs\court-bounce-20260925-110335-545 -NearHand right
#   .\Run-Player-Shots.ps1 -ContactReview contact-review.json -StrokeReview stroke-review.json
#
# Python: project\.venv (see CLAUDE-OVERNIGHT.md "Environment"); falls back to the
# legacy .venv-player if .venv is missing. Never creates environments or installs packages.
param(
    [string]$CalibrationRun = 'runs\court-bounce-20260925-110335-545',
    [string]$ContactReview,
    [string]$StrokeReview,
    [ValidateSet('auto','left','right','unknown')][string]$NearHand = 'auto',
    [ValidateSet('auto','left','right','unknown')][string]$FarHand = 'auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $legacyPython = Join-Path $projectRoot '.venv-player\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $legacyPython)) {
        throw "No Python environment found. Expected $pythonPath (or the legacy .venv-player). Set up project\.venv as described in CLAUDE-OVERNIGHT.md ""Environment""; this script does not create environments or install packages."
    }
    Write-Warning "project\.venv not found; using the legacy environment $legacyPython"
    $pythonPath = $legacyPython
}
Push-Location -LiteralPath $projectRoot
try {
    $correctionPath = Join-Path $CalibrationRun 'calibration-corrections.json'
    if (-not (Test-Path -LiteralPath $correctionPath)) { throw "Calibration corrections missing: $correctionPath" }
    $outputPath = Join-Path $projectRoot ('runs\player-shots-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    $cacheKey = (Get-FileHash -LiteralPath $correctionPath -Algorithm SHA256).Hash.Substring(0,16)
    $cachePath = Join-Path $projectRoot ('runs\pose-cache-' + $cacheKey + '.json')
    $reviewArgs = @()
    if ($ContactReview) { $reviewArgs = @('--contact-review',(Resolve-Path -LiteralPath $ContactReview).Path) }
    if ($StrokeReview) { $reviewArgs += @('--stroke-review',(Resolve-Path -LiteralPath $StrokeReview).Path) }
    & $pythonPath -m tennis_vision.shot_replay --run runs\rally-neural-ball --review runs\rally-event-review-verified --court court.yaml --corrections $correctionPath --pose-weights yolo26l-pose.pt --pose-cache $cachePath --contacts --near-hand $NearHand --far-hand $FarHand --output $outputPath @reviewArgs
    if ($LASTEXITCODE -ne 0) { throw 'Player/shot build failed; original runs are unchanged.' }
    Write-Host "Open: $outputPath\replay.html"
    Invoke-Item -LiteralPath (Join-Path $outputPath 'replay.html')
} finally { Pop-Location }
