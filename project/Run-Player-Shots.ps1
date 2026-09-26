param(
    [string]$CalibrationRun = 'runs\court-bounce-20260925-110335-545',
    [string]$ContactReview,
    [string]$StrokeReview,
    [ValidateSet('auto','left','right','unknown')][string]$NearHand = 'auto',
    [ValidateSet('auto','left','right','unknown')][string]$FarHand = 'auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-player\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Missing .venv-player inference environment. See PLAYER-SHOTS.md.' }
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
