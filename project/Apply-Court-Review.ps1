param([Parameter(Mandatory=$true)][string]$ReviewFile)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-ball\Scripts\python.exe'
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
