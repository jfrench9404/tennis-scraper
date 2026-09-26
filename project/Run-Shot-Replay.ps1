param(
    [string]$Labels,
    [string]$Corrections,
    [ValidateSet('auto','left','right','unknown')][string]$NearHand = 'auto',
    [ValidateSet('auto','left','right','unknown')][string]$FarHand = 'auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-ball\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'The project .venv-ball Python environment is missing.' }
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
