param(
    [string]$SourceRun = 'runs/rally-confirmed',
    [string]$Scene = 'sebbie-scene.json'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $ballPython = Join-Path $PSScriptRoot '.venv-ball/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $ballPython)) {
        python -m venv .venv-ball
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated ball environment.' }
    }
    & $ballPython -c 'import cv2, onnxruntime, onnx, h5py, tqdm'
    if ($LASTEXITCODE -ne 0) {
        & $ballPython -m pip install -r requirements-temporal.txt
        if ($LASTEXITCODE -ne 0) { throw 'Temporal dependencies could not be installed.' }
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
