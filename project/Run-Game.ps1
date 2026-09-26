# Analyze a whole video (or a time range) into a 3D replay. Rerun the same
# command to resume after an interruption; finished stages are skipped.
#
#   .\Run-Game.ps1 -Video "media\sebbie-demo-shortest (1).mp4" -Name game-v1
#   .\Run-Game.ps1 -Video media\sebbie-demo-shortestpar2.mp4 -Name game-v2 -StartSeconds 0 -EndSeconds 64
#   .\Run-Game.ps1 ... -Cpu        # PyTorch on CPU (baseline environment; ~15 s/frame)
param(
    [Parameter(Mandatory=$true)][string]$Video,
    [Parameter(Mandatory=$true)][string]$Name,
    [double]$StartSeconds = -1,
    [double]$EndSeconds = -1,
    [string]$Court = 'court.yaml',
    [string]$Scene = 'sebbie-scene.json',
    # Reviewed calibration to carry as a DRAFT after a camera-consistency check.
    [string]$Corrections = 'runs\court-bounce-20260925-110335-545\calibration-corrections.json',
    [string]$Ffmpeg = '',
    # Continue an existing detection folder (e.g. runs\claude-game-v1-full) instead of runs\NAME\run.
    [string]$RunDir = '',
    [switch]$Cpu
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing .venv. See CLAUDE-OVERNIGHT.md "Environment".' }
if (-not $Ffmpeg) {
    $found = Get-Command ffmpeg -ErrorAction SilentlyContinue
    $Ffmpeg = if ($found) { $found.Source } else { Join-Path $env:USERPROFILE 'Downloads\ffmpeg\ffmpeg.exe' }
}
Push-Location -LiteralPath $projectRoot
try {
    $arguments = @('-m','tennis_vision.game_pipeline','--input',$Video,'--output-root',(Join-Path 'runs' $Name),
        '--court',$Court,'--scene',$Scene,'--corrections',$Corrections,'--ffmpeg',$Ffmpeg,'--keep-awake')
    if ($Cpu) {
        $arguments += @('--weights','yolo11x.pt','--pose-weights','yolo26l-pose.pt','--refine-pose-weights','yolo26l-pose.pt','--device','cpu')
    } else {
        $onnx = 'models\onnx-export'
        if (-not (Test-Path -LiteralPath "$onnx\yolo11x-1280-dynamic.onnx")) {
            & $python -m tennis_vision.export_onnx yolo11x.pt yolo26l-pose.pt
            if ($LASTEXITCODE -ne 0) { throw 'ONNX export failed.' }
        }
        $arguments += @('--weights',"$onnx\yolo11x-1280-dynamic.onnx",'--pose-weights',"$onnx\yolo26l-pose-1280-dynamic.onnx",
            '--refine-pose-weights',"$onnx\yolo26l-pose-1280-dynamic.onnx",'--backend','onnx-directml')
    }
    if ($RunDir) { $arguments += @('--run-dir',$RunDir) }
    if ($StartSeconds -ge 0) { $arguments += @('--start-seconds',$StartSeconds) }
    if ($EndSeconds -ge 0) { $arguments += @('--end-seconds',$EndSeconds) }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Pipeline stopped. Rerun the same command to resume; finished stages are kept.' }
} finally { Pop-Location }
