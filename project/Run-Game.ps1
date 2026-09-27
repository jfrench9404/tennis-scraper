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
    # FFmpeg for review videos. Default: first capable of PATH, opencv-env, Downloads.
    [string]$Ffmpeg = '',
    # Continue an existing detection folder (e.g. runs\claude-game-v1-full) instead of runs\NAME\run.
    [string]$RunDir = '',
    [switch]$Cpu
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing .venv. See CLAUDE-OVERNIGHT.md "Environment".' }

# Same check event_review runs before it creates review\ (-fps_mode needs FFmpeg 5.1+, plus libx264).
# Prints one JSON line on stdout; exit code 0 means capable. Runs from the project root.
function Test-Ffmpeg([string]$Candidate) {
    $line = @(& $python -m tennis_vision.event_review --check-ffmpeg $Candidate)[-1]
    $code = $LASTEXITCODE
    try { $result = $line | ConvertFrom-Json } catch { $result = $null }
    if (-not $result) { $result = [pscustomobject]@{ path = $Candidate; version = $null; reason = "check did not run (exit $code): $line" } }
    [pscustomobject]@{ Ok = ($code -eq 0); Path = $result.path; Version = $result.version; Reason = $result.reason }
}

Push-Location -LiteralPath $projectRoot
try {
    # Check FFmpeg now, not after hours of detection: review rendering is stage 2.
    if ($Ffmpeg) {
        $check = Test-Ffmpeg $Ffmpeg
        if (-not $check.Ok) { throw "-Ffmpeg $Ffmpeg cannot render review videos: $($check.Reason)" }
        $Ffmpeg = $check.Path
    } else {
        # First capable candidate: PATH, then the opencv-env copy (7.1), then the Downloads copy.
        $candidates = @()
        $onPath = Get-Command ffmpeg -ErrorAction SilentlyContinue
        if ($onPath) { $candidates += $onPath.Source }
        $candidates += (Join-Path $env:USERPROFILE 'miniconda3\envs\opencv-env\Library\bin\ffmpeg.exe')
        $candidates += (Join-Path $env:USERPROFILE 'Downloads\ffmpeg\ffmpeg.exe')
        $rejected = @()
        foreach ($candidate in $candidates) {
            if (-not (Test-Path -LiteralPath $candidate)) { $rejected += "$candidate : not found"; continue }
            $check = Test-Ffmpeg $candidate
            if ($check.Ok) { $Ffmpeg = $check.Path; break }
            $rejected += "$candidate : $($check.Reason)"
        }
        foreach ($skipped in $rejected) { Write-Host "Skipped FFmpeg $skipped" }
        if (-not $Ffmpeg) { throw "No capable FFmpeg found; pass -Ffmpeg PATH. Checked:`n$($rejected -join "`n")" }
    }
    Write-Host "Using FFmpeg $($check.Version): $Ffmpeg"
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
