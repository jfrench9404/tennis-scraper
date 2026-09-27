# How is a long run doing? Frames/chunks done, source seconds covered, last progress,
# measured rate, ETA, stalls and wall-clock gaps in the run's log. Read-only: safe to
# use while a run is going, from any PowerShell window.
#
#   .\Get-RunStatus.ps1 -Run runs\claude-game-v1-full
#   .\Get-RunStatus.ps1 -Run runs\claude-game-v1-full -StallMinutes 20 -Log runs\claude-overnight-chain.log
#   .\Get-RunStatus.ps1 -Run runs\claude-game-v1 -Json      # runs\NAME with run\ inside also works
param(
    [Parameter(Mandatory=$true)][string]$Run,
    [double]$StallMinutes = 10,
    [double]$GapMinutes = -1,
    [string[]]$Log = @(),
    [switch]$VerifyHashes,
    [switch]$Json
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
# Paths may be given relative to the current folder or to the project folder.
function Resolve-RunPath([string]$path) {
    if (Test-Path -LiteralPath $path) { return (Resolve-Path -LiteralPath $path).Path }
    $inProject = Join-Path $projectRoot $path
    if (Test-Path -LiteralPath $inProject) { return (Resolve-Path -LiteralPath $inProject).Path }
    return $path
}
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    # The status tool needs only the standard library, so any Python 3.10+ works.
    $found = Get-Command python -ErrorAction SilentlyContinue
    if (-not $found) { throw 'Missing .venv and no python on PATH. See CLAUDE-OVERNIGHT.md "Environment".' }
    $python = $found.Source
}
$invariant = [Globalization.CultureInfo]::InvariantCulture  # "10.5", never "10,5"
$arguments = @('-m', 'tennis_vision.run_status', (Resolve-RunPath $Run), '--stall-minutes', $StallMinutes.ToString($invariant))
if ($GapMinutes -gt 0) { $arguments += @('--gap-minutes', $GapMinutes.ToString($invariant)) }
foreach ($item in $Log) { $arguments += @('--log', (Resolve-RunPath $item)) }
if ($VerifyHashes) { $arguments += '--verify-hashes' }
if ($Json) { $arguments += '--json' }
Push-Location -LiteralPath $projectRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "No readable run at $Run." }
} finally { Pop-Location }
