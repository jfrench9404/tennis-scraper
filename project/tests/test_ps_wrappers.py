"""Contracts for the legacy PowerShell wrappers (issue #10), without running them.

Run-Player-Shots.ps1, Run-Shot-Replay.ps1, Apply-Court-Review.ps1 and
Run-Ball-Review.ps1 must prefer project\\.venv, fall back to their legacy venv, stop
with a message pointing to CLAUDE-OVERNIGHT.md otherwise, never create a venv or run
pip, and keep their parameters and the Python command lines they build unchanged.

The PowerShell parser check runs only where `pwsh` or `powershell` is on PATH (for
example John's laptop, or a CI runner that ships pwsh); elsewhere it skips cleanly.
"""
import base64
from pathlib import Path
import re
import shutil
import subprocess
import unittest

PROJECT = Path(__file__).resolve().parents[1]

# Legacy environment each wrapper may fall back to.
LEGACY_VENV = {
    'Run-Player-Shots.ps1': '.venv-player',
    'Run-Shot-Replay.ps1': '.venv-ball',
    'Apply-Court-Review.ps1': '.venv-ball',
    'Run-Ball-Review.ps1': '.venv-ball',
}

# param() blocks copied verbatim from origin/dev before this change: the public
# interface (names, types, defaults, validation) must stay byte-identical.
PARAMS = {
    'Run-Player-Shots.ps1': """param(
    [string]$CalibrationRun = 'runs\\court-bounce-20260925-110335-545',
    [string]$ContactReview,
    [string]$StrokeReview,
    [ValidateSet('auto','left','right','unknown')][string]$NearHand = 'auto',
    [ValidateSet('auto','left','right','unknown')][string]$FarHand = 'auto'
)""",
    'Run-Shot-Replay.ps1': """param(
    [string]$Labels,
    [string]$Corrections,
    [ValidateSet('auto','left','right','unknown')][string]$NearHand = 'auto',
    [ValidateSet('auto','left','right','unknown')][string]$FarHand = 'auto'
)""",
    'Apply-Court-Review.ps1': "param([Parameter(Mandatory=$true)][string]$ReviewFile)",
    'Run-Ball-Review.ps1': """param(
    [string]$SourceRun = 'runs/rally-confirmed',
    [string]$Scene = 'sebbie-scene.json'
)""",
}

# Lines that build or run the Python commands and name the outputs, copied verbatim
# from origin/dev. Each must still appear exactly once, in this order.
COMMAND_LINES = {
    'Run-Player-Shots.ps1': [
        "    $correctionPath = Join-Path $CalibrationRun 'calibration-corrections.json'",
        "    $outputPath = Join-Path $projectRoot ('runs\\player-shots-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))",
        "    $cacheKey = (Get-FileHash -LiteralPath $correctionPath -Algorithm SHA256).Hash.Substring(0,16)",
        "    $cachePath = Join-Path $projectRoot ('runs\\pose-cache-' + $cacheKey + '.json')",
        "    $reviewArgs = @()",
        "    if ($ContactReview) { $reviewArgs = @('--contact-review',(Resolve-Path -LiteralPath $ContactReview).Path) }",
        "    if ($StrokeReview) { $reviewArgs += @('--stroke-review',(Resolve-Path -LiteralPath $StrokeReview).Path) }",
        "    & $pythonPath -m tennis_vision.shot_replay --run runs\\rally-neural-ball --review runs\\rally-event-review-verified"
        " --court court.yaml --corrections $correctionPath --pose-weights yolo26l-pose.pt --pose-cache $cachePath --contacts"
        " --near-hand $NearHand --far-hand $FarHand --output $outputPath @reviewArgs",
        "    Write-Host \"Open: $outputPath\\replay.html\"",
        "    Invoke-Item -LiteralPath (Join-Path $outputPath 'replay.html')",
    ],
    'Run-Shot-Replay.ps1': [
        "$outputPath = Join-Path $projectRoot ('runs\\shot-replay-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))",
        "$arguments = @('-m','tennis_vision.shot_replay','--run',(Join-Path $projectRoot 'runs\\rally-neural-ball'),",
        "    '--review',(Join-Path $projectRoot 'runs\\rally-event-review-verified'),'--court',(Join-Path $projectRoot 'court.yaml'),",
        "    '--output',$outputPath,'--near-hand',$NearHand,'--far-hand',$FarHand)",
        "if ($Labels) { $arguments += @('--labels',(Resolve-Path -LiteralPath $Labels).Path) }",
        "if ($Corrections) { $arguments += @('--corrections',(Resolve-Path -LiteralPath $Corrections).Path) }",
        "    & $pythonPath @arguments",
        "    Write-Host \"Ready: $outputPath\\replay.html\"",
    ],
    'Apply-Court-Review.ps1': [
        "$reviewPath = (Resolve-Path -LiteralPath $ReviewFile).Path",
        "$outputPath = Join-Path $projectRoot ('runs\\court-bounce-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))",
        "    & $pythonPath -m tennis_vision.calibration_review --run runs\\rally-neural-ball --review runs\\rally-event-review-verified"
        " --court court.yaml --corrections $reviewPath --output $outputPath",
        "    Write-Host \"Calibration + bounce desk: $outputPath\\calibration.html\"",
        "    Write-Host \"3D replay: $outputPath\\replay.html\"",
    ],
    'Run-Ball-Review.ps1': [
        "    & $ballPython -c 'import cv2, onnxruntime, onnx, h5py, tqdm'",
        "    & $ballPython -m tennis_vision.setup_temporal",
        "    $reviewName = 'runs/gridtracknet-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')",
        "    & $ballPython -m tennis_vision.temporal_review --run $SourceRun --output $reviewName --scene $Scene",
        "    Write-Host \"Finished. Open $PSScriptRoot/$reviewName/review.mp4\"",
    ],
}


def read(name):
    return (PROJECT/name).read_text(encoding='ascii')


def code_only(text):
    """Strip comments and string contents; return (code, error) using PowerShell quoting rules.

    Handles '...' ('' escapes), "..." ("" and backtick escapes) and # line comments,
    which is all these wrappers use (no here-strings, no <# #> blocks). Parentheses
    inside "$(...)" subexpressions would be dropped, so the wrappers avoid them.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '#':
            while i < n and text[i] != '\n':
                i += 1
            continue
        if c in '\'"':
            quote, i = c, i+1
            while True:
                if i >= n:
                    return ''.join(out), f'unterminated {quote} string'
                if quote == '"' and text[i] == '`':
                    i += 2
                    continue
                if text[i] == quote:
                    if i+1 < n and text[i+1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(quote+quote)
            continue
        out.append(c)
        i += 1
    return ''.join(out), None


class PowerShellWrapperTests(unittest.TestCase):
    def test_parameters_unchanged(self):
        for name, block in PARAMS.items():
            with self.subTest(name):
                self.assertEqual(read(name).count(block), 1)
                code, _ = code_only(read(name))
                self.assertEqual(code.count('param('), 1)

    def test_python_command_lines_unchanged_and_in_order(self):
        for name, lines in COMMAND_LINES.items():
            with self.subTest(name):
                file_lines = read(name).splitlines()
                positions = []
                for line in lines:
                    self.assertEqual(file_lines.count(line), 1, line)
                    positions.append(file_lines.index(line))
                self.assertEqual(positions, sorted(positions))

    def test_prefers_shared_venv_then_legacy_then_stops(self):
        for name, legacy in LEGACY_VENV.items():
            with self.subTest(name):
                text = read(name)
                shared = re.search(r"Join-Path \S+ '\.venv[\\/]Scripts[\\/]python\.exe'", text)
                fallback = re.search(r"Join-Path \S+ '" + re.escape(legacy) + r"[\\/]Scripts[\\/]python\.exe'", text)
                self.assertIsNotNone(shared)
                self.assertIsNotNone(fallback)
                self.assertLess(shared.start(), fallback.start())
                stop = re.search(r'throw "No Python environment found\.[^\n]*CLAUDE-OVERNIGHT\.md', text)
                self.assertIsNotNone(stop)
                self.assertLess(fallback.start(), stop.start())
                # The environment is resolved before any Python command runs.
                first_call = re.search(r"^\s*& \$(pythonPath|ballPython) ", text, re.M)
                self.assertLess(stop.start(), first_call.start())
                # No other interpreter (e.g. bare `python` on PATH) is ever invoked.
                code, _ = code_only(text)
                self.assertNotRegex(code, r'(?m)^\s*(&\s*)?python(\.exe)?\s')

    def test_never_creates_venv_or_installs_packages(self):
        for name in LEGACY_VENV:
            with self.subTest(name):
                code, error = code_only(read(name))
                self.assertIsNone(error)
                # Only string contents mention pip/venv (printed instructions); no code does.
                self.assertNotRegex(code, r'-m\s+(pip|venv)\b')
                self.assertNotRegex(code, r'\bpip(\.exe)?\b')
                self.assertNotRegex(code, r'\bvirtualenv\b|\bensurepip\b')

    def test_ball_review_prints_install_instruction_instead(self):
        text = read('Run-Ball-Review.ps1')
        self.assertIn("-m pip install -r requirements-temporal.txt", text)
        self.assertIn('requirements-lock-windows.txt', text)
        self.assertIn("throw 'Temporal dependencies are missing; nothing was installed or run.'", text)

    def test_header_comments(self):
        for name in LEGACY_VENV:
            with self.subTest(name):
                header = read(name).split('param(', 1)[0]
                lines = header.splitlines()
                self.assertGreaterEqual(len(lines), 3)
                self.assertTrue(all(line.startswith('#') for line in lines), lines)
                self.assertIn(f'.\\{name}', header)          # usage example
                self.assertIn('CLAUDE-OVERNIGHT.md', header)
                self.assertIn(LEGACY_VENV[name], header)

    def test_static_syntax_sanity(self):
        pairs = {'(': ')', '{': '}', '[': ']'}
        for name in LEGACY_VENV:
            with self.subTest(name):
                raw = (PROJECT/name).read_bytes()
                raw.decode('ascii')                      # Windows PowerShell 5.1 reads BOM-less files as ANSI
                self.assertNotIn(b'\r', raw)             # same LF endings as before
                code, error = code_only(raw.decode('ascii'))
                self.assertIsNone(error)
                stack = []
                for c in code:
                    if c in pairs:
                        stack.append(pairs[c])
                    elif c in pairs.values():
                        self.assertTrue(stack and stack.pop() == c, f'unbalanced {c!r}')
                self.assertEqual(stack, [])
                self.assertNotIn('$(', code)             # keeps code_only's paren accounting exact

    def test_powershell_parser(self):
        shell = shutil.which('pwsh') or shutil.which('powershell')
        if not shell:
            self.skipTest('pwsh/powershell not on PATH; PowerShell parser check not run')
        paths = ','.join("'" + str(PROJECT/name).replace("'", "''") + "'" for name in LEGACY_VENV)
        script = (
            f"$bad = 0; foreach ($p in @({paths})) {{ $t = $null; $e = $null; "
            "[void][System.Management.Automation.Language.Parser]::ParseFile($p, [ref]$t, [ref]$e); "
            "foreach ($x in $e) { Write-Output ($p + ':' + $x.Extent.StartLineNumber + ': ' + $x.Message); $bad++ } }; "
            "Write-Output ('parsed with PowerShell ' + $PSVersionTable.PSVersion); exit $bad")
        encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
        result = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('parsed with PowerShell', result.stdout)


if __name__ == '__main__':
    unittest.main()
