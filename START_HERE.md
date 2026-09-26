# Tennis vision: weekend handoff

Prepared for John and Claude on 2026-09-25. This is a copy of the working project,
not a replacement for the original. No footage has been uploaded anywhere.

## Easiest option: Claude Code on this computer

Open this handoff folder in Claude Code. Give it `CLAUDE_PROMPT.md` and ask it to
read `CLAUDE_HANDOFF.md` before changing anything. Work inside `project/`.
The complete loose folder includes the model weights and two longer source videos.
Existing virtual environments are intentionally not copied.

To see the current working result, open `project/runs/raw-play-ready/replay.html`.
It is still the 20-second baseline, NOT a newly processed game or mesh upgrade.

## Moving the project to another computer

Extract these three sibling ZIPs into the SAME empty destination folder:

1. `claude-tennis-handoff.zip`: instructions, source, tests, reviewed calibration,
   cached poses, and the working replay. Start with this package.
2. `claude-tennis-models.zip`: optional for simply viewing; required for fresh
   inference and the documented pose-cache rebuild. Contains trained weights.
3. `claude-tennis-footage.zip`: the two longer videos for the next analysis.

All archives use matching `project/...` paths. Do not nest one project inside
another. `MANIFEST.json` inventories the complete package by asset category and
SHA-256; absent optional models/footage are expected if only ZIP 1 is extracted.

For a Claude chat, paste the prompt and provide the files its interface supports.
If that environment cannot execute local video/model work, use Claude Code or
have Claude return patches; attaching the prompt alone does not grant file access.

## What to ask for

First a longer-run workflow and a baseline comparison, then 3D player bodies and
racquets. These are development tasks for Claude, not completed features in this
handoff. Read `CLAUDE_HANDOFF.md` for acceptance checks and constraints.

When coming back to Codex, bring the updated folder and a completed
`RETURN_TO_CODEX.md`, including the exact command and result to open.
