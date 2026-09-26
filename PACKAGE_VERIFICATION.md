# Verification of the copied handoff

Run on 2026-09-25 from the handoff's `project/` directory, using the original
project's `.venv-ball/Scripts/python.exe` and Node available on this machine.
Virtual environments are not portable and are not included in the archives.

- `python -m unittest discover -s tests -q -b`: **125 tests passed**.
- `node tests/test_shot_replay_ui.cjs`: passed.
- `node tests/test_player_shots_ui.cjs runs/raw-play-ready/replay.html`: passed.
- `node tests/test_calibration_review_ui.cjs runs/court-bounce-ready/calibration.html`: passed.
- `node tests/test_event_review_ui.cjs`: passed.
- The documented cached replay rebuild ran successfully against the copied
  source, court, review files, pose cache and checkpoint. It reported
  `Using matching focused-pose cache`; no fresh detector inference was needed.
  Output went to the original workspace's `work/claude-handoff-rebuild`, outside
  the handoff, so the supplied baseline files remain unchanged.
- Rebuild preserved review run ID
  `39449302177328c739e679a953b51d001bf297e21541feb1cc38c76265aa9ab9`
  and package ID
  `3c970d6526e0854df89f0742104973372d7834a86b8534aeefe104087baaa4e1`.
  It reports 600 frames, source offset 4740, one serve candidate, one confirmed
  bounce, and 600 displayed roots for each player, matching the baseline.

This is functional regression and relocation/cache verification, not a fresh
installation test, game-length inference run, browser-rendering check, or
measurement of real-world detection/3D accuracy. No new player/racquet mesh is
implemented by this packaging task.

Archive creation checks all ZIP member CRCs and records file SHA-256 hashes in
MANIFEST.json. Existing copied project files are checked against their originals
before sealing. The archives omit bytecode and machine inference settings.
