Please continue John's local tennis-video analysis project in `project/`.
Read CLAUDE_HANDOFF.md and project/RAW-PLAY.md first. Preserve the supplied
baseline, human-reviewed calibration, and all existing review bindings.

The next priorities, in order, are:

1. Process a longer piece of raw footage (ideally a complete game). Two roughly
   three-minute sources are included under project/media; verify their content
   and ask John for game boundaries or more footage if they are not a full game.
   Generalize the sample-specific run scripts, keep progress/ETA, preserve source
   frame timing, and make long processing resumable and bounded in memory.
2. Show actual 3D player bodies/meshes in the virtual replay, driven by the
   available observations and grounded feet, with a wireframe toggle. Keep clear
   which geometry is observed, estimated, or only a visual animation choice.
3. Add 3D racquets associated with the correct players and stroke hands, with
   confidence-aware visibility. Do not treat the rendered racquet as proof of
   contact or claim its orientation is measured when it is not.

Raw footage contains walking, preparation and unrecognizable motion. A swing is
not automatically a shot. Missing observations do not prove an ace or a return.
John described the baseline play as an ace, but it has no exact point-boundary
ground truth. Players may use either hand, forehands on both sides, or two-handed
strokes on either side. Do not impose a fixed handedness-to-shot-type rule.

Start by running the included regression checks and inspecting the current
replay. Work in new output folders. Do not silently rewrite old review IDs,
human labels, or original tracks to make tests pass. Keep all processing local
unless John explicitly approves external upload, paid jobs, or cloud inference.
Do not make spin/RPM claims from these monocular clips.

Implement and verify these upgrades in manageable stages. For each stage report
what actually ran, what was visually checked, remaining limitations, and the
command John should run next. Complete RETURN_TO_CODEX.md so Codex can resume
from the returned folder without needing this entire conversation.
