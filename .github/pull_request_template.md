<!-- Base branch: dev (John promotes dev -> main himself). -->

## Summary

<!-- One or two sentences: what changed and why. Link the task: Closes #N -->

## What ran

<!-- Exact commands and their results (test counts, runtimes, output folders). -->

- [ ] `python -m unittest discover -s tests -q` → 
- [ ] JS suites (shot replay, player shots, calibration, event review) → 
- [ ] Other commands / runs:

## Verified vs not verified

<!-- Be specific. "Browser-checked frame 300 in Chrome" beats "UI works".
     Fake-DOM tests are not browser rendering. Coverage is not accuracy. -->

**Verified:**

**Not verified:**

## Data safety

- [ ] No reviewed files, historic runs, `court.yaml` or review IDs were modified
- [ ] No footage, frame stills or model weights committed or uploaded
- [ ] Observed vs estimated vs render-only geometry stays labelled

## Decisions needed from John

<!-- Questions Claude did not guess on. "None" if none. -->

## How to try it

<!-- The one command John runs, from which folder, and what file to open. -->
