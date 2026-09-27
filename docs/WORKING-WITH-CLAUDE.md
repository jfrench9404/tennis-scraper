# Working with Claude: the day/night cadence

How John and Claude split work on this repo so overnight sessions come back as
pull requests John can review quickly. Claude's own rules are in `CLAUDE.md`.

## Branches

```
claude/12-shot-shortcuts ─┐
claude/13-replay-speed ───┼─► PR into dev ──► you try it on dev ──► PR dev → main
claude/14-ball-trail ─────┘   (Claude opens)                         (you open + merge)
```

- **`main`** is stable: only things you have tried on `dev`. Only you merge into it.
- **`dev`** is integration: every Claude PR targets `dev`. Merge the ones you like, try
  the result from a `dev` checkout, then promote.
- **`claude/*`** holds one task each; delete it after merging (GitHub offers a button).

To promote, open https://github.com/jfrench9404/tennis-scraper/compare/main...dev?expand=1,
check the tests are green, and merge it (or run `gh pr create --base main --head dev`).

## The daily loop

| When | Who | What (time) |
| --- | --- | --- |
| Evening | John | Write or pick tasks as issues (*Task for Claude* form), label tonight's with `tonight`. If a compute job is needed, start it in your own PowerShell window. Then start Claude with the prompt below. (about 15 min) |
| Overnight | Claude | For each `tonight` issue, in priority order: branch → implement → tests → push → PR (draft if anything is unverified or undecided). No long inference inside the session. Finish with one summary comment on a "Night of <date>" issue. |
| Morning | John | Read the night summary. For each PR into `dev`: CI green? Read *Decisions needed* and *Not verified*, then run *How to try it*. Merge into `dev`, or leave review comments. Promote `dev` → `main` when you're happy with it. Check or restart the compute job. (about 20-30 min) |
| Daytime (optional) | Both | Answer the decisions, pair on judgment-heavy work (UI feel, labels, game boundaries), write tomorrow's issues together. |

Feedback goes on the PR itself, as line comments or a review. The next session starts with
"Address the review comments on PR #N", so nothing gets lost in chat.

### Starting a night

Paste into a new Claude Code session opened on this repo:

> Work the open issues labelled `tonight` in priority order, following CLAUDE.md:
> one branch and PR per issue, draft PRs when anything is unverified or needs my
> decision, never run long inference yourself. When you finish or get stuck, post a
> summary comment on the "Night of <date>" issue listing each PR, what was verified,
> and my decisions to make.

## What makes a good overnight task

- **Provable:** acceptance checks a test or command can confirm ("N/P keys jump between
  shot candidates" plus a test), not "make it better".
- **One area of the code:** two PRs that both edit `shot_replay.html` will conflict in
  the morning. Spread tonight's tasks across different files, or chain them in one PR.
- **Decisions listed up front:** anything that is yours (game boundaries, confirming
  events, uploads, visual taste) goes in *Decisions reserved for John*, so Claude
  asks instead of guessing.
- **Right size:** about 1-3 hours of focused work. Split anything bigger.

Poor overnight tasks: accuracy improvements without labelled ground truth, anything
that needs your eyes mid-way, repo-wide refactors, and hours of GPU inference.

## About 8-10 features a night

Doable for code tasks, with three limits:

1. **Review time is the real bottleneck.** Ten PRs are only useful if each takes a
   few minutes to review. That is why the PR template, CI and small scopes matter.
2. **Parallel work needs separate branches and worktrees.** Each feature gets its own
   checkout so they don't trip over each other. Overlapping files mean merge conflicts.
   After you merge some, Claude merges the latest `dev` into the others and re-tests.
3. **Compute is serial and belongs to your terminal.** The laptop GPU does one inference
   job at a time (a 3-minute video is about 7.5 h). Jobs started inside a Claude session
   die when the session ends, which is what stopped the first overnight run. Claude writes
   the command; you run it in PowerShell. `Run-Game.ps1` resumes where it left off.

A realistic night: one compute job running in your terminal, plus several code PRs.

## Where work can run

| Where | Sees footage and weights? | Good for |
| --- | --- | --- |
| Claude Code on this laptop | Yes | Everything, including building replays from cached runs and browser checks |
| Cloud Claude (GitHub @claude mentions, scheduled cloud sessions) | **No**, they never leave this PC | Code-only tasks. Tests and CI already run without footage. |
| GitHub Actions CI (`tests.yml`) | No | The regression suite on every push and PR |

## One-time setup (John)

1. **Merge this workflow PR into `dev`, then promote `dev` → `main`.** The overnight work
   is already in both.
2. **Install and log in to the GitHub CLI** so Claude can open PRs, read your review
   comments and check CI. Install with `winget install --id GitHub.cli`. Then, in a
   **new** PowerShell window (older windows don't see the new install):

   ```powershell
   gh auth login --hostname github.com --git-protocol https --web
   ```

   It shows an 8-character code and opens github.com/login/device. Paste the code
   there and click Authorize. Check with `gh auth status`.
   - **Browser never opens:** go to https://github.com/login/device yourself and
     enter the code.
   - **Certificate / TLS error:** your git also needed Windows' certificate store
     (`http.sslBackend=schannel`), so something on this PC inspects HTTPS. Log in with
     a token instead. On github.com open Settings → Developer settings →
     Personal access tokens → Fine-grained. Create one for `tennis-scraper` only, with
     Contents, Pull requests and Issues set to read/write and Actions set to read. Then
     run `gh auth login --with-token` and paste it at the prompt yourself; never paste it
     into a chat.
   - If you run it in the Claude app's Terminal panel, Claude can read the error.
3. **Create labels:** `claude-task`, `tonight`, `needs-john`, `compute`. Once `gh`
   works, Claude can create these for you.
4. **Protect `main` and `dev`:** in the repo's Settings → Branches (or Rules), add a
   rule for each that requires a pull request and a passing `tests` check before
   merging. On `main`, also block force pushes.
5. **Keep the laptop awake when plugged in:** in Windows power settings, set sleep to
   Never and closing the lid to Do nothing while plugged in. Only you should change these.
6. **Free some disk space:** C: had only a few GB free, which is why the GPU build of
   PyTorch couldn't be installed.
