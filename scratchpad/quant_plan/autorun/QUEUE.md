# AUTORUN QUEUE - the hourly supervisor's contract

Owner: Andrew. Runner: the hourly goose run of `quant_hourly.yaml` (Windows Task Scheduler task
`RWKV-Quant-Autorun`), started by `run_autorun.cmd`. This file says exactly what that run MAY do.

Read the `status_report.py` output first, then act only through the sections below, in order.

If this file and the recipe disagree, this file wins. If anything is unclear, do nothing and log it.

---

## 1. NEVER restart, never touch

These chains are dead on purpose. Their logs have no end marker. Leave their logs, lock files, output
folders and checkpoints alone. Do not score them and do not resume them.

- `it36qat496` - killed near step 250 in the 2026-11-20 ~10:28 machine outage; its checkpoint is not usable.
- `it36qatl1i4` - launched from the post-decay checkpoint after the same outage; its folder is not usable.
- `it36ptq` - finished its real work; only the end marker is missing.

Also never touch `scratchpad/quant_plan/qat_it36qat496/` or `scratchpad/quant_plan/qat_l1i8e1/`.

## 2. LIVE chains, and the restart rule

CURRENT live list (supersedes everything below; edited by hand as chains land). These are the chains
the interactive session launched, and the restart rule below applies to each of them:

`it36shiftrung` (chain 25, the shift-width rungs) -> `it36fbrung` (chain 26, the codec `m` ladder) ->
`it36fbwidth` (chain 27, the codec's coefficient bit-width + the CARD on the codec). Each takes its own
lock file and waits for its predecessor's `=== <name> chain end ===` marker, so a WAITING chain's log
GROWS once a minute and the 45-minute rule never fires for it. A chain the interactive session runs
that is NOT listed here must not be started, killed, restarted or scored by this run.

LIVE chains, in run order (Andrew resumed experiments on 2026-11-22 12:02: "Since corruption checks are
done, we can resume experiments. Sure, try rank-2 note"):

`it36noterank2` -> `it36shrink` -> `it36equalbits` -> `it36combo` -> `it36corrupt2`

All five are LIVE. The 11:21 pause ended when `it36corrupt` finished at 11:41 with five PASSes; the four
chains killed at 11:24 were relaunched at ~12:06 waiting behind `it36noterank2`, and `it36noterank2`
(the rank-2/3 NOTE ladder from Andrew's 12:01 direction) started at 12:05:28.

Each chain takes its own lock file and, while waiting, logs `waiting: prev chain ended=..., matching
processes=N` once a minute. So a WAITING chain's log GROWS and the 45-minute rule below never fires for
it. A chain killed by an earlier stop has no end marker and its log stops growing, which the report
shows as `STALLED?`; for the four relaunched chains that is the OLD kill, not a new stall - read the log
tail before doing anything.

The report prints a `CPU over 5 s:` line (added 2026-11-22, after an F: disk I/O error wedged
`it36shrink` for 8.7 h at 0.000 s CPU while the report still said `GPU BUSY` and every chain log looked
normal). If that line says `WEDGE? live workload processes burned no CPU`, do NOT kill anything: log
`NEEDS ANDREW: wedged workload at <time> - <pid lines>` and STOP. The chains now kill and retry such a
run by themselves (`it36ladder_lib.run`: 20 min with neither log growth nor CPU -> `taskkill /F /T`,
then one retry).

`it36corrupt` (the corruption/finiteness check) is DONE: five points, `=== it36corrupt chain end ===`
at 11:41:38, all five PASS - RESULTS.md rows 17, 15, 16, 18 and the quantization-OFF control of row 0,
50/50 users each, 158,986 raw state observations and 150 p(recall) tensors per quantized point, zero
non-finite. `it36corrupt2` (16 gate-guarded points, four of them `it36noterank2`'s) is the chain that
checks the new rows, and it is last in the queue.

A chain queued BEHIND a corruption chain must add `probe_get_result.py|check_corruption.py` to its own
BUSY_PAT, or it will start on top of the corruption run (the driver's command line matches no other
chain's pattern). `it36shrink`, `it36equalbits`, `it36combo` and `it36corrupt2` all carry those two
patterns.

The chains sequence themselves through their own lock file and their `=== <name> chain end` marker.
Normally you do nothing for them.

Restart a LIVE chain ONLY when all four conditions hold:

1. the status report says `GPU IDLE (no matching workload process)`;
2. that chain's log has no `=== <name> chain end` line;
3. that chain's log file has not changed for more than 45 minutes;
4. host RAM free is at least 10 GiB.

Then run these two commands and log the restart:

    del "scratchpad\quant_plan\<name>_chain.lock"
    powershell -NoProfile -File scratchpad\detach.ps1 -Command ".venv\Scripts\python.exe -u scratchpad\quant_plan\<name>_chain.py > scratchpad\quant_plan\<name>_launcher.log 2>&1"

The chain skips any point whose merged result already exists, so a restart resumes cleanly. A WAITER
that stalled is safe to restart: it re-takes its lock and goes back to waiting for its predecessor.
Restart at most one chain per run. Never restart a chain while any workload process is running.

## 3. Idle-only deploy measurement (run at most once)

Run this ONLY when the report says `GPU IDLE`, no LIVE chain is waiting, and section 5 is empty:

    .venv\Scripts\python.exe scratchpad\quant_plan\rust_speed.py --weights scratchpad\quant_plan\rust_artifacts\it36l2i8_124.safetensors --configs plain,l1i8,l2i8,l2i8s4,l3i8s4 --users 107,136,156 --reps 2 --tag idle_l2i8

It writes `scratchpad/quant_plan/rust_speed_idle_l2i8.json`. **DONE 2026-11-22 11:47** (this section is closed and the file exists, so skip it): `plain` 2,915.1 rev/s, `l1i8` 0.902x, `l2i8` / `l2i8s4` / `l3i8s4` all 2,357.8 rev/s = 0.809x, every config above the 0.75x floor. That file is the official constraint-6 number: the 0.812x/0.800x cells of RESULTS.md rows 15-18, measured while a decay was training, are accurate to about 1% - not the 3% pessimistic that RESULTS.md feared.
If `rust_speed_idle_l2i8.json` already exists, skip this section.

## 4. Record landed results

For every sweep result the report lists as `NOT RECORDED`:

1. add one row to `scratchpad/quant_plan/RESULTS.md` in the existing table format (iteration number,
   method description, card bits, note bits, speed, both log-loss values, both deltas, verdict,
   corruption). The main table has 12 columns since 2026-11-21 23:00 (Andrew's corruption gate):
   write `pending` in the last cell when `it36corrupt` has not scored that row yet, and `-` when the row
   fails the log-loss gate (not a deploy candidate, so it is never checked). Write `PASS` or `FAIL`
   (with the reason) only from `corruption/corruption_corrupt_<tag>.json` - never from a guess. A row
   with 11 cells breaks the table;
2. append the sweep stamp to `scratchpad/quant_plan/autorun/RECORDED.txt`;
3. update the `**Status:**` line of RESULTS.md and add one bullet at the top of the
   `## Live status` list in `scratchpad/quant_plan/NOTES.md`. Name the candidate tag inside the new
   row's Method cell (e.g. `l3i4s4`): the corruption filler below finds a row by its tag when the row
   is not in its built-in map;
4. fill the `corruption` column from any corruption verdicts that exist now, by running exactly this
   (it rewrites ONLY the last cell of each row it can identify, and refuses a row that does not have
   12 cells; add `--dry-run` first if you want to see what it would do):

       .venv\Scripts\python.exe scratchpad\quant_plan\corruption\record_corruption.py

   A verdict that has no row yet is reported and skipped - write the row first, then run it again.
   Never hand-edit a corruption cell. The same run also flips a cell that is still exactly `pending`
   to `-` when that row's own sweep verdict FAILS the log-loss gate (such a row is never a deploy
   candidate, so it is never checked): if you wrote `pending` for a row that then failed, this fixes
   it in the same command.

Then PUSH RESULTS.md, because Andrew's standing instruction is that every update of RESULTS.md goes to
GitHub (2026-11-21). Do it EXACTLY like this, in this order, and nothing else:

    cd /d C:\Users\Andrew\rwkv-anki-autoresearch
    git add scratchpad/quant_plan/RESULTS.md
    git diff --cached --name-status
    git commit -m "Hourly supervisor: record landed sweep results in RESULTS.md"
    git push origin main

Rules for that push, all of them hard:

- `git diff --cached --name-status` (run it, do not use `git status --short --cached`: this checkout's
  git rejects that option with exit 129) must list ONLY `scratchpad/quant_plan/RESULTS.md` (plus, if you also
  edited them in this same run, `scratchpad/quant_plan/NOTES.md` and
  `optimization/GPU_RUN_BOARD.md`). If ANY other path is staged, run
  `git reset` and log `PUSH ABORTED: unexpected staged files` - do not commit, do not push.
- If any command fails (including `index.lock` errors, which mean another git process is running), log
  the error and STOP. Never retry in a loop, never use `--force`, never `git add -A` or `git add .`,
  never commit a checkpoint or a database, never change the branch.
- Do not push anything else, ever. This is the only git operation the supervisor is allowed to do.

**If RESULTS.md was NOT changed this run, do not run any git command at all.**

Rules for the numbers: quote the paired deltas against the control tag the sweep used (the `ref`
field, normally `qnj_it36ctl`), never against the iter36 champion. The gate is the row-0 anchor: the
identical decay with no QAT and no quantization. A row passes only when `ub95 <= +0.0005` on BOTH heads.

## 5. Pre-authorized launch items

None right now, and none are needed: the five LIVE chains of section 2 already own the queue in run
order, so there is no `- [ ]` item for you to launch. Do NOT launch a chain by hand and do NOT add one -
the queue drains by itself.

When the report says `GPU IDLE` while a LIVE chain is waiting (its log says `waiting: ...`), that is a
normal hand-off: the next chain starts on its own within about a minute. It is NOT an instruction to
start anything.

When `it36corrupt2` logs `=== it36corrupt2 chain end ===` and the GPU is idle, there is nothing left to
run: log `QUEUE EMPTY - needs Andrew's decision` and stop. Do NOT invent a new experiment, do NOT extend
an existing chain, and do NOT restart a finished chain.

The CPU-only Rust checks (`rust_finite.py`, `measure_finite_offpath.py`, `measure_offpath_oldbuild.py`)
also exist now, but they are the interactive session's job, NOT yours: each one ends in an artifact that
a human decision depends on.

Andrew adds an item here when he authorizes one. A real item is one raw, unindented line: the box
marker, then the task, then `::`, then the exact command line, then `::`, then the proof of success.
`status_report.py` counts only raw unindented box lines, so keep any example indented like this
paragraph:

    - [ ] <what to run> :: <exact command line> :: <what proves it worked>

## 6. Forbidden, always

- Starting a second GPU workload, or starting anything while a workload process runs.
- Killing, restarting or modifying a run that is in progress.
- Changing the deploy contract, a state budget, the quantization gate, or the bit bars.
- Starting the 3,000/3,000-bit run (vetoed), or any new research direction.
- Deleting, moving or rewriting data, checkpoints, logs, sweep JSONs or result files.
- `git` writes of any kind: no `add`, no `commit`, no `push`, no pull request, EXCEPT the single
  RESULTS.md-only commit-and-push described in section 4 and nothing else.
- Editing a sibling repository, or anything outside `C:\Users\Andrew\rwkv-anki-autoresearch`.
- Editing `quant_sweep.py`, `qat_decay.py`, `it36ladder_lib.py` or any `it36*_chain.py`.
- Letting host RAM free fall below 10 GiB, or running with the RAM guard stale.

## 7. If a corruption verdict is FAIL

Andrew's gate (2026-11-21 23:05): a non-finite card/note state value, or a non-finite `p(recall)`, makes
the recipe undeployable. The fix (a state-growth penalty plus a max cap, or a re-normalization) and the
advisor consult that Andrew asked for are the interactive session's job, not yours.

So: record the verdict exactly as section 4 says, then log
`NEEDS ANDREW: corruption FAIL for <tag> - state-growth fix + advisor consult required` and STOP.
Do NOT change the quantizer or any other source file, do NOT restart the chain that produced the
verdict, do NOT start anything, and do NOT describe the row as passing anywhere.
