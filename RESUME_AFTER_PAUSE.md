# RESUME AFTER THE 2026-08-21 PAUSE

Andrew paused the loop after featA2's decay so the PC could finish the FSRS-7 benchmark
(*"After decay is done, let's pause. I want to free the PC to finish the FSRS-7 benchmark"*).

Everything below is armed-and-verified or complete. **Nothing will start on its own** — every
waiter was disarmed, so the machine stays yours until you run step 1.

## What is DONE and needs no repeat

* **The int64 id fix** (`data_processing.create_sample`) plus its value-comparison guard, the
  identity smoke (`scratchpad/parity3/smoke_id_identity.py`) and the bf16 label audit.
* **Gen-3 `-id` dbs** — `train_db_5k_h1_id3` (1,483,984 entries) / `test_db_5k_id3` (170,384),
  both width 46, both verified id-healthy. Gen 1 and gen 2 are deleted; they were broken.
* **The id-fixed PUBLISHED dbs** — `train_db_5k_h1_fix` (1,483,984, width 24) and
  `test_db_5k_fix`. These are what featA2 trains and evaluates on.
* **featA2 WS + decay** — `featA2_ws_10935.pth` and `featA2_d_10935.pth`.

## The state featA2 was paused in

featA2 completed **WS (15:39)** and **decay**, and was stopped at the start of its eval phase.
Its runner therefore never generated `eval.toml`, which is why resuming needs the eval-only runner
below rather than a re-launch of `run_featA2.cmd` (that would redo 8 h of training).

## ⚠ THE PAUSE LEFT A TERMINAL MARKER IN featA2.log

Killing the eval let the runner's own error branch write `featA2 EVAL_FAILED_-1` and
`DONE_EXIT_25` before the cmd exited. So **`featA2.log` now carries a terminal marker for a run
that did not fail on its merits** -- the decay it depended on completed fine.

Consequence: any waiter polling `featA2.log` fires INSTANTLY. That is the same trap as featB's
dead log, created by the pause itself rather than by a crash. The eval-only runner writes to its
own `featA2_evalonly.log`, so it is unaffected; only re-arming something on `featA2.log` is unsafe.

## Step 1 — finish featA2's eval

```
powershell -File scratchpad\detach.ps1 -Script C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\run_featA2_evalonly.cmd
```

~3-4 h on a free GPU. Writes `result/RWKV-featA2.jsonl` and `result/RWKV-P-featA2.jsonl`.

## Step 2 — featB, the treatment arm

```
powershell -File scratchpad\detach.ps1 -Script C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_ab\featB\run_featB.cmd
```

~11 h. Preflight PASS, param guard 565252, all three db paths on `_id3`.
⚠ Its `featB.log` from the 04:09 id-bug failure was renamed to `featB_failed_idbug_0409.log`, so
the log starts clean. Do not restore it — a stale terminal marker in that file fires any waiter
polling it instantly.

## Step 3 — the verdict

```
.venv\Scripts\python.exe scratchpad\features_ab\verdict.py
```

Applies the bands pre-registered in `optimization/FUTURE_FEATURES.md` **before either number
existed**: >= +0.0010 both modes = adopt and the family ablation earns its ~54 h; +0.0003..+0.0010
= adopt but ablate only 2-3 families; below that = do not adopt and do NOT run per-feature arms.
It also enforces the three artefact checks and refuses to band a partially evaluated arm.

## Step 4 — optional, the champion's TRUE deploy accuracy

```
powershell -File scratchpad\detach.ps1 -Script C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\champ_deploy\run_champ_deploy_resume.cmd
```

Waits for featB, so it can never co-tenant the GPU again. 455 of 2500 users are already banked and
it does NOT delete them, so it resumes rather than restarts. This scores the UNCHANGED iter-53
checkpoint against the id-fixed eval db, which is what it would actually do in Anki — the recorded
0.297523 / 0.265191 was measured on int32-grouped ids that deploy never uses.

## ⚠ Read before resuming

* **Check what every waiter polls before arming it.** A crashed run's terminal marker satisfies a
  downstream grep and fires it instantly. That is why featB's dead log was renamed.
* **Do not run two GPU jobs at once.** On 2026-08-21 the champion eval and featA2 overlapped at
  11,991 of 12,282 MiB and featA2 managed 2 steps in 48 minutes.
* `scratchpad/chain_watch.sh <arm_dir> ...` follows whichever arm is live and alarms on a STALL as
  well as a death. Arm it alongside step 1 or 2.
