# TA3 (Training-Time Testing) arm — notes

Durable notes for the TA3 experiment. Live status lives in `optimization/GPU_RUN_BOARD.md`
(row 10). This file records the lineage, the pre-registered endpoints, the gate history and the
resume hazards, so a future instance does not have to reconstruct them.

Method source: Dang 2026, <https://dangxingyu.github.io/blog/t3.html> (mirror:
`scratchpad/t3_blog.txt`). Consult records: `optimization/PROPOSALS.md` (2026-11-13 and
2026-11-15 entries).

## FINAL VERDICT (2026-11-19): NEGATIVE, arm CLOSED

Andrew stopped the chain mid epoch 4 and closed the arm: **TA3 does not help here.** Recorded in
`optimization/research_log.jsonl` #128 and `optimization/research_5k.md` row 128; the run-board
row 10 reads ABORTED.

Evidence (epoch-matched 750-user screens, delta = plain minus TA3, so negative = TA3 worse):

| epoch | TA3 ahead / imm | plain ahead / imm | ahead delta | imm delta |
|---:|---|---|---|---|
| 1 | 0.309108 / 0.275626 | 0.308803 / 0.275131 | -0.000305 (p 0.9995) | -0.000494 (p 1.0) |
| 2 | 0.309924 / 0.273467 | 0.307044 / 0.274275 | -0.002880 (p 1.0) | +0.000808 (p 5.0e-29) |
| 3 | 0.306259 / 0.273344 | 0.305847 / 0.272732 | -0.000412 (p 0.9994) | -0.000613 (p 1.0) |

Every epoch fails the dual-mode gate. Ahead is worse in all three and the ahead curve
(0.30911 -> 0.30992 -> 0.30626) is non-monotone and stuck at the plain epoch-1/epoch-2 level:
TA3 trained SLOWER rather than better. No TA3 step ever failed, so the code path is not the
explanation.

Two limits on what this result can claim: (a) the pre-registered primary endpoint was the 4-epoch
checkpoint and the abort came at step 138,044 of 173,416, so this is screen-level evidence, not a
completed endpoint; (b) the A/Q split here is within-batch by row rank because 98.2% of batches
hold exactly one user group, which is a much weaker test-half than the paper's multi-epoch
pretraining setting. So this negative does not refute TA3 at scale.

Do not resume this chain. Do not re-run TA3 without a new mechanism-level idea (for example a
by-user A/Q split, which needs a data-pipeline change because the batch stream cannot supply it).

## What TA3 does here

Per optimizer step, `rwkv/ta3.py:training_step` owns the WHOLE gradient:

1. Split the batch into halves A and Q by a **loss-mask change on `has_label`** at time position
   `frac=0.5`. The fetched batch stream stays byte-identical; no by-user split exists (98.2% of
   batches are one user's group: 42,592 of 43,354).
2. Two mode-masked passes on A (ahead group, immediate group) -> `g_A`. The direction is
   `u = g_A / ||g_A[scope]||` (one global L2 norm over the scoped matrices).
3. Trial step: `master -= eta * u` **downhill** (`_shift` takes a DESCENT scale; positive = descent),
   then `copy_downcast_` so the child sees the perturbed weights.
4. Two mode-masked passes on Q at the perturbed weights -> `g_Q` (first-order MAML: no gradient
   through the shift).
5. Restore the master weights with `copy_` from a snapshot (exact, no add/sub round trip).
6. Leave in the master grads the **per-loss-group weighted merge**
   `(N_a^A/N_a)*g_A^ahead + (N_a^Q/N_a)*g_Q^ahead + (N_i^A/N_i)*g_A^imm + (N_i^Q/N_i)*g_Q^imm`,
   where `N` are the whole batch's group row counts and `N^P` the phase's. Clip + optimizer are
   untouched.

Scope = 28 explicit channel-mixer `.weight` matrices (28,672 of 192,800 parameters), listed at
startup by `scope_report`. `RWKV_TA3_SCOPE=cmix|trunk|all`; unknown values are rejected.

Env gates: `RWKV_TA3=1`, `RWKV_TA3_ETA` (0.3), `RWKV_TA3_SCOPE` (cmix), `RWKV_TA3_FRAC` (0.5),
`RWKV_TA3_PROBE=<n>` (measure-only), `RWKV_TA3_PROBE_ETAS`, `RWKV_TA3_CHECK=<n>` (gradient gate).
Inert when unset. TA3 and `RWKV_SAM_RHO` raise if both are set. Failures abort with exit 44 —
a TA3 failure must never degrade into a plain step.

## Pre-registered endpoints (written 2026-11-15, before the chain started)

* Primary: the 4-epoch TA3 checkpoint `scratchpad/ta3_fullrun_20261115/epoch_04/ta3_173416.pth`
  versus the plain 4-epoch checkpoint
  `scratchpad/adaptive_wd/adaptive_fullrun_92_20261105/attempt_04_epoch_04_wd_0.01/adaptive_173416.pth`
  (screen 0.304392 ahead / 0.272142 imm on the fixed 750-user roster).
* Gate: on the same 750-user roster BOTH modes must improve by >= 0.0001 with p < 1e-4.
* Per-epoch screens are **diagnostic only**. No accept/reject, no rollback, no WD switching.
* One chain is a screen-level result. A method-level claim needs independent paired seeds
  (preferably three).
* Nothing else changes: same start pair (QAT46 step-248), same 92-input data, same LR shape and
  offsets, WD 0.01, one fetch worker, 11 GiB VRAM cap, 2-user internal validation.

## Epoch plan (mirrors the plain chain)

| epoch | `EPOCHS` | `STEP_OFFSET` | loads | saves |
|---:|---:|---:|---|---|
| 1 | 1 | 1 | `pretrain/rwkv/opt_qat46/rwkv_iter46_248(.pth)` | `ta3_43354.pth` |
| 2 | 2 | 43,355 | `ta3_43354.pth` | `ta3_86708.pth` |
| 3 | 3 | 86,709 | `ta3_86708.pth` | `ta3_130062.pth` |
| 4 | 4 | 130,063 | `ta3_130062.pth` | `ta3_173416.pth` |

`total_steps = EPOCHS * len(groups)` and `EPOCHS = n` with `STEP_OFFSET = (n-1)*g + 1` trains exactly
epoch `n-1`; `g = 43,354`. Each epoch is 43,354 steps.

Run directory: `scratchpad/ta3_fullrun_20261116/` (the restart after the split fix). The aborted first
attempt `scratchpad/ta3_fullrun_20261115/` is preserved, not resumed.

## Gate history (why the gate is restated statistically)

Recorded in `scratchpad/ta3_gradcheck_20261115/`:

* Weight restore PASS (0.0 drift), optimizer state PASS (0.0 delta at eta=0.3).
* eta=0 must reproduce the plain gradient. It does not exactly: residual relative L2 2.1-12.7%
  (cosine 0.996-1.000) after two fixes (per-mode weights, then mode-masked passes).
* Repeatability control: the **identical plain pass run twice** differs by 0.007-0.109 relative L2
  on this bf16 path. The eta=0 residual sits at that floor; the eta=0.3 effect is 18.7-51.1%,
  above it. Value-level proof that the decomposition is right: `loss_groups_value == average_loss`
  to ~1e-7.
* An exact gate needs fp32 (the TorchScript `_get_loss` rejects float32 today) or a bf16
  nondeterminism investigation (RWKV_CUDA reductions/atomics, cuBLAS workspace). Parked.

Probe (30 real batches, `scratchpad/ta3_probe_20261115/ta3_probe.jsonl`): a normalised downhill
step on A lowers Q's loss at every eta <= 0.3, beating the equal-norm random control 27/30
(Wilcoxon p 1.0e-07 / 2.4e-07 / 2.6e-06) and uphill at every eta. `cos(g_A, g_Q)` all-params +0.387
(positive 28/30). Window eta ~0.1-0.3; turns over at 1.0. Per-element displacement at eta 0.3 is
0.00173, about 1.7x the optimizer lr 0.001. Probe numbers are mechanism evidence, never efficacy.

## Cost

The run is LMDB-fetch-bound (GPU 0-22% on plain attempts), so the extra three passes cost almost
nothing in wall time: measured ~43-45 steps/min with TA3 versus ~40 steps/min plain. One epoch is
about 16 h; the four-epoch chain about 2.7-3 days.

CORRECTED after the real chain ran (2026-11-17/19): epochs 2 and 3 each took about 23.5 h of
training for 43,354 steps, i.e. ~31 steps/min with TA3, and epoch 4 is holding ~32 steps/min. The
smoke's 43-45 steps/min was a short-run upper bound. Budget ~24 h per epoch plus ~50-60 min for its
screen, so the four-epoch chain is about 4 days, not 3.

## Epoch screens (diagnostic; measured 2026-11-17 to 2026-11-19)

`ta3_train.py` runs the standard 750-user screen after every epoch. These are DIAGNOSTIC only: no
accept/reject and no rollback. Deltas are plain minus TA3, so a POSITIVE delta means TA3 is better.
Plain = the epoch-matched plain attempt of `adaptive_fullrun_92_20261105_90fecbaa` (a01-a04).
Paired per-user comparison by `scratchpad/ta3/compare_screens.py`.

| epoch | TA3 ahead / imm | plain ahead / imm | ahead delta (p) | imm delta (p) |
|---:|---|---|---|---|
| 1 | 0.309108 / 0.275626 | 0.308803 / 0.275131 | -0.000305 (0.9995) | -0.000494 (1.0) |
| 2 | 0.309924 / 0.273467 | 0.307044 / 0.274275 | -0.002880 (1.0) | +0.000808 (5.0e-29) |
| 3 | 0.306259 / 0.273344 | 0.305847 / 0.272732 | -0.000412 (0.9994) | -0.000613 (1.0) |
| 4 | pending | 0.304392 / 0.272142 | pending | pending |

All three measured epochs fail the dual-mode gate. The ahead mode is worse in every epoch, so the
interim trend is negative; epoch 4 is still the pre-registered PRIMARY endpoint, and no conclusion
about the method may be drawn before it is scored. TA3's ahead curve (0.30911 -> 0.30992 -> 0.30626)
is non-monotone and sits near the plain epoch-1/epoch-2 level.

## Incidents

### 2026-11-15/16 — epoch 1 aborted at step 4,930 (`mode 'ahead' has no labelled rows in this phase`)

The first chain epoch died at 19:55 on 2026-11-15 and the GPU then sat idle for ~12 h because nothing
was watching. Root cause: the A/Q split used ONE cut at `round(0.5 * T)` on the raw time axis, and some
real batches put a whole loss group on one side of it, so a phase had zero rows for that group and
`_group_grad` raised. Fail-closed, so nothing was corrupted; the last checkpoint pair was
`ta3_4500.pth` + optim. The aborted run dir `scratchpad/ta3_fullrun_20261115/` is preserved as evidence
and is NOT resumed (a clean epoch is worth more than the ~2 h of steps it contains).

Two fixes, both in `rwkv/ta3.py`:

1. FIRST ATTEMPT (rejected by test): cut each group at the midpoint of ITS OWN labelled positions,
   as a TIME threshold. `scratchpad/ta3/test_split.py` showed this still fails: with more than one
   chunk per batch several rows share a time position, and a threshold keeps or drops whole tied
   groups together (`imm full=7 but phases 0/7`).
2. FINAL FIX: `_rank_split` / `_mode_split` split each loss group by RANK over its own labelled rows
   (time order, batch row breaks ties): the first `floor(frac * n)` rows go to phase A, the rest to
   phase Q. A group with fewer than two labelled rows cannot be split and lands whole in phase A,
   where the merge gives it weight 1 and its Q term weight 0 -- exactly its full-batch gradient.
   `training_step` now runs only the (phase, group) passes that have rows and checks that each group's
   phase counts partition its whole-batch count.

Evidence: `scratchpad/ta3/test_split.py` (400 random layouts x 3 fractions plus degenerate and
disjoint-block shapes), and `scratchpad/ta3/scan_batch_layouts.py`, which read the stored
`global_labels` of EVERY training chunk and reproduces the batch layout exactly (with
`RWKV_PROBE_DENSITY=0`, `RWKV_R2_HEAD` unset and the clock fix off, nothing else modifies the label
mask). Result over all 5,000 training users (46,062 chunks, 1,722 s, log
`scratchpad/ta3/layout_scan_lean.log`):

* OLD rule: **55 chunks** (plus 4 in immediate mode) had a loss group with no row on one side of the
  single cut -- 0.13%. The examples are short tail chunks (T = 61..1,841) where all ahead rows sit in
  the first half and all imm rows in the second, e.g. user 121 chunk 8193-8223, T=61, ahead 2 rows
  (both before the cut), imm 30 rows. Any group containing such a chunk could reach a phase with no
  ahead rows, which is the crash.
* Chunks where a group has fewer than 2 labelled rows (the degenerate case that now lands whole in
  phase A): 37 ahead + 6 imm = 43 chunks (0.09%). Chunks with a group entirely absent: 30 ahead +
  4 imm.
* NEW rule: **0 failures**, 0 broken partitions, 0 single rows misplaced.

A 4-worker pool version of the scan hung on 2026-11-16 (workers busy, no results returned for 20 min);
the lean single-process version is the one that finished.

Hardening: `scratchpad/ta3/ta3_train.py` now retries a crashed epoch from scratch (`--retries`, default
2, retry logs `train_retryN.log`) and records the failures in `ta3_state.json`, so a crash no longer
idles the GPU until a human notices.

Also note: the direction `u` is now the unweighted sum of the two A-phase group gradients (the exact
gradient of the A-phase objective, which is what the probe measured). At eta=0 the merge is unchanged,
so the gate result still holds.

## Monitoring and hazards

Check the chain with `.venv/Scripts/python.exe scratchpad/ta3/logcheck.py scratchpad/ta3_fullrun_20261116/epoch_0N/train.log`
(step counts, TA3-vs-plain step lines, tracebacks/skips/aborts, validations, problem markers) and the
controller log `scratchpad/ta3/chain.log`. The final paired gate is
`.venv/Scripts/python.exe scratchpad/ta3/compare_screens.py --ta3 <ta3 tag> --plain adaptive_fullrun_92_20261105_90fecbaa_a04`.
The plain arm's per-epoch screens live in `result/adw_adaptive_fullrun_92_20261105_90fecbaa_a0N.jsonl`
(ahead) and `result/adwP_..._a0N.jsonl` (immediate); epoch 4 is the incumbent.

Corrected 0.04-epoch smoke (2026-11-15 16:53-17:34, `scratchpad/ta3_smoke_fixed_20261115/`): 1,734
TA3 steps, 0 tracebacks, 0 skipped batches, 0 aborts, 0 non-finite steps, 5 internal validations
(0.3650/0.3386 -> 0.3385/0.3137), model+optimizer pairs at 50/500/1000/1500/1734, 0.729 steps/s
(~44 steps/min). Epoch 1 of the chain started 17:36 on 2026-11-15 and reproduced the smoke's step-50
validation exactly.

Smoke #2 after the split fix (2026-11-16 07:48-08:34, `scratchpad/ta3_smoke_fix2_20261116/`, prefix
`ta3smoke2`): 1,734 TA3 steps, 0 tracebacks, 0 skipped batches, 0 aborts, 0 non-finite, 5 internal
validations (0.3668/0.3406 -> 0.3378/0.3139), final pair `ta3smoke2_1734.pth` + optim, and it exercised
multi-chunk groups (a 3-chunk 4,279-row group appears in its log). The chain restarted at 08:37 on
2026-11-16 in `scratchpad/ta3_fullrun_20261116/`.

* Main progress signal is the trainer's own 2-user internal validation line
  (`Mean ahead validation loss: ... imm: ...`) — it uses a separate batch and is not affected by TA3.
* The per-step `[ta3]` line prints the **A-phase ahead-masked pass** statistics, so its `imm` field
  is `0.000` by construction and its `all` field is the A-half ahead loss. This is a logging artifact
  of the mode-masked merge, not a training problem. `key_value_stats` inherits the same bias, which is
  why the chain relies on validation, not on the printed training loss. Train-loss pruning is inert
  unless `RWKV_PRUNE_REF`/`RWKV_VPRUNE_REF` are set, and they are not.
* Resume: `scratchpad/ta3/ta3_train.py` skips an epoch whose checkpoint pair exists and whose log
  passes the checks. It never overwrites or deletes a checkpoint. Re-run it after a crash; the model
  and optimizer files carry the global step in their names.
* `eval_sharded` refuses to overwrite a previous result, so every screen uses a fresh unique tag.
* Do not run any other GPU workload beside the chain. Keep >= 10 GiB host RAM free; the RAM guard
  caps the single fetch worker.
* Keep TA3 training-only and full precision for this first experiment. Do not silently add QAT:
  a deploy claim would need the same verified QAT treatment on both arms.

## Files

* `rwkv/ta3.py` — the implementation (scope, split, probe, check, training_step).
* `rwkv/train_rwkv.py` — the hook (search for `_ta3`).
* `scratchpad/ta3/ta3_train.py` — the 4-epoch chain controller (resumable, `--dry-run` pre-flight).
* `scratchpad/ta3/run_ta3_chain.cmd` — launcher; chain log `scratchpad/ta3/chain.log`.
* `scratchpad/ta3/logcheck.py` — read-only log health check; `scratchpad/ta3/compare_screens.py` —
  paired per-user screen comparison with the project gate.
* `scratchpad/ta3/test_split.py` — unit test of the A/Q split invariants (synthetic label layouts).
* `scratchpad/ta3/scan_batch_layouts.py` + `run_scan_layouts.cmd` — full read-only scan of the training
  LMDB label layouts: how many groups the old split would break, how many the new one does (log
  `scratchpad/ta3/layout_scan.log`).
* `scratchpad/ta3/mk_ta3_smoke_fix2.py` + `run_ta3_smoke_fix2.cmd` — smoke #2 after the fix.
* `scratchpad/detach.ps1` — recreated detached hidden-window launcher (used for the chain).
* `scratchpad/adaptive_wd/mk_ta3_smoke_fixed.py` + `run_ta3_smoke_fixed.cmd` — the required
  0.04-epoch smoke from the real start config (`scratchpad/ta3_smoke_fixed_20261115/`).
* `scratchpad/adaptive_wd/mk_ta3_probe.py`, `mk_ta3_gradcheck.py` — probe and gate builders.
* `scratchpad/ta3_probe_analyze.py`, `scratchpad/ta3_record.py` — probe analysis and record helper.
* Run outputs: `scratchpad/ta3_fullrun_20261115/` (`ta3_state.json`, `epoch_0N/`), screens in
  `result/adw_<tag>.jsonl`.
