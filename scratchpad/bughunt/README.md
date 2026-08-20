# The bug hunt — tooling and the ranked matrix

Andrew, 2026-08-18:

> once GPU is free, do some bug hunting. Obviously not every bug will be caught just by staring at
> the code, so temporarily reduce the number of epochs to 0.05 and the number of eval users to 20,
> and use that for diagnostics. Once you are reasonably confident that there are no bugs left,
> resume the autoresearch loop normally.

**This comes before the next research iteration**, not after it.

## Why execution and not review

The bug *rate* is flat — 21% of commits in both halves of the last 24 days, ratio 1.01× (audit in
`optimization/HISTORY.md`). The recent spike was **exposure**: orchestration edits ran 3.5× baseline
after the power outage while model-code edits were zero. The root cause is **cloning runners across
lineages**, so each new run inherits its nearest ancestor's defects.

Every failure of 2026-08-17/18 would have been caught by **one cheap end-to-end run**, and *none* of
them was visible by reading: the `endlocal` marker, the omitted `LOAD_MODEL_FOLDER`, the two
alpha/guard mismatches, the smoke that inherited its own control. That is the argument for the
diagnostic config.

## What is already done — do NOT redo it

| done | result |
|---|---|
| smoke audit for the false-green class (a control arm inheriting the treatment from the ambient env) | 12 of 49 inherit `os.environ`; only `rgate` was a live risk; fixed and **re-verified by execution** with the contaminating var deliberately set |
| `preflight_runner.py` asserts the `endlocal` ordering | all four live runners pass |
| guard/value desync + artifact-gate checks added to preflight | found and fixed two live gaps (below) |
| full structural sweep of all 223 runners (`sweep_runners.py`) | **one** defect class, 27 runners, one lineage |
| iter 53 verdict, iter 55 verdict, numbering moved to completion order | recorded |

### The sweep result, which is the case for a template

`D1` (a bare `endlocal` before the terminal `DONE_EXIT_` echo, so `%LOG%` expands to empty and the
marker is written to `""`) hits **27 of 223 runners**, in one unbroken lineage from `eval_pava` and
`iter32_kd` through **iter 53**, `run_iter45.cmd` among them. `iter52/54/55/57` are absent — iter 52
came from a different generator.

`D2` (`%VAR%` used before set), `D3` (no `cd /d`) and `D4` (guard/value desync) are **zero** once two
false-positive classes are removed: a `%VAR%` mentioned in a `REM` line is prose, and a runner
invoked via `call` legitimately inherits its caller's environment and working directory. Both were
in the first pass and would have reported six defects that do not exist.

All 27 are finished runs, so **none needs repair** — the exposure is to future clones.

## Findings so far (CPU-only, GPU still booked)

### 1. Two live runner defects — fixed

* **`run_iter55.cmd` (rgate, queued) verified neither training phase's output.** Added two gates
  naming the expected **final step**. An existence test would not have sufficed: the outage left a
  step-8000 checkpoint, so "a `.pth` exists" was true while the model was half-trained.
* **`kdalpha025` announced `alpha 0.9`** for a run whose alpha is `0.25`. Env and guards were
  correct, so the run was never at risk — the **record** was. Fixed in the generator; the diff is
  three prose lines with no executed logic touched.

Both were patched while their waiters were still looping, with byte-exact backups. A `call`ed `.cmd`
is not open until the call.

### 2. rgate's smoke failure was stale — verified by execution

Its `DONE_EXIT_46` log is timestamped 22:25:45; the hermetic fix landed at **22:31:21**, six minutes
later. Re-run **with `RWKV_RGATE=card` deliberately set in the ambient environment** — the exact
contamination that caused the original false green — it now reports `RGATE_ALL_PASS`: OFF 0 keys /
ON 8 keys, delta **+324 params**, inertness exactly `0.000e+00`, gain=0.8 moves. That queue slot is
sound.

### 3. Three-way parity is green on current code

`parity_train_vs_rnn.py` → `PARITY_ALL_PASS`, including cases for both in-flight levers
(`RWKV_CMIX_POW`, `RWKV_RGATE`).

### 4. The state clamp does **not** bind — a three-way-parity question closed

The clamp is implemented in all three paths but with deliberately different granularity: training
clamps **per 32768-step window**, the RNN and Rust deploy paths clamp **every step**. They agree
exactly wherever `||S||_F < tau`, because the factor is then exactly `1.0` and the multiply is
bit-inert — so the entire question is whether the norm ever reaches tau.

Nothing had ever measured it. `parity_train_vs_rnn.py` deliberately **skips** the parity assertion
for a binding tau (the training clamp is CUDA-only, so a CPU run of the training path does not clamp
at all), and no run has ever switched on `RWKV_STATE_CLAMP_LOG`.

Measured on the champion via the deploy RNN path (`state_norm_probe.py`, CPU, wraps `clamp_state` in
its own process so no repo file is touched):

| user | clamp calls | max ‖S‖_F | tau | binding steps | headroom |
|---|---|---|---|---|---|
| 5001 | 30,000 | 9.61 | 300 | **0** | 31× |
| 5002 (giant) | 200,000 | **13.06** | 300 | **0** | 23× |

**Zero binding steps on both** — so training and deploy compute the same quantity, and the documented
divergence is latent rather than live. The clamp is doing its intended job: a safety net for the
divergent A0/A3 configs it was added for, inert on a healthy model.

**The shape of the two rows matters more than either number.** The giant user runs 6.6× more clamp
calls and reaches only 1.36× the norm, so `||S||` **saturates** rather than accumulating with
recurrence length. That is what a contracting recurrence must do: the decay sits at ~0.98 < 1, and
the delta term only ever *removes* state, so both terms contract. So the conclusion does not rest on
two samples — it rests on the mechanism, with two samples confirming it. A user long enough to reach
300 would have to break that contraction first.

⚠ An earlier version of this paragraph cited the 2026-08-17 gloss that the trunk is a "near-pure
exponential-decay accumulator, delta term worth only ~0.15". **That gloss is refuted** — see the
delta-rule section below. The saturation argument is unaffected, because it needs only that both
terms contract, which is true independently of how much *functional* work the delta term does.

## Tools here

| file | what |
|---|---|
| `mk_diag.py` | generates a diagnostic runner + WS toml for a given lever. `mk_diag.py <tag> [KEY=VAL ...]` |
| `sweep_runners.py` | scans every `.cmd` in the repo for the four structural defects, reported by defect so lineages are visible |
| `patch_preflight.py` | added the guard/value-desync and artifact-gate checks to `preflight_runner.py` |
| `patch_iter55_gates.py` | added the two missing step-verification gates to the queued rgate runner |

### Two scalings in `mk_diag.py` that are not optional

`EPOCHS 0.05 × 10,935 groups ≈ 546 steps`, about 10 min per phase. Alone that would be fast and
**hollow**:

* `WARMUP_STEPS 400 → 20` — 400 of 546 steps is 73% warmup, so the LR would never plateau and the
  decay phase would start from a model that never trained at the production LR.
* `VALIDATE_EVERY 1000 → 200` — `1000 > 546` means validation **never fires**, so that path would go
  entirely unexercised while the run looked green. Same shape as the bugs being hunted: a phase that
  silently does not happen.

Deliberately **not** reduced: db, user range, `MAX_TRAIN_GLOBAL_LEN`, fetch processes. KD replays its
dump by step index and hard-exits 43 on a per-step `labels_sum` mismatch, so shrinking the data would
break the KD path — one of the paths most worth exercising. Fewer users would also change
`num_groups`, and with it the step count, silently.

**A diagnostic run's logloss is meaningless** and must never be recorded as a result.

## The ranked matrix — run in this order when the GPU frees

1. **`base`** — champion path, nothing switched on. Proves the harness and confirms `DIAG_STEPS=546`
   by execution. Everything below is uninterpretable until this is green.
2. **`resume`** — the mid-epoch resume (`RWKV_RESUME_SKIP_GROUPS=1` + `make_resume.py`). Highest
   value after `base`: it broke during the outage, **and yesterday's fix has never been executed
   end-to-end**. Kill the WS phase mid-run, resume, confirm `[resume-skip]` and that the final step
   is reached.
3. **`idfeat`** (`RWKV_ID_FEATURES=1`) — the features code has **never run on a GPU**. On the current
   92-dim LMDB it *should* fail; the question being tested is whether it fails **loudly**. A silent
   width mismatch is exactly the bug class. On the critical path for the next phase.
4. **`qat`** — the quant-aware env, which was silently **inert** for every track-2 run until
   `70185c7`. `assert_qat_live.py` covers the config; only a real run covers the kernels.
5. **`decktree`** (`RWKV_DECK_TREE=2`) and **`rgate`** (`RWKV_RGATE=card`) — both CPU-smoked already,
   so these are confirmations rather than discoveries.

## Then, and only then

Build **one canonical runner template** with the guards baked in, so the next run is generated rather
than cloned. The design principle, which is the whole lesson of 2026-08-18:

> **Derive the guard from the value. Do not check the guard against the value.**

Three failures that day were guard/value desyncs. An assert that the two agree catches the third
instance; generating both from one variable makes the first impossible.

`kdalpha025` is the first run named for its lever with **no number in its path** — the number is
assigned at verdict time, per the completion-order convention.

## Delta-rule simplification (Andrew, 2026-08-19) — screened and DEAD, ~25 min of CPU

**The proposal:** simplify the delta rule to cut parameters "for free", as the earlier reduction
ladder did. Two independent reasons it does not work, both measured before spending a 9.2 h run.

### 1. The prize is the wrong *kind*, and it is not free

`a_lora` (9,360) + `k_scale` (5,265) = 14,625 params = 2.62%. But these are **weights, not state.**
The binding deploy budget is per-card **state** (9 B/card, frozen); weights ship once. The old ladder
mattered because 2.76M params had to become shippable at all — that constraint is retired.

Nor can a bias replace them (`delta_rule_screen.py`, 26 sites, per-module):

| target | variance a bias CANNOT reproduce | prize |
|---|---|---|
| `a_lora` | **36.4%** | 1.68% |
| `k_scale` | **20.0%** | 0.94% |
| rank 4→3 | needs mean **3.1 of 4** components for 95% | 0.37% |

### 2. The stronger version — ablate the delta term entirely — is decisively dead

Zeroing `a` on the champion deletes the delta term and nothing else (`delta_ablate_screen.py`,
4 smallest VAL users, paired within user, project's own equalized LogLoss via `get_stats`):

| user | baseline imm | a=0 imm | Δ imm |
|---|---|---|---|
| 5044 | 0.20588 | 0.40362 | +0.19775 |
| 5100 | 0.38223 | 0.53289 | +0.15065 |
| 5063 | 0.17232 | 0.41081 | +0.23850 |
| 5097 | 0.12592 | 0.37100 | +0.24508 |
| **mean** | | | **+0.20799 imm / +0.06029 ahead** |

Accept bar 0.0001; the entire A0→A18 ladder (4.95× smaller) cost +0.00053 imm. This is **~390× that
whole ladder**, and the *smallest* per-user cost is ~1500× the bar. Inference-time ablation is an
upper bound (weights co-adapted), but nothing at this scale retrains into a free simplification.

### The reusable finding: magnitude ≠ selectivity

This refutes the gloss the record carried since 2026-08-17 — "RWKV-7's headline innovation is barely
engaged" — while leaving its measurement intact. The delta term really does move the state-transition
eigenvalue only ~0.15 against a decay of ~0.98. But it is not tuning a decay rate: it performs the
**key-selective removal** that makes the state an associative memory, rank-1 and aimed exactly at the
direction being overwritten. 0.15 of movement spent precisely where needed does work no amount of
uniform decay reproduces.

> **A scalar summary of a mechanism's magnitude says nothing about its selectivity.** Same family as
> iter 51's median-vs-max error: the wrong summary statistic, confidently applied.

### Two method notes

* **My first screen was wrong and arithmetic caught it.** Hooks keyed by `layer_id` merged the
  layer-0 mixers of all five streams into one bucket. The tell: four variance components summing to
  0.819 instead of 1.0, impossible for a purely linear `B(A(x))`. Contaminated 65.4% → corrected
  28.2%, which moved the verdict from "don't bother" to "marginal" — so the bug mattered.
* **Do not pipe a background job through `tail`.** The output file *is* the record; the first
  ablation run kept only its last 22 lines and lost three of four users' rows.

---

# THE HUNT, EXECUTED (2026-08-20) — results

The GPU freed at 05:18 when rgate finished, and the chained waiter started the matrix
automatically at 05:20:28. No idle GPU.

## 1. `base` — PASS, 24m37s

`DONE_EXIT_0`. WS 12 min, decay 11 min, 20-user eval 1m17s — matching the ~25 min the harness was
designed for. All gates fired: artifact-at-expected-step, validation-actually-ran, derived alpha
guard.

**`DIAG_STEPS=546` is now confirmed by execution**, not assumed — `base_ws_546.pth` and
`base_d_546.pth` both exist, which is what the gate requires.

## 2. `resume` — PASS. The highest-value target, and the fix had never been run

The outage broke this path and yesterday's `make_resume.py` fix had only ever been *read*. Staged a
simulated crash (base's step-50/200/400 checkpoints in a fresh dir, final checkpoint absent), then
ran the real recovery:

```
appended keys absent from the source toml: LOAD_MODEL_FOLDER, LOAD_MODEL_NAME
[resume-skip] epoch 0: skipping the first 400 already-trained groups (resume at global step 401)
exit=0     reached step 545/546     res_ws_546.pth written
```

The first line **names the bug it was fixed for**: the old replace-only code silently never wrote
those two keys, which made `train_rwkv` die on an `AttributeError`, swallow it, exit 0, and let the
runner log "WS OK" after 8 seconds.

Writing a correct toml was never the question. Firing the skip, reaching the final step, and not
swallowing an error are — and only execution shows those.

## 3. `idfeat` — PASS, and it changed from a negative probe into a real one

Planned as "does a 92-dim mismatch fail *loudly*". The rebuild finishing at 01:27 turned it into a
genuine end-to-end test of the new-features path: **exit 0, trained to step 546 on the rebuilt DBs,
564,612 params** = 558,212 + 20 new input dims × the 320-wide input FC. Exactly the expected arithmetic.

### ⚠ TWO REAL TRAPS IT SURFACED, both aimed at the champion re-base

1. **THE KD DUMP IS INVALID ON THE REBUILT DBs.** `size` moves for ~30% of users from the dataset
   swap alone, so the batch composition its per-step `labels_sum` checksum guards no longer matches
   — it would hard-exit 43, or worse, distil against mismatched batches. The re-base must either
   regenerate the dump on the new DBs or drop KD. **That is not a small call: iters 32/35/39/45
   banked ~0.0019 from KD.** Found in 25 minutes instead of 9 hours into a re-base.
2. **`write_decay_setup.py` and `write_eval_toml.py` hardcode the OLD db paths** (`train_db_5k_h1`,
   `F:/rwkv_lmdb/test_db_5k`, `label_filter_db`). Any runner built for the new DBs gets a WS phase on
   112-dim data and decay/eval phases silently pointed at 92-dim data. Fix the generators before the
   re-base, not after.

## Still open

`qat` — the quant-aware path, which was silently **inert** for every track-2 run until `70185c7`.
`assert_qat_live.py` covers the config; only a real run covers the kernels.
