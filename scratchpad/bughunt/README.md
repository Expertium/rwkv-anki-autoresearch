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
