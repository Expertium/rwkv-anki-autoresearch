# rwkv-anki-autoresearch — Claude handover

> **GitHub rule (always):** every GitHub comment posted on Andrew's behalf — PR
> descriptions, review replies, issue comments, anything — **must start with the line
> "Written by Claude".** No exceptions. (Also in the global `~/.claude/CLAUDE.md`.)

## 0. Who you are / read this first

You own **`C:\Users\Andrew\rwkv-anki-autoresearch`** (GitHub:
`Expertium/rwkv-anki-autoresearch`). The repo starts **empty** — your first real job is
to vendor the existing RWKV code into it (see §2 and Roadmap step 1).

- **Separate Claude instances own the sibling repos** `C:\Users\Andrew\srs-benchmark`
  (the upstream-clone benchmark + the original RWKV code) and
  `C:\Users\Andrew\fsrs-autoresearch` (the FSRS-7 GPU autoresearch). **Do not edit those
  repos** — they are your *read-only source*. One Claude per repo.
- **The user is Andrew** — hobbyist data scientist. He knows **Python/PyTorch and little
  else**, so spell out non-Python tooling (Rust, `candle`, CUDA toolchains, git internals,
  build systems) rather than assuming it. **He did NOT write this neural net** and can't
  answer deep questions about its internals — *the source code in `srs-benchmark/rwkv` is
  the ground truth; read it and be self-reliant.* When a design choice is yours to make,
  explain the trade-off and recommend, don't quiz him.
- Andrew sometimes hand-edits files; if you see an uncommitted change you didn't make,
  it's intentional — don't revert it, commit it if needed.

## 1. The end goal

A small **RWKV-7 neural network for Anki spaced-repetition scheduling** that:
- runs **CPU-only, inference-only**, with **frozen weights** (no per-user training — the
  net generalizes across users from one frozen checkpoint), and
- is **small, fast, and quantized** enough to **ship inside Anki**.

The path there is the roadmap in §8: reproduce → shrink the data loop → port to Rust →
speed up → reduce params → CPU inference → quantize.

## 2. What the model is, and where it currently lives

**"RWKV"** is the current top entry on the
[open-spaced-repetition/srs-benchmark](https://github.com/open-spaced-repetition/srs-benchmark)
leaderboard — a multi-stream RWKV-7 (RWKV is a *linear-attention RNN* LLM family; the "7"
is the architecture generation) that predicts recall probability from a user's full review
history. It is materially more accurate than FSRS-7 (numbers in §5).

**Architecture** (`rwkv/architecture.py`, `rwkv/config.py`, `rwkv/model/srs_model.py`):
- **Five parallel RWKV-7 stacks**, one per ID granularity: `card_id`, `deck_id`,
  `note_id`, `preset_id`, `user_id`. Each stack is `d_model = 128` (32 × 4 heads),
  **2–4 layers**, channel-mixer factor 1.5–2.0, with LoRA-style low-rank projections for
  the decay / `a` / `v0`-mix / gate terms. The streams let the net pool evidence at
  different scopes (this exact card vs. the whole deck/note/preset/user).
- **SRS heads on top** (`srs_model.py`): a **forgetting-curve** head = a softmax **mixture
  over 128 basis curves** (`w_head`) sampled at **128 points** (`ahead_head`), plus a
  **4-way per-rating head** (`p_head`, Again/Hard/Good/Easy). So the model outputs a
  flexible learned forgetting curve, not a fixed FSRS-style formula.
- **~2.76 M parameters total** (measured from the pretrained `.pth`). This is the baseline
  for the param-reduction work (step 6).

**Two benchmark scoring modes** (`rwkv/get_result.py`, names `FILE_AHEAD` / `FILE_IMM`):
- **`RWKV` ("ahead")** — predict the next review cold from history.
- **`RWKV-P` ("imm")** — the immediate-prediction variant; **the stronger one.**
  Read `get_result.py` + `srs_model.py::forward` for the *exact* definitions — you must
  understand this distinction to match the baseline and to port correctly.

**Source to vendor** (read-only, in `C:\Users\Andrew\srs-benchmark`):
- `rwkv/` — the whole subtree (model, CUDA/C++ kernel `rwkv/model/csrc/`, training,
  eval, data pipeline). See the file index in §10.
- **Cross-repo imports it depends on** (must come along or be re-implemented): `features/`
  (`create_features`), `utils.get_bin`, `config.py` (`Config`, `create_parser`), and
  `setup.py::get_rwkv_extensions` (builds the kernel — a **`CUDAExtension`** when CUDA is
  present, else a **CPU `CppExtension`** fallback). After vendoring, the package must
  `import` and run standalone — verify before anything else.
- **Pretrained weights:** `srs-benchmark/pretrain/RWKV_trained_on_101_4999.pth` and
  `RWKV_trained_on_5000_10000.pth` (2.76 M params each).

## 3. Data + preprocessing (HEAVY — plan storage first)

- **Dataset:** `C:\Users\Andrew\anki-revlogs-10k` — 10 000 users, ~745 M reviews
  (sibling, **read-only — never write there**). `anki-revlogs-3k` is the first 3 000 users.
  `user_order.jsonl` ranks user ids by size.
- The pipeline writes **LMDB** databases (`train_db`, `test_db`, `label_filter_db`):
  1. `python -m rwkv.find_equalize_test_reviews` — builds a ~7 GB helper db that precomputes
     RMSE(bins) bins and which reviews count in the benchmark (the "equalized" test set,
     aligned with the `--short --secs` feature settings).
  2. `python -m rwkv.data_processing --config rwkv/data_processing_config_{train,test}.toml`
     — the train + test LMDBs.
- **⚠ The full 10k preprocess needs ~400 GB of disk.** This is the binding constraint on
  Andrew's machine (§7). Working on the **2k subset (step 2) is the cheap iteration loop**
  for everything downstream — do that early.

## 4. Training

`rwkv/train_rwkv.py` + `rwkv/train_rwkv_config.toml`:
- **WSD LR scheduler** (Warmup–Stable–Decay): ~10 epochs warmup+stable, then ~2 epochs
  decay; switch phases via the config (`TRAIN_MODE = "WS"` / `"D"`).
- **bfloat16, CUDA**, peak LR `7e-4`, 20 000 warmup steps, `MAX_TRAIN_GLOBAL_LEN = 66000`.
  Requires the compiled CUDA kernel. CPU training is supported (`DEVICE = "cpu"`) but
  "dramatically slower" — not practical for full runs.
- The provided weights were trained on users **5000–10000** and **101–4999**, with
  **1–100 held out as validation**.

## 5. Evaluation + the baseline to reproduce

- `rwkv/get_result.py` (CUDA, fast) scores a trained model on a held-out user range.
  **Cross-validation:** model trained on 5000–10000 → evaluate users 1–4999; model trained
  on 101–4999 → evaluate 5000–10000; pool both → 10 000 users.
- **Metric = by-user mean `LogLoss`** (each user weighted equally), plus `RMSE(bins)` —
  same definitions as srs-benchmark.
- **★ BASELINE NUMBERS TO MATCH** (from `srs-benchmark/result_upstream/`, 10 000 users):

  | Variant | mean LogLoss | mean RMSE(bins) |
  |---|---|---|
  | `RWKV` (ahead) | **0.29743** | 0.05438 |
  | `RWKV-P` (imm) | **0.26600** | 0.03212 |

  (For scale: FSRS-7 is ~0.32 by-user on the 3k subset — the RWKV nets are clearly better.)
- **"Reproduce" = match the mean LogLoss within a small tolerance**, not bit-for-bit.
  Cross-version SGD + bf16 nondeterminism makes exact reproduction unrealistic; aim for
  ~matching aggregate LogLoss/RMSE(bins). (This is unlike the FSRS bit-exact speedup
  project — here parity is statistical.)
- **Plan of attack (confirmed with Andrew):** first **reproduce the table from the
  provided `.pth` weights** via `get_result.py` — no training. Once it matches, **move all
  further work to the 2k loop** (roadmap step 2: train ids 1–1000 → eval 1001–2000, then
  swap for full 2 000-user coverage) and don't routinely touch the 400 GB 10k pipeline again.

**Acceptance tolerances** — apply to **both** scoring modes (`ahead` *and* `P`)
independently; a change passes only if **both** pass:
- **Parity (Rust port, roadmap step 3):** each mode's mean LogLoss must be **within ±0.0005**
  of the Python reference. This is the gate for "the Rust port is correct."
- **Efficiency-regression budget (roadmap steps 4, 5, 7 — speedups, param reduction,
  quantization):** each mode's mean LogLoss may **rise by at most +0.0015** relative to its
  **parity-verified Rust baseline**. Within budget → keep; over → reject. (A pure speedup
  should cost ≈0; the +0.0015 is headroom for param-cutting and quantization to spend.)

## 6. CPU inference — already half-built (central to the goal)

RWKV-7 has an **exact RNN (sequential/recurrent) formulation** mathematically equivalent
to the parallel CUDA training kernel. It's already implemented:
- `rwkv/model/rwkv_rnn_model.py` (`RWKV7RNN`) + `rwkv/model/srs_model_rnn.py` +
  `rwkv/run_as_rnn.py` already run **a single user on CPU** from the saved weights
  (`run_as_rnn_config.toml`: `DEVICE = "cpu"`, `DTYPE = "float"`).
- This RNN-mode path is the **starting point for steps 6–8** (Rust port, CPU inference,
  quantization). Inference one-token-at-a-time needs no custom CUDA — pure tensor ops,
  ideal for `candle`/Anki. (The `CppExtension` CPU kernel build is a fallback if you need
  the chunked form on CPU, but RNN-mode is likely enough for inference.)

## 7. Host machine + build caveats

- **Andrew's PC:** RTX 4070 (**12 GB VRAM** — less than the 24 GB 3090 this code was
  developed on; the model is tiny so bf16 training should fit, but you may need to lower
  `MAX_TRAIN_GLOBAL_LEN`), Ryzen 9 5950X (16c/32t), 64 GB RAM, 1 TB M.2 SSD + 4 TB external
  USB HDD. **The ~400 GB preprocessed dataset is the storage constraint** — put the LMDBs
  where there's room (M.2 if it fits; otherwise the 4 TB USB, which is slower I/O).
- **⚠ CUDA build caveat (from the srs-benchmark handover):** on this machine the RWKV CUDA
  extension has failed to build — **system CUDA toolkit (13.2) vs the cu126 torch wheel**
  mismatch (and MSVC/`vcvars` must be on PATH on Windows). Expect to fix the toolchain
  (install a matching CUDA toolkit, or a torch build matching the system CUDA) before GPU
  training/eval works. The **CPU `CppExtension` / RNN-mode path sidesteps this** for
  inference work.
- **Native Python** here (no Docker, unlike fsrs-autoresearch). Later, **Rust** (step 3).
  Run from PowerShell. Use a venv; install torch matching your CUDA situation.

## 8. The roadmap (Andrew's plan)

1. **Reproduce existing results on 10k.** Train RWKV on the first 5 000 users (ids 1–5000),
   evaluate on the second 5 000 (5001–10000); then **swap** train/test and repeat. Match
   the §5 baseline. **Start by reproducing from the *provided* weights** (no training);
   a fresh exact-split train is optional after that.
2. **Move to a 2k loop** — train ids **1–1000 → evaluate 1001–2000**, then **swap**
   (train 1001–2000 → eval 1–1000) for full 2 000-user coverage. This is your **fast
   iteration workbench** for everything below; build it right after step 1 and don't
   routinely touch the 400 GB 10k pipeline again.
3. **Implement RWKV in Rust** (likely [`candle`](https://github.com/huggingface/candle),
   HF's minimalist Rust tensor/ML library). Port the **RNN-mode recurrence** (§6) — no
   custom CUDA needed. **Verify parity** with the Python implementation: both modes'
   LogLoss within **±0.0005** of Python (the §5 parity gate). This Rust engine is what
   ultimately runs inside Anki.
4. **Speed it up WITHOUT changing architecture/training** — pure-performance wins (op
   fusion, killing redundant recompute, better batching/memory layout, cutting allocation
   churn). A pure speedup should keep both LogLosses ≈unchanged; stay within the **+0.0015**
   regression budget (§5) vs the Rust baseline.
5. **Reduce the parameter count** while keeping LogLoss within the **+0.0015** budget (and
   ideally **improving** it) — via hyperparameter tuning, architecture search, pruning, or
   distillation. Baseline = 2.76 M params @ 0.266 (RWKV-P). The dream is an algorithmic
   change that *lowers* LogLoss while shrinking. The most "research-y" step — measure every
   change on the 2k loop, keep the wins.
6. **CPU-only, inference-only** (training stays on GPU). The RNN-mode path (§6) is the
   start. End state: usable inside Anki with **frozen weights**.
7. **Quantize.** Read the two papers and pick an approach (adapt — both target 14B-scale
   RWKV; ours is 2.76 M), keeping both LogLosses within the **+0.0015** budget:
   - **RWKV-edge** — *Deeply Compressed RWKV for Resource-Constrained Devices*
     ([arXiv 2412.10856](https://arxiv.org/abs/2412.10856)): a compression **suite**
     (architecture optimizations + post-training compression), **3.4–5× memory reduction**,
     edge-device focus.
   - **RWKVQuant** — *Quantizing the RWKV Family with Proxy-Guided Hybrid of Scalar and
     Vector Quantization* ([arXiv 2505.03803](https://arxiv.org/abs/2505.03803)): PTQ built
     for RWKV's quirks (non-linear ops that block fusion; near-uniform weights that hurt
     clustering) — a **proxy-guided hybrid of scalar + vector quantization** with codebook
     optimization, **~3-bit, <1% accuracy loss, 2.14× speedup**.
   Quantized weights pay off in the Rust/candle CPU path for Anki.

Steps 4, 5, and 7 are naturally **iterative** (propose a change → measure LogLoss + speed +
size on the 2k loop → keep it only if it passes the §5 tolerances) — i.e. a lightweight
autoresearch loop, hence the repo name. Keep an append-only log of what you tried and the
deltas so dead ends aren't re-run.

## 9. Working norms

- **Be self-reliant on RWKV internals** — Andrew didn't write the net. The source in
  `srs-benchmark/rwkv` is ground truth; verify facts against it, not memory.
- **Parity discipline:** verify against the reference on a **small fixed verification
  user-set** defined early (mirror srs-benchmark's `test_users.json`: a few small + a few
  large + some random, seeded) so checks are fast and comparable. The numeric gates live in
  §5: **±0.0005** for Rust-port parity (step 3), **+0.0015** regression budget for
  efficiency changes (steps 4/5/7) — and **both** scoring modes (`ahead` and `P`) must pass.
- **Git:** commit/push only when asked; for non-trivial pushes branch off `main`; end commit
  messages with the `Co-Authored-By` trailer. GitHub comments start "Written by Claude".
- When a step is ambiguous (exact split, quant target, candle vs other Rust ML lib), state
  the trade-off and your recommendation rather than guessing silently.

## 10. Key files (in `srs-benchmark/rwkv`, to vendor)

| Path | What |
|---|---|
| `architecture.py` | the 5-stream RWKV-7 config (d_model, layers, LoRA dims per ID module) |
| `config.py` | ID-encoding dims, time-feature periods, `RWKV_SUBMODULES` |
| `model/srs_model.py` | the SRS model (training mode): feature FC + the 5 RWKV stacks + curve/rating heads |
| `model/srs_model_rnn.py` | the SRS model in **RNN (sequential) mode** — CPU inference |
| `model/rwkv_model.py` | core `RWKV7` (parallel/CUDA training form) |
| `model/rwkv_rnn_model.py` | core `RWKV7RNN` (recurrent form) |
| `model/rwkv_ops.py` | kernel wrapper + a pure-PyTorch `reference_rwkv7` |
| `model/csrc/**` | the CUDA/C++ kernel (`rwkv7_cuda.cu`, `parallel_scan.cu`, `rwkv7.cpp`) |
| `train_rwkv.py` / `train_rwkv_config.toml` | training entry + config (WSD scheduler) |
| `get_result.py` / `get_result_config.toml` | evaluation (CUDA) — produces the `RWKV` / `RWKV-P` jsonls |
| `run_as_rnn.py` / `run_as_rnn_config.toml` | **single-user CPU inference** (RNN mode) |
| `data_processing.py`, `prepare_batch.py`, `data_fetcher.py` | dataset → LMDB → batches |
| `find_equalize_test_reviews.py` | builds the helper db (test-review alignment + RMSE bins) |
| `parse_toml.py`, `utils.py` | config + small helpers |
| *(parent)* `features/`, `utils.get_bin`, `config.py`, `setup.py` | shared deps to vendor |

## 11. Optimization loop (steps 4–5–7) — THE PROTOCOL (canonical; mirror in `optimization/PROTOCOL.md`)

> **⚠ SUPERSEDED GATE:** the work is now in the **research phase** — the live acceptance gate
> (both modes improve ≥0.0003 vs the current champion, params ≤225k, card/note state fixed) is in
> the **"Optimization state"** section below, NOT the iter0 +0.0015 gate described here. The rest of
> this section (logging discipline, the Wilcoxon speed protocol, Rust-parity invariant, the
> training-resume mechanism) is still current. Keep it for those; use the research gate for accept/reject.

Steps 4 (speed), 5 (param reduction), 7 (quantize) run as ONE iterative autoresearch loop.
Follow this exactly — Andrew has flagged sloppiness, so do every step every iteration.

**Scope / allowed changes:** both **exact** (float-noise) and **inexact** (accuracy-affecting)
changes — training, hyperparameters, AND architecture. Biggest wins first, but per Andrew
(2026-06-27): **bank cheap size/speed wins that barely move LogLoss first; don't push the
champion close to the +0.0015 threshold early** (the champion's distance from the threshold is
the remaining budget for ALL future iterations — burning it early starves them).

**Two hard INVARIANTS (never change):** (1) hierarchy `card→note→deck→preset→global` (5 chained
streams in that order); (2) inputs — the model must still run on the *same preprocessed 92-dim
data* / existing LMDBs. No new/changed inputs.

**The 5 gates — a change is KEPT only if ALL pass:**
1. **LogLoss (both modes):** ahead AND imm by-user-mean LogLoss not worse than **iteration 0**
   by >**+0.0015**. (A pure/exact change ≈0; a real rise is a red flag, not budget to spend.)
2. **Review count ("size"):** per-user equalized review count IDENTICAL to iter0 (it's a
   property of the data+filters; any change = a pipeline bug).
3. **State size:** per-card RNN state (card_id stream) **≤ iter0** (13,056 floats / 51.0 KiB).
4. **Hierarchy** preserved. 5. **Inputs** unchanged.
GPU training speed is **untimed** (prefer it not balloon, but it doesn't gate).

**Eval recipe (FIXED every iteration):** train users **1–100**, eval **101–200** (all 100),
bf16 CUDA `python -m rwkv.get_result --config rwkv/get_result_config_iterN.toml` → by-user mean
of `result/RWKV-iterN.jsonl` (ahead) + `RWKV-P-iterN.jsonl` (imm). Training recipe = **WSD**:
WS 18 epochs (558 steps, `train_rwkv_config_iterN.toml`) then **D** 2-epoch cosine decay
(`..._iterN_decay.toml`, loads the WS-final ckpt) — the decay phase matters (it's what landed
the iter3 champion). **Rust-parity invariant:** `verify_rust.py` (3-user float32) must pass for
the champion arch before "shipping" (re-export trace + match the trained model bit-exactly).
**RUN IT WITH `RWKV_WEIGHTS=reference/rwkv_iter36_124.safetensors`** -- the trace_user py_pred is the
iter36 champion's; verify_rust's DEFAULT (rwkv_ref_558) and other models (iter45 etc.) will MISMATCH
(that is wrong-weights, not a regression). Confirmed 2026-06-29 bit-exact: dpred ~3e-7, |rust-python| 0.000000.

**Speed = batch throughput via simultaneous paired Wilcoxon (protocol point 7–8):**
- **Lock CPU freq** (admin, once/session): `powercfg -attributes SUB_PROCESSOR
  75b0ae3f-bce0-45a7-8c89-c9611c25e100 -ATTRIB_HIDE` ; `powercfg /setacvalueindex SCHEME_CURRENT
  SUB_PROCESSOR PROCFREQMAX 3400` ; `... PROCTHROTTLEMIN 100` ; `... PROCTHROTTLEMAX 100` ;
  `powercfg /setactive SCHEME_CURRENT`. (`PROCFREQMIN` is not a valid alias — pin the perf
  state instead. Restore: `PROCFREQMAX 0`, `PROCTHROTTLEMIN 5`.)
- **One trial** = run *before* (champion) and *after* (candidate) **simultaneously**, each
  pinned to **3 threads**, each looping the **same frozen pre-chosen batch set** for a fixed
  wall-clock **T≈20–30 s**; count reviews each finishes → one paired point. Pairing the *trial*
  (not the batch) keeps pairs independent + cancels external load + avoids tail bias.
- Repeat **20 trials** (drop 1–2 warm-ups); accept the speedup only if **one-sided Wilcoxon
  signed-rank p < 0.01**. (**Andrew 2026-06-28: use 20 trials, not ~10** — `wilcoxon_speed.py`
  default is now `--trials 20`.) (Power: n all-same-sign pairs → p≈2⁻ⁿ, so 20 consistent trials
  clear p<0.01 with wide margin.) Batch throughput = stepping many *independent* card-streams in
  parallel (per-card is inherently sequential); batching is an exact, free speedup. Build via the
  config-driven Rust bench + a Python Wilcoxon driver.

**Logging — DO NOT BE SLOPPY (Andrew flagged this twice):** `optimization/logbook.py` appends to
`log.jsonl` and regenerates `log.md` (table excludes `comment`). EVERY iteration gets ALL fields:
`number, timestamp, logloss{ahead,imm}, params, state_kib, throughput, wilcoxon_p,
review_count_check, logloss_tolerance_check, state_size_check, summary(≤15 words, BEFORE),
comment(after; jsonl only)`.
- **Throughput (rev/s) is MANDATORY for every ACCEPTED iteration** — measure it then and there
  (`python optimization/measure_throughput.py <ckpt.pth>`); rejected → `n/a`. Never "pending".
- **`wilcoxon_p` is MANDATORY for every ACCEPTED iteration** — run the paired Wilcoxon trial
  (champion-vs-candidate) and record p; rejected → `n/a`.
- Plain ASCII in shell-written values (an em-dash mojibakes). Log dead ends with a why-comment.

**Tooling (`optimization/`):** `model_stats.py` (params + per-card state), `gate.py` (computes
the gates + appends a record; `--no-write` to dry-run), `logbook.py`, `measure_throughput.py`,
`PROTOCOL.md`. Use `.venv/Scripts/python.exe`, `OMP_NUM_THREADS=7`.

**Training survives the ~5-min session teardowns** (which kill bg/detached jobs) via
**foreground + resume-from-checkpoint**: ckpts every 100 steps; resume by copying
`{prefix}_optim_{step}.pth` → `{prefix}_{step}_optim.pth` and setting LOAD_MODEL /
LOAD_MODEL_NAME=`{prefix}_{step}` / STEP_OFFSET=step+1.

## Optimization state (research phase, 100/100 workbench)

> Numeric record = `optimization/log.md` (the CANONICAL regenerated table -- now has a Research-phase
> section fed by `research_log.jsonl`; rebuild via `python optimization/logbook.py rebuild`) + the source
> jsonls (`research_log.jsonl`, `baseline_log.jsonl`, `log.jsonl`, `quant_log.jsonl`, `qat_log.jsonl`).
> `research_log.md` keeps the verbose per-experiment notes; `HISTORY.md` = superseded plans + the full
> pre-2026-06-30 snapshot. **Log EVERY research experiment to `research_log.jsonl` + rebuild log.md.**
> This section keeps ONLY the current champion, deploy config, acceptance gate, lesson bank, live state, ops.

### Workbench + baselines
- **Workbench:** eval 101-200 (`--short --secs`), MAX_TRAIN_GLOBAL_LEN=66000, sc8k
  8192-chunk db, WS(+decay), **augmentation OFF** (RWKV_AUGMENT_SEED=1234) + RWKV_DETERMINISTIC=1 +
  RWKV_EMPTY_CACHE_EVERY=0 -> run-to-run variance ~0. Eval `python -m rwkv.get_result` (CUDA, JIT-on ->
  REQUIRES the `@torch.jit.ignore` fix on `quant_aware_rwkv7`).
  **★ TRAIN-DATA SHIFT (2026-06-30): the champion recipe is now "1 epoch on ~1500 users (1000-2499,
  `train_db_sc8k_1500`) + decay", NOT "100 users x15 epochs"** -- data variety won decisively (see CHAMPION).
  Future gated experiments should use the 1500-user recipe (~16 min WS + ~5 min decay + ~4 min eval = ~25 min)
  and compare vs the NEW champion (0.309706/0.276357), NOT the old 0.314807/0.280200. (The old 100u/`train_db_sc8k`
  workbench remains a cheap proxy for quick arch sanity checks, but accept/reject is on the 1500-user recipe.)
- **Baseline-to-beat (accuracy TARGET, fp32, NOT deployable):** the d=128 2.76M model trained on 1-100, eval
- **Baseline-to-beat (accuracy TARGET, fp32, NOT deployable):** the d=128 2.76M model trained on 1-100, eval
  101-200 = **ahead 0.320295 / imm 0.281913** (eval via arch-swap `scratchpad/architecture_old_d128.py`).
- **Iteration-0 reference:** d=128, 2,762,884 params, ahead 0.374046 / imm 0.319475 (historical floor).

### CHAMPION = H=2/K=16 on the 1500-user data-variety recipe  (d=32, 2 heads x K=16; 193,724 params)
- arch `[1,4,3,3,3]` (card,deck,note,preset,user), d_model=32 split as **2 heads x 16 (K=16)** via the NEW
  K<32 CUDA kernel -- this HALVES the per-card WKV state (1088->576 floats; model_stats confirmed) at ~same
  params, ~half the WKV-kernel work, and **~1.16x faster GPU training (WS 1.182 vs 1.020 steps/s)**. Trained on
  users 1000-2499 (`train_db_sc8k_1500`), 1 epoch WS (3351 steps) + 0.27-epoch cosine decay (904 steps). ckpt
  `scratchpad/exp_h2k16/h2k16d_904.pth`; weights `reference/champ_h2k16.safetensors`. Recipe env = RWKV_N_HEADS=2
  RWKV_HEAD_DIM=16 + HP {peak_lr 1e-3, warmup 200, wd 0.01, clip 0.25} + RWKV_EMPTY_CACHE_EVERY=0.
- **fp32: ahead 0.309723 / imm 0.276566** (eval 101-200) -- accuracy PARITY with the prior H=1 champion
  (champ_1500d 0.309706/0.276357; both modes within 0.0002, far inside the +0.0015 efficiency budget), and BEATS
  the d=128 baseline by +0.0106 ahead / +0.0053 imm. Accepted as a **SIZE/SPEED win** (state halved + faster),
  NOT on the +0.0003 monotonic gate. **⚠ HPs NOT YET re-tuned** for this smaller-model + larger-data regime
  (Andrew 2026-06-30) -- re-tune may turn parity into an accuracy gain (IN FLIGHT, see LIVE STATE).
- **★ KEY FINDINGS:** (1) DATA VARIETY beats repetition -- "1 epoch on ~1500 varied users" >> "15 epochs on
  100 users" (drove the prior champion jump; the d=32 model is DATA-limited, so the path forward is MORE DATA,
  scale toward 5k). (2) K<32 UNBLOCKED -- the WKV kernel is now K-dynamic (any K dividing 32), so H=2/K=16 gives
  the 2x-smaller-state + faster model that makes 5k-user training practical. PRIOR champions kept as refs:
  champ_1500d (H=1/K=32, 0.309706/0.276357), decay15 (100u, 0.314807/0.280200).
- **DEPLOYED (the OFFICIAL comparison number = quant + low-rank, via the Rust engine) [[champion-logloss-deployed]]
  -- PENDING (engine port needed):** the Rust engine is still K=32-hardwired (`rust/rwkv-infer/src/model.rs`
  H/C dims) -> needs a K<32 port before deployed rev/s + deployed logloss can be measured for H=2/K=16. The card
  state is now **two 16x16 per layer** (per-head), which RE-FRAMES the state-quant problem (per-head low-rank of
  16x16, NOT one 32x32) for the outsourced sibling loop at `C:\Users\Andrew\rwkv-state-quant`. Prior H=1 deployed
  config (reference): rank-2 int4 low-rank card+note + int4 shifts = card 96 B + note 288 B (both hard targets
  met: card <=0.15 KB, note >=2x). int2 factors DEFERRED (4-level + Hadamard CONFIRMED DEAD in logloss; Frobenius
  anti-correlated) -- see [[deploy-known-issues]]. **This repo's Claude does NOT work the sibling folder** -- stay
  on GPU speedups + the smaller model + more data.

### ACCEPTANCE GATE (research phase) -- accept iff ALL hold (record binary accepted/rejected per iter):
1. "size" (equalized review count, 101-200) IDENTICAL to champion (data-integrity; any change = pipeline bug).
2. params <= **225,000**.   3. card AND note per-entity state UNCHANGED (deck/preset/global MAY grow freely).
4. ahead improves by >= **0.0003** vs the CURRENT champion.   5. imm improves by >= **0.0003**.
=> accept ONLY changes that improve BOTH modes by >=0.0003 (a monotonic champion). [[research-acceptance-gate]]
**EXCEPTION -- SIZE/SPEED changes** (e.g. H=2/K=16): judged on the **efficiency budget** instead -- accept if
both modes stay within **+0.0015** of the champion AND the change shrinks state and/or speeds training (it
Pareto-dominates at accuracy-parity). H=2/K=16 was accepted this way (halved card state, 1.16x faster, accuracy
within 0.0002). Such a change MAY shrink card/note state (gate #3 is for accuracy-research iters, not these).
Two HARD INVARIANTS (never change): hierarchy card->note->deck->preset->global; same preprocessed 92-dim
inputs / existing LMDBs (no new/changed inputs).
**5k-PHASE METHODOLOGY (Andrew 2026-07-01) -- full text in `optimization/research_5k_notes.md`:** the 5k
research phase (train 1-5000 / eval 5001-10000; old d=128 model eval'd on 5001-10000 as the target) keeps
the same >=0.0003-BOTH-modes gate + params <=225,000, and ADDS: (a) **LogLoss recorded WITH (fake)
card- AND note-state quantization** -- beat the old fp big model *while* quantized (sibling `rwkv-state-quant` is
writing the fast fake-quant kernel; copy later); (b) card+note state sizes FIXED, but deck/preset MAY grow
~5-10x and global up to ~100x; (c) WS FIXED at 2 epochs, decay = WS x ratio, ratio in [1/10, 1/2.5] (decay
0.2-0.8 epochs, ALSO quant-aware) -- add decay_ratio as an `hp_tuner_5k.py` lever; (d) HP-tune FIRST,
then re-tune after accumulated small changes OR a major one; (e) every change must be Rust/CPU-deployable
in Anki -- no GPU-only tricks in the shipped model; (f) BEFORE HP tuning, sweep MAX_TRAIN_GLOBAL_LEN (the
WKV batch dim) over ~100 steps each and fix the largest batch that ALMOST maxes the 12 GB VRAM (fastest
training; batch size is structural so LR/warmup tune after it; don't go below 66000 = data drops) --
**DONE 2026-07-02: MAX=110000** (peak 38,968 rev/s @ 9.44 GB; 132k thrashes, -25%); (g) **Wilcoxon
early-pruning (2026-07-02):** run order = old-model eval -> ONE champion-HP run logging per-step WS train
logloss (RWKV_STEP_TRACE; NOT decay) -> HP tune; candidates then check one-sided Wilcoxon (candidate vs
champion, paired by step, growing window) every 300 steps and ABORT iff BOTH modes worse at p<1e-4
(exit 42 + .pruned.json with estimated finals = champ_final + cand@s - champ@s -> front-table `logloss`
column says exact|estimated). Champion accept = `python optimization/promote_champion_5k.py` (auto-replaces
optimization/champion_5k.json = the prune ref; never hand-edit). Pairing needs identical db/MAX/seeds.
[[research-acceptance-gate]]

### LESSON BANK -- do NOT re-run these (full numbers in log.md / HISTORY.md)
- KEPT: SRS heads 128->64 * card->deck rebalance (compensation deck>preset>user, NOT note) * card 2->1 layer
  * 4-epoch decay * **HP tuning (peak_lr 7e-4->1e-3, clip 0.5->0.25, epochs->15) = the big win; the model was
  undertuned** * scoped state-quant card int4 + note int8 ~free * QAT makes card int2 + note int4 ~free
  (warm-started) * **LOW-RANK rank-2 int4 card/note WKV state BEATS int2 -- smaller AND more accurate (rank-2
  keeps ~98.7% energy)** * shifts must be quantized for honest deploy size (RWKV_QUANT_SHIFTS).
- FAILED/REJECTED: FC/head-width 4->2 (imm +0.053) * note 3->2 layer-cut (iter38, +0.0018) * all-streams
  blanket state-quant (long-recurrence user/global sink it) * note int4 PTQ (>2x budget) * weight PTQ int8/int4
  (no speed win) * QAT from scratch (iter40, +0.0118 -- MUST warm-start) * naive low-rank QAT (iter46; STE
  can't guide a structural rank change -> low-rank stays PTQ, int-quant stays QAT) * capacity adds at 100
  users: num_curves/points 64->128, channel_mixer 1.0->1.5, WS 18 epochs, 8-epoch decay -- ALL reject =>
  **the d=32 model is DATA-limited at 100 users, not capacity-limited; training levers are the wins.**
- DATA-DROP bug (FIXED): `get_groups` silently skips any batch with size>MAX_TRAIN_GLOBAL_LEN. At the old
  MAX=20000 the early loop trained on ~5% of the data; MAX=66000 = full coverage (worth ~0.013 imm -- larger
  than the entire early iter0->iter36 loop). Iter-to-iter rankings stayed valid (same subset) but absolute
  quality was on a biased slice.
- GPU-training speedups (arch-agnostic, non-gating): `torch._foreach_*` for copy_downcast_/grad-transfer +
  skip grad_norm/log_model when wandb off + JIT restored via `@torch.jit.ignore` on `quant_aware_rwkv7` (the
  QAT-lowrank `torch.linalg.svd` had SILENTLY broken TorchScript -> would crash plain WS/eval) = ~1.38x over
  the no-JIT body. `torch.compile` is Windows-blocked (no Triton). Gate parallelism (run_qat_eval.sh NPROC)
  made the Rust gate ~8x faster.
- **DONE (was BLOCKED): K<32** -- the WKV kernel is now K-DYNAMIC (any K dividing 32; byte-identical at K=32,
  K=16 parity-verified) and H=2/K=16 is the new champion: halved per-card state (1088->576 floats) + 1.16x
  faster training at accuracy-parity. The GPU path is K<32-ready; the **Rust inference engine is NOT yet**
  (`model.rs` K=32-hardwired) -> a K<32 Rust port is the next deploy-side task.
- STILL DEFERRED: CUDA graphs (variable shapes, ~1.1-1.3x only); Stateful-BPTT carry SHELVED (smaller chunks
  don't speed training; the verified stateful WKV kernel is done + UNCOMMITTED) [[stateful-bptt-shelved]].
- **TIER 1 DEPLOYED (2026-07-01):** the cudaMalloc/cudaFree->`torch::empty` caching-allocator scratch (WKV
  fwd+bwd scan, kills the synchronizing `cudaFree`, bit-exact ~1.3-1.44x microbench) is now the LIVE production
  `rwkv/model/RWKV_CUDA.cp312-win_amd64.pyd` (SHA256 == the bit-exact-validated build). Real-world WS steps/s
  A/B still deferred (falls out of the next training run).
- **TENSOR CORES -- PROFILED + DEAD (2026-07-01, hard numbers, `scratchpad/prof_wkv.py`).** The ONLY matmuls
  (scan `rwkv7_scan_kernel`+`rwkv7_add_kernel`) are **<=1.1% of WKV GPU time, 0.74% at B16xT30000** (realistic
  5k shape); the other 96% is the per-timestep matrix-VECTOR warp-shuffle recurrence (backward `final` ~61%,
  fwd `final`/`base` ~12/11%, bwd `base` ~11%) which tensor cores CANNOT touch. Amdahl ceiling <1% => the cheap
  "tensor-core the scan" win is DEAD. The only TC path is a from-scratch chunked-matmul (fla delta-rule) rewrite
  of the recurrence -- multi-day + parity-risky (K=16 underfills TC tiles); revisit ONLY if 5k proves too slow.

### SPEED -- where GPU training time actually goes (RE-DIAGNOSED 2026-06-30) [[gpu-training-speed-levers]]
- **Fetching is already HIDDEN -- NOT a lever.** `data_fetcher.get()` waits ~2.5-3 s on the FIRST batch then
  ~3-7 ms/step (7 workers + FETCH_AHEAD=5 fully hide prep+IPC); the input `.to(device)` H2D is ~0 ms on the
  critical path (~21 MB batch). Async-pinned prefetch / mp.Queue swap / vectorizing prepare() buy ~nothing.
  (This CORRECTS the earlier "fetch overlap 1.5-1.85x" claim, which was wrong about the mechanism.)
- **Cheap win = `RWKV_EMPTY_CACHE_EVERY`** (env added; default 1 = byte-identical). The per-step
  `torch.cuda.empty_cache()` (first 1000 steps, an OOM-fragmentation guard) costs ~118 ms/step.
  **VALIDATED 2026-06-30 (scratchpad/run_ectest.cmd, 320-step WS on train_db_sc8k):** every=1 -> 0.932 steps/s,
  every=0 -> 1.047 steps/s = **1.12x, NO OOM** (exit 0). Numerics-neutral (allocator only). Full 1.12x only for
  runs <=1000 steps (only the first 1000 steps clear); for WS-15 (~2400 steps) ~5% overall. ADOPT every=0 for
  research runs (model is tiny ~6/12 GB -> no frag-OOM risk).
- **Real lever = the WKV-kernel compute floor (fwd 140 + bwd 403 = ~543 ms/step, ~80% of the step).** Only a
  smaller model / K<32 kernel / bigger batch moves it. **PARTLY BANKED:** H=2/K=16 (K<32, now champion) cut
  ~half the WKV-kernel work for a net 1.16x WS speedup; bigger effective batch is the remaining headroom.
  Param breakdown (~193k): 5 RWKV streams 75.5% (deck 4L 21.6%, note/preset/user 3L 16.2% each, card 1L 5.4%),
  SRS heads 16.0%, input FC 8.4%; ~10.4k params per d=32 layer.

### LIVE STATE (2026-06-30 late, post-h2k16)
- **★ QUANT PORT DONE (2026-07-03): the sibling's research is FINISHED and its machinery is IN-REPO.**
  Sibling final result: **e150_pq @ ~352 b/card** (rank-1 PQ m2b8 WKV + int4 shifts + 1.5-ep QAT) = VAL
  **+0.0010 imm / -0.0003 ahead** vs fp32 (compressed BEATS fp32 on ahead). Ported: fused QAT CUDA kernels
  (full-matrix int-N + rank-1 low-rank with PQ branch, 150-490x over the Python loop), PQ codebook
  `reference/pq_cb_m2b8.txt`, shift-QAT (JIT-annotated here; sibling ran NO_JIT), architecture int3 +
  RWKV_QAT_SHIFT_SCOPE, and train_rwkv **LR+WD clobber fixes** (optim load silently restored saved
  lr/initial_lr/weight_decay over config/env -- affects EVERY warm-started run) + non-finite loss/grad
  guards. QAT env: `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4 RWKV_QAT_PQ=reference/pq_cb_m2b8.txt
  RWKV_QAT_FUSED=1`. Validated here: plain path bit-exact vs golden; PQ parity 3.2e-07; int-N parity
  7.5e-04; 25-step QAT smoke green. Full detail: `optimization/research_5k_notes.md` "Quantization port";
  sibling log `rwkv-state-quant/research_log_h2k16.md`. Weights `reference/qat_pq_ep150.safetensors` (local).
- **★ H=2/K=16 WON -> NEW CHAMPION (see CHAMPION section).** fp32 ahead 0.309723 / imm 0.276566 (eval 101-200,
  100 users) = accuracy PARITY with champ_1500d (within 0.0002), per-card state HALVED (576 floats), WS train
  1.16x faster (1.182 vs 1.020 steps/s). Logged to research_log.jsonl (`h2k16` row) + log.md rebuilt. Weights
  `reference/champ_h2k16.safetensors` (from `scratchpad/exp_h2k16/h2k16d_904.pth`). K<32 kernel is K-DYNAMIC
  (any K dividing 32), byte-identical at K=32, K=16 parity-verified (`scratchpad/test_k16_wkv.py`). arch.py
  decoupled: `d_model = N_HEADS*HEAD_DIM`, env `RWKV_N_HEADS`/`RWKV_HEAD_DIM` (default 1/32). Rebuild kernel:
  `scratchpad/run_build_k16.cmd`. ⚠ Rebuild FAILS if any process holds `RWKV_CUDA.*.pyd`.
- **NEXT (IN FLIGHT): re-tune HPs** for the smaller-model + larger-data regime (Andrew 2026-06-30: "model got
  smaller AND dataset got larger"). Use `optimization/hp_tuner.py` (greedy coord descent, resumable) on the
  H=2/K=16 + 1500u recipe (env RWKV_N_HEADS=2 RWKV_HEAD_DIM=16). Levers: peak_lr, warmup, wd, clip, decay ratio.
  A re-tune may turn accuracy-parity into an accuracy GAIN. Champion to beat: 0.309723/0.276566.
- **DEFERRED until the champion is FINALIZED (post-HP-tune)** -- doing these now then again after tuning = wasted
  work, since tuning changes the weights:
  - **SIBLING quant-loop reframe** (`C:\Users\Andrew\rwkv-state-quant`) [[deploy-known-issues]]: h2k16 won -> card
    state is now TWO 16x16 per layer -> the low-rank problem reframes to per-head rank-r of 16x16 (not one 32x32).
    Will need champ_h2k16 weights + re-dumped 16x16 STATES (the sibling runs ITS engine on the input traces).
    Input traces themselves are arch/weight-independent. This repo's Claude does NOT do the sibling's research.
  - **TRACE EXPORT (sibling reference data): NO LONGER NEEDED (Andrew 2026-06-30) -- do NOT run or recover.**
    The 500+500 export to the sibling's `reference_big` (users 6000-6999) ran this session and landed 836/995
    existing-user traces (159 lost when workers crashed on dataset-absent users; `export_features_fast.py` since
    fixed to skip-missing + per-user try/except). Andrew: we don't need the export anymore -- the 836 traces stand,
    NO recovery, NO further trace export. (Supersedes the old 50+50 plan + the `run_export_5050.cmd` path.)
- **ACTIVE AGENDA (Andrew):** [DONE] fetching (empty_cache banked) -> [DONE] 1500u data champion (variety wins)
  -> [DONE] H=2/K=16 2x-smaller-state model (won) -> [IN FLIGHT] re-tune HPs -> then scale DATA toward 5000 users
  (the proven lever; the smaller+faster model makes it practical). META-GOAL: 5k-user training practical.
- **UNCOMMITTED code (commit-when-asked):** `rwkv/model/csrc/**` (K-DYNAMIC kernel: rwkv7_cuda.cu,
  rwkv7_cuda_time_parallel_{forward,backward}.h, parallel_scan.cu/.h + stateful WKV kernel) + rwkv7.cpp;
  `rwkv_model.py` (K-divides-32 assert relax); `architecture.py` (N_HEADS/HEAD_DIM decouple + env overrides);
  `train_rwkv.py` (EMA + HP env overrides + aug seed + RWKV_EMPTY_CACHE_EVERY); `data_processing.py`
  (label_filter-optional); `rust/rwkv-infer` (sort-fix + per-column + Hadamard/4level low-rank, the last two
  confirmed-dead). NOTHING committed this whole arc -- commit-when-asked.
- Tuner `optimization/hp_tuner.py` (greedy coord descent, resumable) ready -- run SPARINGLY (after a big arch
  change like h2k16, or accumulated small changes). Lit-review queue: `optimization/LIT_REVIEW.md`.

### Ops
- **Compaction (ONLY sanctioned way):** run `claude-automation/request_compact.ps1 -Focus "<carry-through>"`
  then yield idle and STOP beating the heartbeat. `/compact <focus>` fires only from a FRESH (<=30 min) +
  focus-bearing flag (stale/empty = purged). Never hand-create `pending_compact.txt`. The injector is 24/7
  (ClaudeLoopController every 3 min; acts only on a stale heartbeat) and may inject EXACTLY `/compact <focus>`
  or a short `Continue` -- nothing else.
- **ESC-PROOF detached launches:** Esc / session teardown tree-kills Claude's Bash/PowerShell bg jobs INCLUDING
  training. Launch each training as a self-contained `.cmd` via `scratchpad/detach.ps1` (WMI Win32_Process ->
  parented to WmiPrvSE, survives); log to a STABLE repo path (`scratchpad/*.log`, NOT the rotating session
  temp); end the .cmd with `echo DONE_EXIT_%ERRORLEVEL%`. MONITOR via OS truth (poll the log / Get-Process /
  ckpt mtime) -- detached runs give NO tool-completion event. A Bash watcher gives notifications but is itself
  Esc-killable (re-arm it each turn; the training survives). Beat the heartbeat each working turn
  (`claude-automation/beat.ps1`). **Do NOT kill the FSRS benchmark PIDs (the ~80000s-CPU python procs).**
- **DATA FACT:** anki-revlogs-10k has NO absolute timestamp / review-id (anonymized; raw `revlogs` parquet =
  card_id, day_offset [integer DAY counter], rating, state, duration, elapsed_days, elapsed_seconds). Time-of-
  day is UNRECOVERABLE -> a time-of-day input feature is impossible here. elapsed_seconds (time-since-last) is
  already an input.
- Quant papers: `scratchpad/{rwkvquant,rwkvedge}.txt` (poppler installed; the Read tool handles PDFs). Use the
  CURRENT session's scratchpad dir for transient logs (it rotates on teardown -- check task-output paths).
