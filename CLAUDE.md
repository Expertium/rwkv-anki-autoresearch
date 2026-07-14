# rwkv-anki-autoresearch — Claude handover

> **GitHub rule (always):** every GitHub comment posted on Andrew's behalf — PR
> descriptions, review replies, issue comments, anything — **must start with the line
> "Written by Claude".** No exceptions. (Also in the global `~/.claude/CLAUDE.md`.)

## 0. Who you are / read this first

You own **`C:\Users\Andrew\rwkv-anki-autoresearch`** (GitHub:
`Expertium/rwkv-anki-autoresearch`). (The repo started empty; the RWKV code has long since been
vendored in — see §2/§10 for what lives where. Roadmap steps 1–3 are DONE.)

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
- **CUDA build: RESOLVED long ago** (torch cu130 wheel + VS2022 vcvars64; the kernel builds and is
  the live production `.pyd`). Rebuild via `scratchpad/run_build_k16.cmd` — fails only if a process
  holds `RWKV_CUDA.*.pyd` (use `setup.py build_ext` WITHOUT `--inplace` for an isolated build then).
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

## Optimization state (5k phase: train 1-5000 / eval 5001-10000)

> Numeric record = `optimization/log.md` (the CANONICAL regenerated table -- now has a Research-phase
> section fed by `research_log.jsonl`; rebuild via `python optimization/logbook.py rebuild`) + the source
> jsonls (`research_log.jsonl`, `baseline_log.jsonl`, `log.jsonl`, `quant_log.jsonl`, `qat_log.jsonl`).
> 5k-phase verbose per-iteration notes = `research_5k_verbose.md` (AI-only; research_5k.md's summary
> column is capped at <=20 words, Andrew 2026-07-13; `research_log.md` = the CLOSED 100/100-era log);
> `HISTORY.md` = superseded plans + the full pre-2026-06-30 snapshot. **Log EVERY research experiment
> to `research_log.jsonl` + research_5k.md row + research_5k_verbose.md section + rebuild log.md.**
> This section keeps ONLY the current champion, deploy config, acceptance gate, lesson bank, live state, ops.

### Workbench + baselines
- **5k phase (CURRENT):** train 1-5000, eval 5001-10000, budget 2 WS ep + tuned-ratio decay,
  MAX_TRAIN_GLOBAL_LEN=110000 (swept), quant-aware logloss. Baseline-to-beat = the old d=128 model
  (`pretrain/RWKV_trained_on_101_4999.pth`, unquantized) eval'd on 5001-10000 (PENDING, needs eval data).
  Front table `optimization/research_5k.md`; full methodology + status `optimization/research_5k_notes.md`.
- **Run env (all phases):** **augmentation OFF** (RWKV_AUGMENT_SEED=1234) + RWKV_DETERMINISTIC=1 +
  RWKV_EMPTY_CACHE_EVERY=0 -> run-to-run variance ~0. Eval `python -m rwkv.get_result` (CUDA, JIT-on ->
  REQUIRES the `@torch.jit.ignore` fix on `quant_aware_rwkv7`).
- **Historical 100/100 + 1500u workbench refs** (eval 101-200, MAX=66000, sc8k dbs): champion recipe was
  "1 ep on 1500 users (1000-2499) + decay" (data variety >> repetition; ~25 min/experiment -- still useful
  for cheap sanity checks). d=128-on-1-100 baseline = 0.320295/0.281913 (arch-swap
  `scratchpad/architecture_old_d128.py`); iteration-0 floor = 0.374046/0.319475.

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
  NOT on the +0.0003 monotonic gate. HPs are re-tuned as part of the 5k phase (methodology d), not on 1500u.
- **★ KEY FINDINGS:** (1) DATA VARIETY beats repetition -- "1 epoch on ~1500 varied users" >> "15 epochs on
  100 users" (drove the prior champion jump; the d=32 model is DATA-limited, so the path forward is MORE DATA,
  scale toward 5k). (2) K<32 UNBLOCKED -- the WKV kernel is now K-dynamic (any K dividing 32), so H=2/K=16 gives
  the 2x-smaller-state + faster model that makes 5k-user training practical. PRIOR champions kept as refs:
  champ_1500d (H=1/K=32, 0.309706/0.276357), decay15 (100u, 0.314807/0.280200).
- **DEPLOY config (the sibling's FINAL locked recipe `q72u`, research CLOSED 2026-07-07; results ported
  here 2026-07-08) [[champion-logloss-deployed]]: 72 b/layer = 9-BYTE CARD, 27 B note, 256x compression.**
  Format per layer: m2b12L learnable shift catalog (2 chunks x 4096 entries, 48 b) + JOINT-UV b10 WKV
  catalog (per head ONE 10-bit code into a 1024-entry concat(u,v) 32-dim catalog, 20 b) + 1-bit norms (4 b).
  VAL penalty vs fp32 **+0.00114/+0.00021 (seed 1234) and +0.00115/+0.00040 (seed 4321)** — 2/2 seeds pass
  with margin; best-ever robustness (imm nbad 96-98/400); imm is ~seed-noise-FREE under joint coding.
  **Artifacts (ported to our `reference/`):** `qat_pq_q72u.safetensors` + `pq_cb_wkv_q72u.txt` +
  `pq_cb_shift_q72u.txt`. **Deploy env (Rust):** `RWKV_STATE_LOWRANK_SCOPE=card:1:int4,note:1:int4
  RWKV_QUANT_SHIFTS=1 RWKV_LOWRANK_PERCOL=1 RWKV_LOWRANK_PQ=reference/pq_cb_wkv_q72u.txt
  RWKV_SHIFT_PQ=reference/pq_cb_shift_q72u.txt RWKV_PQ_NORM_BITS=1`. **QAT recipe:** warm-start champion,
  2.0-ep plain QAT (no rotation/anneal/KD), BOTH cbs learnable (`RWKV_QAT_PQ_LEARN=1
  RWKV_QAT_SHIFT_PQ_LEARN=1`), `RWKV_QAT_NORM_BITS=1 RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3`, NO_JIT.
  **The full engine (joint cb + warm search + norm quant) IS in OUR `rust/rwkv-infer` since `1d3b5b8`**
  (byte-identical champion eval verified from the parent build). Full detail: sibling
  `research_log_h2k16.md` + explainer `how_state_compression_works.md`.

### ACCEPTANCE GATE (research phase) -- accept iff ALL hold (record binary accepted/rejected per iter):
1. "size" (equalized review count, 101-200) IDENTICAL to champion (data-integrity; any change = pipeline bug).
2. params <= **225,000**.   3. card AND note per-entity state UNCHANGED (deck/preset/global MAY grow freely).
4. ahead improves by >= **0.0003** vs the CURRENT champion.   5. imm improves by >= **0.0003**.
6. **p-gate (Andrew 2026-07-08):** paired per-user one-sided Wilcoxon (candidate vs champion, same 5000
   eval users) gives **p < 0.0001 in BOTH modes** -- `python optimization/paired_pvalue.py` (zero GPU cost,
   reads the result jsonls; exit 0 = pass). Record both p-values in research_5k.md's `p-value` column.
   Applies to accuracy accepts only (SIZE/SPEED-exception accepts claim parity, not improvement -> exempt).
=> accept ONLY changes that improve BOTH modes by >=0.0003 AND pass the p-gate (a monotonic champion).
[[research-acceptance-gate]]
**EXCEPTION -- SIZE/SPEED changes** (e.g. H=2/K=16): judged on the **efficiency budget** instead -- accept if
both modes stay within **+0.0015** of the champion AND the change shrinks state and/or speeds training (it
Pareto-dominates at accuracy-parity). H=2/K=16 was accepted this way (halved card state, 1.16x faster, accuracy
within 0.0002). Such a change MAY shrink card/note state (gate #3 is for accuracy-research iters, not these).
Two HARD INVARIANTS (never change): hierarchy card->note->deck->preset->global; same preprocessed 92-dim
inputs / existing LMDBs (no new/changed inputs).
**RESEARCH-PHASE CONDUCT (Andrew 2026-07-10) -- for the phase after HP tuning + the deck/preset/global
state-size ladders:** (1) try LOTS of different tweaks of both the ARCHITECTURE and the TRAINING
PIPELINE, from different FAMILIES of ideas (not many variants of one); (2) if an idea BARELY misses the
logloss threshold, don't give up early -- try a slightly different implementation of the same idea first;
(3) MIX literature review (optimization/LIT_REVIEW.md) with self-generated ideas; (4) spend AT LEAST 50
iterations (NOT counting HP-tuning trials) before even considering declaring "nothing left to improve";
(5) (Andrew 2026-07-13) NEVER declare a FAMILY "closed" after one iteration -- writing off a family
needs at least 3-5 distinct in-family variants; 1-2 rejects = "0/N so far, deprioritized", not closed.
[[research-phase-conduct]]
**5k-PHASE METHODOLOGY (Andrew 2026-07-01) -- full text in `optimization/research_5k_notes.md`:** the 5k
research phase (train 1-5000 / eval 5001-10000; old d=128 model eval'd on 5001-10000 as the target) keeps
the same >=0.0003-BOTH-modes gate + params <=225,000, and ADDS: (a) **LogLoss recorded WITH (fake)
card- AND note-state quantization** -- beat the old fp big model *while* quantized. Env UPDATED 2026-07-08
to the final q72u recipe (fixed champion codebooks, no cb-learning -- that upgrade needs per-run
cb-export->eval wiring, queued): `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
RWKV_QAT_PQ=reference/pq_cb_wkv_q72u.txt RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_q72u.txt
RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3 RWKV_QAT_NORM_BITS=1 RWKV_QAT_FUSED=1 RWKV_NO_JIT=1` (JIT on the
grafted q72u paths unverified -- A/B once at champion-run launch); (b) card+note state sizes FIXED, but deck/preset MAY grow
~5-10x and global up to ~100x; (c) WS FIXED at **1 epoch** (2->1 Andrew 2026-07-09 via the champ5k_b1
budget A/B: 2nd epoch adds nothing -- ahead -0.00006 p=0.31, imm +0.00043 BETTER p=6e-62), decay = WS x
ratio, ratio in [1/10, 1/2.5] (ALSO quant-aware), decay_ratio is an `hp_tuner_5k.py` lever; (d) HP-tune FIRST,
then re-tune after accumulated small changes OR a major one; (e) every change must be Rust/CPU-deployable
in Anki -- no GPU-only tricks in the shipped model; (f) BEFORE HP tuning, sweep MAX_TRAIN_GLOBAL_LEN (the
WKV batch dim) over ~100 steps each and fix the largest batch that ALMOST maxes the 12 GB VRAM (fastest
training; batch size is structural so LR/warmup tune after it; don't go below 66000 = data drops) --
**DONE 2026-07-02: MAX=110000** (peak 38,968 rev/s @ 9.44 GB; 132k thrashes, -25%); (g) **Wilcoxon
early-pruning (2026-07-02):** run order = old-model eval -> ONE champion-HP run logging per-step WS train
logloss (RWKV_STEP_TRACE; NOT decay) -> HP tune; candidates then check one-sided Wilcoxon (candidate vs
champion, paired by step, **last-1500-paired-steps window** -- RWKV_PRUNE_WINDOW, 0=old full window;
changed 2026-07-08 after the 0p0014 audit: full-window drags stale early history -> ~2k-step lag on late
regressions AND would kill late-bloomer configs) every 300 steps and ABORT iff BOTH modes worse at p<1e-4
at TWO CONSECUTIVE checkpoints (RWKV_PRUNE_PERSIST=2, added 2026-07-09: the identical-config null control
champ5k_r1-ep1-vs-b1 showed autocorrelated drift transients hit imm p~1e-15 under the NULL -- single-mode
p is overconfident; the persist rule guards the joint test. No false fire in the control itself.)
⚠ SCOPE (2026-07-09 decay_ratio_0p1 FALSE-KILL audit): prune ONLY candidates at MATCHED regularization
vs the reference -- train-loss pruning is sign-biased against regularization levers (wd=0.1 ran train-hot
vs the wd=0.01 champion trace yet WON eval both modes; its WS-identical twin got killed at imm p=3e-45 --
drift scales with config, no fixed alpha calibrates across bases). HP-TUNER trials therefore run WITHOUT
train-loss pruning; they use the REPLACEMENT **VALIDATION-based prune** (Andrew 2026-07-09): validate
every 500 steps, die iff BOTH modes' val loss >= champion's val at the same step + per-mode delta
(RWKV_VPRUNE_DELTA_AHEAD=0.004 / _IMM=0.006) at 2 consecutive val checkpoints from step 1000
(RWKV_VPRUNE_MIN_STEP/PERSIST). EARLY window by necessity (Andrew's flat-curve catch: val curves are
~flat past 2500 -- ahead range only 0.004 -- so late thresholds catch nothing; at 1000-2000 curves drop
~0.01/1000 steps and disasters gap +0.004-0.011 vs twin-null <=0.0025/0.0029). Sign-correct for
regularization, magnitude-based; late-emerging regressions intentionally run to an honest eval.
RWKV_VPRUNE_REF=champion_5k.json (carries val_step/val_ahead/val_imm; promote_champion_5k --val-trace
embeds them; train_rwkv writes <trace>.val.jsonl when STEP_TRACE is on).
(exit 42 + .pruned.json with estimated finals = champ_final + mean(diff over last 300 paired steps) ->
front-table `logloss` column says exact|estimated). Champion accept = `python optimization/
promote_champion_5k.py` (auto-replaces optimization/champion_5k.json = the prune ref; never hand-edit).
Pairing needs identical db/MAX/seeds.
[[research-acceptance-gate]]

### LESSON BANK -- do NOT re-run these (full numbers in log.md / HISTORY.md)
- **TUNE-EVAL SUBSET OVERFIT (2026-07-12, champ5k_t1):** the 200-user tune-eval (5001-5200) is for
  COARSE ranking only -- a +0.0008/+0.0010 subset win (in-subset paired imm p=5e-8!) INVERTED to
  -0.0005/-0.0007 at n=5000. Sub-0.001 effects measured on 200 users do NOT transfer; confirm on the
  full eval before adopting. Champion HPs (wd 0.01, dropout 1.0, beta2 0.999, cb_lr 1x, peak_lr 1e-3,
  warmup 200, clip 0.25, decay_ratio 0.25) are CONFIRMED at 5k -- don't re-tune without new structure.
  **REMEDY ADOPTED (Andrew 2026-07-12): future HP tuning uses a 1000-user tune-eval (5001-6000)** --
  SE ~sqrt(5)x smaller, resolves ~0.001 effects. Wired: hp_tuner_5k EVAL range + trial template now
  passes the range explicitly; write_eval_toml default 5200->6000. When tuning reopens: re-record the
  tuner baseline on 5001-6000 FIRST (old journal rows are 200-user, not comparable); sub-0.001
  verdicts still need full-eval confirmation.
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
  the no-JIT body. `torch.compile` WORKS on Windows now (STALE-CLAIM FIX 2026-07-03: triton-windows 3.7.1
  is in the venv; smoke test compiles + runs bit-correct) -- but it was 0.79x on a mixer-like chain at our
  tiny C=32 sizes (per-call overhead; 4070 too few SMs for max_autotune_gemm), so it must EARN its way in
  via a real profile A/B, and needs RWKV_NO_JIT (Dynamo can't trace ScriptModules). Gate parallelism
  (run_qat_eval.sh NPROC) made the Rust gate ~8x faster.
- **DONE (was BLOCKED): K<32** -- the WKV kernel is now K-DYNAMIC (any K dividing 32; byte-identical at K=32,
  K=16 parity-verified) and H=2/K=16 is the champion. ~~OUR rust/rwkv-infer is still K=32-hardwired~~
  RESOLVED: `1d3b5b8` ported the sibling's full engine (K-dynamic + PQ + joint cb + warm search).
- **QUANT ENDGAME LESSONS (sibling, 2026-07-04..07, full ladder in its research_log_h2k16.md):** per-card
  cost is INDEX bits -- catalog size is FREE (amortized): fewer/bigger chunks + huge learnable catalogs beat
  the product form on BOTH shift (m2b12) and WKV (joint-uv b10) sides * JOINT coding of correlated vectors
  buys robustness + seed-stability more than mean * rotation lever CLOSED (absorbed by learnable m=1
  catalogs; negative on big catalogs; only "won" on capacity-starved rungs that died as seed luck) * EMA at
  decay-tail = nil (3 confirmations); 2-seed weight soup HURTS (breaks weight<->cb co-adaptation) * norm
  axis bottoms out at 1 bit (0-bit fixed norms = +0.004 cliff) * ⚠ SEED-PAIR DOCTRINE: at-the-gate passes
  with margin < ~0.001 imm / ~0.002 ahead are UNRESOLVABLE by one run (64-b and 56-b "wins" both died on the
  seed test); any thin-margin verdict needs the exact recipe re-run at a second RWKV_AUGMENT_SEED.
- STILL DEFERRED: CUDA graphs (variable shapes, ~1.1-1.3x only); Stateful-BPTT carry SHELVED (smaller chunks
  don't speed training; the verified stateful WKV kernel is done + committed) [[stateful-bptt-shelved]].
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
- **RE-PROFILED 2026-07-03 at the 5k regime (H=2/K=16, MAX=110000, RWKV_PROFILE_STEP env hook in
  train_rwkv): the WKV floor is NO LONGER dominant.** Plain step = 578 ms GPU: elementwise/other 78%, WKV
  recurrence 18%, gemm 5% => the chunked-matmul (fla delta-rule) rewrite is DEAD as a priority (addresses
  <=18%); the new top surface is the PyTorch elementwise mass.
- **torch.compile: WORKS on Windows (triton-windows in venv; the old "blocked" claim was STALE — Andrew
  caught it 2026-07-03) but SHELVED at an honest 1.05x.** Whole-graph compile hits Python 3.12's fixed
  C-recursion cap in Dynamo (RecursionErrors swallowed by the NaN-except -> HOLLOW steps -> a fake 1.27x
  profile); mixer-scoped compile is clean + deterministic but only 365 vs 384 ms. Costs (NO_JIT switch,
  warmup, recompile risk, numerics break) outweigh 5%. Plumbing kept: RWKV_COMPILE=1 + RWKV_NO_JIT=1 +
  scratchpad/train_bigstack.py. LESSON: count "Exception caught" before trusting any run's numbers.
- **DETERMINISTIC-INDEXING SPEEDUP BANKED 2026-07-03 (1.5x plain step, BIT-EXACT):** RWKV_DETERMINISTIC=1
  cost 251 of the 578 ms (sort-based index_add from 2 gather sites). Fixes: **PermGather** (srs_model --
  stream gather is a permutation+pads -> backward = index_select by the inverse permutation; escape hatch
  RWKV_PERM_GATHER=0) + **flat-row time_shift_gather** (rwkv_model -- row index_select instead of
  gather-expand-C -> det backward sorts B*T keys not B*T*C). Both verified by 10-step E2E training traces
  BIT-IDENTICAL to the old path. Det step 578->384 ms (det tax now ~57 ms). **STACKED total: the full
  quant-aware deterministic step = 4,122 -> 450 ms (9.2x); a 5k champion run ~= 4-5 h.**
- **QAT KERNEL SPEEDUP BANKED 2026-07-03 (the big one): quant-aware training was 7.1x slower than plain**
  (4,122 ms/step, 87% in the qat_lr kernels -- would have made every methodology-(a) 5k run ~30-40 h).
  Three bit-exact fixes in `qat_lr_rank1` + call sites: (1) skip-step elision (~half of rows are query
  duplicates whose truncation result was computed then discarded), (2) BLOCK-PARALLEL PQ codebook search
  (was single-threaded ~8k serial FMAs/step while 255 threads idled) with first-strict-min tie-breaking,
  (3) warp-0-scoped power iteration (__syncwarp instead of ~6 block barriers x <=64 iters). Result:
  **QAT share 3,577 -> 96 ms/step (37x), full step 4,122 -> 651 ms (6.3x); quant-aware now costs ~13%
  over plain. BIT-EXACT verified** (32-tensor golden fwd+bwd, int-N + PQ paths, both shapes) + deploy
  parity re-run (max REL 3.2e-07). Goldens: `scratchpad/qat_speed/golden_gen.py gen|check`.

### LIVE STATE (2026-07-13)
- **★ RESEARCH ITER 10 REJECTED (2026-07-13 19:48): warmup-only KD from the d=128 teacher
  (Andrew's idea; 800-step annealed target mix from a stored dump, checksum-guarded) = ahead
  0.306907 / imm 0.278222 -- WORSE both modes (-0.000277/-0.000329 vs champ5k_b1, p=1.0 both).**
  Trajectory = iter 9's exactly: led val early (-0.0026/-0.0046 @ step 500), washed out by WS
  end, finished slightly negative. **EARLY-TRAINING-INTERVENTION family 0/2 (shrink-perturb,
  KD warmup) -> DEPRIORITIZED, not closed (conduct rule 5, Andrew 2026-07-13: closing a family
  needs 3-5 in-family variants)** -- so far head starts do not survive 6554 hard-label steps at
  the 1-ep budget; untried variants if revisited: longer/never-zero KD window, KD into decay,
  permutation init.
  KD machinery stays in-repo (RWKV_KD_DUMP_OUT / RWKV_KD_MIX + exit-43 checksum guard, 78caceb).
  ⚠ OPS: the 2-parallel-shard eval WEDGED ON THE CHAMPION ARCH (both shards frozen 66+ min at
  11.7/12 GB, 100% util, full-core CPU each -- two mega-users collided; the iter-5
  elevated-VRAM-only scoping was TOO NARROW). Fix = kill tree + sequential-resume evalfix
  (run_iter10_kd_evalfix.cmd). **RULE UPDATED: ALL evals run SEQUENTIAL shards** (~45 min slower
  than a clean parallel run, never wedges = unattended-safe; iter11 .cmd already updated).
  **Iter 11 = additive GRADE EMBEDDING (Andrew's idea) REJECTED (2026-07-14 01:24): ahead
  0.307481 / imm 0.278801 -- worse both modes (-0.000851/-0.000908, p=1.0), ~2x cross-seed
  noise = real harm, no seed-pair needed.** The 4x32 zero-init bypass around the input MLP
  (RWKV_GRADE_EMB=1, +128 params) distorts the shared trunk more than it helps -- grade info
  was never bottlenecked (4 of 92 dims through the 128-wide fc). Val looked champion-level all
  run; the harm only showed at full eval. GRADE-REPRESENTATION family 0/1, deprioritized
  (rule 5); untried variants: per-stream embeddings, grade-emb into the SRS heads, LayerNorm on
  the bypass. Hook stays (env-gated, default off = byte-identical).
  **Iter 12 = SRS-HEAD RESOLUTION 64->128 REJECTED (2026-07-14 07:01): ahead 0.306899 / imm
  0.278134 -- no effect (-0.000270/-0.000241 vs champ5k_b1, p=1.0 both, inside the ~0.0004
  cross-seed band = the deck/preset null signature).** The 100u "capacity adds fail" lesson does
  NOT flip at 5k for this lever: 64 curves / 64 points are enough resolution for the
  forgetting-curve mixture. Val sat at champion parity all run (WS-end +0.0003/+0.0010),
  consistent with the null. CAPACITY-AT-5K family 0/1 so far. Clean ~5.6h run (WS 2h32m, decay
  38m, sequential eval 2h24m), no incidents.
  **Iter 13 = CHANNEL MIXER 1.0->1.5 REJECTED (2026-07-14 12:41): ahead 0.306788 / imm 0.278164
  = -0.000159/-0.000271 (p=0.999/1.0), no-effect signature. CAPACITY-AT-5K family 0/2** (head
  resolution, FFN width): the d=32 trunk is not capacity-limited at 5k -- the d=128 gap lives
  elsewhere. LAST QAT-ERA ITERATION.
  **★ METHODOLOGY SWITCH (Andrew 2026-07-14) -- supersedes methodology (a) for the research
  phase:** (1) **QAT PARKED until research closes** -- ALL screening runs (both tracks) are
  PLAIN bf16, JIT on, no codebooks (saves ~2h20m/run; plain step 0.385 s vs 1.41 quant-aware);
  ONE quant-aware run of the final champion at the very end, NO per-accept confirmations.
  champion_5k.json (QAT deploy truth, champ5k_b1) FROZEN; plain screening champion ->
  optimization/champion_5k_plain.json (promote_champion_5k.py --out flag added; plain
  candidates use RWKV_VPRUNE_REF=champion_5k_plain.json). Plain vs QAT-era logloss NOT
  comparable. (2) **TWO RESEARCH TRACKS, ~12h alternating blocks, two tables in
  research_5k.md:** Track 1 = improve the d=32 model (gate unchanged: >=0.0003 both + p<1e-4
  both, params <=225k). Track 2 = ABLATE the old d=128 model; gate =
  50,000*(LL_after-LL_before)/(params_before-params_after) <= 0.0001 in BOTH modes (params
  strictly decrease; "before" = current track-2 champion; rows A0,A1,...). Track-2 anchor A0 =
  d=128 arch retrained through OUR plain 1-ep pipeline at MAX=66000 (fits 12 GB; the 12-ep
  upstream .pth is NOT budget-comparable; ~6.5h train + ~6h eval, unshardable VRAM-wise).
  Context: the whole d=128->d=32 collapse = 0.0002/50k ahead, 0.00026/50k imm -- the bar
  demands ~2-2.6x better than that average. A0 also A/Bs the 1-ep budget at 14x params. TODO
  at A0 launch: env-based arch-module selector in architecture.py (NOT the KD-dump file-swap).
  (3) **POWER-USER-AWARE EVAL LANDED (eval_sharded.py rewritten, dry-run tested):** users >=1M
  work (56 = 11.3% of eval work on 5001-10000; top-7 ~2.1M) run SOLO first (one process,
  7 threads), then 2 parallel LPT shards, then merge -- one call does all phases; worst
  concurrent pair ~2x below the wedge scale; ~1.8x over sequential; resume-safe per phase;
  --solo-threshold 0 = old behavior; RWKV_EVAL_SHARD_DIR overrides the shard dir. d=128 evals
  stay UNSHARDED (one alone ~9 GB). First E2E = the champ5k_plain eval -- watch phase-B VRAM.
  **★ ITER 14 = champ5k_plain ACCEPTED (2026-07-14 15:53) = THE PLAIN SCREENING CHAMPION:
  ahead 0.303734 / imm 0.273448** (n=5000; 3h07m pipeline: WS 91 min @ 0.82 s/step wall, decay
  22 min, eval 75 min). QAT TAX measured at n=5000: +0.002896/+0.004445 (p=0.0) vs champ5k_b1.
  Gap to the d=128 target now +0.0073/+0.0085 (was +0.0102/+0.0134 QAT). Promoted ->
  optimization/champion_5k_plain.json (ckpt champ5kplaind_1638.pth + WS trace + val trace =
  the PLAIN vprune ref for track-1 candidates); champion_5k.json (QAT) FROZEN. The phased eval
  E2E'd FLAWLESSLY: solo 9 min (mega-user 3.9 GB), phase B ~1.8 GB combined (no wedge
  exposure), 1.9x over sequential.
  ⚠ FIXED EN ROUTE: iter-11 RWKV_GRADE_EMB hook broke JIT-on construction (TorchScript
  resolves attrs in dead branches; hidden all QAT era by NO_JIT) -> @torch.jit.ignore
  indirection in srs_model.py, smoke-tested both hook states. train_rwkv swallowed that
  traceback with exit 0 -- the .cmd artifact gate caught it (always gate phases on artifacts).
  **-> NOW: TRACK 2 ANCHOR A0 RUNNING (4th launch, detached pid 20332, 17:02, verdict ~07:15
  tomorrow):** the ORIGINAL d=128 arch (2,762,884 params, in-log confirmed) retrained through
  the plain pipeline via the NEW RWKV_ARCH_MODULE env hook (architecture.py bottom: exec's a
  standalone config file, replaces DEFAULT_ANKI_RWKV_CONFIG wholesale -- bypasses all
  default-build env hooks; scratchpad/architecture_old_d128.py verified). **MAX=32768 -- THE
  TRACK-2 STANDARD (pairing needs it identical across all track-2 runs).** Launch saga:
  MAX=66000 THRASHED (11.85/12 GB WDDM spill, 40 s/step -- the 100u-era "66000 fits" fact
  doesn't transfer, 5k packs fuller groups) and 49152 still thrashed (13.3 s/step, allocator
  bloat on 3x16384 packing); 32768 = 2x16384 clean packing -> 3.6 GB, 1.06 s/step, ~22k
  steps/epoch. ⚠ COVERAGE FACT (probe 2026-07-14): max single batch in train_db_5k_h1 =
  16,384 tokens -> ZERO data drop at ANY MAX >= 16,384 (the "don't go below 66000 = data
  drops" rule was sc8k-era, NOT true of the 5k db). TWO LATENT BUGS FIXED en route:
  (1) train_rwkv's blanket NaN-except now prints the real traceback (bare asserts have empty
  str(e) -- it had hidden the hollow-compile run and this); (2) utils.KeyValueAverage
  .get_value returned via bare assert n>0 -- early groups can have ZERO equalize-counted
  reviews (first seen at small MAX), and the throw landed AFTER backward but BEFORE
  optimizer.step = silently skipped weight updates; now returns NaN (wandb-only consumer).
  Eval = SINGLE process (--shards 1 --solo-threshold 0; d=128 can't share 12 GB). Ends with
  informational paired vs base5k (the 1-ep-budget check at 14x params). A0's finals + val
  trace = the track-2 "before" anchor + its vprune ref.
  Track-1 queue (plain era, ~3h/iter): prehead output gate, cross-head readout mix, loss-term
  reweighting, permutation init (LOW). Track-2 queue after A0: layer cuts / d_model cuts /
  mixer cuts / LoRA dims / head-width cuts, ranked by expected ratio-efficiency.
- **★ RESEARCH ITER 9 REJECTED (2026-07-13 12:58): shrink-perturb init (lam=0.5, fresh seed 777,
  RWKV_INIT_BLEND hook, else exact champion recipe) = ahead 0.307373 / imm 0.278926 -- WORSE both
  modes (-0.000744/-0.001033 vs champ5k_b1, p=1.0 both), beyond the ~0.0004 seed noise = real harm,
  no seed-pair needed.** Trajectory lesson: the warm init LED the champion's VAL curve all WS
  (-0.010 @ step 1000 shrinking to -0.0006 @ 3500) yet ended net NEGATIVE at full eval -- mid-WS
  val leads from a warm start do NOT predict the final verdict. Both lam endpoints (~0 =
  from-scratch champion, ~1 = the 2-ep budget A/B) are champion-level and the midpoint sits below
  -> **data-driven-init scheme A (shrink-perturb at lam=0.5) rejected; family DEPRIORITIZED,
  not closed (conduct rule 5); lam probe {0.3,0.7} judged not worth GPU for now; scheme B
  (permutation init) queued LOW.** The RWKV_INIT_BLEND hook stays (eed7cb5,
  env-gated, plain path untouched). Artifacts: scratchpad/iter9_sp/, result/RWKV[-P]-iter9_sp.jsonl.
  **-> NOW: iter 10 = warmup-only KD from the d=128 teacher** -- machinery committed 78caceb:
  train_rwkv RWKV_KD_DUMP_OUT teacher-dump mode + RWKV_KD_MIX annealed target-mix student mode
  (per-step labels-checksum pairing guard, mismatch = exit 43 never a silent skip; srs_model
  get_loss(kd_mix=) mixes TARGETS exactly -- BCE/CE are linear in the target; window 800 WS steps,
  alpha 1->0; clear RWKV_KD_MIX before decay -- decay replays the epoch-0 stream). Sequence: dump
  smoke KDSTEPS=3 (d=128 VRAM check) -> full 800-step dump (~20 min, scratchpad/iter10_kd/dump
  ~0.9 GB) -> run_iter10_kd.cmd (~4.7h). ⚠ the dump .cmd FILE-SWAPS rwkv/architecture.py --
  never overlap with any other rwkv launch. Queue after 10: SRS-head resolution 64->128 (capacity
  re-test at 5k data -- the 100u "capacity rejects" lesson was data-limitation-scoped), channel
  mixer 1.0->1.5, prehead output gate, cross-head readout mix, loss-term reweighting.
- **★ STATE-SIZE LADDER CLOSED (2026-07-13 08:04): 0 accepted rungs across 5 iterations (4-8).**
  Per-stream arch hooks live (d6fca68): `RWKV_STREAM_HEADS` (H=1 doubles that stream's per-entity
  WKV state ~param-free) + `RWKV_STREAM_LAYERS` (~10.4k params/layer). Verdicts (all paired vs
  iter 2 champ5k_b1, n=5000): **deck H=1** (iter 4) null p=1.0; **preset H=1** (iter 5) null p=1.0;
  **user H=1** (iter 6) NEAR-MISS +0.000345/+0.000258 (imm short by 0.000042, in-seed p 1e-20/1e-29);
  **user H=1 + 4L** (iter 7) mode TRADE (ahead -0.000299 / imm +0.000604); **iter 8 lad_user1b =
  the seed-pair test of iter 6 (seed 4321) came back NULL** -- ahead 0.306674 (-0.000044, p=0.88) /
  imm 0.278039 (-0.000146, p=1.0) = the deck/preset no-effect signature. **Iter 6's signal did not
  replicate -> substantially SEED LUCK; reject stands per the pre-declared branches.** LESSONS:
  (1) no stream is state-capacity-limited at d=32/H=2 -- 2x recurrent memory clears nothing;
  (2) ⚠ in-seed Wilcoxon p (even 1e-29) measures per-user delta consistency, NOT cross-seed
  robustness -- cross-seed spread on the SAME recipe is ~0.0004 both modes, so **any single-run
  margin < ~0.0005 defaults to seed-pair confirmation before acting**; (3) widened vprune
  (0.006/0.008) ran clean across a seed change. Artifacts: scratchpad/lad_user1b/ (laduser1bd_1638
  + cbs), result/RWKV[-P]-lad_user1b.jsonl; pipeline template = scratchpad/lad_user1b/
  {run_lad_user1b.cmd,lad_user1b_ws.toml} (vprune-ON candidate runs; exit-42 branch; sequential
  sharded eval + gate in-.cmd).
  ⚠ EVAL-SHARD VRAM LESSON (2026-07-12): 2-parallel-shard eval WEDGES on elevated-VRAM rungs
  (K=32 streams: chunk-state buffers ~+0.8 GB/shard on 1M-token batches -> WDDM oversubscription,
  100% GPU util at 10-50x slow). RULE: such rungs -> sequential shards (get_result resumes
  per-shard) then eval_sharded relaunch-skip-merge; template in run_lad_user1b.cmd.
- **-> NOW: the >=50-iteration RESEARCH PHASE [[research-phase-conduct]]** (many idea FAMILIES,
  arch + training pipeline, lit review + own ideas, retry near-misses as variants). Queued seeds:
  warmup distillation from the d=128 teacher (design in notes), data-driven init (shrink-perturb/
  permutation-init), cross-head readout mix (PHA analog), LIT_REVIEW.md queue. Iter numbering
  continues from 9. Champion unchanged = iter 2 champ5k_b1 (0.306629/0.277893, 193,724 params).
- **★ HP TUNING CLOSED (2026-07-12): champ5k_t1 (the tuner winner: wd 0.01->0.2 + dropout_scale
  1.0->0.5) REJECTED at full eval** -- ahead 0.307174 / imm 0.278570 = WORSE than champ5k_b1 by
  0.000545/0.000677 (p=1.0 both) despite winning tune-eval 5001-5200 by +0.0008/+0.0010.
  **champ5k_b1 REMAINS CHAMPION; its HPs are confirmed vs 19 alternatives** (peak_lr, warmup, wd,
  clip, decay_ratio, adamw_beta2, dropout_scale, cb_lr_mult all settled at champion values on the
  full-eval verdict). ⚠ LESSON (bank + research_log note): the 200-user tune-eval CANNOT resolve
  sub-0.001 HP effects -- even in-subset paired p=5e-8 inverted at n=5000; any future sub-0.001
  tuner verdict needs full-eval confirmation before adoption. Round-2 levers wired + kept
  (RWKV_ADAMW_BETA2 / RWKV_DROPOUT_SCALE / RWKV_CB_LR_MULT, defaults byte-identical). The
  VALIDATION prune (replaced the sign-biased train-loss rule mid-tuning) ran the whole descent
  clean: 0 kills, no false fires, joint-AND correctly spared single-mode transients (incl.
  cb_lr_mult=10's imm-only breach); its estimated-logloss formula is now window-mean x
  fitted-alpha anchored on the baseline journal row (fa724c0). Trial .cmds now GATE every phase
  on exit codes (d289d9a, after a WS crash cascaded into decaying a step-50 ckpt -- caught before
  the journal). NEXT = state-size ladders (deck <=5x -> preset <=10x -> global <=50x, FULL-eval
  gate each rung), then the >=50-iteration research phase [[research-phase-conduct]].
- *(2026-07-08 era below)*
- **★ FIRST 5k CHAMPION PROMOTED (2026-07-08 18:23): champ5k_r1 = ahead 0.306572 / imm 0.278323**
  (quant-aware q72u + per-run learned cbs, n=5000 both modes, eval 5001-10000). Behind the d=128 fp
  target (0.296385/0.264905) by +0.0102/+0.0134 -- THE GAP THE PHASE NOW CLOSES. champion_5k.json
  carries ckpt champ5kd_3277.pth + cb_wkv_final/cb_shift_final + the 13108-step WS trace (= Wilcoxon
  prune ref). Pipeline wall-clock ~7.0h clean (WS 5h @ ~1.36 s/step real, decay 72 min, eval 66 min
  2-sharded, GPU-bound at 2 shards -> 2 stays the default). TWO LATENT BUGS hit+fixed en route:
  (1) LEARN=1 optim resume param-group mismatch at the WS->decay seam (f71f43b -- cb groups now
  register pre-load when the saved state has them, moments resume); (2) per-user lmdb env leak in
  get_benchmark_info killed eval shard 0 at user 2007 with a bogus ENOENT swallowed to exit 0 --
  the n=5000 finish gate caught it (7d095e3 -- env now opened once/process). Results recorded:
  research_log.jsonl + research_5k.md (p-value col = 1.0/1.0 vs target, honest) + log.md rebuilt.
- **★ LIVE LOSS PLOT (2026-07-08, Andrew asked):** `scratchpad/liveplot/liveplot.py` = matplotlib
  window, champion-vs-candidate WS train loss (ahead+imm panels), EMA-smoothed, paired one-sided
  Wilcoxon p + mean delta per panel, warmup-end + decay-start vlines, 15 s refresh. Auto-discovers
  the newest `*_ws_trace.jsonl` (tuner trials AND champion runs both set RWKV_STEP_TRACE), champion
  ref = champion_5k.json embedded trace -> works for ALL runs; switches to a new trial automatically.
  Relaunch: `detach.ps1 -Script scratchpad/liveplot/run_liveplot.cmd` (survives Esc; close window to
  stop). NOTE: WMI-launching pythonw GUI directly stalls at 0 CPU -- use the .cmd wrapper.
- **★ BUDGET A/B RESOLVED + ADOPTED (2026-07-09 01:40): champ5k_b1 = NEW CHAMPION at HALF budget.**
  WS 1 ep (6554) + 0.25 ep decay (1638), otherwise champ5k_r1's exact recipe. Full-eval finals
  **ahead 0.306629 / imm 0.277893** -- paired vs r1: ahead -0.000058 (p=0.31, indistinguishable),
  imm +0.000430 BETTER (p=6.1e-62). The 2nd WS epoch (same 5000 users reshuffled) adds NOTHING
  (data-variety lesson holds at 5k). SIZE/SPEED accept; **1-ep budget now standard for ALL 5k runs**
  (tuner trials AND research runs; champion pipeline ~3.5h: WS 2h27m + decay 37m + eval 89m).
  Adoption executed: promoted (champion_5k.json = ckpt champ5kb1d_1638.pth + its cbs + 6554-step
  trace = the new prune ref), hp_tuner WS_EPOCHS=1, 2-ep journal archived
  (tuner_5k_log_2ep_era.jsonl), new baseline recorded (5001-5200: 0.294490/0.270492), tuner loop
  RELAUNCHED (1-ep era; 2-ep prune verdicts for peak_lr 7e-4/1.4e-3 will be re-tested at 1 ep).
  Pre-ship note: the final champion should get ONE full-budget (2 ep) confirmation run.
- **★ HP TUNING RUNNING (launched 2026-07-08 18:35, detached pid 4468):** hp_tuner_5k `loop` --
  coordinate descent over peak_lr/warmup/wd/clip/decay_ratio, trials are self-recording full-recipe
  .cmds (WS 2ep + decay + tune-eval 5001-5200, LEARN=1 cbs, Wilcoxon-pruned vs champ5k_r1's trace).
  Baseline recorded (5001-5200 subset: 0.294204/0.270881). Journal optimization/tuner_5k_log.jsonl;
  loop log scratchpad/tuner5k/loop.log; ~6h/full trial, prunes much cheaper. Monitor armed.
- **FETCH WORKERS = 4 EVERYWHERE (Andrew 2026-07-08, RAM):** every training/eval launch uses
  NUM_FETCH_PROCESSES=4 (was 7-10; each worker holds ~2.6 GB at MAX=110000, fetch is over-provisioned --
  ~4 ms get() waits; worker count never affects batch content/order). Already set in: hp_tuner_5k
  (NUM_FETCH), write_decay_setup, write_eval_toml, champ5k_r1_ws.toml (the copy-from template for future
  hand-written WS tomls). Check any NEW toml against this.
- **★ EVAL CPU PATH VECTORIZED (2026-07-08, byte-identical):** extract_p / get_stats / run() raw-gathers
  were per-review Python loops (300k-user cost: extract_p 308->118 ms, get_stats 1151->87 ms x2/user);
  now numpy dict(zip)+searchsorted (`_eq_gather`), exact dtypes preserved. Verified: 6-trial exact-equality
  harness (scratchpad/eval_speed/stats_ab.py ALL_PASS) + E2E GPU A/B 3 users = result jsonls BYTE-IDENTICAL.
  RNN/trace callers auto-fallback to the old loop (tensor dicts). champ5k_r1's eval picks it up.
  FOLLOW-UP at eval launch (~16:40): sample per-shard VRAM/GPU-util -> maybe --shards 3-4 for future evals.
- **★ SHIFT-PQ SEARCH KERNEL BANKED (2026-07-08, direction #3): quant-aware step 1.207 -> 0.996 s/step
  (1.21x; stacked 1.65x over NO_JIT today).** ~45% of the q72u step was the learnable shift-PQ search
  running eager torch.cdist().argmin() (sqrt+clamp+argmin over a never-needed ~1.8 GB N x 4096 distance
  matrix, 16 calls/step). New `rwkv7_pq_argmin` CUDA kernel (row-tiled, SUB-templated, first-strict-min
  ties = cdist semantics; 5.9 vs 23.9 ms/call): index-identical on 330k-row + exact-tie tests, QAT
  goldens BITEXACT_PASS after rebuild, escape hatches RWKV_SHIFT_SEARCH_KERNEL=0 (-> matmul tier) /
  RWKV_SHIFT_SQ_SEARCH=0 (-> cdist). CPU tensors auto-fallback (RNN/Rust paths untouched). ⚠ DISCOVERY:
  the compiled frozen env is NOT run-to-run bit-reproducible (3-arm A/B: identical-env controls diverge
  ~step 27; per-step trace noise <=3e-4, weight drift 1.7e-2 @ 110 steps) — bit-exact E2E gates are
  unattainable under it; unit-level index proofs + noise-class drift comparison are the standard now
  (Wilcoxon prune pairing unaffected: zero-mean noise). Wall-clock gap CLOSED (1184 ms GPU-busy / 1207
  wall = GPU-bound; host-side lever dead). Plain step re-profiled 385 ms = flat tail confirmed.
  Champion-run training now ~4.6 h. Details: research_5k_notes.md "Speedups banked" 2026-07-08.
- **★ QUANT RESEARCH CLOSED + FULLY PORTED (2026-07-08).** The sibling (`rwkv-state-quant`) finished its
  bit-descent 2026-07-07: final champion **q72u = 72 b/layer (9-byte card)**, 2-seed-confirmed, details in
  the CHAMPION "DEPLOY config" block above. Its full 2026-07-07 code stack (CUDA joint-uv/norm-quant/warm
  search + train_rwkv QAT wiring + the complete Rust engine) landed here in `1d3b5b8` (the sibling's Claude
  verified byte-identical champion eval from OUR build); the RESULTS layer (champion artifacts ->
  `reference/`, deploy env, methodology-(a) QAT env in `hp_tuner_5k.py`, lesson bank) ported 2026-07-08.
  Open follow-ups from the port: (i) ~~per-run learnable-cb wiring~~ DONE 2026-07-08 (LEARN=1 in QAT_ENV;
  resolve_run_cbs.py repoints env at WS->decay and decay->eval seams; champion_5k.json carries
  ckpt+cb_wkv+cb_shift; a champion's evals/deploys use ITS OWN cbs), (ii) ~~JIT unverified~~ RESOLVED
  2026-07-08 (scratchpad/jitab A/B/C): TorchScript FIXED on the grafted paths (instance-bool shift_pq_on +
  jit.ignore fake_pq_shift + typed kd tuple) but JIT vs NO_JIT is a WASH (1.643 vs 1.658 s/step);
  **ADOPTED + FROZEN 5k-family env = NO_JIT + the sibling's sanctioned round-4 flags (COMPILE=student +
  ROT_CACHE + FAST_EMB + EMA_FOREACH + NO_MEMFILL) = 1.207 s/step (1.37x). Never flip flags inside the
  family. ⚠ COMPILE runs MUST call vcvars64 first (no cl.exe -> inductor errors swallowed by the
  NaN-except as hollow skipped batches, exit 0). q72u-era quant-aware step at MAX=110000 = 1.21 s (the
  old ~450 ms predates joint-search/shift-PQ/learnable cbs); champion run ~= 5.6 h**, (iii) 5k-phase
  state-size gates: card/note budgets should now be interpreted against the 72-b deploy format.
- *(2026-07-03 era below)*
- **★ QUANT PORT DONE (2026-07-03): the sibling's research is FINISHED and its machinery is IN-REPO.**
  Fused QAT CUDA kernels (full-matrix int-N + rank-1 low-rank with PQ branch, 150-490x over the Python
  loop), PQ codebook `reference/pq_cb_m2b8.txt`, shift-QAT (JIT-annotated here; sibling ran NO_JIT),
  int3 + RWKV_QAT_SHIFT_SCOPE, and train_rwkv **LR+WD clobber fixes** (optim load silently restored saved
  lr/initial_lr/weight_decay over config/env -- affected EVERY warm-started run) + non-finite loss/grad
  guards. Validated here: plain path bit-exact vs golden; PQ parity 3.2e-07; int-N 7.5e-04; 25-step QAT
  smoke green (`scratchpad/qat_parity/`). Deploy recipe + numbers: see CHAMPION section "DEPLOY config".
- **★ QAT KERNELS OPTIMIZED 37x (2026-07-03, bit-exact):** see the SPEED section -- quant-aware 5k runs
  are back to ~6-7 h (were headed for ~30-40 h). Profile hook added: `RWKV_PROFILE_STEP=N` +
  `RWKV_PROFILE_COUNT` in train_rwkv -> bucketed kernel self-time summary, then exit.
- **★ TELEGRAM BRIDGE LIVE (2026-07-03):** Andrew can steer this session from his phone + sees mirrored
  output (see Ops). His injected messages arrive Esc-first (interrupt, then message).
- **★ 5k LMDB BUILD RUNNING (launched 2026-07-03, detached, 6 threads):** `scratchpad/run_build_5k.cmd` ->
  6 sequential resumable steps (find_equalize 5001-10000 -> test_db 5001-10000 (F:) -> train_db 1-5000 (C:)
  -> find_equalize 1-5000 -> test_db 1-5000 -> train_db 5001-10000 (F:)); log `scratchpad/build_5k.log`;
  ~2-4 days. Eval data for 5001-10000 lands FIRST so the d=128 baseline eval can start before the train_dbs
  finish. Monitor via OS truth; the 6 configs are `rwkv/*_5k_*.toml` (PROCESSES=6).
- **★ EVAL SHARDING READY (2026-07-03, Andrew-approved):** `optimization/eval_sharded.py --config
  <eval toml>` = 2-process size-balanced (LPT) full eval, ~1.5-2x wall-clock, numerics-IDENTICAL
  (additive USERS_FILE selector in get_result; merge + means printed). d=32 evals only (two d=128s
  OOM); E2E smoke pending -- watch the first champion-era sharded eval. Details in notes.
- **★ BASELINE-TO-BEAT LANDED (2026-07-03): d=128 on 5001-10000 = ahead 0.2964 / imm 0.2649**
  (0.296385/0.264905, n=5000 both modes, fp unquantized; consistent with the published 10k-pooled
  0.29743/0.26600; recorded in research_5k.md; result jsonls result/RWKV-base5k*.jsonl; arch restored).
- **⚠ GPU HOLD (Andrew 2026-07-04): do NOT launch GPU training/evals — he is running his own quant
  experiments. Champion run waits for his GO.**
- **★ STEP3 DONE 2026-07-04 07:00 (train_db_5k_h1 complete, exit 0; STEP4 find_equalize 1-5000 running).
  `count_groups_5k.py` run: GROUPS_PER_EPOCH = 6554 → groups_5k.json (hp_tuner prereq DONE). Champion-run
  arithmetic: 2 WS ep = 13,108 steps + decay 0.2–0.8 ep → total ~14.4k–18.4k steps ≈ 1.8–2.3 h clean.
  EVERYTHING for the champion run is staged — only the GPU hold gates it.**
- **★ TONIGHT'S DIRECTION (Andrew 2026-07-08, supersedes the NEXT list below where they differ):**
  (1) ADD CODEBOOK LEARNING to 5k runs (per-run learnable cbs: train with RWKV_QAT_PQ_LEARN=1 +
  RWKV_QAT_SHIFT_PQ_LEARN=1, export each run's learned cbs, point that run's quant-aware EVAL + any
  deploy at ITS OWN exported cbs — the promote/champion flow carries cb artifacts with the ckpt);
  (2) TURN JIT ON (A/B TorchScript on the grafted q72u paths: parity + speed; drop RWKV_NO_JIT if clean)
  -> compaction about here; (3) hunt any remaining speedups (profile the q72u quant-aware step — joint
  search / shift-PQ / norm paths are new surface; check the sibling's speed-round flags for portable
  wins); (4) FIRST REAL 5k CHAMPION RUN (champion-HP, quant-aware, RWKV_STEP_TRACE -> promote);
  (5) HP TUNING (hp_tuner_5k); (6) STATE-SIZE KNOBS in this order, each until gain <0.0003 (the phase
  threshold) or its ceiling: deck up to 5x -> preset up to 10x -> global up to 50x. **RULE (write-down,
  Andrew 2026-07-08): card and note state sizes REMAIN FIXED — the only exception is an architectural
  change that makes a card/note state-size change INEVITABLE (not a tuning knob, a structural
  consequence).** (7) then any architectural improvements at my discretion (queued ideas: warmup
  distillation, data-driven init, cross-head readout mix, LIT_REVIEW).
- **NEXT (per methodology g), in order once data allows:** (1) ~~d=128 baseline eval~~ DONE (above);
  (2) ONE champion-HP 5k run with per-step WS trace (RWKV_STEP_TRACE) + quant-aware forward -> promote via
  `promote_champion_5k.py`; (3) HP tune -- `hp_tuner_5k.py` REPOINTED to FULL 5k 2026-07-03 (train 1-5000
  @ MAX=110000, tune-eval 5001-5200, QAT env in every trial's WS+decay+eval, proxy-era journal archived to
  tuner_5k_log_proxyera.jsonl; PREREQ after STEP3: `python optimization/count_groups_5k.py` -> groups_5k.json).
  ALL live 5k tooling now trains on 1-5000 and evals on 5001-10000 ONLY (verified sweep 2026-07-03); the
  100u/1500u dbs are no longer referenced by anything live (kept on disk, C: has 383 GB free). Any TIMING
  numbers taken while build workers run are fetch-contaminated; take final numbers with the build idle.
- Queued analysis (task #18, Andrew 2026-07-03): **irreducible-entropy estimate** -- cross-model
  residual covariance of the TWO disjoint-trained d=128 .pths on users 1-100 (seen by neither) ->
  irreducible-Brier -> Beta-translated LogLoss floor; + constant-retention baselines H(p-bar).
  Design in notes "Queued analysis" section; needs build STEP4+5 (test data for 1-100); ~30 min GPU.
- Queued research ideas: data-driven init (shrink-perturb / permutation-init, post-HP-tune -- notes
  "Queued idea" section); **warmup-only distillation from the d=128 teacher** (Andrew 2026-07-03: soft
  targets from `RWKV_trained_on_101_4999.pth` for the first ~200-800 steps only, annealed 1->0, then hard
  labels so the student can surpass the teacher; STORED-dump design -- teacher+student can't share a
  process (module-level arch config) -- full design + gate fit in the notes "Queued idea" section;
  post-HP-tune; test SEPARATELY from data-driven init, both touch early training); cross-head readout
  mix (PHA analog, LIT_REVIEW, low-med). Lit-review queue: `optimization/LIT_REVIEW.md`. Everything
  through the quant port is COMMITTED + pushed (local == GitHub).

### Ops
- **Compaction (ONLY sanctioned way):** run `claude-automation/request_compact.ps1 -Focus "<carry-through>"`
  then yield idle and STOP beating the heartbeat. `/compact <focus>` fires only from a FRESH (<=30 min) +
  focus-bearing flag (stale/empty = purged). Never hand-create `pending_compact.txt`. The injector is 24/7
  (ClaudeLoopController every 3 min; acts only on a stale heartbeat) and may inject EXACTLY `/compact <focus>`
  or a short `Continue` -- nothing else Claude-originated. (Since 2026-07-03 the **Telegram bridge**
  (`claude-automation/telegram_bridge.py`, task `ClaudeTelegramBridge`) additionally injects messages
  AUTHORED BY ANDREW from his authenticated Telegram account + mirrors chat output to his phone -- human
  steering, not self-injection. Master switch `telegram_bridge_active.txt`; see automation README.)
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
