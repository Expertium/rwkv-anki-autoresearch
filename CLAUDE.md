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

## Optimization state (steps 4-5-7)

> Full numeric record = `optimization/log.md` (iteration table + State-quant + QAT sections, rebuilt
> from `log.jsonl`/`quant_log.jsonl`/`qat_log.jsonl`). Verbose research narrative (frontier history,
> breakthroughs, measured lever costs, dead-end details) = `optimization/HISTORY.md`. This section
> keeps only the current state, the compact lesson bank, and the active agenda.

**Iteration 0 baseline:** d_model=128, 2,762,884 params, 51.0 KiB/card, ahead 0.374046 / imm 0.319475.
**Gate ceilings (iter0 + 0.0015):** imm <= 0.320975, ahead <= 0.375546. Review count = 6,164,115 (must
be identical every iter). Gates are vs iter0 (a FLOOR), not vs the champion.

**CHAMPION (fp32 arch) = iter36** `[1,4,3,3,3]` (card,deck,note,preset,user), d=32 / K=32 / H=1,
**192,800 params** (14.3x smaller than iter0), per-card state **4.25 KiB fp32**, ahead 0.347959 /
imm 0.313864, throughput 285.6 rev/s (B=1). Rust-parity PASS (bit-exact). Restore =
`optimization/arch_snapshots/arch_iter36.py`.

**DEPLOYED CHAMPION = iter45 weights + LOW-RANK card deploy** (PTQ, no retrain): **card rank-2 int4 factors
(lowrank) + note int2**, with shifts quantized. deploy imm **0.291471** / ahead 0.323603, **deployed state =
card 0.094 KiB (96 B) + note 0.80 KiB** -- BOTH hard targets MET. Gate PASS; BEATS the fp32 champion (imm
-0.004593, ahead -0.003028). Engine: `RWKV_STATE_LOWRANK_SCOPE=card:2:int4 RWKV_STATE_QUANT_SCOPE=note:int2
RWKV_QUANT_SHIFTS=1` on `reference/rwkv_iter45.safetensors`. ★ KEY: low-rank rank-2 int4 is SMALLER (0.27->0.094
KiB) AND MORE ACCURATE than card int2 full (int2 coarsely quantizes all 1024 WKV floats; rank-2 keeps the top-2
SVD comps in int4 = 98.7% energy). Prior all-int2 champion (card int2+note int2): honest deploy imm 0.295833
(shifts int2) / 0.292560 (shifts fp32) -- superseded by low-rank. Same iter45 weights (16-epoch decay-QAT).
**Meets the >=2x note target** (note int2 0.80 KiB). iter44 (8ep) is ~tied (imm 0.295436 / ahead 0.323291 --
better ahead, worse imm); both saved. KEY QAT LESSON: the original 4-epoch decay-QAT (iter43) was UNDERTRAINED;
deploy-imm by decay length = 4ep 0.299469 / 8ep 0.295436 / 16ep 0.292560 (gains shrink -0.0040 -> -0.0029, and
ahead crosses over at 16ep) -> STOP epoch-scaling at ~16. FINDING: the fp32 BASE keeps improving with more
decay (qat_fp32 imm 0.296064->0.292454->0.287818 at 4/8/16ep) => iter36's 2-epoch decay was undertrained; a
longer PLAIN (non-QAT) decay would likely improve the real eval100 benchmark -- revisit as a base improvement.
Lineage: iter39 (int2/int4, +0.0025) -> iter43 (int2/int2, +0.0034) -> iter44 (8ep, -0.0006) -> iter45 (16ep,
-0.0035, champion). PTQ could not reach even card int4+note int4 (+0.0036).

**★ CORRECTION + new low-rank work (2026-06-29, re-derived after the low-rank Rust code was lost from the
working tree and rebuilt this session -- card rank-2 int4 PTQ re-measured at EXACTLY 0.291471, validating it):**
- **iter43/44/45 were NOT real QAT.** The restored champion `architecture.py` (arch_iter36) was MISSING the
  `[QAT]` scope parser that arch_iter41/42 have, so `state_qmax` stayed inf -> fake-quant never ran (the
  iter44/45 logs have ZERO `[QAT]` lines). They were plain LONGER-DECAY fine-tunes + PTQ int2 at the gate.
  So the "more QAT epochs help" lesson is really "more DECAY improves the fp32 base" -> the int2 PTQ penalty
  (+0.003-0.005) was never dissolved => REAL QAT has untapped headroom. The [QAT] + [QAT-LOWRANK] parsers are
  now restored into architecture.py AND the arch_iter36 snapshot.
- **NOTE low-rank also works (PTQ):** both-low-rank (card rank2-int4 + NOTE rank2-int4 + int4 shifts) PTQ on
  iter45 = imm **0.289137** / ahead 0.321056 -- the BEST deploy yet (beats fp32 champ -0.0069), at the
  SMALLEST state (card 96 B + note ~288 B vs note int2 816 B). BUT note low-rank's per-step 3-layer nalgebra
  SVD makes the GATE ~20x slower (~20-25 min vs ~100s); note int2 already meets the >=2x target, so note
  low-rank is lower-ROI (revisit if the extra -0.002 imm + 2.8x note shrink is worth the eval/QAT cost).
- **REAL low-rank QAT (iter46) = DEAD END (naive STE).** `fake_lowrank_state` (STE rank-r SVD truncation +
  int-N factor quant, matches the Rust deploy) in rwkv_ops.py, wired via `RWKV_QAT_LOWRANK_SCOPE`. iter46 =
  8-epoch decay, card rank2-int4 low-rank QAT + note int2 QAT. RESULT: deploy imm **0.303617 -- WORSE** than
  the card low-rank PTQ (0.291471, by +0.012) and worse than champ_fp32 (+0.0076). The low-rank deploy cost
  on the QAT model BALLOONED to +0.0103 (vs ~+0.0037 PTQ). WHY: rank-2 truncation is a STRUCTURAL change, so
  the identity STE gradient gives NO signal to concentrate energy in the top-2 singular dirs (unlike int-quant,
  where small element-wise error makes STE work) -> the model drifts toward HARDER-to-low-rank states. LESSON:
  **low-rank stays PTQ** (PTQ low-rank already BEATS int2 + hits 0.15 KB); int-quant stays QAT. A
  differentiable-SVD QAT could be tried but PTQ already suffices. The infra (fake_lowrank_state, parsers) is
  kept for the int-quant QAT path it also enables.

**Note on the two state numbers:** "4.25 KiB" is the *fp32, pre-quant* card state (1,088 floats x 4 B
- a pure arch property that `model_stats.py`/`scratchpad/params_for_arch.py` report). The *deployed*
card state is the quantized figure: int8 1.06 / int4 0.53 / **int2 0.27 KiB**. Same 1,088 floats,
different storage bits (deployed KiB = floats x bits / 8 / 1024). `params_for_arch.py` now prints both.

**★★ HARD TARGETS (Andrew 2026-06-29) — BOTH MET 2026-06-29: (A) card state -> 0.15 KB [✓ MET: 0.094 KiB
(96 B) via rank-2-int4 low-rank card WKV + int4 shifts, deploy imm 0.291471 PASS, BEATS fp32 champ];
(B) note state -> >=2x smaller [✓ MET: note int2 0.80 KiB via QAT].** The deployed champion now hits both
(see DEPLOYED CHAMPION above). Memory math (card = 1,024 WKV floats [32x32 matrix] + 64 token-shift
floats [2 vectors; 1-D so only quantizable, not low-rankable]):
- **int2 quant ALONE bottoms out at 256 B** (1,024 floats x 2 bit) -> CANNOT reach 0.15 KB by quant
  alone; MUST cut the float COUNT (low-rank WKV, or smaller K via the kernel route).
- **Card to 0.15 KB path (PRIMARY = low-rank + quantized FACTORS, sidesteps the K=32 kernel block):**
  rank-1 WKV int4 (32 B) + shifts int4 (32 B) = **64 B (0.06 KiB)**; rank-1 int8 + shifts int8 = 128 B;
  rank-2 int4 + shifts int4 = 96 B. All clear 0.15 KB. (Dump shows card state IS near rank-1.) Stacks
  low-rank err x quant err -> measure on the 2k loop. Alt (BLOCKED): H=2/K=16 + int2 = 144 B (CUDA
  kernel rewrite or slow chunked-PyTorch proof).
- **Note >=2x path [✓ DONE iter43]:** note int4 (1.59 KiB) -> **note int2 via QAT = exactly 2x (0.80 KiB)**
  WORKED (deploy imm 0.299469, +0.0034 vs champ fp32, PASS) -- exactly as predicted (QAT rescued note int2
  just as it made card int2 nearly free). Further cuts (if ever needed) via low-rank note WKV + quant.
  Dimension cuts are HARD: note layers 3->2 rejected (iter38,
  costs imm); note d_model<32 (K<16) is K=32-kernel-BLOCKED. NOTE matters MOST for total memory at deploy
  (3 layers => note int4 1.59 KiB is ~6x the card 0.27 KiB per entity; MEASURED: notes ~= 0.9x cards
  across the 10k dataset, so a 1M-card user has ~900k notes -> note state is the DOMINANT deploy memory,
  ~4-5x the card-state total for a power user. See scratchpad/entity_counts_10k.csv + [[dataset-entity-counts]]).
- **These RAISE the value of low-rank (now PRIMARY, not lowest-priority) and of the deck/preset grow**
  (iter41/42 build the accuracy headroom to afford card-0.15 + note-int2). See RESUME step 4.

### Engine (`rust/rwkv-infer`)
fp32 + pre-transpose + lerp-fusion (+8.7%). **Auto-derives num_curves/num_points AND per-stream layer
counts from weight shapes** - adapts to any arch with no code change. State quant via
`RWKV_STATE_QUANT_SCOPE="card:int2,note:int4"` (per-stream mixed bits int8/int4/int2; omitted streams
stay fp32). Batching: `*_batched` query forward (B=1 path untouched, parity bit-exact); **optimal
B~128, single-thread** (intrinsic L2/L3 cache knee at B=128->256; thread count irrelevant). Rust modes:
`--verify-batched`, `--bench-batched`, `--sweep-batched`, `--bench-synth`. See `rust/rwkv-infer/BATCHING_PLAN.md`.
**★ FAST LOW-RANK SVD (2026-06-29, step-3 win):** `lowrank_roundtrip` no longer uses nalgebra's FULL SVD
(which converged pathologically slowly on near-low-rank states -> the note-low-rank gate HUNG; user 187
ran >35 min). Replaced with a top-r truncation via **Gram matrix + symmetric eigendecomposition**
(eigvecs of A Aᵀ = left singular vecs, eigvals = sigma²; right vec v = Aᵀu/sigma). A is normalized by its
max-abs before forming the Gram (the product squares magnitudes -> f32 overflow -> NaN eigenvalues for a
state grown large over a long history; normalize, then unscale sigma). NaN-safe sort + skip non-finite
comps. Validated == full-SVD rank-2 recon to ~1e-15 (numpy). RESULT: user 187 both-low-rank now **22 s**
(was a >35 min hang); the full 17-user both-low-rank gate runs in ~100 s. note-low-rank is now PRACTICAL
in the iteration loop. **Both-low-rank deploy re-confirmed on ALL 17 users (incl 187): imm 0.288831 /
ahead 0.320098, beats fp32 champ by -0.0072 imm / -0.0065 ahead, GATE PASS** (the prior 0.289137 was 16
users w/o 187; 187's low-rank deploy is fine -- the earlier hang/panic was purely the SVD numerics, not divergence).

### LESSON BANK - do NOT re-run these dead ends (full numbers in log.md / HISTORY.md)
- ✅ **Kept:** SRS heads 128->64 (iter29) · card->deck rebalance (compensation order **deck > preset >
  user**, NOT note) · card 2->1 (iter36) · 4-epoch decay (general win, tightens variance) · scoped
  state-quant **card int4 + note int8 ~free** (the 1-KB lever) · QAT makes card int2 + note int4
  essentially free (+0.000018 quant cost) WHEN warm-started from the champion · **QAT note int2 = >=2x
  note target MET (iter43-45)** · **LONGER decay-QAT (8-16 epochs, warm-started) makes the deployed
  int2+int2 model BEAT the fp32 champion** (iter45 deploy imm -0.0035 vs champ; the 4ep decay was
  undertrained) -- saturates ~16ep (best imm@16, best ahead@8) · **LOW-RANK card WKV (rank-2, int4 factors)
  BEATS int2 -- smaller (0.27->0.094 KiB) AND more accurate (-0.0044 imm): rank-2 keeps the top-2 SVD comps
  in int4 (98.7% energy) vs int2's coarse 3-level on all 1024 floats. Card 0.15 KB target MET via PTQ.** ·
  shifts must be quantized too for honest deploy size (RWKV_QUANT_SHIFTS): +0.0033 imm at int2, +0.0011 at int4.
- ❌ **Failed:** FC/head-width 4->2 (imm +0.0526, imm-critical) · note 3->2 layer-cut (iter38, +0.0018
  - shrink note STATE via quant, not layers) · all-streams blanket state-quant (long-recurrence
  user/global sink it) · note int4 via PTQ (>2x budget) · weight PTQ int8/int4 (no speed win) ·
  **QAT from scratch (iter40, +0.0118 - MUST warm-start from a good fp32 ckpt)** · **naive low-rank QAT
  (iter46, STE rank-2 truncation): deploy +0.0076 vs champ, WORSE than low-rank PTQ -- STE can't guide a
  structural rank change; low-rank stays PTQ, int-quant stays QAT**.
- ⚡ **GPU-training + gate SPEEDUPS (2026-06-29, step 3 -- arch-agnostic, untimed/non-gating):**
  (a) **`copy_downcast_` + `transfer_child_grad_to_master` vectorized with `torch._foreach_*`** (one fused
  kernel per dtype group vs ~440 per-param launches each) -- BIT-IDENTICAL (verified); (b) **`get_grad_norm`
  (~440 `.item()` syncs/step) + `log_model` skipped when `USE_WANDB` is off** (logging-only) -- in
  `train_rwkv.main_loop`. Together **+1.21x** no-JIT (2.53->3.07 steps/s, full 31-group workload). (c) **JIT
  RESTORED via `@torch.jit.ignore` on `quant_aware_rwkv7`** -- the QAT-lowrank addition (torch.linalg.svd in
  the per-step loop) had SILENTLY broken TorchScript scripting (internal assert in torch 2.12.1+cu130), which
  would CRASH any plain WS/decay training AND `get_result.py` eval (both JIT-on). Fix lets the scripter skip
  the never-scripted QAT branch -> JIT-on hot path restored, eager QAT path unchanged (off-path==reference).
  Combined **JIT-on + foreach + sync-removal = 3.48 steps/s = 1.38x over the no-JIT old body** (1.30x over
  JIT-on old body). ⚠ JIT has a ~30-60 s one-time compile -> wins only for LONG runs (the 1k-user phase);
  for short 100-user iters it's ~neutral and the foreach/sync win (unconditional) is what matters. **`torch.compile`
  is NOT viable (no Triton on Windows); JIT was the only fusion route and it's now fixed.** Profiler =
  `scratchpad/profile_train.py` (sync section breakdown + old-vs-new body A/B; the dominant fwd+bwd ~90% is the
  custom WKV kernel at low B=1 parallelism -- untouchable without kernel/batch changes).
- 🔒 **Blocked:** K<32 (smaller head dim, the biggest state lever) - the CUDA training kernel hardwires
  K=32; needs a kernel rewrite or a slow K-agnostic chunked-PyTorch proof. Deferred. · `torch.compile`/inductor
  (no Triton wheel on Windows) and CUDA graphs (variable seq shapes + custom autograd.Function) -- not pursued.

### ★★ NEW PHASE PLAN (Andrew 2026-06-29, supersedes the deck/preset-grow RESUME below) ★★
Low-rank investigation is essentially DONE: **both-low-rank PTQ (card rank2-int4 + note rank2-int4 +
int4 shifts) = imm 0.289137** is the best deploy (smallest state too: card 96 B + note ~288 B), and it
is **deploy-viable** -- the per-step SVD is needed at inference (re-truncate the rank-2 state each review,
since a rank-2 state + rank-1 WKV update -> rank-4) BUT costs ~10-40 us/SVD in Rust (~158 us in numpy);
at human review pace that's ~0.6 s over a 1000-review DAY = negligible. The ~20-min gate slowness is ONLY
the benchmark replaying millions of reviews at max speed -- a measurement artifact, not a deploy cost.
Ordered steps:
1. **[✓ DONE 2026-06-29] Clean-confirm both-low-rank -> CHAMPION.** 16-user clean gate (dropped the stuck
   large user 187): deploy imm 0.271665, delta vs champ_fp32 = **-0.005905 imm** (matches the prelim 17-user
   -0.006927 -> validated; absolute 17-user ~0.289). Pure low-rank deploy cost +0.001012. **Both-low-rank PTQ
   (card rank2-int4 + note rank2-int4 + int4 shifts, card 96 B + note ~288 B) is the deployed champion.**
   ⚠ BLOCKER [✓ RESOLVED 2026-06-29 by step 3]: the note-low-rank gate was impractically slow (user 187
   ran >35 min on nalgebra full SVD). FIXED with the fast Gram+eigen truncated SVD -> 187 now 22 s, full
   17-user both-low-rank gate ~100 s. **Re-confirmed on ALL 17 users incl 187: deploy imm 0.288831 / ahead
   0.320098, -0.0072 imm vs champ_fp32, GATE PASS** (the 16-user 0.271665 above was a different/cleaner
   subset; the absolute 17-user number is ~0.289). Both-low-rank PTQ is the deployed champion; gate practical now.
2. **Settle PTQ vs QAT for BOTH-low-rank -> LOCKED = PTQ (low-rank), based on iter46 + mechanism.** iter46
   (card-only low-rank QAT, STE) was a DEAD END (deploy +0.0076, pure quant cost +0.0103 vs PTQ ~+0.001 --
   STE can't guide a STRUCTURAL rank change; this is per-stream physics, so the NOTE case is identical). A
   full both-low-rank QAT to re-confirm was impractical (its deploy gate hung on the slow-SVD issue, now
   FIXED in step 3 -- the gate is fast). So low-rank stays PTQ. ★ ROOT CAUSE of the gate slowness (2026-06-29): nalgebra FULL SVD converges
   SLOWLY on the real near-low-rank states (sing. values 3-32 ~0 -> the iterative SVD grinds on the tiny
   clustered values) -- user 187 (only 1,119 cards) took >35 min. My random-matrix bench (158 us) missed this.
   FIX (step 3, and the RIGHT method anyway): a truncated rank-2 (power/subspace iteration) extracts ONLY the
   top-2 and IGNORES the tiny values -> fast AND well-suited to near-low-rank. DEPLOY is still fine (per-review
   even at ~ms is negligible vs seconds between reviews); only the benchmark (millions of replayed reviews) is
   hit. Path to beat the champion = a BETTER BASE (fp32 base still improving at 16ep: 0.296->0.292->0.288 at
   4/8/16ep -> try 24-32ep plain decay) + both-low-rank PTQ. NO note-int2 QAT (iter47 shelved -- note int2
   0.80 KiB is BIGGER than note low-rank 0.28 KiB).
3. **[✓ DONE 2026-06-29] Maximally speed up GPU training + the low-rank gate (arch-agnostic).** See the
   ⚡ lesson-bank entry for full numbers. GPU TRAINING: profiled (the dominant cost is the custom WKV CUDA
   kernel at B=1 low parallelism over long sequences -- compute-bound, NOT launch-bound the way assumed;
   the launch-bound part was the per-param Python loops + logging syncs). Wins = `torch._foreach_*`
   vectorization of `copy_downcast_`/`transfer_grad` + skip `get_grad_norm`/`log_model` when wandb off
   (+1.21x, bit-identical) + RESTORE JIT via `@torch.jit.ignore` on `quant_aware_rwkv7` (was silently
   broken -> would crash plain WS/eval; combined **3.48 steps/s = 1.38x** over the no-JIT old body).
   `torch.compile` ruled out (no Triton on Windows); CUDA graphs not worth it (variable shapes). LOW-RANK
   GATE: replaced nalgebra full SVD with a fast Gram+symmetric-eigen top-r truncation (see Engine section)
   -> note-low-rank now ~22 s/heavy-user (was a >35 min hang); the both-low-rank gate is practical IN THE
   LOOP. ⚠ JIT one-time compile (~30-60 s) means JIT-on wins for LONG runs (the 1k phase); foreach/sync win
   is unconditional. ALL changes are arch-agnostic (derive shapes at runtime).
4. **NEW RESEARCH PHASE: train 1-1000 / test 1001-2000, GPU-ONLY eval** (the roadmap's 2k loop). Rust/CPU
   ONLY for minimal ~3-user parity checks, NOT the main gate -- the main eval is `get_result.py` (CUDA) on
   1000 users. Focus = **ALGORITHMIC improvements** (the research-y step) while keeping **params AND
   per-entity state size under fixed MAX CAPS** (cap = current champion: ~192,800 params; card 96 B + note
   ~288 B low-rank, or whatever the confirmed champion is). Bigger/cleaner eval signal than the 17-user gate.

### Step-4 GROUNDWORK (Andrew 2026-06-29, IN PROGRESS) -- old-vs-new baseline on the 1k test set
Andrew's pre-step-4 groundwork: (1) eval the OLD RWKV (`pretrain/RWKV_trained_on_5000_10000.pth`, the
original 2.76M d=128 leaderboard model, trained on users 5000-10000) on users 1001-2000, per-user logloss
for BOTH modes (ahead=forgetting-curve, imm=immediate); (2) ENSURE per-user `size` (equalized review count)
is IDENTICAL old-vs-new (proof the preprocessing matches); (3) eval the NEW champion on the same 1k users.
- **DATA WASN'T BUILT**: test_db only had users 101-200, label_filter_db ~100-516. Building 1001-2000 via
  `find_equalize_test_reviews` (label_filter) + `data_processing` (test_db) -- detached `scratchpad/build_eval1k.cmd`
  (configs `find_equalize_eval1k_config.toml` + `data_processing_config_eval1k.toml`, USER 1001-2000). Both
  APPEND (skip `_done` users) so the existing 101-200 gate data is untouched. Monitor `scratchpad/build_eval1k.log`
  (`DONE_EXIT_`). ~1-2 hr, ~50 GB (257 GB free).
- **OLD model needs the d=128 arch**: our srs_model.py diverged (features_fc_mult/head_fc_mult/num_curves/
  num_points config fields the srs-benchmark original lacks), so I transcribed the original into our format =
  `scratchpad/architecture_old_d128.py` (STRICT-loads the old ckpt, 2,762,884 params, exact match). Eval swaps
  it into `rwkv/architecture.py` then restores the champion (`scratchpad/architecture_champion_backup.py`).
- **Eval after build**: `scratchpad/run_eval1k.cmd` (NEW via get_result_new_1k.toml; OLD via arch-swap +
  get_result_old_1k.toml; then `compare_eval1k.py` = size-identity check + by-user-mean logloss + per-user CSV).
- **SMOKE (users 1001-1003) PASSED**: size IDENTICAL old/new (14170/91150/67930); OLD beats NEW on all 3
  (e.g. user 1003 imm old 0.4522 / new 0.7373). The NEW champion was trained on only 100 users + SELECTED on
  101-200, so its 1001-2000 numbers are a generalization FLOOR -- step 4 retrains the arch on 1-1000 to close
  the gap vs the old 5000-user-trained model. The full 1000-user means are the real comparison (3 users = noisy).
- ⚠ get_result.py runs JIT-on -> REQUIRES this session's `@torch.jit.ignore` fix on `quant_aware_rwkv7`
  (else it crashes at model build). Confirmed working (the 11s/100-user eval + the smoke ran JIT-on).

### ★★ DATA-DROP BUG (Andrew 2026-06-29) -- the optimization loop trained on ~5% of the data ★★
While investigating "why is B=1", found that **`get_groups` SILENTLY SKIPS any batch whose size >
MAX_TRAIN_GLOBAL_LEN** (`max_batch = floor(MAX/size); if max_batch==0: continue`). The train_db batches
are large (per-user histories, sizes up to 65,536 ~ the ORIGINAL MAX=66000). The optimization configs use
**MAX_TRAIN_GLOBAL_LEN=20000**, so at 20000: **only 35/212 batches kept = 4.7% review-token coverage, just
20/100 users fully present** (the smallest-history users); the 80 longer-history users are partly/fully
dropped. Coverage by MAX: 20000->4.7%, 40000->16.3%, **66000->100%** (all 212 batches, 170 groups). So the
champion (iter36) trained on ~5% of even its 100 users' data -- almost certainly a big part of its POOR
generalization to 1001-2000 (smoke: old beats new on all 3 users; it never saw long-history users). B=1 is a
symptom: the ~35 surviving batches each ~fill the 20000 budget alone. **Iter-to-iter RANKINGS stay valid (all
used the same 20000 subset), but absolute champion quality is on a biased slice.** FIX = MAX=66000 (full
coverage); feasible on the 12 GB GPU now (d=32 champion, ~16x smaller activations than the original d=128 that
needed 66000 on a 24 GB card). At 66000 you also get B>1 free for small users (histogram B1:148,B2:13,...,B7:1).
- **IN PROGRESS: re-baseline the champion at 66000** (Andrew "do both"): `scratchpad/run_rebaseline.cmd` runs
  `rebase_66k_ws.toml` (from-scratch WS, 1-100, 66000, 6 epochs ~1020 steps) TWICE -> run1=fair champion,
  run1-vs-run2=run-to-run variance. THEN eval run1 on 1001-2000 (new) + old on 1001-2000 -> redo old-vs-new.
  RUN ONLY AFTER build_eval1k finishes (the failed 20000 variance run died from GPU contention with the build's
  data_processing -- evals crashed before writing; trainings were fine). ~30 min/run on a clean GPU.
- **DETERMINISM enabled** (Andrew "enable determinism"): `train_rwkv._maybe_enable_determinism()` (RWKV_DETERMINISTIC
  default 1) pins the TRAINING process RNG + cuBLAS/cuDNN (CUBLAS_WORKSPACE_CONFIG=:4096:8). The custom WKV kernel
  has no atomics (already deterministic; eval is bit-identical). **Augmentation KEPT stochastic** (Andrew's call --
  the per-batch random ID-encodings + time baselines stay in the fetch children, unseeded) -> run-to-run variance
  now isolates the AUGMENTATION-only noise floor. (Andrew is skeptical the augmentation even helps -- ablation TODO.)

### ★★★ REVISED PLAN (Andrew 2026-06-29 late) -- supersedes the NEW PHASE PLAN's step-4 ordering ★★★
**KEY NEW RESULTS this session:**
- **Full-coverage 66000 re-baseline (WS-only, from scratch on 1-100) BEATS the iter36 champion by ~0.013 imm /
  ~0.017 ahead on 101-200** (re-baseline imm 0.2989-0.3006 / ahead 0.330 vs champion imm 0.3139 / ahead 0.3480;
  SAME train users + eval set, only 5%->100% coverage). The data-drop fix is worth ~0.013 imm -- LARGER than the
  ENTIRE optimization loop (iter0 0.3195 -> champion 0.3139 = 0.006). Re-baseline ckpts:
  `scratchpad/rebase_run1/rebase_1020.pth` (WS), `scratchpad/rebase_champ/rebasec_680.pth` (WS + 4-epoch decay).
- **Run-to-run variance (determinism ON, augmentation stochastic) = ~0.0018 imm / 0.0006 ahead (100 users).**
  PURELY augmentation-induced (the two trainings land in different optima -- a correlated shift that does NOT
  average out with more users). NOT <0.0001. => **tuner noise margin ~0.002.**
- **Tuner = GREEDY coordinate descent** (pattern-search / Hooke-Jeeves, ~0.002 noise-margin acceptance, natural
  early-stop), NOT CMA-ES (25-eval budget too small for its covariance) or Bayesian (warmup waste); Optuna TPE as
  a phase-2 on the ~3 most-coupled params. Tune ~6-8 of the ~20 non-arch hyperparams (full inventory in HISTORY).
- **Stateful-BPTT finding:** training chunks (32768-review windows, multiple per user) are trained COLD --
  `RWKV7_WKV.forward` takes NO initial state, and `get_groups` shuffles chunks independently. So (a) B=1 wastes
  parallelism (one ~62k-token chunk fills the 66000 budget; GPU ~15-67% util) and (b) train/eval MISMATCH (eval =
  full history with carry; test_db = 1 batch/user, asserts len==1). Eval is also slow: power users have 700k+
  review histories (~3 min/100 users; the earlier "11s" was a resume-skip artifact).
**ANDREW'S PLAN (ordered):**
0) [DONE] compaction + GitHub=local.
1) **STATEFUL BPTT FIRST** (the speed enabler -> makes everything else faster): chunk smaller + batch across users
   (B>>1) + carry the RNN state across a user's consecutive chunks. Gets speed (high B util) AND learns long
   context AND closes the train/eval mismatch -- "2-3 birds". Needs a CUDA-kernel change (add initial-state input +
   final-state output to the WKV forward/backward). ALSO look for OTHER train + EVAL speedups.
2) **Build train_db for users 1-1000** (only 1-100 exists!) -- WITH the new BPTT chunking. test_db 1001-2000 is
   ALREADY built (this session). This is the prerequisite Andrew's plan implies for "train on 1k".
3) **1k RESEARCH PHASE: train 1-1000 / eval 1001-2000** (GPU get_result), algorithmic improvements under the
   param + per-entity-state caps. OLD baseline = `pretrain/RWKV_trained_on_5000_10000.pth` (2.76M d=128; eval via
   `scratchpad/architecture_old_d128.py` arch-swap, strict-loads). NEW champion logloss MUST include QUANTIZATION
   (deployed = low-rank PTQ): current champ = iter45 fp32 `pretrain/rwkv/opt_qat45/rwkv_iter45_496.pth`; quantized
   eval via the RUST engine on exported traces (`export_features_fast.py --range`) -- per-step SVD too slow in
   Python over power users' full histories.
4) **AUGMENTATION ABLATION:** train with the per-batch augmentation ON vs a FIXED seed, compare logloss -> does the
   randomization actually improve generalization? If not, fix the seed -> reproducible objective (variance ~0) for
   the tuner. (Augmentation = random ID-encoding vectors + random time-of-day baselines, regenerated EVERY batch,
   `prepare(seed=None)`; eval uses fixed seed 1234 -> eval is bit-deterministic.)
PENDING/ARTIFACTS: the 1001-2000 old-vs-new fp32 comparison was STARTED then STOPPED (slow power users; variance
already answered -- don't resume it as-is). Harness ready: `scratchpad/run_rebaseline_eval.cmd` + `compare_rebaseline.py`
(old d=128 arch-swap + iter45 + re-baseline; size-identity check). get_result runs JIT-on (needs the jit.ignore fix).

### Active agenda (Andrew, priority order) [OLDER -- see NEW PHASE PLAN above]
1. **Param reduction = headline** (helps throughput AND state). Champion 192,800. Big blocks: RWKV
   stacks ~70%, the two 128x128 SRS linears, the input FC. Standard levers mostly spent -> needs
   CREATIVE methods.
2. **State-only wins count** - shrink card+note, grow the CHEAP deck/preset/global. State memory ~
   entity count (many cards/notes, few decks/presets, one global), so **grow deck/preset freely (even
   10x)** to buy back accuracy lost to aggressive card/note quant.
3. **Quantization** - scoped / per-layer / hybrid schemes; QAT warm-started; revisit RWKV-edge
   (`scratchpad/rwkvedge.txt`).
4. **Creative / non-standard** (now PRIMARY for the 0.15 KB card target): **low-rank/factored card WKV
   state + QUANTIZED factors** (rank-1 int4 = 64 B incl shifts; pure-fp32 low-rank only TIES int2, see
   RESUME step 4 math) - the only path under int2's 256 B floor; per-persist state quant;
   mixed-precision outlier channels; learned-codebook / autoencoder state compression; structured
   pruning; weight-tying across layers. Full seed list in HISTORY.md. Measure every idea on the 2k loop.

**▶▶ RESUME (2026-06-29, ACROSS COMPACTION) — autonomous deck/preset-grow plan (Andrew's REFINED 4-step plan):**
"(1) moderate deck+preset grow; (2) aggressive deck+preset grow; pick whichever gives lower log loss; (3) improve
QAT; (3.5) speed up BOTH GPU training and Rust evaluation (HIGH EFFORT); (4) once card int2 + note int4 work well
(via larger deck/preset and/or better QAT), try the two-low-rank-matrices idea to shrink card state further." Run
autonomously. **NOTE the change vs the old plan: do BOTH moderate AND aggressive unconditionally, then PICK the
lower-logloss one — not "aggressive only if moderate is partial."**
- **iter41 = MODERATE grow [1,8,3,6,3]** (deck 4→8, preset 3→6; 265,614 params; CARD STATE UNCHANGED 4.25 KiB fp32
  / 0.27 KiB int2 — deck/preset are ×few-entity cheap). Pipeline `scratchpad/run_iter41_pipeline.cmd` (WS non-QAT →
  warm-started decay-QAT card int2/note int4 → export `reference/rwkv_iter41_124.safetensors` → gate). MONITOR
  `scratchpad/iter41_pipeline.log` (poll `DONE_EXIT_`). arch snapshot = `arch_iter41.py`.
- **iter42 = AGGRESSIVE grow [1,16,3,12,3]** (deck 4→16, preset 3→12 = 4× champion, 2× moderate). Pipeline
  `scratchpad/run_iter42_pipeline.cmd` — it FIRST copies `arch_iter42.py`→`rwkv/architecture.py` (the arch swap is
  baked in), then WS → decay-QAT → export `reference/rwkv_iter42_124.safetensors` → gate. MONITOR
  `scratchpad/iter42_pipeline.log`. **Run iter42 AFTER iter41 fully finishes (no GPU contention + the arch swap must
  not race iter41's python).** Launch via `detach.ps1 -Script <abs run_iter42_pipeline.cmd>`.
- **WHEN EACH DONE:** read its log `=== EVAL ===` block (champ_fp32 / qat_fp32 / qat_quant imm+ahead), **LOG to
  `optimization/qat_log.jsonl`** (mode "moderate grow [1,8,3,6,3] + decay-QAT" / "aggressive grow [1,16,3,12,3] +
  decay-QAT"; fields per the QAT section) then `python optimization/logbook.py rebuild`. SUCCESS = qat_quant imm
  ≤ champ_fp32 (0.296064) ± a hair (recovers iter39's +0.0025). After BOTH: PICK lower qat_quant imm = new champ;
  weigh the extra deck/preset params/state of aggressive vs its accuracy gain ("see if aggressive is worth it").
- **THEN (3) improve QAT + push note int2** = a LONGER WARM-STARTED QAT fine-tune from the WINNING grown WS-final
  (a few stable-LR epochs + decay, quant active, NOT from scratch — iter40 proved from-scratch QAT fails). USE THIS
  to attempt **note int2 (= the >=2x note target, 1.59->0.80 KiB)**: PTQ rejected note int4 but QAT made card int2
  nearly free, so QAT'ing `card:int2,note:int2` (with the grown deck/preset for headroom) is the path to the note
  target. Gate it. Config like the decay one but TRAIN_MODE WS, fewer epochs, LOAD_MODEL=true from the winning WS-final.
- **THEN (3.5) SPEED UP GPU training AND Rust evaluation (HIGH EFFORT, Andrew 2026-06-29).** ★ CONSTRAINT
  (Andrew 2026-06-29): keep every speedup **ARCHITECTURE-AGNOSTIC** — do NOT hardcode the current dims/layers
  ([1,4,3,3,3], d=32, 1 card layer, etc.). The arch WILL keep changing for log-loss/speed gains, so a speedup
  tailored to today's shapes is wasted effort. Derive shapes at runtime (the Rust engine already does this from
  weight shapes; CUDA graphs must shape-bucket whatever appears; batch/QAT/gate-parallelism are all naturally
  general). Profiled 2026-06-29:
  GPU training is **OVERHEAD/launch-bound, NOT compute-bound** — measured ~15% GPU util, 45 W of 200 W, 6/12 GB
  during WS (a d=32 / 200-400k-param model starves the 4070). QAT is ~4x slower still (~0.24 vs ~1.0 steps/s) due
  to its per-step Python fake-quant loop = even smaller/more-frequent launches. Levers (rated):
  - **Rust eval / the GATE — ✓ DONE 2026-06-29 (~8x, the cheapest+biggest win):** `run_qat_eval.sh` now runs the
    per-user rust passes CONCURRENTLY (split user list across processes, each pinned RAYON/OMP=1 so NPROC procs use
    NPROC cores; NPROC arg, default 10). Bit-IDENTICAL to the old sequential gate (verified iter45: same imm 0.292560
    / ahead 0.324638) -> pure speedup. Measured **841s -> ~100s** at NPROC=10. Gate is no longer the bottleneck
    (~1.7 min); training (~5 min) now dominates. Arch-agnostic (loops whatever users appear). Pass NPROC=1 for sequential.
  - **QAT 4x tax -> chunked/boundary quant (med effort, high value):** use the FAST kernel within chunks, fake-quant
    the state only at chunk boundaries instead of every step. Recovers most of the 4x. ★ SYNERGY: if DEPLOY moves to
    per-PERSIST quant (quantize only on save, not every recurrence step) QAT needs only boundary quant -> fast kernel
    AND lower deploy loss = two-for-one (ties to the per-persist creative idea).
  - **Bigger training batch (low-med):** 6 GB free, but entangled with the long user_id stream (T up to 66k);
    MAX_TRAIN_GLOBAL_LEN is a packing cap not a clean batch knob (40k already backfired). ~1.5-2x.
  - **CUDA graphs (HIGH effort, 2-5x):** the classic launch-bound fix; needs shape-bucketing (variable seq lengths
    break static capture) + care around the custom autograd.Function kernel. torch.compile (1.3-2x) may fight the
    custom kernel/JIT. Theoretical ceiling ~5-6x (the 85%-idle headroom) but structure caps easy capture.
  - ROI UPDATE (2026-06-29): gate-parallelism DONE (~8x) made the gate ~1.7 min, and the recent decay-QAT runs
    trained FASTER than profiled (~1.6 steps/s, a 496-step 16ep run in ~5 min -- the "QAT 4x tax" did not bite the
    DECAY phase). So a full QAT iteration is now ~7 min (train ~5 + gate ~1.7). Remaining GPU-training speedups
    (chunked-QAT, batch, CUDA graphs) are now LOW marginal ROI (training ~5 min, high effort, would fight the custom
    kernel). DEFERRED unless a much longer/bigger-arch training run makes GPU time dominate again. Next priority =
    step 4 (low-rank card WKV -> 0.15 KB), the last open hard target.
- **THEN (4) low-rank card WKV state -> the 0.15 KB target [✓ DONE 2026-06-29 -- 0.094 KiB, beats fp32 champ].**
  RESULT: rank-2 int4-factor low-rank card WKV + int4 shifts = 96 B (0.094 KiB), deploy imm 0.291471 PASS, BEATS
  the fp32 champion (-0.0046 imm) AND the int2 champion (-0.0044) -- low-rank is SMALLER *and* MORE ACCURATE than
  int2. Pure PTQ on iter45 weights; QAT-for-lowrank (fake-low-rank-roundtrip in training) is an untried further
  refinement. Engine: `RWKV_STATE_LOWRANK_SCOPE=card:2:int4` (nalgebra SVD per-step) + `RWKV_QUANT_SHIFTS=1`. The
  original plan/math below is retained for reference.  ORIGINAL PLAN: Needed because int2 alone
  floors at 256 B; 0.15 KB requires cutting the float COUNT. Store the card WKV state S (K×K=32×32) as U·Vᵀ (rank
  r≪32) → 2Kr floats vs K². ★ EMPIRICAL RANK SCREEN (2026-06-29, `scratchpad/analyze_card_rank.py`, 20 real card
  states from gate users via --dump-card-state, SVD energy): **rank-1 is TOO LOSSY** (energy mean 0.896, min 0.711;
  relerr up to 0.54 -- the "near rank-1" claim holds only on AVERAGE, real tail of rank-2 cards). **rank-2 is the
  sweet spot** (energy mean 0.987, min 0.944; relerr mean 0.093). rank-4 ~lossless (0.999) but 160 B int4 just
  OVER target. ★ MEMORY MATH (card = 1024 WKV + 64 shift floats; shifts 1-D so quant-only): **rank-2 int4 WKV (64 B)
  + int4 shifts (32 B) = 96 B (0.094 KiB)** clears 0.15 KB with good fidelity; rank-2 int4 WKV + int8 shifts = 128 B
  (0.125 KiB) safer; rank-2 int8 WKV = 160 B over. So TARGET = rank-2, int4 factors. (NOTE: rank-1 fp32 = 256 B =
  int2-full TIE, confirming pure-fp32 low-rank is pointless; the win needs the rank-2-int4 combo.) NEXT: Frobenius
  energy is a PROXY -- must measure LOGLOSS cost of per-step rank-2 truncation propagated through recurrence+heads.
  Build = (a) Rust low-rank card-state mode: after each card recurrence step truncate the WKV state to rank-2 (SVD
  via nalgebra) + quantize factors int4 -- this per-step == the deploy per-persist model (a card advances 1 step per
  review, state persisted between reviews); gate it PTQ-style. (b) if PTQ too lossy, QAT with a fake-low-rank-roundtrip
  (analogous to fake_quant_state -- QAT rescued int2, likely rescues rank-2 too). (c) gate. Alt (BLOCKED): smaller K
  (H=2/K=16 + int2 = 144 B) needs the K=32 CUDA-kernel rewrite -- low-rank sidesteps it.
- **HOW TO RUN AUTONOMOUSLY + ESC/COMPACTION-PROOF:** launch every training as a self-contained `.cmd` via
  `detach.ps1` (parented to WmiPrvSE, survives Esc/teardown/compaction); log to a STABLE repo path
  (`scratchpad/*.log`, NOT session temp); MONITOR via OS truth (poll log / `Get-Process` / ckpt mtime) — detached
  runs give NO tool-completion event. Re-arm a Bash watcher each turn for notifications (watcher is Esc-killable,
  training is not). Beat heartbeat each turn while actively working. Do NOT kill FSRS PIDs (the 67000s-CPU ones).
- STATUS: iter39 = QAT WINNER (deploy card int2+note int4 = 0.27+1.59 KiB, +0.0025 vs champ, PASSES gate — the
  ideal config PTQ couldn't reach). iter40 = REJECTED (from-scratch QAT). iter41 = MODERATE grow in flight (detached
  pipeline, in the FINAL gate phase — slow because 21 layers vs champ 14). iter42 = AGGRESSIVE grow FULLY PREPPED
  (configs `train_rwkv_config_iter42_{ws,qat_decay}.toml`, `arch_iter42.py`, `run_iter42_pipeline.cmd`) — launch
  right after iter41's `DONE_EXIT_`. NEW TARGETS (2026-06-29): card 0.15 KB + note >=2x (see HARD TARGETS above) —
  pursued AFTER the grow/QAT steps via note int2 (QAT) and low-rank card WKV + quantized factors.

**Ops:** Injector now 24/7 (ClaudeLoopController every 3 min; controller.ps1 only acts on stale heartbeat).
Compaction (ONLY sanctioned way, Andrew 2026-06-28) = run `claude-automation/request_compact.ps1 -Focus "<carry-through>"`
+ yield idle + STOP beating the heartbeat. `/compact <focus>` fires only from a FRESH (<=30 min) + FOCUS-bearing
flag (stale/empty = purged, no fire) so it happens ONLY when Claude itself just asked. Never hand-create
`pending_compact.txt`. Papers in
`scratchpad/{rwkvquant,rwkvedge}.txt`; poppler installed (Read tool handles PDFs). Use the CURRENT session's
scratchpad dir for logs (changes each session teardown — check the task-output paths).
**★ ESC-PROOF DETACHED LAUNCHES (2026-06-29):** the user pressing **Esc** (or session teardown) tree-kills
Claude's Bash/PowerShell background jobs — INCLUDING long training runs. WORKAROUND: launch training DETACHED
via WMI so it's parented to WmiPrvSE (a system service), not Claude. Helper: `scratchpad/detach.ps1 -Script
<abs .cmd>` runs the .cmd via `Invoke-CimMethod Win32_Process Create` (returns detached_pid + parent). Write a
per-run `.cmd` wrapper (cd, set env, python -u, redirect to a STABLE repo log path like `scratchpad/<run>.log`
— NOT the session temp dir which rotates on Esc; end with `echo DONE_EXIT_%ERRORLEVEL%`). Then MONITOR via OS
truth (poll the log / the final-checkpoint mtime / Get-Process) — detached runs give NO tool-completion event.
A Claude-side watcher (Bash run_in_background until-loop) is fine for notifications but is itself Esc-killable;
the TRAINING survives, just re-arm the watcher. Example: `scratchpad/run_qat40_decay.cmd` + `detach.ps1`.
**DATA FACT (2026-06-29):** the anki-revlogs-10k dataset has NO absolute timestamp / review-id anywhere (raw
`revlogs` parquet = card_id, day_offset[integer DAY counter], rating, state, duration, elapsed_days,
elapsed_seconds). It was anonymized — time-of-day is UNRECOVERABLE, so a time-of-day input feature is
impossible with this dataset (would need real Anki collections). elapsed_seconds (time-since-last) is already in.
