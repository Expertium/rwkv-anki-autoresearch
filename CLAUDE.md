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

## 0a. Simplified Technical English -- how to WRITE to Andrew (2026-08-18)

**Write chat replies in ASD-STE100 Simplified Technical English.** The rules:

- Write short sentences. Use 20 words maximum for an instruction, 25 for a description.
- Write one idea in one sentence.
- Use the active voice. Use simple tenses.
- Use the same word for the same thing every time. Do not change words for variety.
- Do not use idioms, metaphors, or slang.
- Do not use a noun cluster of more than three words.
- Write six sentences maximum in a paragraph.
- Keep the articles "a" and "the". Put complex data in a list or a table.

**Scope: CHAT REPLIES ONLY.** This file, `research_5k_verbose.md`, `PROPOSALS.md`, the run notes and
the commit messages keep their current style. Those documents must carry mechanism, caveats and
numbers, and the density is deliberate. The [[claude-md-bloat]] problem is length, not sentence style.

**One limit, stated honestly:** STE also fixes an approved vocabulary of approximately 900 words,
one meaning per word. That list is licensed, so I cannot check words against it. I follow the writing
rules fully and only approximate the vocabulary rule. The standard permits technical names and
technical verbs, so `logloss`, `checkpoint`, `decay`, `quantize` and `Wilcoxon` stay.

**If a rule would drop a caveat, keep the caveat and shorten the sentence instead.** Accuracy first.

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
  `user_order.jsonl` ranks user ids by size. **Everything built so far uses this set.**
- **★ SECOND DATASET — `C:\Users\Andrew\anki-revlogs-10k-id` (REAL epoch-ms IDs + corrected
  `review_time`), built by `scratchpad/dataset_id/`.** Same layout + same 1:1 user numbering, so it
  is a drop-in source for a future preprocessing pass. This is what unblocks every timestamp
  feature in `optimization/FUTURE_FEATURES.md`; full detail in the Ops section's DATA FACT bullets.
  Also read-only. Staging copy: `anki-revlogs-10k-id-raw`.
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

- **★ STANDING AUTHORIZATION: RUN EXPERIMENTS WITHOUT ASKING (Andrew 2026-08-15, verbatim:
  *"you don't have to ask for permission to run more experiments"*).** Launching the next iteration
  off the ranked queue -- writing its runner, training, evaluating, logging the verdict, and moving
  to the next -- is ORDINARY WORK, not a decision needing sign-off. Do not end a turn with "say the
  word and I'll launch"; launch it and report the result. This is what [[work-autonomously]] already
  said and the loop had drifted from.
  **★★ TIGHTENED 2026-08-30, and this is the operative form: "not to stop unless there is a
  decision for *me* to make or unless the GPU/CPU is already busy and there is nothing to do but
  wait."** Exactly TWO reasons may end a turn without continuing: (1) a decision that is genuinely
  Andrew's -- the list below; (2) compute is busy AND nothing else can be advanced. **Both halves
  of (2) are required.** A busy GPU does not license stopping while CPU work exists, and it almost
  always does: the next runner, a CPU screen that could redirect a queued run, a guard, a smoke, a
  verdict to log, the record to update. Everything else is ordinary work -- do it and report it.
  Never end a turn with "say the word and I'll launch" or a menu of options; launch it. When
  something IS his call, ask it while continuing everything that does not depend on the answer.
  [[work-autonomously]]
  **What still needs Andrew:** (a) anything that changes the DEPLOY CONTRACT or state-size budget
  (e.g. spending bits on the +1 norm/index bit), (b) the ~4-day 10x endgame run, (c) deleting the
  LMDBs / starting the features rebuild, (d) anything that breaks a stated constraint of his
  ("keep the current quantization recipe", the two hard invariants), (e) a genuine fork in research
  DIRECTION where the queue is empty or exhausted. Cost alone does not require asking -- a 5.5 h or
  13 h queued iteration is routine.

- **Be self-reliant on RWKV internals** — Andrew didn't write the net. The source in
  `srs-benchmark/rwkv` is ground truth; verify facts against it, not memory.
- **Parity discipline:** verify against the reference on a **small fixed verification
  user-set** defined early (mirror srs-benchmark's `test_users.json`: a few small + a few
  large + some random, seeded) so checks are fast and comparable. The numeric gates live in
  §5: **±0.0005** for Rust-port parity (step 3), **+0.0015** regression budget for
  efficiency changes (steps 4/5/7) — and **both** scoring modes (`ahead` and `P`) must pass.
- **★ THREE-WAY PARITY — ALWAYS CHECK TRAIN vs EVAL vs CPU INFERENCE (Andrew's standing
  directive, 2026-07-26).** Whenever you add or change anything that touches the model's
  inputs, outputs, or objective, explicitly ask: *what does training optimize, what does
  eval score, and what will CPU inference (Python RNN + Rust) actually compute?* All three
  must be the same quantity. Write the answer down in the iteration's notes; a mismatch is
  a silent correctness bug that no gate catches, because each path looks self-consistent in
  isolation. The three real cases that motivated the rule:
  1. **PAVA was trained but never evaluated** (found 2026-07-26). The rectifier lived only
     inside the loss — `curve_probs` was returned unrectified — so every reported ahead
     number scored a model that differed from the one we intended to ship. It survived from
     iter 23 to iter 30 unnoticed.
  2. **The probe duration disagreed with the pipeline's own convention.** Probes imputed
     the train-set median (`scale_duration(6433)` = −0.121) while query rows — the
     pipeline's existing "no press yet" encoding — carry a literal 0.0. Now unified on 0.0.
  3. **The rectifier does not exist in `rust/rwkv-infer` at all**, so the deploy path could
     not have reproduced either version. On the port plan.
  4. **★ THE INTERVAL ITSELF DIVERGES (found 2026-08-30, and it is live in the champion).**
     TRAIN and EVAL use the dataset's `elapsed_seconds` = `answer(k) - answer(k-1)`, i.e.
     **end-to-END**. A live Anki scheduler computes `now() - last_review_time` (jschoreels fork,
     `rust/rwkv.rs:322`), which is **end-to-START** and structurally cannot be anything else --
     `duration(k)` has not happened when the prediction is made. The two differ by exactly
     `duration(k)`. **It is sharper for us than for anyone else: `duration` is input feature 7 and
     the deploy contract already zeroes the most recent review's duration** for precisely this
     reason -- so we remove it from the features and hand it back inside the interval.
     Measured leak size: at a FIXED end-to-start gap, `duration(k)` still predicts failure at
     **AUC 0.618** against a shuffled-within-bin floor of 0.4996 (`duration_leak_probe.py`,
     2.18 M gaps); it moves the interval by >=10% on 11.1% of same-day rows and 0.00% of longer
     ones. Being measured on the arm that matches deploy: `scratchpad/features_ab/e2s`, control
     `featA2`. Full write-up `scratchpad/hybrid100k/INTERVAL_HANDOFF.md` sections 8-10.
     ⚠ **A cross-project lesson came with it: an interval change can move the REVIEW COUNT, and
     then the comparison is confounded before any model runs.** srs-benchmark's `delta_t > 0`
     (`features/base.py:284`) deleted 0.172% of reviews whose corrected gap floored to zero, and
     those rows were **2.7x easier than average** (6.09% failure vs 16.14%) -- deleting the easiest
     rows raises mean logloss by itself, and it was **two thirds** of the effect being reported
     (+0.000331 -> +0.000111 once sizes matched). Our pipeline is immune (no such filter; rows are
     kept and only marked via `label_is_equalize`) and it is **CHECKED, not assumed**:
     `scratchpad/features_rebuild/compare_db.py` asserts entry-count equality and is phase 0 of the
     arm. Both pass -- train 1,483,984 / test 170,384, identical in both arms.
  5. **★★ THE QUERY-ROW CLOCK LEAK (found 2026-09-10, priced 2026-09-11 at +0.0032 imm / +0.0003
     ahead on arm 1).** Every `-id` clock column (`t_since_any_review`, sibling gap, same-card
     interval, tenure, deck age, the daily cycles) is measured to the RECONSTRUCTED show time
     `id - taken_millis`, and the query row (= the imm prediction) and the PAVA probe rows keep them.
     Anki CAPS `taken_millis` (60 s default), so a slow review's reconstructed show time is late by its
     excess, and a 2 s - 30 min "gap before the card" fails at 21-30% against 15% -- the model reads
     the current review's own difficulty from its clock. A live scheduler predicts at the real show
     time, right after the previous answer, and sees ~0. Train and eval agreed with each other and
     both disagreed with deploy -- the exact shape this section exists for. **Fix:**
     `RWKV_CLOCK_AT_PREV_ANSWER=1800` (`rwkv/clock_fix.py`; train, eval, Python deploy, trace
     exporter; Anki-fork rule in `DEPLOY_FUNCTIONS.md` section 5), in EVERY new run. **The rule:** a
     prediction made at show time may use only what exists at show time, and a show time
     reconstructed from the log is not one. Detail: `research_5k_verbose.md` "THE QUERY-ROW CLOCK
     LEAK"; measurement `scratchpad/leak/`.
  Practical prompts: does the eval path apply every train-time transform that belongs to
  the model (rather than to the loss)? Do train/eval/deploy feed the same value for inputs
  that are unavailable at deploy time? Does the Rust engine implement it? Note that `imm`
  comes from the rating head and `ahead` from the curve head, so a curve-side change moves
  only one of the two gate modes.
  **THE TOOL (2026-07-26): `scratchpad/parity3/parity_train_vs_rnn.py`** — feeds identical
  weights + inputs through RWKV7 (parallel/training) and RWKV7RNN (recurrent/deploy) and
  requires ~1e-6 agreement. CPU-only, seconds to run, one subprocess per env combination
  (ScriptModule bakes the first construction's flags). **Add a case to it whenever you add
  an arch env flag** — that is the cheap check that would have caught STRIP_CMIX /
  STRIP_L0_VLORA / STATE_CLAMP living only in `rwkv_model.py` for a whole track-2 phase.
  Two vacuity traps it guards and yours should too: randomize the zero-init params (W_o and
  the scale linears zero out most of the recurrence) and assert the output scale is
  non-trivial before comparing.
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

## 10b. THIS repo's layout — the file map

> **⚠ KEEP THIS MAP CURRENT:** whenever files/folders are added, renamed, or deleted (and at
> every housekeeping pass), check this section and update it. Stale maps are worse than none.

- **Root:** `CLAUDE.md` (this handover) · `README.md` · `INPUT_FEATURES.md` (the 92-dim input
  table; future timestamp-features plan → `optimization/FUTURE_FEATURES.md`) · `.gitignore` ·
  `requirements.txt` · `setup.py` (CUDA/C++ kernel build) · `config.py` + `utils.py` +
  `features/` (vendored cross-repo deps — needed for imports) · `build_dataset.py` ·
  `test_users.json` · `verify_rust.py` (Rust-parity gate) + `export_rnn_trace.py` +
  `make_reference.py` (its trace/reference companions).
- **`rwkv/`** — the vendored+evolved package: **`id_features.py`** (NEW 2026-08-16 — `RWKV_ID_FEATURES=1`,
  default OFF and structurally inert: the 21 real-timestamp columns, the measured normalization
  constants, the negative-gap clamp, and `input_width()`, which `srs_model.py` + `srs_model_rnn.py`
  now BOTH call instead of each hardcoding `card_features_dim = 92`); `deck_tree.py` (NEW 2026-08-16 — `RWKV_DECK_TREE=L`: the
  parent-map loader, the ancestor walk, and the shared `build_module_data` grouping helper that
  `prepare_batch.insert_probes` now also calls); `architecture.py` (5-stream config + env hooks +
  RWKV_ARCH_MODULE), `config.py`, `train_rwkv.py`, `get_result.py` (eval), `data_processing.py` /
  `prepare_batch.py` / `data_fetcher.py` / `find_equalize_test_reviews.py` (data pipeline),
  `run_as_rnn.py` (CPU RNN mode), `parse_toml.py`, `utils.py`; `model/` = `srs_model.py`,
  `srs_model_rnn.py`, `rwkv_model.py`, `rwkv_rnn_model.py`, `rwkv_ops.py`, `csrc/` (CUDA kernel;
  the built `RWKV_CUDA.pyd` is untracked). Live tomls only (the ~120 closed-era iterN run
  configs were git-rm'd 2026-07-15; git history keeps them).
- **`optimization/`** — tooling + the canonical record. `DEPLOY_FUNCTIONS.md` (the 2 Anki-fork target functions: `review_features` = the 5 stream states, `retrievability_head` = 1-P(Again); + the AGPL caveat) · `DEPLOY_SPEED_LOG.md` (the SINGLE `review_features` states/s table: 200-user paired Wilcoxon, size-identical + logloss-within-0.0005 assertions; distinct from `cpu_speed_log.md`, which pairs trials) ·  Record: `research_5k.md` (front tables,
  4dp) · `research_5k_notes.md` (methodology) · `research_5k_verbose.md` (per-iter detail,
  AI-only) · `research_log.jsonl` (5k source of truth) · `log.md`/`log.jsonl` (regenerated
  canonical table — `python optimization/logbook.py rebuild`) · `research_log.md` (CLOSED
  100/100-era log) · `HISTORY.md` (superseded plans + archived CLAUDE.md live-state) ·
  `FUTURE_FEATURES.md` · `LIT_REVIEW.md` · **`CONTENT_EMBEDDINGS.md`** (CONTINGENT, 2026-08-19 --
  if a dataset with CARD CONTENT ever appears: Andrew's decision is
  `paraphrase-multilingual-MiniLM-L12` at **int8, 118 MB, NO pruning**, on-device at all times
  because users edit cards on mobile. Carries why multilingual beat the AnkiHub/med skew concern,
  why pruning to ~29 MB was declined, that a LEARNED reduction is forbidden by Andrew's own
  anti-skew argument while a fixed random projection is not, and the +5.5% param cost of a 384-dim
  input to our 558k trunk. Nothing is scheduled) · **`DATASETS.md`** (which review dataset to train on:
  the four sets on this machine, the FSRS-Anki-20k VERDICT = do NOT train on it (disk 1.5 TB,
  no note/deck/preset, 4.3% leakage), the card-id-is-not-a-fingerprint lesson, the
  augmentation-off/byte-identical-epochs finding, and the 2-line users-vs-epochs ablation that
  settles data-limited-or-not) · **`CPU_INFERENCE.md`** (the deploy-speed
  scoreboard: why param cuts have NOT yet bought CPU rev/s, and the Rust port that gates
  the real answer; bench `cpu_infer_bench.py`) · `PROTOCOL.md` (iter0-era mirror of §11) ·
  **`PROPOSALS.md`** (the RANKED research queue + the 3-agent generation protocol + the standing
  constraints a proposal must satisfy; write rankings here immediately -- a compaction ate the
  2026-08-10 list's tail) ·
  `STATEFUL_BPTT_PLAN.md` (shelved). Champions: `champion_5k.json` (QAT deploy truth, FROZEN) ·
  `champion_5k_plain.json` (track-1 plain) · `champion_5k_track2.json` (A0 anchor) ·
  `champion_5k_history.jsonl`. Tools: `logbook.py`, `gate.py`, `paired_pvalue.py`,
  `promote_champion_5k.py`, `eval_sharded.py`, `hp_tuner_5k.py` (+ old `hp_tuner.py`),
  `model_stats.py`, `measure_throughput.py`, `wilcoxon_speed.py`, `count_groups_5k.py`,
  `entropy_floor.py`, `quant_ptq.py`, `soup.py`. Journals: `tuner_5k_log.jsonl`
  (+ `_2ep_era`/`_proxyera` archives), `tuner_log.jsonl`, `baseline_log.jsonl`, `qat_log.jsonl`,
  `quant_log.jsonl`, `cpu_speed_log.md`. `arch_snapshots/` = per-iter architecture.py snapshots
  (100/100 era).
- **`reference/`** — deploy + parity artifacts: `pq_cb_{wkv,shift}_q72u.txt` (the q72u deploy
  codebooks), `pq_cb_m2b8.txt`, `ref_metrics.json`, `weight_names.json`, `rpv_*.json`
  (Rust-parity vectors); `.safetensors` untracked by design. ⚠ Its June trace is for the OLD
  d=128 `rwkv_ref_558` and is NOT reproducible by current Python — see §11.
- **`reference_a18/`** (NEW 2026-07-26) — the self-consistent track-2 parity trace: A18's
  weights + traces + `ref_metrics.json` (which now records the checkpoint AND arch module
  actually exported). Self-contained at exactly 0.000e+00; this is what `PARITY: PASS` was
  measured against. Regenerate with `RWKV_REF_DIR=<dir> python export_rnn_trace.py`; only
  `ref_metrics.json` is tracked, the bulk artifacts are gitignored like `reference/`'s.
- **`rust/rwkv-infer/`** — the Rust CPU inference engine (`src/{main,model,fast,pava}.rs`,
  `BATCHING_PLAN.md`, `TRACK2_PORT_PLAN.md`); K-dynamic + full PQ/joint-cb/norm-quant engine
  since `1d3b5b8`; track-2 arch (GRU head, per-layer cmix skips, state clamp) + the PAVA
  button API since 2026-07-26, parity-verified. `pava.rs` = the rectifier + interval solver
  and the crate's only unit tests.
- **`vendor/jschoreels_anki/`** (NEW 2026-07-25, Andrew's directive) — READ-ONLY reference
  copy of the RWKV code from `github.com/JSchoreels/anki`, an Anki fork shipping the OLD
  2.76M RWKV as a live scheduler: `rust/` (mod.rs = the whole engine incl. the 5-stream
  chain `review_features`, the p(recall) head `retrievability_head` = 1−P(Again), state
  serialization and a `StateCompression` scheme; bulk.rs; matmul.rs = macOS Accelerate
  BLAS; **x86_simd.patch = AVX2/FMA `dot_product`+`add_scaled_in_place`, the most portable
  speed win for our engine, which has NO SIMD**; two CPU benches) + `python/` (his shipped
  torch RNN package). **`NOTICE.md` = provenance + ⚠ AGPL-3.0-or-later: copying any of it
  into `rust/rwkv-infer` makes that AGPL-derived — fine for shipping inside Anki, but flag
  it to Andrew first and keep provenance comments.** `INDEX.md` = the function map.
- **`scratchpad/`** — per-run pipelines + shared helpers. Tracked per run: `.cmd` + tomls +
  `*_ws_trace.jsonl` (+ champions' final cbs `cb_{wkv,shift}_final.txt`). Shared:
  `write_decay_setup.py`, `write_eval_toml.py`, `detach.ps1`, `liveplot/`,
  `architecture_old_d128.py`. **`parity3/`** (2026-07-26) = the three-way-parity harnesses §9
  requires: `parity_train_vs_rnn.py` (RWKV7 parallel vs RWKV7RNN recurrent on identical
  weights — ADD A CASE PER NEW ARCH ENV FLAG), `trace_selfcontained.py` (is a parity trace
  reproducible by current Python? run this FIRST when a gate looks wrong — ⚠ it honours
  `RWKV_REF_DIR` only since the 2026-07-27 fix; before that it hardcoded `reference/`),
  `buttons_py_vs_rust.py` (the 4 button intervals, Python vs Rust), `smoke_qat_jit.py` (CPU,
  seconds: proves QAT compiles as a ScriptModule, dispatches to the jit-ignored kernel, and
  matches eager bit-for-bit — i.e. `RWKV_NO_JIT=1` is not structurally required by QAT).
  **`smoke_id_identity.py`** (2026-08-21 -- do TRAINING and DEPLOY agree on WHICH ROWS ARE THE SAME
  ENTITY? Compares the actual PARTITION, not merely entity counts, and **proves its own
  non-vacuity** by re-running against a simulated int32 store and requiring >=1 case to detect it;
  8 do, and they are exactly the two real bugs. Users are picked by NaN-metadata rate (0.0 / 66.8 /
  99.6%) -- a smoke sampling only user 1 would have passed on the broken build. ⚠ The simulation
  MUST use `torch.tensor(float64, dtype=int32)`, which SATURATES; `numpy.astype(int32)` WRAPS and is
  nearly injective, so a wrap-based simulation reports the guard vacuous everywhere. Getting that
  backwards was this file's first version.)
  `smoke_id_features_width.py` (2026-08-16 — the §9 three-way check for `RWKV_ID_FEATURES`, which
  CANNOT live in `parity_train_vs_rnn.py` because that harness is single-stack: asserts the training
  class, the deploy RNN class and `CARD_FEATURE_COLUMNS` agree on the width at BOTH 92 and 112).
  `smoke_deck_tree_rnn.py` (2026-08-16 — the deck tree in the RNN DEPLOY path: an all-inactive
  parent map must reproduce the tree-off forward exactly, a real one must not).
  **`deck_tree/`** (NEW 2026-08-15/16) = the deck-hierarchy lever's own tooling:
  `build_parent_maps.py` -> **`parent_maps.parquet` (TRACKED, 4.6 MB — a RUN DEPENDENCY of any
  deck-tree iteration; users 1-7500, 94.55% resolve, 0 cycles)**, `verify_lmdb_link.py` (do the
  LMDB's stored deck ids resolve? 49.21% of reviews can walk up >=1 level), `level_reach.py` (the
  per-level reach + depth histogram that set L), `shape_cost.py` (padded kernel volume per stream
  — ⚠ it does NOT count B, which is what the singleton blunder turned on), `smoke_inert.py` (the
  lever is byte-identical with the flag off), `smoke_tree.py` + `run_smoke_tree.cmd` (off / null /
  real; `parent_maps_null.parquet` is derived scratch, gitignored).
  **`id_features/`** (NEW 2026-08-16) = `smoke_id_features.py`, the inertness + leakage smoke for the
  `-id` feature rebuild (prefix invariance at 0.000e+00 is the one that catches a whole-table
  statistic). **`optimizer_regime/`** (NEW 2026-08-16) = `muon_gap_over_training.py` (Muon-vs-AdamW
  train gap by decile on the iter-29/iter-26 MATCHED pair) + `ns_steps_dose.py` (the NS step-count
  screen). **`iter50_decktree/`** = the finished run.
  **`eval_pava/`** = the rectified-eval pipeline +
  `check_imm_identical.py` (⚠ its premise is WRONG in bf16 — see below) + `decompose_duration.py`
  (splits the rect-vs-unrect ahead delta) + `run_mode3_noise.cmd` (the noise control).
  **PROBE-INSERTION NOISE (the one rule to carry):** probes are +4 rows per scored review, which re-buckets sequences by length and reorders bf16 reductions, so a rectified eval is NOT numerically comparable to an unrectified one -- **compare rect-to-rect only**. The magnitude is channel- and model-dependent and on `ahead` it is ZERO (iter 31, n=500, `RWKV_EVAL_PAVA=3`: +0.000000 +/- 0.000014, p=0.33, an exact coin flip), so **measure the control for the model in hand** (~30 min at n=500) rather than quoting a fixed number. Mode semantics: `m3-m0` = noise, `m2-m3` = the clean duration cost, `m1-m2` = pooling. Full measurements (A18's imm +0.000280 scaling with recurrence length, and why LogLoss convexity makes the bias one-signed): `HISTORY.md`.
  **`dataset_id/`** (2026-07-15/16, was MISSING from this map —
  Andrew flagged it 2026-07-26) = **the builder for the real-timestamp `anki-revlogs-10k-id`
  dataset**: `run_build_id.cmd` (download -> extract -> build) + `download_from_hf.py` +
  `extract_7z.py` + `build_parquet_id.py` (the real work; `build_parquet_upstream.py` is the
  anonymizing original it was adapted from) + `stats.proto`/`stats_pb2.py` (locally compiled
  protobuf) + `parent_id_probe{,2,3,4}.py` and `deck_depth_by_review.py` (the deck-tree evidence
  quoted in `FUTURE_FEATURES.md`). **`iter32_kd/`** = the distillation run + `check_dump.py`.
  **`iter33_dur/`** = the duration-fix run (`RWKV_AHEAD_PROBE_ONLY=1` + `PROBE_DENSITY=1.0`,
  MAX 16384, RECTIFIED eval). **`ds20k/`** (2026-07-27) = the FSRS-Anki-20k evaluation:
  `scan20k.py` (hand-decodes each user's protobuf header, 64 B instead of ~2.7 MB) +
  `overlap2.py` (review-timestamp fingerprint, the CORRECT one) + `overlap.py` (the card-id
  version, kept only as the counter-example — shared decks propagate card ids, so it reports
  64% overlap instead of 4.3%). Verdict + method: `optimization/DATASETS.md`.
  Untracked on disk: ckpts (`*.pth`), logs, mid-run cb snapshots
  (gitignored since 2026-07-15). ⚠ Champion ckpts live here UNTRACKED (the champion jsons point
  at them) — single-machine artifacts; losing the disk loses the ckpts, not the record.
- **`result/`** — eval outputs, untracked (`RWKV-<tag>.jsonl`, `RWKV-P-<tag>.jsonl`,
  `*.nanskip.jsonl`).

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

**Two hard INVARIANTS (never change):** (1) hierarchy `card→deck→note→preset→global` (5 chained
— ⚠ CORRECTED 2026-08-09, Andrew's catch: this doc said card→NOTE→DECK for weeks, but every
architecture file incl. the vendored original executes card→DECK→NOTE; the code is and always
was the ground truth (`RWKV_SUBMODULES` in config.py orders feature columns only, not
execution). Note the actual order is NOT monotone fine-to-coarse (notes ≈ 0.9× cards, decks
≈ 56/user) — a "true fine-to-coarse swap" is a legitimate cheap future candidate. —
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
the iter3 champion).
**Rust-parity invariant:** `verify_rust.py` must pass for the champion arch before "shipping". **⚠ IT DOES NOT RUN THE ENGINE** -- it scores `reference/rust_pred_*.json` left by an earlier manual run, so `RWKV_WEIGHTS` cannot affect its verdict. Correct procedure: run the binary **from the repo root** (it resolves `reference/trace_user_*` relative to CWD) -> it writes `preds/rust_pred_*.json` -> copy those into the reference dir -> `python verify_rust.py`.
**★ WHEN A PARITY GATE LOOKS WRONG, RUN `scratchpad/parity3/trace_selfcontained.py` FIRST.** It feeds a trace's own features back through the Python RNN at review 0 and checks the stored `py_pred` reproduces -- a stale reference is far likelier than a broken engine, and that is exactly what a red gate meant in July (the June trace predated the d=128 -> d=32 move, so nothing could reproduce it). **Fix by REGENERATING, not archaeologising:** `RWKV_REF_DIR=<newdir> python export_rnn_trace.py` lands a fresh trace beside the old one. ⚠ That tool itself had a bug until 2026-07-27 (it honoured `RWKV_CHAMP_CKPT` but hardcoded `reference/`), so distrust any verdict from before that date.
**STATUS: PARITY PASS, twice.** A18 (imm 0.000035 / ahead 0.000044) and the iter-31 champion (imm 0.000008 / ahead 0.000001, i.e. 62x and 500x inside the 0.0005 tolerance), against `reference_a18/` and `reference_iter31/` respectively -- both self-contained at exactly 0.000e+00. The iter-31 run was the first to exercise GRU N=3 and a real learned PAVA end-to-end; the engine loaded both from the checkpoint unaided. **★ THIRD PASS 2026-08-11 -- the INTERLEAVED champion is now certified too** (`reference_iter41/`, exported from `i41_d_10935.pth` with `RWKV_INTERLEAVE=1` + the `_cnd` arch; self-contained at 0.000e+00): imm 0.000000 / ahead 0.000000 on BOTH engines, max per-review 4.78e-06 (fast) / 1.25e-06 (candle), and the sequential path stayed BIT-IDENTICAL to iter-31's green preds through the refactor. So the old warning that "both traces are SEQUENTIAL" is discharged. ⚠ Front-loaded placement only; a future `RWKV_ILV_SPREAD` adoption would need a fresh port + trace.
Exact env strings and the full debugging narrative: `HISTORY.md` (archived 2026-08-10).

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
> **Every research_log.jsonl entry + research_5k.md row records `nan_users` / "NaN users"** (eval users
> skipped by the NaN guard; Andrew 2026-07-16) — backfilled for all prior iters (all 0 except iter19=1, A0=7).

### Workbench + baselines
- **5k phase (CURRENT):** train 1-5000, eval 5001-10000, budget 2 WS ep + tuned-ratio decay,
  MAX_TRAIN_GLOBAL_LEN=110000 (swept), quant-aware logloss. Baseline-to-beat = the old d=128 model
  (`pretrain/RWKV_trained_on_101_4999.pth`, unquantized) eval'd on 5001-10000 = **ahead 0.296385 /
  imm 0.264905 (n=5000)**, measured 2026-07-03 by `scratchpad/run_base5k_eval.cmd`; restricted to the
  VAL half 5001-7500 (the only set candidates are scored on) it is **0.294612 / 0.263561**, `size`
  identical to ours on all 2500. (This line read "PENDING, needs eval data" until 2026-07-26 -- it was
  stale by three weeks. `result/RWKV-base5k.jsonl` had the answer all along.)
- **★ THE GAP TO THAT BASELINE IS THE TRAINING BUDGET, NOT THE ABLATIONS (decomposed 2026-07-26,
  after Andrew asked "how did this happen? We were doing much more efficient ablations").** At
  IDENTICAL architecture and IDENTICAL 2,762,884 params, our own retrain **A0** scores 0.298342 /
  0.267858 -- so **+0.00373 / +0.00430 was already spent before a single parameter was cut**. The
  entire 2.76M -> 558k ladder (A0 -> A18) cost only **+0.00096 / +0.00053**, and iter 31 handed back
  -0.00039 / -0.00075: the net price of being **4.95x smaller is +0.00057 ahead and -0.00022 imm**
  (i.e. on imm the small model BEATS the same-recipe d=128 anchor). Fair label for the remainder is
  "our 1.25-epoch recipe vs upstream's ~12", since A0 also differs in augmentation (off), peak LR
  (1e-3 vs 7e-4), warmup (200 vs 20,000) and MAX (32768 vs 66000).
  **Two consequences.** (1) The budget gap is SHARED by every iteration, so it cancels in every gate
  and no ranking is affected -- do not "fix" it mid-phase. (2) It is where ~0.004 lives, so the
  research-close plan's 2-ep confirmation run was undersized. **DECIDED (Andrew 2026-07-26): a
  10x-epoch-budget run, ONCE, at the very end — after the algorithmic loop AND after new input
  features. See "THE ENDGAME, ORDERED".** Do not run it earlier "just to see".
  ⚠ This check was DESIGNED IN and its answer was never promoted: `research_5k_verbose.md` planned
  "1-ep-budget check at d=128 rides along free: if A0 ~ the 12-ep upstream number...". It ran, came
  back 0.004 short, and the headline framing kept comparing a 1-epoch model to a 12-epoch one without
  saying so -- which is what makes the gap read as an ablation regression. It is not one.
  **NOT a data-drop bug (checked, dead end):** `get_groups` silently skips any chunk with
  `size > MAX_TRAIN_GLOBAL_LEN` (train_rwkv.py:247-250, the old MAX=20000 incident), but the largest
  chunk in `train_db_5k_h1` is 16,384, so MAX=32768 drops **0 of 46,062 chunks / 0 of 667,525,912
  rows**. MAX only controls how many chunks batch together here.
  Front table `optimization/research_5k.md`; full methodology + status `optimization/research_5k_notes.md`.
- **Run env (all phases):** **augmentation OFF** (RWKV_AUGMENT_SEED=1234) + RWKV_DETERMINISTIC=1 +
  RWKV_EMPTY_CACHE_EVERY=0 -> run-to-run variance ~0. Eval `python -m rwkv.get_result` (CUDA, JIT-on ->
  REQUIRES the `@torch.jit.ignore` fix on `quant_aware_rwkv7`).
- **Historical 100/100 + 1500u workbench refs** (eval 101-200, MAX=66000, sc8k dbs): champion recipe was
  "1 ep on 1500 users (1000-2499) + decay" (data variety >> repetition; ~25 min/experiment -- still useful
  for cheap sanity checks). d=128-on-1-100 baseline = 0.320295/0.281913 (arch-swap
  `scratchpad/architecture_old_d128.py`); iteration-0 floor = 0.374046/0.319475.

### HISTORICAL CHAMPION (d=32 era, fully superseded)
The H=2/K=16 / 193,724-param champion on the 1500-user data-variety recipe, its q72u deploy config and the two findings that came out of it are archived to `HISTORY.md` (2026-08-10). The two that still matter are stated where they are used: **DATA VARIETY BEATS REPETITION** (1 epoch on ~1500 varied users >> 15 epochs on 100) and the WKV kernel is **K-dynamic**. The frozen QAT deploy truth (`champion_5k.json`, 0.306629/0.277893 quant-aware, 9-byte card / 27-byte note at 256x compression) is unchanged and is what the endgame's quant arm re-bases against; its recipe and artifacts are in the DEPLOY block of `HISTORY.md`.

### ACCEPTANCE GATE (research phase) -- accept iff ALL hold (record binary accepted/rejected per iter):
1. "size" (equalized review count) IDENTICAL to champion (data-integrity; any change = pipeline bug).
   **★★ THE GATE IS LINEAGE-SCOPED, AND THE LINEAGE IS DEFINED BY THE LABEL FILTER DB (Andrew
   2026-09-01: "with e2s we need to start counting size from a new baseline" -- measured 2026-09-02,
   and the trigger is NARROWER than the dataset generation).**
   **`size` IS the stored `label_is_equalize` count**, which comes from `LABEL_FILTER_LMDB_PATH`.
   Verified against the db itself: per-user equalized counts read out of `test_db_5k_e2s` match the
   `size` field in `RWKV-e2sc.jsonl` exactly (users 5001/5137/5613/6104/7499).
   **=> THE e2s SWITCH DID *NOT* MOVE `size`, so the published baseline carries across it
   UNBROKEN.** Four eval dbs spanning the whole published lineage -- `test_db_5k` (July), `_fix`
   (08-21), `_fixc` (08-31), `_e2s` (08-30) -- give **0 per-user mismatches out of 2,500 and an
   identical 128,800,080 total**. Mechanism: all four set `LABEL_FILTER_LMDB_PATH = label_filter_db`,
   and our pipeline has **no `delta_t > 0` filter** (srs-benchmark does, which is why the same
   interval change deleted 0.172% of *their* reviews; we keep every row and only mark it).
   So the gate still BRIDGES the e2s transition and can still catch a pipeline bug there -- more
   useful than a re-base would have left it.
   **What DOES move the baseline is swapping the LABEL FILTER, i.e. moving to `-id`**
   (`label_filter_db_id`). Measured: user 5001 scores 12,625 on `test_db_5k_id3` vs 12,615 on
   `test_db_5k_e2s`; 3 of 6 sampled users differ. That is why featB cannot be gated on `size`
   against featA2 and is not a champion candidate.
   **★ COROLLARY, and it is a free integrity check: gen 3 and gen 4 SHARE `label_filter_db_id`, so
   their sizes MUST be identical.** A difference is a build bug, not a dataset property. Wired as
   phase 3 of `run_rebuild4.cmd` via `scratchpad/features_rebuild/compare_equalize.py`
   (non-fatal -- by then the dbs are built and verified, so a mismatch is information for a human).
   **TOOLS:** `optimization/size_baseline.py snapshot|check <lineage> <result.jsonl>`, with
   `optimization/size_baseline_published.json` already snapshotted from e2sc (2,500 users,
   128,800,080). Both tools refuse to pass vacuously and both were proven able to FAIL: the size
   check catches a 1-review perturbation on 3 of 2,500 users; `compare_equalize` returns 1 across
   label filters. **Snapshot the `-id` baseline from featB when it lands** -- a lineage with no
   baseline must not be gated by treating its first candidate as the reference.
   **⚠⚠ CORRECTION 2026-09-01, and I told Andrew the wrong thing first: WE ALREADY HAVE
   `delta_t > 0`.** I wrote "we have no `delta_t > 0` filter", generalising this file's own line
   about the LMDB builder (`data_processing` genuinely has none -- every review row is stored) to
   the whole pipeline. But `find_equalize_test_reviews.py` calls **`create_features`**, i.e.
   srs-benchmark's own code, and `features/base.py:284` is `df[df["delta_t"] > 0]`. That is
   precisely why our `size` reproduces their published jsonls. **Two different filters, one name:
   the SCORED set has always honoured `delta_t > 0`; the STORED rows never did.**
   **THE REAL GAP, and it is narrower and live:** `find_equalize` reads the parquet DIRECTLY and
   never went through `get_rwkv_data`, so the filter was evaluated on **END-TO-END** gaps while
   training and eval had moved to end-to-start. With `SECS = true` that filter is **not**
   interval-independent (`delta_t := elapsed_seconds / 86400`, base.py:127/227), so WHICH rows
   floor to zero depends on the definition. Consequence: rows whose end-to-start gap is zero
   stayed in the scored set, so the model was scored on reviews srs-benchmark's own rule deletes.
   **FIXED (Andrew 2026-09-01: "We should have delta_t > 0 though, to make our methodology closer
   to that of srs-benchmark").** `find_equalize_test_reviews.py` now applies the SAME two
   functions `get_rwkv_data` calls, in the same order, gated on the DATASET -- not a second
   implementation, because the two datasets need different formulas (published subtracts THIS
   review's duration, `-id` the PREVIOUS one) and the wrong one is silently wrong.
   **MEASURED, 60 published eval users / 3,151,582 rows: 0.1907% of rows leave, and they are
   1.46x EASIER than average (7.07% vs 10.31% failure, -9.8 sigma)** -- so removing them RAISES
   mean LogLoss on its own. Consistent with Andrew's 0.172% on the full 10k in srs-benchmark.
   ⚠ **Small samples flip the sign of the difficulty claim**: 8 users said 1.24x easier, 24 users
   said 0.86x *harder*, 60 users says 1.46x easier at -9.8 sigma. Do not quote a direction from a
   handful of users; the probe now prints a sigma and refuses a verdict under 2.
   **★★ AND THE SCORED SET IS RE-SELECTED, NOT MERELY SHRUNK.** The folds come from
   `TimeSeriesSplit` over the SURVIVING rows, so dropping rows shifts every boundary and reviews
   ENTER the scored set too -- measured **-1437 / +87** on user 5402. I expected a subset and the
   smoke caught me. **=> a per-user LogLoss is computed over a DIFFERENT set of reviews, not a
   smaller one, so old numbers cannot be corrected proportionally; the lineage needs its own
   baseline.**
   **WHAT GEN 4 DOES (my call, delegated by Andrew: "I'll let you figure out what size number to
   consider 'correct'"):** gen 4 builds against a NEW **`label_filter_db_id_e2s`**
   (`rwkv/find_equalize_id_e2s.toml`, phase 0d of `run_rebuild4.cmd`). `label_filter_db` and
   `label_filter_db_id` are **NOT** touched -- featB is scored against the latter and rebuilding
   in place would silently re-base a finished experiment.
   **=> THE "CORRECT" SIZE FOR FUTURE new-features + e2s RUNS IS WHATEVER GEN 4 PRODUCES.**
   Snapshot it with `size_baseline.py snapshot id_e2s` from the first gen-4 eval, and gate every
   later candidate against that.
   **⚠⚠ AND THE CONSEQUENCE FOR THE CURRENT CHAMPION, WHICH SHOULD NOT BE LEFT IMPLIED: `e2sc`
   HAS THE SAME GAP.** It trains and evaluates on end-to-start intervals but is scored on an
   **end-to-end-selected** set, because `label_filter_db` is deliberately untouched. So the
   champion's 0.297888 / 0.265676 includes ~0.19% of reviews srs-benchmark's own rule would
   delete, and those reviews are 1.46x EASIER than average -- i.e. the published number is
   slightly FLATTERING as a leaderboard-comparable figure.
   It is not wrong as a GATE: every published-lineage run shares the same filter, so candidate
   comparisons are unaffected, and that is why this is not being fixed reflexively. Fixing it
   means rebuilding `label_filter_db` and re-basing the whole published lineage -- **Andrew's
   call, and only worth making if published-lineage work continues.** If the features are adopted
   (Andrew 2026-09-01: "we will almost certainly adopt timestamp features"), all future work is
   `-id` and the question retires with the lineage. Flagged so the endgame's honest-number step
   does not inherit it silently.
   **THE TRADE I ACCEPTED, stated rather than buried:** this forfeits the clean gen3-vs-gen4 Bug C
   measurement, because gen 4 now differs from gen 3 in two ways instead of one. It is unavoidable
   at acceptable cost -- `label_is_equalize` is **baked into the LMDB at build time**, so adopting
   the filter later means a second ~4 h rebuild. Methodology alignment was a directive; the Bug C
   number was a nice-to-have I had invented.
   Verified by EXECUTION, not by reading: the modified filter ran on users 5001/5002 and moved
   them **12,625 -> 12,595 (-0.238%)** and **229,050 -> 227,995 (-0.461%)** against
   `label_filter_db_id`. Smoke: `scratchpad/features_rebuild/smoke_equalize_e2s.py`.
   **⚠ COST: the label-filter phase is ~3 h, NOT the "~1 h" this repo has assumed since gen 2.**
   Measured on a representative STRIDE of 20 users (not the first 20, which are
   unrepresentatively small), under featB's eval load: PROCESSES=4 gives 1.80 s/user = **5.00 h**,
   PROCESSES=8 gives 1.05 s/user = **2.92 h**. Set to 8; both are upper bounds since the real run
   starts after featB. **So the full gen-4 chain is ~3 h label filter + ~4 h dbs = ~7 h**, not the
   ~4 h the rebuild alone suggests -- budget accordingly. 8 workers are safe here despite gen 3's
   warning, which is about `data_processing`'s whole-user matrices, not this; and the phase is
   RESUMABLE, so an OOM costs a restart rather than the phase.
   **✓ ACTUAL: 49 min 41 s (06:34:12 -> 07:23:53), i.e. 3.5x FASTER than the 2.92 h projection.**
   The projection was measured while featB's eval was competing for CPU and was flagged as an
   upper bound, so the caveat held -- but **the contention factor is ~3.5x, which is the number
   worth carrying**: a CPU cost measured beside a live run overstates the free-machine cost by
   roughly that much on this box. Measure timings on a quiet machine or label them as ceilings;
   the same correction already applies to the CPU-side profiling numbers in the speed section.
2. params <= **225,000**.   3. card AND note per-entity state UNCHANGED (deck/preset/global MAY grow freely).
4./5. **★ CURRENT RULE (Andrew 2026-08-10, TIGHTENED): each mode's RAW improvement vs the CURRENT
   champion must be >= 0.0001 in BOTH modes.** No rounding step -- a raw +0.000088 now FAILS.
   **WHY it moved:** the previous rule (>=0.0001 *after 4-dp rounding*, i.e. raw >= 0.00005) put
   the bar BELOW the measured noise floor. Iters 41/43/44 are three structurally different
   schedules at identical capacity, mutually indistinguishable at |delta| <= 7.5e-5 -- so a raw
   0.00005 bar could accept a difference the data cannot resolve. 0.0001 sits above that floor.
   History: >=0.0003 (original) -> raw >=0.00005 via 4-dp rounding (2026-07-19, first applied to
   iter 26) -> raw >=0.0001 (2026-08-10).
   ⚠ NO PAST ACCEPT IS INVALIDATED -- checked: the smallest accepted margins are iter 39
   (+0.000158/+0.000153) and iter 35 (+0.000153/+0.000271), both clear the new bar; iter 38 was
   already rejected for missing the OLD bar. And iter 36 stands as a directed accept regardless.
   ⚠ The floor is BUDGET-DEPENDENT: if research moves to the 1/3 training budget, the calibration's
   c41-vs-c43 null measures the noise floor THERE, and the bar should be re-derived from it rather
   than carried over.
6. **p-gate (Andrew 2026-07-08):** paired per-user one-sided Wilcoxon (candidate vs champion, same 5000
   eval users) gives **p < 0.0001 in BOTH modes** -- `python optimization/paired_pvalue.py` (zero GPU cost,
   reads the result jsonls; exit 0 = pass). Record both p-values in research_5k.md's `p-value` column.
   Applies to accuracy accepts only (SIZE/SPEED-exception accepts claim parity, not improvement -> exempt).
=> accept ONLY changes that improve BOTH modes (RAW >=0.0001 in each, Andrew 2026-08-10; was
raw >=0.00005 via 4-dp rounding, and >=0.0003 before that) AND pass the p-gate (a monotonic
champion).
[[research-acceptance-gate]]
**★ EXCEPTION -- CURVE-SIDE changes (Andrew 2026-08-12): "ahead better, imm not (statistically)
significantly worse."** For levers that touch ONLY the curve/ahead objective, requiring imm to also
improve by >=0.0001 demands a side effect the mechanism cannot produce. Verified for iter 46's
self-distillation: the teacher is `.detach()`ed so no gradient reaches the rating head, and the imm
objective is `p_loss` = cross-entropy on **`label_rating`**, which the lever never touches -- it
rewrites `label_y`, which feeds only `curve_loss` / `curve_raw_loss` / the PAVA probe target. imm can
therefore move ONLY through the shared trunk. (§9 already said this in general: "a curve-side change
moves only one of the two gate modes".)
⚠ **`label_y` DOES reach one imm-side term** -- `p_binary_loss`, srs_model.py:1128 (**stale: it is
now :1365**) -- but `pbin_scale=0` in this recipe so it is skipped. **If `RWKV_PBIN_SCALE` is ever
turned on, a curve-side lever starts softening the imm objective too and this exception no longer
applies.**
**⚠⚠ AND THAT CAVEAT NAMES THE WEAKER PATH WHILE OMITTING THE STRONGER ONE (found 2026-08-19, while
pre-registering kdalpha025's gate rule). EXTERNAL-TEACHER KD IS *NOT* CURVE-SIDE.** `RWKV_KD_ALPHA`
rewrites **BOTH** objectives from the same `kd_mix` tuple, gated on the same
`if kd_mix is not None`, with the same alpha:
* `srs_model.py:1263` -- `label_y = alpha*teacher_curve + (1-alpha)*hard`  ->  curve / **ahead**;
* `srs_model.py:1354` -- `_km2_target = alpha*teacher_p + (1-alpha)*one_hot(label_rating)`, and
  `p_loss` is then REPLACED by soft-target CE against it  ->  rating / **imm**.
So a KD-alpha lever is a **direct** lever on the imm objective, not an indirect one through the
trunk. **The exception's verification is correct FOR ITER 46** (self-distillation rewrote only
`label_y`); the trap is generalising "KD rewrites `label_y`" to "KD is curve-side". Any KD-alpha
iteration -- iter 55, kdalpha025, and any successor -- gets the **BOTH-MODES** rule.
This also gives iter 55 a cleaner mechanism than the record has: its imm **-0.000116** did not have
to travel through the shared trunk, which would be a surprisingly large indirect effect. alpha 0.9
replaced 90% of the imm TARGET with teacher probabilities, so the rating head inherited the teacher's
miscalibration head-on. **Before applying the curve-side exception to any lever, grep for every site
that consumes `kd_mix` / the lever's tensor -- do not reason from which tensor it is named after.**
**THE RULE:** accept iff ahead improves by **raw >=0.0001 with p<0.0001**, AND imm is **not
significantly worse** = NOT (imm mean declines AND the one-sided paired Wilcoxon for "candidate
worse" gives p < 0.05). **Both halves of the harm test are load-bearing, and iter 44 is why:** its
imm mean moved -0.000001 (nominally worse) while the RANK test said candidate BETTER at p=1e-4 --
most users improved slightly, a few worsened a lot. A rank-only guard fails a magnitude-null
iteration; a mean-only guard fires on noise. Tool: `paired_pvalue.py --curve-side` (exit 0 = pass;
`--harm-alpha` tunes the 0.05). It also NOTES, without failing, an imm decline larger than the
7.5e-5 noise floor that the rank test did not call worse -- look before promoting.
**SCOPE -- use it only for curve-side levers** (self-distillation, PAVA lambda, ahead-target and
monotonicity changes, duration handling). Trunk / optimizer / capacity / topology changes keep the
BOTH-modes rule, because those genuinely can move both. Precedent: iter 36 (PAVA lambda 0.1->0.2)
was this exact shape and was directed-accepted on a 5.9:1 ahead-for-imm trade.
**EXCEPTION -- SIZE/SPEED changes** (e.g. H=2/K=16): judged on the **efficiency budget** instead -- accept if
both modes stay within **+0.0015** of the champion AND the change shrinks state and/or speeds training (it
Pareto-dominates at accuracy-parity). H=2/K=16 was accepted this way (halved card state, 1.16x faster, accuracy
within 0.0002). Such a change MAY shrink card/note state (gate #3 is for accuracy-research iters, not these).
Two HARD INVARIANTS (never change): hierarchy card->deck->note->preset->global (the CODE's order,
corrected 2026-08-09 -- the doc had note/deck swapped vs what every arch file executes); same preprocessed 92-dim
inputs / existing LMDBs (no new/changed inputs).
**★ THE NUMBERS THE FOUR IN-FLIGHT RUNS GET (decided 2026-08-18):** `kdalpha` -> **52** (still
vacant, and its own slug), `cmixpow` -> **55**, `rgate` -> **56**, `decayshape` -> **57**. QAT#2 was
renumbered **56 -> 54** to close the last gap -- free ONLY because its number was in no path
(`scratchpad/qat_tax/`, `qtaxg_i45kd_*`). **⚠ So `iter54_cmixpow/` and `iter55_rgate/` now have
LYING slugs** -- `exp` in `research_log.jsonl` is the identity, directory digits are not. Result is
a contiguous 45-57 with no permanent hole.
**★ ITERATION NUMBERING = COMPLETION ORDER (Andrew 2026-08-18):** *"it's better to just order
iterations by the time they finished rather than by the time they were queued"*. **Assign the number
when the VERDICT is recorded**, so `iter N` means the Nth result and the log reads as a history of
what was known when. The old queue-time convention broke whenever runs finished out of order --
QAT#2 became 56 because 52-55 were reserved for queued runs, then finished before iter 53. **That
is the only violation in the log and it is GRANDFATHERED**; history is not renumbered, because the
numbers are load-bearing in run dirs, checkpoint prefixes, `champion_5k_track2.json` and commits.
**★ THE STRUCTURAL FIX -- new runs must NOT put the number in the directory or checkpoint prefix.**
Name the run for its lever (`rgate`, `cmixpow`); `exp` in `research_log.jsonl` is the stable
identity and `number` is assigned at verdict. Baking a number into a path is precisely what forced
the old convention: once `scratchpad/iter55_rgate/i55_*.pth` exists the number cannot move.
(No tooling change needed -- `logbook.py`/`gate.py` record the number, they never assign it. The
four runs in flight need no exception: their chain order 52 -> 54 -> 55 -> 57 is already ascending.)

**RESEARCH-PHASE CONDUCT (Andrew 2026-07-10) -- for the phase after HP tuning + the deck/preset/global
state-size ladders:** (1) try LOTS of different tweaks of both the ARCHITECTURE and the TRAINING
PIPELINE, from different FAMILIES of ideas (not many variants of one); (2) if an idea BARELY misses the
logloss threshold, don't give up early -- try a slightly different implementation of the same idea first;
(3) MIX literature review (optimization/LIT_REVIEW.md) with self-generated ideas
-- **★ TIGHTENED TO STRICT ALTERNATION (Andrew 2026-08-19): "1 invented, 1 adopted, 1 invented,
1 adopted".** Not a ratio to satisfy on average -- the provenance column must alternate row by row.
**WHY IT WAS NEEDED: iters 51, 54, 55, 56, 57, 58 are SIX CONSECUTIVE `invented` rows**, i.e. "mix"
had silently degraded to "self-generated only" while the doc still claimed a mix was happening.
**The next iteration is therefore `adopted`** -- sourced from a paper or a real RWKV repo, with the
source named in the `provenance`/`change` fields, not merely inspired by one. Andrew also framed
this as a change of pace, so treat it as a search-diversity mechanism rather than bookkeeping: the
invented ideas come from this trunk's own measurements and therefore inherit its blind spots, which
is exactly what an external source does not. ⚠ Pair it with his 2026-08-19 steer -- **stop chasing
0.0001** -- so an `adopted` slot means a LARGE-EFFECT idea from outside, not a small one merely
because it has a citation.
(4) spend AT LEAST 50
iterations (NOT counting HP-tuning trials) before even considering declaring "nothing left to improve";
(5) (Andrew 2026-07-13) NEVER declare a FAMILY "closed" after one iteration -- writing off a family
needs at least 3-5 distinct in-family variants; 1-2 rejects = "0/N so far, deprioritized", not closed.
[[research-phase-conduct]]
**5k-PHASE METHODOLOGY (Andrew 2026-07-01) -- full text in `optimization/research_5k_notes.md`:** the 5k
research phase (train 1-5000 / eval 5001-10000; old d=128 model eval'd on 5001-10000 as the target) keeps
the same >=0.0003-BOTH-modes gate + params <=225,000, and ADDS: (a) **LogLoss recorded WITH (fake)
card- AND note-state quantization** -- beat the old fp big model *while* quantized. Env UPDATED 2026-07-08
to the final q72u recipe (fixed champion codebooks, no cb-learning -- that upgrade needs per-run
cb-export->eval wiring, queued).
**★ BOTH CODEBOOKS REPLACED 2026-08-12 -- THE q72u PAIR IS d=32-SHAPED AND MUST NOT BE USED ON THIS
TRUNK.** `pq_cb_shift_q72u.txt` is C=32 and hard-FAILS a shape assert at C=80; `pq_cb_wkv_q72u.txt`
stays dimensionally valid (K=16 either way) and so fails SILENTLY -- it measures **worse than random**
here (held-out 1.0107 vs 0.9576 for 1024 random directions, against the 1.0 encode-to-zero bound).
Refits: **`reference/pq_cb_wkv_c80_b10.txt`** (same header/1024 rows -> identical deploy state size;
error 1.0107 -> 0.3973) and **`reference/pq_cb_shift_c80_m2b12.txt`** (24 b/vector, same bits as
q72u). The 4-arm probe matrix priced the WKV swap at **+0.003235 ahead / +0.004183 imm of recovered
PTQ cost for zero extra bytes**, and showed the WKV side is ~14x the shift side (whole shift cost
+0.000365/+0.000720), so m2b12 is the deploy choice and quantization work belongs on the WKV half.
**CURRENT ENV:** `RWKV_QAT_LOWRANK_SCOPE=card:1:int4,note:1:int4
RWKV_QAT_PQ=reference/pq_cb_wkv_c80_b10.txt RWKV_QAT_SHIFT_PQ=reference/pq_cb_shift_c80_m2b12.txt
RWKV_QAT_SHIFT_SCOPE=card:int3,note:int3 RWKV_QAT_NORM_BITS=1 RWKV_QAT_FUSED=1 RWKV_NO_JIT=1
**RWKV_QAT_PQ_LEARN=1 RWKV_QAT_SHIFT_PQ_LEARN=1**` (JIT on the grafted paths unverified -- A/B once at
champion-run launch).
**★★ BOTH DEPLOY CATALOGS ARE STALE ON THE GEN-5 TRUNK -- REFITTED 2026-09-08, BEFORE THE QAT ARM
SPENT A GPU-HOUR ON THEM (`scratchpad/qat_gen5/FINDING.md`).** Second instance of the q72u failure,
and silent for the same reason: d_model/H/K are unchanged since the 2026-08-12 fit, so every shape
assert passes and only a RANDOM control at the same budget can expose it (1.0 = encode to zero).
Measured on 15,854 WKV states / 23,787 shift vectors dumped from the certified `reference_realcyc`
model, held out BY USER: **WKV `pq_cb_wkv_c80_b10` = 0.9416 against a random control of 0.9582**
(1.7% better than random; a refit gives 0.5185), and **shift `pq_cb_shift_c80_m2b12` = 1.0352 (TS) /
1.0811 (CS), i.e. WORSE THAN ENCODING TO ZERO** (a refit gives 0.3933 / 0.3814 on a held-out user).
**=> GEN-5 AND LATER RUNS USE `reference/pq_cb_wkv_gen5_b10.txt` +
`reference/pq_cb_shift_gen5_m2b12.txt`** -- headers byte-identical to the live pair, so deploy state
size is UNCHANGED, and both smoke-load in the Rust engine. The old pair is kept and is still correct
for the published-features trunk, which is what every recorded QAT number was measured on.
⚠ This is a REPAIR, not a ranking: reconstruction error cannot rank two WORKING catalogs (the
learned catalog that cut the tax 45% reconstructs worse than its frozen start), but a catalog past
the encode-to-zero bound is the q72u signature, and that swap was worth +0.003235/+0.004183.
⚠ OWED before arm 2 launches: re-run the ~10 min procedure on **ws10's** WS-final (the model arm 2
actually branches from) and write `reference/pq_cb_{wkv,shift}_ws10_*.txt` -- arm 2's phase 0
REFUSES without them.
⚠ **WIDENING the corpus was tried and DROPPED the same day.** The dump is fast (Rust) but needs a
parity TRACE, and the trace export is the Python deploy RNN at ~25 rows/s -- user 112 alone is
790,369 rows = ~8.8 h. More importantly the benefit is unmeasurable by this instrument: reconstruction
cannot rank two WORKING catalogs, and both refits are now far below the random control, so wanting
0.3933 -> 0.1902 is the exact inference 2026-08-15 says not to make. `export_rnn_trace.py` takes
`RWKV_REF_USERS` (default unchanged) if a real reason ever appears.

**★ THE TWO LEARN FLAGS ARE ADOPTED AS DEFAULT (2026-08-13) -- put them in EVERY quant-aware run
including the endgame's arm 2.** They cut the measured QAT tax **45.4% / 43.9%** (+0.004185/+0.006219
-> **+0.002286/+0.003486**, n=2500) for **zero deploy bytes and zero wall-clock** (0.3333 steps/s,
identical to frozen catalogs -- the learning is kernel-side atomicAdd). Mechanism: most of the "tax"
is the catalog going stale *within the run* as QAT moves the weights, which a frozen catalog cannot
track. ⚠ Shape invariant enforced at upload: role-mode learnable catalogs need `m <= 4` (the kernel's
`rec_idx_chunk` slot stride is 8 ints); joint mode is `m == 1` and exempt.
⚠ **A fitted codebook is validated by SHAPE and used on CONTENT** -- re-fit whenever d_model, H, or
the state distribution changes, and check it against a random-catalog control (CPU, minutes:
`scratchpad/qat_tax/wkv_cb_staleness.py`). Nothing in the pipeline warns you.
(b) card+note state sizes FIXED, but deck/preset MAY grow
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
- **★★ CURRENT STATE, MEASURED 2026-08-31/09-01 ON THE d=80 INTERLEAVED TRUNK (quiet machine).
  Everything above this bullet is the d=32 era; the numbers below are the live ones.**
  Full narrative + every closed lead: `optimization/DISPATCH_PLAN.md`.
  * **BANKED, BIT-EXACT: `perm_gather` on the INTERLEAVED path (`srs_model.py:1249`). +5.5%
    throughput, GPU kernel 1,213 -> 892 ms/step (-26.4%).** It was wired on the SEQUENTIAL gather
    (`:1080`) and missed on the interleaved one, which has been the champion's path since iter 41.
    Verified BIT-IDENTICAL over 40 steps (`scratchpad/dispatch/cmp_traces.py`), so **no re-base and
    no seed pair** -- and the champion's recorded numbers stand.
    **⚠ SUPERSEDED BY UPSTREAM, AND MY VERSION WAS NEVER COMMITTED (2026-09-02).** Xemorr's
    **PR #3 (`f6d7505`, 2026-08-23)** had already done this **a week before I "found" it**, and
    does strictly more: `perm_gather` AND `perm_scatter` on the interleaved path. Merged into
    `main`; my redundant hunk was removed surgically rather than reverted, because the same file
    carried unrelated uncommitted work. **The lesson is cheap and I paid full price for it:
    check the remote before writing the fix** -- `git fetch` costs seconds and I did not do it
    until the commit step.
    Verified on OUR champion db before letting `main` move: forward bit-identical over 1,141,200
    elements, gradients bit-identical for all 354 params
    (`scratchpad/parity3/smoke_interleave_permscatter.py`). ⚠ That smoke was **skipping silently**
    here -- it hardcodes `train_db_5k_h1`, deleted 2026-08-30 -- so it now honours `RWKV_SMOKE_DB`
    with the old path as the default. **A guard that cannot run is not a guard.**
  * **⚠ `train_rwkv.py`'s own profiler comment ("237 ms of GPU kernel time inside a ~1450 ms step")
    IS STALE -- it is dated 2026-07-27 and interleaving landed 2026-08-11.** Do not build anything
    on it; it is what made the whole phase open on "85% dispatch-bound, CUDA graphs first".
  * **Interleaving costs 372 ms/step of GPU time** (1,213 with, 841 without) -- real, but only ~38%
    of the gap to that stale figure, so it does not fully explain it. NOT a revert proposal: iter 41
    is an accepted accuracy win and the protocol leaves GPU speed untimed.
  * **`RWKV_EMPTY_CACHE_EVERY=1` IS CORRECT AND IS WORTH 26%** -- `=0` measured 17.8k vs 24.0k
    rev/s, with GPU kernel time 1,369 -> 4,321 ms/step. ⚠ My argument for turning it off was
    backwards: I measured an 8.2 GB peak on a 12.28 GB card **while the flag was active** and read
    that as proof it was unnecessary. **Before removing a control, ask whether the evidence against
    it was produced by it.**
  * **The 273 `cudaStreamSynchronize`/step are CAUSED BY `empty_cache`** (they vanish at `=0`), and
    are not independently removable: `=0` replaces them with 80 `cudaFree` calls at ~15 ms EACH plus
    `Command Buffer Full` stalls.
  * **CUDA graphs are CONDITIONAL, not queued.** 18,756 launches/step cost 218 ms of a 1,121 ms
    step, but graph capture needs STABLE ADDRESSES and `empty_cache` every step is the opposite.
    Test coexistence FIRST; if they cannot, it is 19% against a measured 26% and graphs lose.
  * MAX downward is closed (65536 already optimal). MAX=98304 is +14.3% rev/s but **-34% optimizer
    steps** -- a phase-5 tuning lever, not a speedup.
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

### CURRENT STATE (trimmed 2026-07-26 — KEEP THIS SECTION SHORT: champions, live run, queue, live rules. Superseded chronology is in `optimization/HISTORY.md` "5k-era LIVE STATE archive"; per-iteration detail in `research_5k_verbose.md`; numbers in `research_5k.md` + `log.md`)

**★ THE TWO TRACKS HAVE MERGED (Andrew 2026-07-26).** The track-2 A-series is CLOSED at A18.
Work continues as ONE lineage on the A18 trunk, numbered as track-1 iterations in
**research_5k.md's FIRST table**. ⚠ The old track-1 `params <= 225,000` cap is RETIRED — it
belonged to the d=32 track; this lineage's size story is the 4.95x reduction (flagged to
Andrew, not silently dropped).

#### CHAMPION = iter 53 `iter53_muonlora` (Muon extended to the LoRA matrices) -- promoted 2026-08-18 08:04
**ahead 0.297523 / imm 0.265191** on the VAL half (n=2500) = **+0.000174 / +0.000184** vs iter 45 at
p=3.5e-08 / 2.7e-54. size 0/2500, nan_users 0, **params 558,212 EXACTLY unchanged**, card/note/deck
state 2,880/1,440/5,760 all unchanged (an optimizer change has no weights), throughput 1860.3 rev/s
(vs 1833.5 -- identical within noise). Training cost ~1% (0.907 vs 0.920 steps/s).
ckpt `scratchpad/iter53_muonlora/i53_d_10935.pth`; `champion_5k_track2.json` points at it.
**THE LEVER, ~2 lines:** `RWKV_MUON_INCLUDE_LORA=1`. The Muon grouping excluded any param whose name
contains `lora` or `scale`, so the **27,520** rank-4/rank-2 LoRA projections (104 tensors, 4.9% of the
model) had always run on AdamW. They move to Muon **in their own group at `weight_decay = 0.0`** --
the value they already had in `other_params` -- so the optimizer is the ONLY variable (verified in
code: the group carries an explicit `weight_decay: 0.0`).
**★ THE MECHANISM IS CONFIRMED, NOT ASSUMED, and it is why this worked.** The 2026-08-16 finding that
**Muon pays here as a REGULARIZER rather than a faster optimizer** predicted this directly, and the
paired WS traces reproduce its signature: over deciles 2-10 iter 53's **TRAIN**-loss advantage on
ahead oscillates around **ZERO** (mean ~+0.00001) while the **HELD-OUT** gain is **+0.000174**. A
generalization gain with no optimization gain behind it. (imm: train ~+0.00012 vs +0.000184 held out.)
**=> THE PRODUCTIVE OPTIMIZER AXIS IS COVERAGE, NOT DESCENT QUALITY.** The 2026-08-16 note demoted
PolarExpress/NorMuon because they refine the DESCENT -- the half that stopped paying. This confirms
the other half pays. The only 2-D params still on AdamW are the 26 `*scale*` tensors.
**★ OPEN, and this result promotes it -- the LoRA norms have NO BRAKE.** Deployed `||W||_F` is **+62%**
over the champion's and does NOT saturate (+22.9 -> +36.4 -> +70.6% at steps 1k/2k/10.9k). It stops
only because the LR schedule anneals to zero: Muon's step is fixed-norm x LR, this group has `wd=0`,
nothing restores it. **At the 10x endgame the same mechanism runs ~6x longer, so that run must either
carry weight decay on the LoRA group or re-measure.** PROPOSALS rank 8 is now the natural follow-up,
as **Muon PLUS decay**.
**DEPLOY: nothing to port** -- training-only, forward pass untouched, params identical.
**⚠ PRE-REGISTERED PREDICTION WAS WRONG** (`scratchpad/iter53_muonlora/PREREG.md`): I predicted
null-or-harm because a rank-4 bottleneck exists *to* concentrate. It does overshoot (LoRA goes from
0.52 to 0.83 of shape-matched-random spread, past the 0.81 where the rest of the model sits) and it
helps anyway. Detail + the two measurement lessons: `research_5k_verbose.md` iter 53.

#### THE CHAMPION LINEAGE (full blocks archived to `optimization/HISTORY.md` 2026-08-10 -- this table replaces ~150 lines of superseded champion detail)

| iter | what changed | rect ahead / imm (VAL half) |
|---|---|---|
| A18 | the d=80 trunk; width ladder closed at 4.95x smaller | 0.299302 / 0.268390 |
| 31 | + PAVA + GRU N=3 + Muon (3-change graft) | 0.298909 / 0.267637 |
| 32 | + full-run distillation from the d=128 teacher | 0.300268 / 0.267262 |
| 34 | + the MAX=65536 tuned recipe (HP tuning; the phase's largest gain) | 0.298970 / 0.266217 |
| 35 | + KD restored at seed 4321 (the seed pair) | 0.298816 / 0.265946 |
| 36 | + PAVA lambda 0.1 -> 0.2 (directed accept, a 5.9:1 trade) | 0.298338 / 0.266027 |
| 39 | + KD alpha 0.5 -> 0.9 | 0.298180 / 0.265875 |
| 41 | + interleaving (and a reorder that iter 43 later showed is free) | 0.297889 / 0.265479 |
| 45 | + KD kept through the decay phase (alpha 0.5), zero code | 0.297697 / 0.265375 |
| **53** | **+ Muon extended to the LoRA matrices (own wd=0 group), ~2 lines** | **0.297523 / 0.265191** |

⚠ iters 32 and 34 are not directly comparable to their neighbours: the gate basis changed to the RECTIFIED metric at iter 33, and iter 34 changed the training budget. Per-iteration detail: `research_5k_verbose.md`. Full superseded champion blocks (env strings, ckpt paths, caveats): `HISTORY.md`.

### ★★★ THE PLAN, ORDERED (Andrew 2026-08-30) — THIS SUPERSEDES EVERY EARLIER ORDERING

> *"Let's stop making the model smaller and just focus on speedups -> then finish eval of featA ->
> then do featB -> then more algorithmic improvements -> then final HP tuning **with QAT on** ->
> then the final run with QAT and larger epoch budget -> then we'll see if there is anything left
> to squeeze from CPU inference."*

His reason, and it is the honest one: *"it's getting hard to keep track of all the experiments."*
So this list is the single source of order. Anything not on it is not scheduled.

| # | phase | state |
|---|---|---|
| 1 | **Training speedups** (dispatch-bound) | **DONE** — +5.5% bit-exact (PermGather on the interleaved path); four leads closed. `optimization/DISPATCH_PLAN.md` |
| 2 | **Finish the features CONTROL eval** | **DONE** — featA2 0.298186 / 0.265588; prices the Bug A fix at +0.000148 / +0.000169 |
| 3 | **featB** — the new-features arm | **DONE** (0.297884 / 0.263217). Then gen 4 (Bug C fix) -> gen4base, and gen 5 (real-time cycles) -> realcyc, the lineage's reference. Detail archived to `HISTORY.md` 2026-09-11 |
| 4 | **More algorithmic improvements** | the research loop, gate unchanged. **★ STOP CRITERION (Andrew 2026-09-02, TIGHTENED the same day to 0.2950 ahead \| 0.2640 imm; was 0.2960 \| 0.2650): keep going -- adopted AND invented, alternating -- until LogLoss is at most that.** **Re-affirmed 2026-09-02 (Andrew: "write down to alternate between adopted (literature/Github repos) and invented"): STRICT row-by-row alternation, adopted = a paper or a real repo named in `provenance`, invented = ours.** **★ SUPERSEDED 2026-09-04 20:35 -- Andrew: "Let's skip KD."** No teacher retrain, no born-again KD; the features lineage runs KD-OFF for the rest of phase 4 and the GPU days go to the endgame's own run. (The costing that led there: `scratchpad/teacher_gen5/TIMING.md` -- a useful d=128 teacher is ~4 GPU days.) The sentence that follows is kept as history. **The KD TEACHER RETRAIN WAS SCHEDULED INSIDE THIS PHASE (Andrew 2026-09-02: "we can retrain the teacher later as part of the algorithmic improvements plan"): a d=128 model native to the FINAL feature layout (109 dims if realcyc promotes, else 114), because the 92-dim teacher cannot be re-laid-out (teacher_114 screen). Until it exists the features lineage runs KD-off, and every candidate is gated against a KD-off baseline.** His reason, and it is a net of two opposing effects: *"we'll be using more epochs for the final run, but we'll also use QAT."* The 10x budget BUYS ~0.0042 (the 2026-08-11 calibration's 3x step of +0.002, projected), while QAT COSTS **+0.002286 ahead / +0.003486 imm** even with learnable catalogs -- so the endgame is not a free improvement and the research phase must arrive with headroom, not merely at the old bar. On featB's basis imm (0.263217) still clears it and **ahead (0.297884) is binding, now ~0.0029 away**. ⚠ gen 4 moves the absolute basis (its e2s-selected equalize set removes ~0.19% of rows that are 1.46x EASIER, which RAISES mean LogLoss by itself), so read the target on the gen-4 lineage's numbers, never on featB's. First lever queued: `realcyc` (real-time cycles replace the pseudo day-offset ones + row 11, Andrew's directive) |
| 5 | **Final HP tuning, WITH QAT ON** | ⚠ NEW — all prior tuning was PLAIN |
| 6 | **The final run: QAT + larger epoch budget** | **RUNNING since 2026-09-08** (Andrew: "keep current model size ... train WS, then experiment with the decay stage"): ws10 DONE, arm 1 DONE, arm 2 (QAT) decaying; the leak-free rerun `w10lf` is queued. See LIVE |
| 7 | **CPU inference** — whatever is left to squeeze | `optimization/CPU_INFERENCE.md` |

**★ "STOP MAKING THE MODEL SMALLER" CLOSES A WHOLE LINE OF WORK.** The parameter-ratio gate, the
<=100k hybrid arms, and the FSRS-core V1/V2 experiments are all DONE or SHELVED. Do not propose a
size reduction as an iteration. Size is now whatever falls out of phases 4-6.
* iters 60 (arm A) and 61 (arm B) are the recorded verdicts; both rejected, and together they
  showed the parameter-efficiency curve has a knee below 558k and that widening the feature
  pathway does not substitute for recurrent capacity.
* **V1 (FSRS-7 card core) is SHELVED, wired and verified but never trained.** 16/16 parity checks
  green, inert when off, 488,858 params. It is shelved on SPEED, not on doubt: 0.166 steps/s,
  ~9x slower than an arm (~32 h/run), because it replaces a fused CUDA kernel with a Python loop
  of ~40 ops per review into a step that is 85% dispatch-bound. Full state:
  `optimization/HYBRID_100K.md` sections 13-14. If phase 1 lands a large dispatch win, V1 becomes
  cheap enough to reconsider — that is the only condition under which it returns.

⚠ **PHASE 2 NAMING: Andrew said "finish eval of featA", and `featA` is COMPLETE.** The arm with no
number is **featA2** — the control retrained on the Bug-A-FIXED published dbs
(`train_db_5k_h1_fix`), trained but never evaluated. That is what phase 2 means; `run_featA2_evalonly.cmd`
exists. featA's own numbers are on the OLD dbs and are not a valid control for featB.

⚠⚠ **PHASE 3 IS CONFOUNDED AND ANDREW SHOULD SEE THIS BEFORE IT RUNS.** featB changes TWO things at
once, because `elapsed_end_to_start` (landed 2026-08-19) is gated on the DATASET, not on a flag: it
runs whenever `review_time` is present, which is true of every `-id` build. `train_db_5k_h1_id3` was
built 08-24. **So featB = new features AND end-to-start together, and no `-id` database with
end-to-end intervals exists.** The features A/B cannot separate them.
**The cheap resolution needs no rebuild:** the interval question is a ONE-LINE transform on the
PUBLISHED set (`elapsed_seconds - duration(k)/1000`), because `duration` and `elapsed_seconds` are
both public columns. That isolates the interval definition with features held fixed, and it is the
same experiment as the srs-benchmark hand-off. Details + the per-dataset formula trap:
`scratchpad/hybrid100k/INTERVAL_HANDOFF.md`. The alternative -- a 4th `-id` generation with the
correction disabled -- is ~23 h of preprocessing to un-bundle what one line answers for free.

**★ PHASE 5 IS NEW AND IS NOT A REPEAT.** Every previous HP tune was PLAIN; the champion HPs were
confirmed against 19 alternatives without QAT. QAT changes the loss landscape (the tax is
+0.002286/+0.003486 even with learnable catalogs), so the plain optimum is not known to be the
quant-aware optimum. Budget it as a real phase, not a re-confirmation.

**What phase 6 inherits, unchanged from the old plan:** two arms (plain 10x, then a warm-started
QAT fine-tune on its final -- NOT QAT from scratch, iter 40 measured that at +0.0118), the 10+2
WS:decay split, augmentation stays OFF, and the three 1-epoch assumptions (warmup 200, wd,
dropout) to reconsider first. The detail was archived to `HISTORY.md` on 2026-09-11 (phase 6 is
running; see LIVE).

---

### (archived 2026-09-11) THE 2026-07-26 ENDGAME DETAIL + THE 2026-08-18 BUG HUNT
Both sections moved VERBATIM to `optimization/HISTORY.md` "CLAUDE.md LIVE archive (moved out 2026-09-11)". Phase 6 is now executing (ws10 -> arm 1 -> arm 2, see LIVE), and the questions they carried are settled there: arm 2 = the SAME 2-epoch decay from the shared WS with the QAT env added (a single-variable A/B); the 10+2 split; augmentation OFF; KD skipped except QAT-KD; wd/dropout re-screened on ws10's own validation trace (still fit-limited). The bug hunt ran in 2026-08 (`scratchpad/bughunt/README.md`: the runner sweep, the diagnostic configs, the `preflight_runner.py` checks); its standing form is the dry-run rule in LIVE RULES. **Two lessons from them stay live:** (1) a KD dump's `labels_sum` checksum proves LABEL alignment only -- ANY input-side change (augmentation, interval definition, features, the clock fix) silently invalidates a dump while the checksum keeps passing, so the live-teacher QAT-KD path is the safe one; (2) QAT compiles as a ScriptModule (`scratchpad/parity3/smoke_qat_jit.py`, CPU half green), but the GPU half (bit-exactness of a real step, the steps/s A/B) was never run -- moot while runs use `torch.compile`, which needs `RWKV_NO_JIT=1`.

#### LIVE
**⟶ 2026-09-11 08:47 -- ★★ PHASE L: THE CLOCK LEAK IS MATERIAL -- imm +0.0032 / ahead +0.0003 on arm 1's checkpoint** (users 5001-5300: lkc30m - lkc0 = **+0.003152 +/- 0.000454 imm**, worse on 273/300, p=2.2e-47; **+0.000256 ahead**, worse on 198/300; the 3 h arm adds only +0.000129, so **T = 1800**). Harness exact (control = arm 1 at max |d| 0); both modes inside the pre-registered bands (`scratchpad/leak/leak_report.txt`). **ARM 1's imm CLAIMS ARE WITHDRAWN** (~0.265 at deploy-faithful inputs: misses 0.2640 and trails the d=128 model's 0.263586); its ahead claims move ~0.0003 and stand. It is the cost for a model TRAINED with the leak; a leak-free model recovers part of it. The features' whole imm gain (-0.0024) is SMALLER than this. Recorded: `research_log.jsonl` (`leakcf`, status screen), `research_5k.md` row, verbose "THE QUERY-ROW CLOCK LEAK", section 9 case 5.
**THE QUEUE WAS RE-PLANNED BY HAND at 08:55:** stopped the four leaked-base waiters -- QAT-KD (28684), budget curve (29148), decay length (36528) and the end-of-queue w10lf (3724) -- cmd first, no markers (a note in each log). **The chain is now: arm 2's eval (running since 08:50; probe OK on the learned catalogs; then phase D) -> `w10clk` at T=1800 (3644, automatic) -> cmixgraft (36904) -> `w10lf` (NEW: `scratchpad/w10lf/wait_cmixgraft_then_lf.cmd`, pid 16676, log `wait_lf2.log`, T fixed at 1800, refuses 150 on cmixgraft's 90; stub-tested on nine paths and run 75 s in the foreground).** cmixgraft stays ahead of w10lf because its question is ahead-side (the leak barely touches ahead), and a capacity win would change which architecture the leak-free base should have. **TO BE BUILT on w10lf's WS, all with the flag, before w10lf ends (~09-14):** the catalog refit, a leak-free QAT arm (the deploy QAT decay AND QAT-KD's control), QAT-KD (teacher = w10lf's decayed final), the budget curve (from `w10lf_ws_22000` / `_55000`), the decay-length pair. Why re-base rather than keep the ws10 versions: quantizing card/note state can strip the precision the leak needs, so a leaked model's imm QAT tax can be too large; and part of the budget's +0.0015 imm may be the model learning the leak better. ⚠ **Read arm 2's imm tax with that caveat.**

**⟶ 2026-09-10 23:10 -- ★★ ANDREW: "I think we should do a leak-free run anyway. Nothing that cheats should make it into the final version that will be shipped." -- THE LEAK-FREE 10-EPOCH RUN IS BUILT AND ARMED (`scratchpad/w10lf/`).** Not conditional on phase L. `run_w10lf.cmd` (generated by `mk_w10lf.py [T]`) = ws10's WS FROM SCRATCH (109,350 steps, auto-resume) + arm 1's 2-epoch decay + arm 1's rectified eval, with `RWKV_CLOCK_AT_PREV_ANSWER` in ALL THREE phases, so no weight ever trained on a leaking clock column; it is the BASE of the shipped model. Env asserted equal to ws10's and arm 1's except the lever (plus `OMP_NUM_THREADS=7` in the WS, which ws10 lost by accident -- CPU threads only). ~47 h. PREREG `scratchpad/w10lf/PREREG.md` (P1 imm +0.0005..+0.0030 / ahead +0.0000..+0.0006 worse than arm 1; P4 imm criterion ~50%). Stub-tested on five paths incl. crash -> `make_resume` -> finish (**ws10's resume branch had never executed**), dry run `scratchpad/w10lfdry/` (0.05 + 0.05 epochs, WS validation every 250 steps). **PLACEMENT: END of the queue by default** -- `wait_decaylen_then_lf.cmd` (pid 3724) fires on the decay-length waiter's marker, takes T from `clk_decide.py` (3 -> 10800, else 1800), regenerates + preflights both runners, dry run, then the run; stub-tested on nine paths. **If phase L says MATERIAL, move it behind cmixgraft BY HAND and re-base QAT-KD / budget / decay-length on it** (at a large leak their imm conclusions on a leaked base are unsafe: KD from a leaked teacher can pass the leak on); w10clk (pid 3644) stays armed until that decision. **★ AND THE FIX GREW ONE CHANNEL FIRST, because a 38 h run must not keep a measured leak:** `scratchpad/leak/residual_channels.py` counted what `clock_fix` left unshifted (users 5001-5040, T=1800): a shift past **UTC midnight** changes dow/doy/is_weekend/the review-time cycles on 0.013% of query rows, and those rows fail at **0.29 vs 0.14** -- a real leak. **Now fixed** (`shift_features`: crossing iff delta > seconds since UTC midnight, read from the STORED tod pair; the calendar pairs rotate back one day, is_weekend re-derived; deploy via `SHIFTED_COLUMNS`; `DEPLOY_FUNCTIONS.md` section 5). Smoke part 5 checks it against the builder's own formulas at the moved time on nine synthetic show times (err <= 7e-16; doy on 1 January approximated, 4.3e-3). **Still unshifted, measured:** Anki-day columns (0.0036% of query rows) and creation-batch counts (0.021%), at P(fail) 0.10 / 0.16 vs 0.14 = ~3e-7 nats per query row, the estimator's bias floor; closing them needs an LMDB rebuild -- **Andrew 2026-09-10: "no LMDB rebuild", the residual is ACCEPTED.** Smoke PASS on all five parts; flag-off still bit-identical.

**⟶ 2026-09-10 22:05 -- THE CLOCK-LEAK FIX IS BUILT AND PROVEN INERT; THE LEAK-FREE BRANCH IS BUILT, NOT YET ARMED.** `RWKV_CLOCK_AT_PREV_ANSWER=<T s>` (`rwkv/clock_fix.py`, `8914d1a`): query + probe rows moved to the previous answer for in-session gaps, a labelled real row's ahead t minus its target's gap; train/val/eval via `prepare_batch.prepare`, Python deploy via `run_as_rnn.imm_predict` + `clock_fix.ahead_t` (+ the trace exporter's t for Rust); the Anki-fork rule is `DEPLOY_FUNCTIONS.md` section 5. **Default off = byte-identical, PROVEN not argued:** `scratchpad/leak/smoke_clock_fix.py` shows prepare() bit-identical to the pre-hook file on gen-5 test/train chunks at probe densities 0 / 1.0 / 0.08 and on a published e2s chunk (the chain's later phases import it fresh), the ON query transform equal to phase L's counterfactual on its 12 columns, every label shift equal to an independent recomputation, and deploy-vs-LMDB parity (the same rows shift, delta within bf16, flag-off agreement not degraded). ⚠ delta is recovered from the bf16-stored gap: ~1% of delta, up to ~18 s at T=1800 -- a storage limit, not a path disagreement. **`scratchpad/w10clk/`** = arm 1 + the flag and nothing else (`mk_w10clk.py [T]`, both-direction asserts, guards 56 = ON banner / 57 = the chunk line only `apply_to_sample` prints; dry run `scratchpad/w10clkdry/`; stub-tested on four paths; PREREG + `verdict_w10clk.py`). **PLACEMENT IS AUTOMATIC (armed 22:04):** `wait_qateval_then_clk.cmd` (pid 3644) waits for arm 2's eval (which holds phase L and phase D) to end `DONE_EXIT_0`, then `clk_decide.py` applies the pre-registered rule to phase L's jsonls -- MATERIAL => dry run + w10clk next (T=10800 instead if the 3 h arm costs >= +0.0005 imm more; the runner is regenerated and preflighted first), below MATERIAL or a failed harness => `DONE_EXIT_137`, skipped. cmixgraft's waiter v1 (35108) was stopped in gate 0 and replaced by `wait_clk_then_cmixgraft.cmd` (pid 36904, same log, refuses only on 130 = arm 2's eval failed). Both waiters stub-tested on eleven paths, `clk_decide.py` on five synthetic outcomes. Either way every NEW runner sets the flag from now on.

**⟶ 2026-09-10 20:35 -- ★★ A QUERY-ROW CLOCK LEAK IN THE WHOLE `-id` LINEAGE, reported on Discord and CONFIRMED in our own data (`scratchpad/leak/`).** Every clock column is measured to the reconstructed SHOW time `id - taken_millis`, and `add_queries` keeps all new columns on the query row (data_processing.py:540, argued as "the scheduler knows when it is showing a card"). That argument FAILS: Anki CAPS `taken_millis` (60 s default, 30 of 40 users) and a capped review's show time comes out late by its excess; uncapped reviews show a second channel (timer reset on edit / leave-and-return). Raw data, users 5001-5040, 2.36 M reviews (`mechanism.py`): 71.4% of reviews come <1 s after the previous answer (P(fail) 15.1%, 0.08% capped); 2 s - 30 min gaps are 11.2% of reviews at P(fail) 21-30% and 12-36% capped, and UNCAPPED ones still fail at 20-28%; the gap bucket alone carries 0.009 nats. A deploy scheduler predicts at show time, right after the previous answer, and sees ~0 s. **LEAKING ROWS: query rows (imm) and PAVA probe rows (the pre-answer button intervals = the rectified ahead metric). REAL rows are legitimate** (finished reviews; deploy rebuilds them from the same revlog fields). The ahead LABEL time carries a smaller version (the next review's own excess) -- not yet covered. It fits the record: the features' gain was imm -0.0024 vs ahead -0.0003, the LOO put +0.005076 imm on `t_since_any_review`, and the ablation located the imm asymmetry in the query row's clock. **Until measured, "imm stop criterion MET" and "beats d=128 on imm by 0.0015" are SUSPENDED** (the d=128 model has no timestamp features, so no leak). **MEASUREMENT ARMED as PHASE L at the START of arm 2's eval runner (~1 h, never fatal):** `get_result_cf.py` patches `insert_probes` (a spawn probe proves fetch workers see it; production files untouched) and moves query + probe rows to "card shown at the previous answer" -- t_since_any_review -> 0, sibling gap / same-card interval / cumulative minus the in-session gap, daily sin/cos rotated back -- unit-tested on real gen-5 chunks (T=0 changes nothing; T=1800 changes only in-session query/probe rows, never a real row). Arms on arm 1's `w10p_d_21870`, users 5001-5300: lkc0 (control), lkc30m, lkc3h. PREREG + decision rule `scratchpad/leak/PREREG.md` (imm at 30 min >= +0.0010 = MATERIAL: build the fix env-gated in train/eval/deploy with a parity case and queue a leak-free decay branch from ws10). **The general rule this adds to §9 three-way parity: a prediction made at SHOW time may use only what exists at show time, and the reconstructed show time itself carries the current review's excess.**

**⟶ 2026-09-10 19:00 -- ARM 2's EVAL NOW ENDS WITH A TAX DECOMPOSITION (phase D, ~2.5 h, `scratchpad/qat_decomp/`).** Same checkpoint, users 5001-5300, two more evals each = deploy MINUS one VALUE quantizer: `noshift` (S) and `nowcb` (WKV codebook + norm off, rank-1 kept at int8; C). R = T - S - C is rank-1 + drift and cannot be split -- a whole-QAT-off arm is DELIBERATELY absent (cell 3: structural quantization makes the fp score garbage). Decides the bit split by a pre-registered rule (PREREG P4; my expectation MARGINAL) and bounds every catalog lever. Proven before arming: `assert_arm.py` 9/9 incl. six refusals on the FINAL config; `check_banners.py` fails a real deploy log for both arms; 4 stub variants of `run_decomp.cmd` + the modified eval runner reach `DONE_EXIT_0` (the bad-catalog variant still stops at 72). Inserted via `mk_w10qat_eval.py` (regenerated runner = the old one byte-for-byte + 8 lines; backup `run_w10qat_eval.cmd.bak_before_decomp`) while waiter 16064 was still polling. **Never fatal:** logs to `scratchpad/w10qat/decomp.log`, always returns, so cmixgraft and later links shift ~2.5 h. ⚠ The fused rank-1 kernel needs a FINITE factor level -- `card:1` (fq=inf) falls to the per-timestep Python loop -- which is why `nowcb` uses int8. ⚠ NOT wired into the KD runner (`mk_w10qatkd.py` asserts the eval folder appears twice; phase D adds a third).

**CHAIN NOW FIVE DEEP (19:06): ... -> budget curve (29148) -> DECAY-LENGTH pair (pid 36528, `scratchpad/w10d3/wait_budget_then_decaylen.cmd`, generated by `mk_wait_decaylen.py`): dry w10d3, dry w10d1, then w10d3 (3-epoch decay, ~12 h) and w10d1 (1-epoch, ~6 h), control = arm 1. A GPU-idle guard, not a priority claim -- displace it by stopping 36528 (cmd FIRST) while it loops in gate 0.** Its gate counts terminal lines in `wait_budget.log`, which already holds one SPURIOUS `DONE_EXIT_100`; it fires on the SECOND and refuses (120) only if that one is again 100. Stub-tested on five paths (one line = keeps waiting; 100 = refuse; 0 and 101 = run; failed dry run = 121), 70 s in the foreground, then detached. GPU booked to ~09-15.

**⟶ 2026-09-10 16:05 -- ARM 2 VALIDATION PREVIEW (step 8,042; NOT the verdict; `scratchpad/w10qat/PREREG.md` "PREVIEW").** On the same 10 users vs arm 1 at matching steps, the tax falls from +0.0097 / +0.0128 at step 50 to ~+0.003 ahead / ~+0.005 imm by step 2,000 and is FLAT through 8,000. Two readings: (1) **~90% of the 2-epoch QAT phase buys no further closure** -- a cost fact for every future QAT A/B; (2) if the preview transfers as it did for arm 1 (within 0.0004), the budget does NOT shrink the tax and deployed imm (~0.267) misses the 0.2640 criterion that full precision meets -- which makes the QAT-KD arm the most valuable run in the queue.

**★★★ 2026-09-10 13:25 -- ANDREW: "Yep, do KD" -- QAT WITH DISTILLATION FROM THE FULL-PRECISION TWIN IS BUILT AND QUEUED (`scratchpad/w10qatkd/`).** This supersedes "Let's skip KD" (2026-09-04, a research-lineage decision about a 4-day teacher retrain) FOR QAT ONLY. The arm = arm 2 exactly + `RWKV_QAT_KD=1.0` + `RWKV_QAT_KD_TEACHER=scratchpad/ws10/w10p_d_21870.pth` (arm 1's decayed final: same WS, same batches, fp) -- verified single-variable by diffing the env at the decay call. `w10qat - w10qatkd` IS what KD does to the tax. Live teacher, not a dump: the teacher forwards the same batch each step, so alignment holds by construction. lambda 1.0 == target-mix alpha 0.5 (iter 45's decay strength); ⚠ unlike iter 45's dump KD it leaves the PAVA probe targets hard. PREREG `scratchpad/w10qatkd/PREREG.md` (P1 both modes >= +0.0001, expected 10-30% of the tax; P3 the KD loss trace must sit ABOVE arm 2's from the first steps; P4 falsifier), verdict `verdict_w10qatkd.py`.
**★★ THE LIVE QAT-KD PATH WAS STALE AND WOULD HAVE POISONED THE RUN -- FIXED AND PROVEN BEFORE USE.** `RWKV_QAT_KD` (task22, d=32 era) built the teacher's CURVE target with the plain `forgetting_curve(w, t)`, while under `RWKV_GRU_HEAD` the student's own curve is `gru_forgetting_curve(w, s, d, t)` -- a KD term pulling the curve head toward a curve no model computes. Now `qat_kd_targets()` + `build_qat_kd_teacher()` are module-level in `rwkv/train_rwkv.py` and mirror `_get_loss` branch for branch (inert when `RWKV_QAT_KD` is unset; the loop calls them). **`scratchpad/qatkd/smoke_qatkd.py` (CPU, real model and gen-5 chunk, the code the loop calls): at teacher == student the KD gradient is 2.9e-07 of the hard-loss gradient (it must vanish) vs 0.79 with the old formula; the stripped QAT-env teacher matches the plain model at 0.0e+00 and the unstripped one differs by 1.08.** **The self-consistency test is the reusable instrument: a distillation target is correct iff, with teacher == student, the KD gradient vanishes.** Any KD path in this repo can be checked that way in minutes of CPU.
**CHAIN NOW FOUR DEEP: arm 2 eval (16064) -> cmixgraft dry+real (35108) -> QAT-KD dry+real (28684, `wait_cmixgraft_then_qatkd.cmd`) -> budget curve (29148, `scratchpad/w10b2/wait_qatkd_then_budget.cmd`, renamed from `wait_cmixgraft_then_budget.cmd` and re-pointed; it refuses only on the KD waiter's 110 = arm 2's eval failed).** All stub-tested (six paths) and foreground-tested. GPU booked to ~09-14.
**⚠ OPS RULE, learned at 13:15 the same day: STOP A WAITER'S cmd.exe FIRST, THEN ITS CHILDREN.** I stopped the budget waiter's conhost and ping first; its dying cmd ran one more step with broken exit codes and took the `DONE_EXIT_100` refusal branch although no `DONE_EXIT_90` existed. It launched nothing -- but the launch branch was equally reachable, and would have put a dry run on the GPU beside a gate-critical run.

**★★★ 2026-09-10 12:30 -- ARM 2's EVAL WOULD HAVE USED THE *START* CATALOGS; FIXED 19 h BEFORE PHASE C.** `run_w10qat.cmd` trains both catalogs (`RWKV_QAT_PQ_LEARN=1` + `RWKV_QAT_SHIFT_PQ_LEARN=1`), and its phase C inherited that env -- so it would have evaluated with `reference/pq_cb_{wkv,shift}_ws10_*.txt` while the weights were co-adapted to the LEARNED catalogs, which had already moved **43% (WKV) / 45% (shift)** in relative L2 by step 4000. **The learned centroids are module globals, NOT part of the checkpoint**: train_rwkv exports them to `<SAVE_FOLDER>/<prefix>_{wkv,shift}cb_<step>.txt` and nothing loads them back. `qtaxd_cblearn` -- the run that measured the recorded +0.002286/+0.003486 -- re-points its eval env at the exported files with learning off (`run_cblearn.cmd`, "EVAL ENV"); arm 2's runner was built from the PLAIN runner and lost that step. The number would have been neither the deploy number nor comparable to the recorded tax.
**FIX, in this order:** stopped ONLY the runner cmd (pid 8224) -- the decay python kept running, verified by its log growing; appended a note (no marker) to `w10qat.log`; wrote **`run_w10qat_eval.cmd`** (generated by `mk_w10qat_eval.py`, env verbatim, then the learned-catalog re-point + learning OFF, then a **10-user PROBE that must LOAD `scratchpad/ws10/w10q_d_{wkv,shift}cb_21870.txt` with `learnable=False`** and sit in the sanity band vs arm 1 before the ~10 h eval runs); armed **`wait_decay_then_qateval.cmd`** (pid 16064), which fires when `decay_state.py` sees the trainer gone AND the step-21870 checkpoint and both catalogs present (a crash without them refuses). Both validated by EXECUTION: stubbed good/bad variants (the bad one, loading the start catalog, stops at `DONE_EXIT_72`), and the real waiter in the foreground past its first `goto`. `sanity_probe_gate.py` grew `--base` (default unchanged; reproduces the recorded tax exactly).
**⚠ ANDREW'S "EVALUATE QAT RUNS QUANTIZED" RULE COULD NOT SEE THIS, and the lesson is its sharper form: quantization WAS on, with the wrong catalog. "Quantized" is not enough -- the eval must quantize with the catalog that SHIPS, so guard the loaded PATH, not the banner that says quantization is on.**
**PREREG + verdict written before the number:** `scratchpad/w10qat/PREREG.md` (Q1 tax band ahead +0.0015..+0.0030; Q2 imm pays more; Q3 neither mode meets the stop criterion at deploy -- arm 1 + prior tax forecasts 0.2999 / 0.2656; Q4 probe engagement) and `verdict_w10qat.py`. **And it closes two of the QAT-retry routes on paper:** (a) "refit on the deploying model" is already inside arm 2 (the catalogs LEARN against its own loss); (b) "chunk structure at constant bits" is MOOT on the WKV side (already m=1 per head; coding heads jointly needs a 2^20+ catalog). **The zero-byte axis that remains is the SPLIT of the frozen bytes between WKV and shift** -- card = 110 WKV + 72 shift bits (checked against the arch: 2 card layers, 3 shift vectors), and m2b12 -> m2b8 on shift pays for b10 -> b12 on WKV at 4 bits under budget. Priceable only by a logloss A/B. **Screened the same day, held out by user on ws10's corpus: WKV b12 buys >=6.6% reconstruction (0.5587 -> 0.5219, thin corpus), shift m2b8 costs +11% (TS 0.48 -> 0.53, CS 0.40 -> 0.44), far from random (1.07) -- the route is ALIVE, not free, and worth one A/B only if arm 2 puts the cost in the codebook term.** Side finding: one catalog scores 0.36 on a user inside its fit and 0.56 on the same user held out, so WKV states are strongly user-specific.
**cmixgraft's OPTIMIZER STATE WAS MISSING TOO, found by reading the load path before its dry run could hit it.** `expand_ckpt.py` wrote only weights; the decay loads `<ws>_<step>_optim.pth` unconditionally, and ws10's could not be copied because 12 restored `W_k`/`W_v` change group ((1,1) dummies on AdamW -> (80,80) on Muon), so `load_state_dict` refuses. **`scratchpad/cmixgraft/expand_optim.py`** maps ws10's state BY NAME through the real `get_optimizer` -- 391 params keep moments bit-for-bit, 30 grown start fresh -- and verifies it in a third process with a synthetic step. **CHAIN ARMED (pid 35108): arm 2 eval `DONE_EXIT_0` -> cmixgraft DRY RUN (refuses on any swallowed exception) -> cmixgraft**, stub-tested on its fail/hollow/clean paths. **(SUPERSEDED 13:25 -- the QAT-KD arm now sits between cmixgraft and the curve; see the 13:25 entry.) Then the budget curve (was pid 7004, `scratchpad/w10b2/wait_cmixgraft_then_budget.cmd`): polls the cmixgraft WAITER's log (not `cmixgraft.log`, which a refused dry run never creates), refuses only on its `DONE_EXIT_90` (= arm 2's eval failed), then BOTH dry runs before either real run, then w10b2, w10b5.** Stub-tested on four paths. GPU booked to ~17:00 on 09-12; a QAT-retry built sooner displaces the curve by stopping pid 7004 while it still loops in gate 0.
**⚠ GENERAL RULE FROM BOTH: a branch that starts from a TRANSFORMED checkpoint must transform everything the resume path loads -- weights, optimizer state, and any exported side-car (catalogs) -- and each of those is a separate file that nothing checks for you.**

**★★★★ 2026-09-10 03:53 -- ENDGAME ARM 1 IS IN: ahead 0.297577 / imm 0.262066 (n=2,499, size 0/2,499, nan_users 0, params 563,652). THE BUDGET PREMISE IS REFUTED ON AHEAD AND CONFIRMED ON IMM.** Logged as a `baseline` row (`research_log.jsonl`, `research_5k.md`, `research_5k_verbose.md` "ENDGAME ARM 1", `log.md` rebuilt). PREREG: `scratchpad/w10plain/PREREG.md`; verdict tool `verdict_w10plain.py`.

| vs | ahead | imm |
|---|---|---|
| realcyc (1+1 ep) -- CLEAN, single variable, diffed | **+0.000506** | **+0.001526** |
| the 2026-08-11 projection | +0.0042 | +0.0042 |
| old 2.76M d=128 at ~12 ep (basis worth ~0.0001) | **-0.002954** | **+0.001520** |
| stop criterion | 0.2950 -- **NOT met, 0.0026 short** | 0.2640 -- **MET, 0.0019 spare** |

**Q1: 12% of the projected ahead gain, under the pre-registered `<+0.0015` line.** The projection was log-linear from ONE measured 3x step and the PREREG named this as the ordinary way such extrapolations fail. **But do not report this as "budget does nothing": imm's +0.001526 is the LARGEST single move in the gen-5 lineage**, ~10 accepted iterations' worth from one lever. Budget pays the rating head and not the curve head. Mechanism visible in the curves: ahead validation is flat from ~epoch 1 and **the decay moves it +0.0001, i.e. not at all**, while imm improves throughout.
**★★ Q2 IS THE NEW RESULT AND IT IS ANDREW'S CALL: CAPACITY BINDS AT THE REAL BUDGET.** Every capacity reject in the record was measured at ~1.25 epochs, where the model could not use capacity. At 12 epochs the 4.95x reduction is **free on imm (we BEAT the big model by 0.0015) and costs ~0.003 on ahead**. Not clean (the d=128 model cannot forward 109 dims -- `teacher_114`), but the basis is worth ~0.0001 against a 0.003 gap. ⚠ **This collides with his 2026-09-10 preference: recovering it means growing the per-card state, the deploy budget he would rather not move.** So "stop making the model smaller" has a measured price for the first time.
**⚠ ARM 1 IS FULL PRECISION -- TRAINED AND EVALUATED (Andrew asked 2026-09-10). Its 0.297577 / 0.262066 is NOT a deploy number.** Verified by grep, not memory: the only `RWKV_QAT_*` variable in `run_w10plain.cmd` or `run_w10plain_eval.cmd` is **`RWKV_QAT_COMPILE=1`, which despite its name is the torch.compile speed flag and has nothing to do with quantization**; there are no QAT banners anywhere in the eval logs. The same is true of `realcyc` and `gen4base`, so **the whole gen-5 lineage is plain and arm1-vs-realcyc is a clean plain-vs-plain comparison.** Arm 2 is the one carrying the QAT stack (`LOWRANK_SCOPE`, `PQ`, `SHIFT_PQ`, `NORM_BITS`, both LEARN flags) and its number will be the deploy one, higher by the tax. ⚠ The flag name is a live trap -- `RWKV_QAT_COMPILE` reads as a quantization flag in every runner and is not one; methodology (a)'s "quant-aware logloss" has been unsatisfiable on this trunk since the 2026-08-12 inert-env finding, so no gen-5 research number is quant-aware.

**★ AND THERE IS ONE CAPACITY DIRECTION THAT COSTS NO DEPLOY BYTES, which matters because Andrew has just said he would rather not grow the state.** Gate #3 has always read "card AND note per-entity state UNCHANGED (**deck/preset/global MAY grow freely**)" -- those streams are per-DECK and per-USER, not per-card, so widening or deepening them does not touch the 9 B/card, 27 B/note contract. iter 49 tested exactly that (restoring the user/preset layer-0 channel mixers, +4.7% params) and returned a null -- **at 1.25 epochs, which is the budget Q2 just showed cannot expose a capacity limit.** So the coarse-stream ladder is UNTESTED at the real budget rather than closed.
⚠ **It is not cheap AT FULL BUDGET: a capacity change alters the architecture, so it cannot branch off ws10's WS checkpoint and needs its own ~38 h WS** (more, since a wider model is slower). Andrew 2026-09-10: *"It would be nice to try to grow deck/preset/global, but 38 h is a pretty steep price."*
**✓ SCREENED FIRST, CPU-ONLY, ~75 s per checkpoint: `scratchpad/capacity_screen/FINDINGS.md` + `stream_rank.py`.** Effective rank of each stream's final block output on realcyc (1.25 ep) vs arm 1 (12 ep) -- same architecture, same data, budget the only difference. **A WEAK POSITIVE, and it must not be sold as more: the variance TAIL broadened on every stream** (99%-variance dims 58->62 card, 43->48 note, 62->64 deck, 61->66 preset, 61->65 user; 90% dims up on four of five) so the coarse streams now need 64-66 of their 80 dims, **but the PARTICIPATION RATIO disagrees** and falls on deck (7.4->4.8) and user (10.0->9.2). The model concentrated its top directions while spreading its tail. Zero dead dims at either budget. => neither idle nor obviously starved; the lever is not killed and 38 h is not justified.
**THE LADDER, and option B is the trap:** **(A) ~10.4 h** -- restore the six stripped `user_id`/`preset_id` channel mixers (iter 49's lever, +26,070 params) on a model WARM-STARTED from ws10 with each restored mixer's output projection **zero-initialised**, so the model is bit-identical to ws10 at step 0 (a channel mixer is `x + cmix(x)`) and the 2-epoch decay trains the new capacity; control = arm 1 exactly. **(B) ~23 h at a reduced 2-5 ep WS -- DO NOT RUN: a LARGER model needs MORE budget to saturate, so a reduced-budget arm under-trains exactly the side that is supposed to win, reintroducing the confound that made iters 49/60/61 uninterpretable.** **(C) ~48-60 h** at the full budget, only if A wins.
**✓ OPTION A IS BUILT AND VERIFIED, 2026-09-10 -- `scratchpad/cmixgraft/` (NOT yet launched; arm 2 owns the GPU until ~17:30 on 09-11).** `expand_ckpt.py` writes **`scratchpad/ws10/w10g_ws_109350.pth`** = ws10's WS-final with the six user/preset channel mixers grown from their `d_model=1` dummy shapes (**563,652 -> 641,862 params, +78,210**), refusing unless exactly six mixers change shape and all six `W_v` are exactly zero. `smoke_graft.py` measures **0.000e+00** against arm 1 over 300 deploy predictions in two separate processes, with a **PERTURBED control at 1.132e-06** so the identity cannot pass vacuously. `mk_cmixgraft.py` generates the runner from **arm 1's own runner** by substitution, asserting in BOTH directions (no stale token survives; every substitution fires) -- and its direction-2 grep caught the `run_fixc_arm.cmd` failure shape on the first run, a `echo ===== w10plain START` banner that no substitution touched. Preflight PASS; the param guard (641862) is verified against arm 1's own log convention (563652).
⚠ **A stripped channel mixer is NOT absent from the checkpoint** -- `rwkv_model.py:578-581` builds it from a `d_model=1` dummy config to keep checkpoint interchange working -- so the graft is a SHAPE replacement of 30 tensors, and the first version of `expand_ckpt.py` looked for missing keys, found none, and refused. Worth knowing before anyone writes another checkpoint transformer here.
⚠ **OWED BEFORE LAUNCH: the DYNAMIC dry run.** This is a new phase shape -- the first branch to load a GRAFTED checkpoint whose architecture differs from ws10's -- which is exactly the trigger the dry-run rule names. It cannot run now: co-tenant GPU work is forbidden during a gate-critical run and arm 2 is one.
**QUEUE ORDER AFTER ARM 2 (my call, stated so it can be overridden): `cmixgraft` (10.4 h) THEN the budget curve `w10b2`/`w10b5` (~19 h).** The curve is retrospective -- it says whether the endgame's ~4 days bought anything a 9 h branch would not have -- while the capacity probe is prospective and is the thing Andrew asked about. Both are built and preflight-PASS.
⚠ **State A's asymmetry before running it: a WIN is decisive, a NULL is WEAK** -- two epochs is far less than twelve for brand-new parameters, so A bounds capacity's value from BELOW. A null must not be quoted as "capacity does not bind"; that is exactly the error iter 49 already caused once.

**=> THE FORK THE PREREG NAMED IS NOW LIVE.** Architecture is closed (three screens), out-of-family structure adds nothing (FSRS-7 at a fitted weight of 0.02), and budget is now measured at +0.0005. **The ahead stop criterion is not reachable by any lever now known** -- and QAT still has ~0.0023 to add. imm is already past its criterion with room.
**IMMEDIATE CONSEQUENCE FOR THE QUEUE: the budget-curve branches are now the highest-value ones, not a nicety.** If 10x buys +0.0005, `w10b2` (2.01 ep) and `w10b5` (5.03 ep) say whether ahead saturated at 2 epochs -- i.e. whether the endgame's ~4 days bought anything a 9 h branch would not have. They are BUILT and preflight-PASS. ⚠ They use `RWKV_DECAY_FROM_STEP`, which no run has exercised -- fire `mk_dryrun.py` on them first.

**⚠⚠ 2026-09-10 04:00 -- ARM 2 LAUNCHED ITSELF AND WAS HOLLOW; KILLED AT 04:09 WITH 33 WASTED STEPS.** 255 of 288 batches raised `torch._inductor.exc.InductorError: RuntimeError: Compiler: cl is not found`, and `train_rwkv`'s per-batch except swallowed each one ("Exception caught. Nan from RWKV-7? Skipping batch"), so it stepped at 3.7 steps/min on ~11% of the data while every banner and the `[qat-assert] PASS` looked right. **The ordcut signature exactly: a banner proves construction, not execution.**
**WHY IT IS NEW, and no earlier run could have found it:** `RWKV_QAT_COMPILE=1` has been in the standard env since 2026-07-30 and every plain run (ws10, arm 1) compiles to pure Triton, which needs no host compiler; the QAT graph graph-breaks differently and asks inductor for a C++ kernel. **No earlier QAT run hit it because `scratchpad/qat_tax/run_arm.cmd` never set `RWKV_QAT_COMPILE` at all** -- QAT + compile is a combination that had never run here. `cl.exe` is not on PATH on this box; vcvars64 exists.
**FIX: `call vcvars64.bat` + a `where cl.exe` guard (DONE_EXIT_52) in `run_w10qat.cmd`, NOT `RWKV_QAT_COMPILE=0`** -- that would make arm 2 differ from arm 1 in two ways, and this A/B exists to isolate exactly one (compile is explicitly numerics-perturbing: "small trajectory perturbation acceptable", `train_rwkv.py:101`). Killed in the safe order, waiter cmd FIRST, so no marker could fire downstream; `wait_qat2.log` ends at "launching arm 2" and `w10qat.log` carries no terminal marker.
**VERIFIED BY A DRY RUN, which is the trigger the rule already names (a new phase shape):** `mk_dryrun.py` regenerated the 0.05-epoch / 20-user copy from the FIXED runner -> `CL_OK`, **0 tracebacks**, real steps. 10 min of GPU against a multi-day branch.
**⚠ AND THE DRY RUN CORRECTS ARM 2's BUDGET, which this file had wrong: the "~6.2 h decay" copied arm 1's PLAIN decay time.** QAT runs at ~0.33 steps/s on this trunk (measured on both qat_tax decays, 0.3276 / 0.3298) against arm 1's ~0.98, so **21,870 QAT steps is ~18-23 h, and arm 2 is ~28-33 h end to end** with its ~10 h quant-aware eval. Budget it as a day and a half, not sixteen hours.
⚠ **The dry run's own post-warm-up rate is 0.250 steps/s, i.e. 24% BELOW the 0.33 the compile-free qat_tax decays held** -- so `RWKV_QAT_COMPILE=1` looks like a net LOSS under QAT (the QAT kernels graph-break the fused region, so dynamo pays guard overhead with little left to fuse). **Deliberately NOT acted on:** turning it off would add a second variable to the one A/B that exists to isolate QAT, and 6 h of GPU is far cheaper than an uninterpretable tax number. Worth a standalone A/B outside a gate-critical run -- it would speed up every future QAT run.
**★ FREE FINDING FROM THE REFIT, and it answers a question the record had never asked: A PQ CATALOG DOES NOT SURVIVE A BUDGET CHANGE.** `run_fit_catalogs.cmd` scored the gen-5 pair on **ws10's** states: WKV **0.7596** (it scores 0.39 on realcyc's own states) against a random control of 0.9599, and shift **0.6614 TS / 0.6767 CS** against 0.9638 / 0.9682. Still working -- well inside the encode-to-zero bound, so this is not the q72u failure -- but badly degraded, and the refit recovers to **0.5587** (+26.4%), oracle 0.3126. **=> 10x the budget moves the state distribution enough to stale a catalog fitted on the same architecture and the same data.** Headers are byte-identical to the live pair, so deploy state size is unchanged.

**★★ 2026-09-10 -- THE VALIDATION CURVES SAID SO FIRST, AND THAT IS NOW A REUSABLE INSTRUMENT.** Written before the eval finished and confirmed by it: val predicted ahead ~+0.0002 / imm ~+0.0019, the gate gave +0.000506 / +0.001526 -- both directions right, both within 0.0004. **So a decay branch's 10-user validation endpoint is a usable EARLY READ at zero GPU cost**, provided the reference run's endpoint on the SAME 10 users is read beside it. Not a substitute for the gate (row-weighted, 4 dp, 10 users). Tool: `scratchpad/ws10/plot_losses.py` -> `loss_curves.png`. The original entry follows. Plot + tool: `scratchpad/ws10/plot_losses.py` -> `loss_curves.png` (parses ws10's WS log and arm 1's decay log; zero GPU). Both runs validate on the SAME 10 users (5001-5010, 594,215 rows, `VALIDATE_USERS_START/END` in every toml), so realcyc's decay endpoint is a directly comparable reference:

| | val ahead | val imm |
|---|---|---|
| realcyc, 1.25 epochs | 0.3212 | 0.2976 |
| **ws10 + arm 1, 10+2 epochs** | **0.3210** | **0.2957** |
| move | **-0.0002** | **-0.0019** |

**The endgame premise is +0.0042 ahead** (the 2026-08-11 calibration's 3x step of +0.002, projected log-linearly). Validation says ~5% of that on ahead. On imm it says roughly half.
**⚠ THIS IS A PREVIEW, NOT THE VERDICT, and the two differ in ways that can matter:** validation is 10 users ROW-weighted at 4 decimal places; the gate is 2,499 users BY-USER on the rectified metric. But a 20x gap on ahead is not a weighting artifact, so read arm 1's ahead number as likely to come in near realcyc's 0.298083.
**WHAT THE SHAPE ALREADY SHOWS, and it is the mechanism:** ahead validation is flat from ~epoch 1 (0.3240 at 1 ep -> 0.3230 at 4 -> 0.3209 at WS end) and **the decay moves it by +0.0001, i.e. not at all** -- the stable phase had nothing left for the decay to convert. imm keeps improving throughout and the decay does move it (-0.0012). So the two heads are budget-limited to very different degrees, which no single "+0.0042" number can express.
⚠ Do NOT read this as "the budget curve is settled" -- w10b2/w10b5 exist to measure the SHAPE and arm 1 is one endpoint. It does raise the prior that the curve saturates early on ahead.

**★★★ ANDREW 2026-09-10: "We've already worked on reducing the QAT tax, but maybe we should try again."** Directive accepted; the tax is the right target and the arithmetic says so -- realcyc 0.2981 - 0.0042 (10x budget) + 0.0023 (tax) = 0.2962 against a 0.2950 stop criterion, so the tax is what keeps the criterion out of reach. **SEQUENCE IT AFTER ARM 2, which re-prices it:** the +0.002286 / +0.003486 on record was measured at the 1.25-epoch budget, and arm 2 minus arm 1 is the first single-variable measurement at 10 epochs. Optimising against a stale target is the mistake the q72u catalogs already cost us once.
**DO NOT RE-RUN THESE -- all closed on MECHANISM, not on null results:** the rank-1-friendly regulariser (iter 47 moved the rank-1 floor 43% card / 75% note and the loss did not follow, so the tax is NOT in that term -- rank-2 states, softer or scheduled lambda all aim at the same empty term); 2-bit norm, per-stream norm ranges, learnable norm levels and per-stream catalogs (the LEARNED catalog already absorbs all four -- its centroids are not unit-norm, spread widened 2.43x, and card/note occupy DISJOINT regions with 0.78% shared mass); the teacher swap (iter 54, exact tie -- the tax does not live in the teacher); QAT from scratch (iter 40, +0.0118 -- it MUST warm-start); a longer QAT phase (closure saturates by ~0.37 epochs, so even 1.5 ep of decay is 4x what it needs).
**WHAT IS LEFT, by elimination: the CODEBOOK and NORM terms, and the WKV half is ~14x the shift half.** The WKV bits ladder (8/10/12/14 = 0.4580/0.3776/0.3224/0.2844) is the obvious lever and it is NOT free: +2 bits is ~+1.25 B on the frozen 9 B/card budget (+14%), i.e. a DEPLOY CONTRACT change and therefore **Andrew's call**.
**✓ ANSWERED 2026-09-10, and it is a CONDITIONAL grant, not a blank one -- Andrew: "I'd rather not increase card/note state size, but if that's the only way, alright."** So the budget MAY move, and it moves LAST. **Order: exhaust the zero-byte routes, prove they are exhausted, and only then spend bytes.** Two zero-byte routes are open and neither is on the CLOSED list: (a) **re-fit the catalogs on the model that actually deploys** -- `run_fit_catalogs.cmd` already does this for ws10, and the q72u and gen-5 incidents both show a stale catalog is worth more than any ladder rung; (b) **the chunk structure at CONSTANT total bits** -- on the shift side m2b12 beats m4b6 by 1.9x at identical bits, so the WKV side's m/b split is a free variable that the bits ladder (all at m=1) never varied. Spending bits is the fallback if both come back null.
⚠ **And the price of ASKING is a full A/B, not a ladder read.** Reconstruction error cannot rank two working catalogs (2026-08-15), so a bits rung costs ~10 h decay + ~10 h quant-aware eval to price honestly. Budget one rung, not a sweep. ⚠ And that ladder is RECONSTRUCTION-only, which 2026-08-15 showed cannot even rank two working catalogs -- so it must be priced by a logloss A/B, never by reconstruction error. Ask him whether the state-size budget may move BEFORE building anything on it; if it may not, the remaining room is better centroids at fixed bits, and the only instrument that has ever moved that number is making the catalog LEARNABLE, which is already on.
**★★ 2026-09-09 23:38 -- ARM 1 KILLED ITSELF AT `DONE_EXIT_28` WITH ITS DECAY COMPLETE, AND THE CAUSE IS THAT MY OWN FIX THAT MORNING WAS HALF A FIX.** The 6.2 h decay finished normally (`scratchpad/ws10/w10p_d_21870.pth`, 23:37, final val ahead 0.3210 / imm 0.2957) and the runner then failed its own artifact guard: `if not exist "%DIR%\w10p_d_21870.pth"` looked in `scratchpad\w10plain` while a **decay-only run writes into the SOURCE run's folder**, `scratchpad\ws10`. That is the SAME wrong-directory bug I caught mid-run at 10:12 and fixed only in `write_eval_toml.py`. **A fix that repairs one consumer of a wrong path and leaves the other is not a fix** -- and the LIVE RULES entry claiming the catch "covers every endgame branch uniformly" was wrong: all SIX remaining branches carried the guard identically and would each have burned a full decay before dying. Nothing was lost; ~15 min of GPU idle.
**RECOVERY, in this order:** stopped the arm-2 waiter FIRST (a non-zero marker in a shared log is a live trigger), verified the checkpoint, **proved phase C's toml step by running it** (the fallback resolves `[ckpt-fallback] none in scratchpad/w10plain; decay.toml points at scratchpad/ws10, found 12` -> `w10p_d_21870.pth`), then wrote **`scratchpad/w10plain/run_w10plain_eval.cmd`** = phase C only, env copied verbatim, guard pointed at `ws10`, a FRESH log (`w10plain_eval.log`) because `w10plain.log` now carries a terminal marker. ASCII-only, CRLF, preflight PASS, and **validated by STUB EXECUTION** end to end (DECAY_OK -> EVAL_OK -> DONE_EXIT_0) before launching. Running since 23:41 (pid 20552); arm-2 waiter re-armed as **`wait_w10plain_then_qat_v2.cmd`** on the new log (pid 15352, exercised 340 s in the FOREGROUND past its first `goto` before detaching, per the LF/goto rule). Verdict ~02:40.
**THE DURABLE FIX: all seven runners patched (backups `*.bak_before_ckptdir`) and `preflight_runner.py` grew the missing half of its own check** -- any `if not exist "...<decay prefix>_<step>.pth"` must resolve to the decay's SOURCE folder. Proven non-vacuous against the pre-fix backup (FAILS) and all seven fixed runners PASS. ⚠ The general lesson is the one the dry-run entry already argues: **the STATIC check I added that morning did not cover the guard, and the DYNAMIC dry run would have caught it in 10 minutes** -- these branches were exactly the structural novelty its own trigger names, and I did not fire it.
**★ FEATURE AUDIT 2026-09-09 (Andrew's ask, CPU-only, beside arm 1): `scratchpad/feataudit/FINDINGS.md`.** Read the gen-5 LMDB BACK rather than the derivations, because a column can be derived correctly and then be dropped, mis-ordered or coded so its undefined case is a legal value. **Nothing is discarded**: every raw `-id` column is consumed, the only two absent from the input vector are `state` (Andrew's directive, and verified a NO-OP -- 108 runners incl. iter 53 already set `RWKV_ZERO_FEATURES=22`) and `parent_id` (it derives the deck depth); stored width 69+40=109 matches `w10_ws_109350`'s `features2card`; no leak. **And the `-id` build does NOT lose card metadata vs published** -- 32 users, `note_id_is_nan` 34.84% in BOTH, paired per-user difference +0.000 pp, i.e. the same ROWS; it is missing at the source (a deleted card keeps its revlog rows, loses its card row). ⚠ Consequence worth knowing: on ~35% of rows the deck and preset streams collapse to one synthetic entity, so two of five scopes carry nothing `user_id` does not. **Six coding warts, none an outright loss, all build-time so all needing a rebuild.** Sharpest: **`scaled_sibling_gap` is undefined on 97.4% of rows and writes exactly 0.0, which is INSIDE the defined range** and de-standardizes to "a sibling was reviewed 3.1 h ago" -- `scaled_deck_age_at_review` uses the same sentinel and is fine only because `deck_id_is_nan` flags it, and the sibling column never got its flag. Also: `note_id_is_nan` == `deck_id_is_nan` == `preset_id_is_nan` bit-identically on 100% of rows (one failed card join sets all three, so 2 dims are pure duplicates); `t_since_any_review`'s normalization constant predates its 2026-08-19 end-to-start change, leaving the column at mean -1.11 rather than 0 (affine, so the input FC absorbs it); `is_default_deck` is 100% zero; the decade/century cycles are near-duplicates (r 0.99 review-vs-first, `cyc36500_cos` std 0.011), which is part of why realcyc tied. **Nothing changed** -- fixing any of it costs a ~7 h rebuild and a re-base of the lineage arm 1 is the control for.
**★★★ 2026-09-08 -- THE PHASE TURNED. ANDREW: "keep current model size and do as you suggested:
train WS, then experiment with the decay stage."** He pushed back on a phase-5 recommendation built on
seven consecutive training-side rejects and no architecture at all ("I'd be shocked if there is no
low-hanging fruit"), and on the cost of 10+2 experiments ("it already takes hours, and 10+2 takes >a
day"). The answer to both is ONE expensive WS run that every later experiment branches from.
* **✓ ws10 IS DONE (2026-09-09 17:18): 109,350 steps, ONE attempt, no resume, zero tracebacks,
  38 h wall clock at a steady 0.80 steps/s.** `w10_ws_109350.pth` is the shared branch point for
  every decay experiment. **Arm 1 launched itself at 17:21** -- decay toml verified
  (`LOAD_MODEL_NAME = w10_ws_109350`, `EPOCHS = 2.0`, **`VALIDATE_EVERY = 2000`**, gen-5 db) and
  the param guard reads **563,652**, so the feature flags reached the fetch workers. Expect arm 1
  ~02:30 on 09-10, then the catalog refit and arm 2, all automatic.
* **★ THE REGULARISATION QUESTION IS SETTLED, at the pre-registered checkpoint: STILL FIT-LIMITED
  AT THE FULL BUDGET.** The ahead train/val gap over the late two thirds runs +0.0055 -> +0.0070
  (4 ep), +0.0065 -> +0.0076 (8 ep), **+0.0065 -> +0.0083 (10 ep)** while validation improves
  throughout (0.3237 at 10k -> **0.3209** at the end). iter 65's re-screen condition is NOT met, so
  SAM, wd and dropout stay OUT of the decay queue -- and that is now an answer rather than a trend.
* (superseded) **RUNNING: `ws10`** (`scratchpad/ws10/`, detached, log `ws10.log`) = the endgame's SHARED 10-epoch
  WS phase. Single variable vs realcyc: `EPOCHS` 1 -> 10, i.e. 109,350 steps. Measured **0.79-0.88
  steps/s** including one validation + checkpoint per 1,000 steps -> **~38-40 h**, finishing ~18:00 on
  2026-09-09. `VALIDATE_EVERY=1000` gives 109 resume points and the runner AUTO-RESUMES (make_resume
  + `RWKV_RESUME_SKIP_GROUPS=1`, capped at 40 attempts), because the decay-checkpoint rule below
  makes an unattended 40 h run the one real failure mode.
* **CHAINED: `w10plain`** (`scratchpad/w10plain/`, waiter gates on ws10's `DONE_EXIT_0` line AND the
  `w10_ws_109350.pth` artifact) = **ENDGAME ARM 1**, a 2-epoch decay off the shared checkpoint plus
  the rectified VAL eval, ~10.6 h. It is the REFERENCE every later decay-only branch is gated against.
* **THE CHAIN IS TWO DEEP AND UNATTENDED: ws10 -> arm 1 -> catalog refit -> arm 2** (waiters
  34032 and 34040, both WMI-detached, both exercised in the FOREGROUND past their first `goto`
  before detaching -- the LF/goto rule). Each gates on the previous job's SUCCESS line, never on a
  bare terminal marker. **`run_fit_catalogs.cmd` sits between arm 1 and arm 2**: it converts the
  ws10 WS-final to safetensors (seconds -- `export_weights_only.py`, NOT the ~25 rows/s trace
  export), dumps a corpus with the Rust engine using the EXISTING `reference_realcyc` traces
  (which are model-independent inputs), refits both catalogs, and then scores the realcyc-fitted
  pair on ws10's states -- a free answer to a question the record has never asked, namely whether
  a PQ catalog survives a budget change. It runs BETWEEN the two arms rather than beside arm 1 so
  its k-means never competes with a training run's fetch workers; ~15 min of idle GPU buys zero
  contention.
* **ARM 2 IS BUILT AND ITS DESIGN QUESTION IS ANSWERED BY THE SHARED-WS SHAPE** (`scratchpad/w10qat/`,
  generator + runner, preflight PASS, phase-0 refusal verified BY EXECUTION at rc 51). The old plan
  asked whether arm 2 should be (A) a warm-started QAT fine-tune on arm 1's FINAL or (B) a second
  full 10x with QAT throughout, and recommended A. Branching gives a third and better option: **the
  SAME 2-epoch decay from the SAME WS checkpoint with the QAT env added and nothing else changed.**
  It is how QAT is actually deployed here (decay-only, warm-started -- iter 40 says it MUST
  warm-start), and it is a genuinely SINGLE-VARIABLE A/B, which neither A nor B can be: A adds ~2
  epochs on top of arm 1, B changes the whole trajectory. **So arm 2 - arm 1 IS the QAT tax at the
  10-epoch budget, with nothing to subtract.** Budget ~6.2 h decay + ~10 h QUANT-AWARE eval (not the
  ~2.9 h a plain eval takes). Phase 0 REFUSES without `reference/pq_cb_{wkv,shift}_ws10_*.txt` --
  see the catalog-staleness entry in the QAT section; fitting them from ws10's WS-final is ~10 min
  of CPU and is a launch prerequisite, not a nicety.
* **=> A DECAY-ONLY BRANCH COSTS ~10.4 h**, today's price, in the 10+2 regime.
  **THE QUEUE, REVISED 2026-09-08 after the gap screen came back negative:**
  1. **arm 1 `w10plain`** -- the plain 10+2 number and the control every branch is gated against.
  2. **arm 2 `w10qat`** (BUILT) -- the QAT tax, now a single-variable A/B against arm 1.
  3. **DECAY LENGTH, `mk_w10decay.py <epochs>`** (BUILT for 1.0 and 3.0, both preflight PASS) --
     the queue's real lever. The 10+2 split was chosen on COST (6+6 = 89 h vs 10+2 = 53 h) and this
     file already says the ~+0.0006 credited to a longer decay is "being SPENT, not disproven",
     because iter 34's decay_ratio gain is confounded with a budget change. A shared WS checkpoint
     turns that from a 43 h question into a 9 h one, with arm 1 as an exact control.
     ⚠ It varies decay length at a FIXED 10-epoch WS, so total budget moves with it. That is the
     endgame's operational question, NOT the fixed-total-budget WS:decay de-confound, which stays
     Andrew's call. Do not report one as the other.
  4. **BUDGET CURVE, `mk_w10budget.py <ws_step>`** (BUILT for WS 22,000 and 55,000 = 2.01 and 5.03
     epochs, both preflight PASS) -- arm 1 tests the endgame's ENDPOINT, this tests its SHAPE.
     "+0.0042 at 10x" is a log-linear extrapolation from ONE measured 3x step, and under WSD the
     stable phase is at constant LR, so decaying from an intermediate checkpoint IS a
     shorter-budget run: one expensive WS buys several budget points for the price of their
     decays. Needs `RWKV_DECAY_FROM_STEP` (added to `write_decay_setup.py`, default unset = the
     latest checkpoint, so every existing runner is byte-identical) and the runner GUARDS that the
     pin was honoured -- an ignored pin silently reproduces arm 1 under a curve point's tag.
     **Eval is 300 users (~20 min), coarse BY DESIGN**: the question is linear-or-saturating at a
     scale of 0.002-0.004, well above a 300-user resolution, and the 200-user lesson forbids
     settling sub-0.001 effects there. Arm 1's own 2,499-user jsonl already contains those 300
     users, so its curve point is FREE. ⚠ A curve point is NOT a gate candidate; never log one as
     an iteration.
  5. decay SHAPE (iter 56 `linear`: sub-bar at 1.25 ep but real on imm at p=6e-12; a 2-epoch decay
     gives the shape more room than a 0.25-epoch one did).
  **SAM and the other regularisers are NOT in this queue** -- the gap screen below says iter 65's
  re-screen condition is not met.
* **★ THE REGULARISATION SCREEN IS RUNNING FOR FREE, AND SO FAR IT SAYS NO** (`scratchpad/ws10/
  gap_trace.py`, reads ws10's own log; zero GPU). iter 65 deprioritised SAM/wd/dropout on the
  finding that the model is FIT-limited at 1.25 epochs, with an explicit condition attached --
  re-screen "on the 10x endgame's checkpoints, where a gap CAN exist". ws10 validates every 1,000
  steps on 594,215 held-out rows, so the screen needs no run of its own. **Through 8.0 epochs the ahead train/val gap is FLAT** -- +0.0055 -> +0.0070 at 4 epochs and
  +0.0065 -> +0.0076 at 8, both inside the noise of a 400-step train window, while validation
  keeps creeping down (0.3237 at 10k -> 0.3230 at 44k -> 0.3224 at 88k). => regularisation stays
  unmotivated at four fifths of the budget; re-run the screen at WS end, which is the
  pre-registered checkpoint, before queueing any of those branches.
* **⚠ AND A CAUTION ABOUT THE +0.0042 PREMISE, stated now so it is not a surprise later: the
  STABLE-phase validation is nearly exhausted by epoch 1.** ahead val goes 0.3342 (step 1k) ->
  0.3237 (10k) -> **0.3230 (44k)**, i.e. -0.0007 over three further epochs. That is EXPECTED under
  WSD -- at constant peak LR the loss sits at an LR noise floor and the gain is converted by the
  DECAY, so this is not evidence against the budget premise. But it does mean the premise is
  UNCONFIRMED until arm 1's decay lands, and arm 1 is therefore the number that decides whether
  phase 6 was worth its ~4 days.
* ⚠ **A units trap the screen had to be fixed for: training `imm_loss` is the 4-way CROSS-ENTROPY
  of the rating head, while validation `imm` is the BINARY logloss of 1-P(Again).** Different
  quantities, which is the entire ~-0.24 "imm gap" a naive read shows. Only AHEAD is comparable
  between the two logs.
* ⚠ **Two things the shared checkpoint BAKES IN, named now because they are not discoverable later:**
  wd and dropout act during WS, and `WARMUP_STEPS` stays 400 (0.37% of this run vs upstream's ~9%) so
  that budget is the single variable. A branch cannot re-open either.

**★★ THE ARCHITECTURE QUESTION IS ANSWERED, AND IT REDIRECTS THE PHASE
(`scratchpad/arch_2026-09-07/FINDINGS.md`; ~2 h of CPU, zero GPU).** Four screens:
1. the curve head's hidden is effectively **2.4-dimensional** and is a SUFFICIENT STATISTIC of its
   own input -- adding the whole 80-d trunk output helps on **0 of 6 users**. A richer ahead head is
   dead, and so is replacing the curve family.
2. realcyc vs the pretrained d=128 model: **residual correlation 0.9957** across a 5x parameter change
   AND a complete feature-layout change. The 2026-07-03 "family saturated" result extends.
3. **THE OUT-OF-FAMILY CONTRAST (2026-09-08): FSRS-7 ADDS NOTHING.** Per-review predictions from
   `srs-benchmark`'s own `script.process` (34-param dual-stability FSRS-7, fitted per user; plus
   FSRS-6 and FSRS-7-`--default` as the version-bump and personalization controls), 340,601 rows on
   4 VAL users. **Best blend weight with realcyc: 0.02, gain +0.00001** (0.00 on three of four
   users); leave-one-user-out stacking is NEGATIVE. The d=128 partner takes **weight 0.53 for
   +0.00086**. => the remaining ahead error is not a blind spot this family shares; what pays is
   capacity and BUDGET, which is exactly what ws10 spends.
   **★ METHOD, and it would have inverted the verdict: RESIDUAL CORRELATION OVERSTATES
   DECORRELATION WHEN THE MODELS DIFFER IN QUALITY** -- FSRS-7's 0.9760 reads as structure and the
   fitted weight says it is FSRS's own error. Price a disagreement with a FITTED blend weight.
   ⚠ The join guard earned its keep: our dumps key on `review_th` (the row predicted FROM), FSRS on
   the row PREDICTED, and a direct join gave label agreement **0.7726 against a chance rate of
   0.745**. `scratchpad/fsrs7/label_map.py` rebuilds the map from the DATA, so the fix cost minutes
   instead of 2.7 h of re-dumping.
**=> DO NOT open a new architecture family on "there must be low-hanging fruit". Three independent
screens now say the ahead residual is a FIT gap.** Levers that remain untested are trunk-side and
cheap-to-fit rather than richer: cross-stream state fusion, extra rounds via layer reuse, multi-step
delta on the coarse streams, a richer feature encoder.

**⟶ ARCHIVED 2026-09-11: every LIVE entry dated BEFORE 2026-09-08 moved VERBATIM to `optimization/HISTORY.md` "CLAUDE.md LIVE archive (moved out 2026-09-11)"** -- iters 45-69, the gen 2-5 rebuilds, the 2026-08-17/08-19 chain states, the featB/gen4base/realcyc ops. Per-iteration detail stays in `research_5k_verbose.md`; numbers in `research_log.jsonl` / `research_5k.md`. **What those entries established that is still OPERATIVE:**
* **THE LINEAGE.** gen 5 (`*_id5`: real-time cycles, 109-dim input, 563,652 params) is the features lineage's db and **`realcyc` (0.298083 / 0.263592, n=2,499) is its reference** -- realcyc tied gen4base and was ADOPTED as the input layout on Andrew's directive. Iters 62-69 were all gated against it and **all REJECTED**: 62 `lorawd` tie, 63 `durdrop` regression, 64 `ordcut` ahead -0.0032, 65 `sam` regression, 66 `hord` tie leaning positive, 67 `muonscale` imm +0.000109 with ahead a certified null, 68 `eqw` regression in both modes, 69 `muongates` clean null. Size baseline for gen 4/5: `optimization/size_baseline_id_e2s.json` (126,657,015 scored reviews). **User 6701 is EXCLUDED from the gen-4/5 VAL set (n = 2,499, `eval_sharded.py --exclude 6701`)** -- its eval OOMs deterministically at 36.09 GiB; every gate pairs on the intersection.
* **STILL OPEN FOR ANDREW, from those entries:** (1) the **Pareto half-accept** -- iter 67 cleared the bar on imm at p=5e-12 while ahead was a certified null; the gate as written rejects it. (2) **Chunk-continuous training** (`scratchpad/proposals_2026-09-06/CHUNKING_CORRECTION.md`): the EVAL does not chunk (median 1 chunk/user) but TRAINING does (median 4, 12.5% of train rows within 2,048 of a boundary), so training computes rows from a cold state that eval and deploy never impose. The stateful WKV kernel is done; steps 1-3 are multi-day.
* **DISK.** `F:/rwkv_lmdb/train_db_5k_h1_id5` is a JUNCTION to `C:\rwkv_lmdb\train_db_5k_h1_id5` (moved 2026-09-03: fetch wait 3.4 s -> 0.022 s per step); `test_db_5k_id5` is a real directory on F:. `train_db_5k_h1_id3` was DELETED (Andrew 2026-09-03). ⚠ The junction list in LIVE RULES predates both.
* **RUST PARITY IS CERTIFIED FOR GEN 5:** `reference_realcyc/`, PASS at imm 0.000000 / ahead 0.000000, max per-review 3.41e-06. `FEATURE_DIM` is derived from `features2card.0.weight`. `export_rnn_trace.py` uses the shared `get_rwkv_data` path under `RWKV_ID_FEATURES=1` (before 2026-09-07 the `-id` lineage could not export a trace at all).
* **DECIDED THEN, STILL IN FORCE:** augmentation OFF; KD SKIPPED for the research lineage (Andrew 2026-09-04: "Let's skip KD") -- QAT-KD (2026-09-10) is the one exception; the LoRA Muon group keeps **wd=0** (iter 62: its norm growth is gradient-driven and restoring, and damping it bought nothing); `RWKV_trained_on_5000_10000.pth` is DISQUALIFIED as a teacher (it trained on our VAL and TEST halves, and no gate would catch the leak).
* **FEATURE RELIANCE** (LOO + group ablation on featB's checkpoint, n=300): `t_since_any_review` carries the WHOLE clock gain (+0.001325 ahead / +0.005076 imm); the pseudo cycles were dead weight. ⚠ **Read this again beside the 2026-09-10 clock leak: the column the model relied on most is the one that leaks.**
* **DATA FACTS.** `review_time = id - taken_millis` = SHOW time (verified in the data). `elapsed_seconds` can go negative because it is diffed in protobuf order and re-sorted afterwards; it is clamped to 0. Bug B (the `insert_probes` KeyError on `-id`) is ROW ORDERING caused by the 60 s `taken_millis` cap, not an id collision; unpairable targets are filtered and reported. Ids are stored int64 and `nan_id_fill` is exact, but **the id fixes' ACCURACY value is UNMEASURED** (featA vs featA2 straddled the sentinel fix `c7883dc`). **Two runs are comparable only if their DATABASES are -- date a db by when it was BUILT.** LMDB `map_size` is SPARSE on Windows: measure with `GetCompressedFileSize`. Feature ablation by name: `RWKV_ABLATE_FEATURES` (unknown names raise).
* **CLOSED ON MECHANISM -- do not re-propose:** delta-rule removal or `a`-simplification (zeroing `a` costs +0.208 imm; "barely engaged" is an eigenvalue statement, not a functional one); the negative-eigenvalue `a` range; the four norm/catalog levers; ahead-vs-imm routing (iters 46, 48); rank-1 regularisation (iter 47); PolarExpress (p(1) < 1 is load-bearing); a new intermediate scope (iter 50); withholding the current duration (iters 18, 33, 63); explicit regularisers at the 1.25-epoch budget (iter 65); the shared-logit ordinal term (iter 64) unless its scale is decoupled.
* **METHOD RULES, each of which cost a run or a false alarm:** a banner proves construction, not execution -- read the first minutes of a WS log for `Traceback`; smoke a flag-gated LOSS term through the real `get_loss` on a real chunk; a lever that adds a train-only Parameter must add its name to `run_as_rnn.TRAIN_ONLY_KEYS` and prove the deploy loader still loads its checkpoint; run `scratchpad/parity3/smoke_scripted_eval.sh` before any launch that touches `srs_model.py` / `rwkv_model.py` (only a PLAIN eval scripts the model); validate an engagement criterion against a control that CANNOT have changed; a typical-case statistic (a median, a resting value) cannot bound a worst case -- report the max; hold METRIC, TRAJECTORY and SIGNAL fixed before attributing a difference; never time a run inside compile warm-up; cost a new stream on B (sequences) as well as on rows; run a minutes-of-CPU screen before every build (four screens re-ranked six candidates); CPU training of this arch runs with `DTYPE = "bfloat16"` (~5 min/step) and is a real end-to-end harness for loop-level changes -- run it at BelowNormal priority beside a GPU run.
* **OPS RULES:** gate a waiter on the SPECIFIC success line, never on a bare `DONE_EXIT_`, and never reuse a log that already carries a marker (a failure marker in a shared log is a live trigger for everything downstream); check a detached waiter is still ALIVE an hour after arming -- a returned pid proves the launch, not the life; after any reboot, check `Get-ScheduledTask` for logon-triggered jobs before trusting the GPU is free (the finished GRU/FSRS tasks were disabled 2026-09-02); a giant-user eval needs ~40 GB of FREE SYSTEM RAM (the WDDM shared pool), so never overlap it with a rebuild; `data_processing`'s writer queue is now bounded (a progress bar measures the producer, not the store); write the terminal marker BEFORE `endlocal` (preflight asserts it); a stub that replaces a batch command must be called with `call` or `cmd /c` (a bare `.cmd` does not return).
* **BUDGET CALIBRATION (2026-08-11): gate AND screen at FULL budget only.** A short-budget run compresses effects to ~0.65 of full size and roughly doubles the effective accept bar on ahead.

### ★★ THE DEPLOY CONTRACT -- ONE QUANTITY IN ALL THREE PATHS (Andrew, 2026-07-27)
> *"Everywhere (train+eval+CPU inference): duration of the most recent review zeroed out + PAVA + no piecewise correction. And yes, train with zeroing as iter 33."*

**The contract:** 1. the most recent review's duration zeroed, 2. PAVA rectification applied, 3. no piecewise ahead correction (`RWKV_NO_AHEAD_RESIDUAL=1`, in every run).
**The gate is therefore the RECTIFIED metric** (`RWKV_EVAL_PAVA=1`) from iter 33 on. Pre-iter-31 rows are unrectified and NOT comparable; do not retro-score them (the rect-vs-unrect delta is model-dependent: A18 +0.003588 vs iter 31 +0.001893).

**Three measured facts that still drive decisions:**
- **Training under PAVA halves the deploy rectification cost** -- A18, never trained under the constraint, pays +0.003588 on ahead; iter 31 pays +0.001893.
- **~70% of the deploy penalty is the lost current-row duration, ~30% is PAVA pooling** (+0.001451 vs +0.000611, from the mode-2/mode-3 decomposition). So PAVA lambda can only ever attack the smaller half.
- **Probe-insertion noise is channel- and model-dependent, and ZERO on ahead** (iter 31, n=500, `RWKV_EVAL_PAVA=3`: ahead +0.000000 +/- 0.000014, p=0.33). Measure the control for the model in hand rather than quoting a fixed magnitude; the "never compare rectified to unrectified at the 0.0001 gate" rule stands for **imm**.

**⚠ `RWKV_ZERO_FEATURES` WAS MISSING FROM `rust/rwkv-infer` -- FOUND AND FIXED 2026-07-27, AND IT WAS A LIVE BUG.** The mask lives inside the Python module, so the exported trace carries RAW features and the engine consumed columns Python had thrown away. With the mask applied, iter 31's max per-review |rust-python| falls from 1.59e-3 to **2.28e-6** (~700x). It survived the gate because the gate scores MEAN LogLoss and the column's weight is small. **This CORRECTS section 11's old explanation of the per-review spread as "accumulated float divergence" -- it was a formula error, and a large per-review spread should be read as a SIGNAL that the two paths compute different formulas.** Fix: `model.rs::load` zeroes the named input columns of `features2card.0.weight` once at load (zeroing input column j == zeroing feature j, since `y = Wx+b` is linear).
**→ OPEN RECOMMENDATION (Andrew's call):** bake the mask into the exported safetensors at export time instead of applying it from an env var at load, so the deploy artifact is correct for any consumer -- Anki will not be setting `RWKV_ZERO_FEATURES`.

Full narrative (the global-vs-surgical zeroing design, the iter-33 implementation fork, and the superseded gate-vs-deploy question): `HISTORY.md` (archived 2026-08-10).

#### QUEUE
**★★★ SPEEDUP PHASE CLOSED 2026-07-30 — ADOPTED STACK = 1.68x, and MAX=65536 IS ACCEPTED.**
Andrew 2026-07-30: *"Accept it, do compaction and then run the HP tuner."* Full measurements in
`optimization/TRAINING_SPEED.md`; the operative facts:

**THE STANDARD TRAINING ENV — put ALL of these in every new run `.cmd` from now on:**

> ★ **2026-09-01: nothing to add here for the PermGather speedup — it is a CODE fix, on by
> default** (`RWKV_PERM_GATHER` defaults to "1"; `=0` is the escape hatch). It is bit-identical, so
> runs before and after it are directly comparable and no env var records which side you are on.
> Worth +5.5% throughput. Detail in the speed section above and `optimization/DISPATCH_PLAN.md`.

    set RWKV_MUON_BATCHED=1     REM batched Newton-Schulz, 35x fewer matmul dispatches
    set RWKV_NO_JIT=1           REM required by torch.compile (worth ~0 alone: 1.003x)
    set RWKV_QAT_COMPILE=1      REM fuses the 26 mixer forwards

**★★★ AND THE END-TO-START DBS, IN TRAIN *AND* EVAL (Andrew 2026-08-30, verbatim: "e2s should be
used both in train AND eval. That should be the new default for all future runs").**

    TRAIN_DATASET_LMDB_PATH = "F:/rwkv_lmdb/train_db_5k_h1_e2s"
    set RWKV_VAL_DB=F:/rwkv_lmdb/test_db_5k_e2s
    set RWKV_EVAL_DB=F:/rwkv_lmdb/test_db_5k_e2s

**WHY IT IS A DEFAULT AND NOT AN EXPERIMENT: it closes a train/deploy divergence** (§9 case 4). A
live Anki scheduler computes `now() - last_review_time` = **end-to-START**, and structurally cannot
do otherwise, because `duration(k)` has not happened when the prediction is made. Training on
end-to-END fed the model a quantity deploy can never supply -- and one that correlates with the
outcome (at a fixed gap, `duration(k)` predicts failure at AUC 0.618). We already zero the most
recent duration as a FEATURE for exactly that reason, then handed it back inside the interval.
**Set them EXPLICITLY; the defaults in `write_eval_toml.py` / `write_decay_setup.py` stay on the old
paths so existing runners remain byte-reproducible** -- the same convention as the speed flags above.

⚠ **THREE CONSEQUENCES, none optional:**
1. **THIS RE-BASES THE CHAMPION.** Every number in the record -- iter 53 included -- is end-to-END.
   An e2s run is NOT comparable to them, so the champion recipe must be re-run on the e2s dbs to
   establish the new baseline before any candidate is gated against it.
2. **★ THE KD DUMP MUST BE REGENERATED FIRST, and nothing will tell you if it is not.**
   `C:\rwkv_kd_dump\t128_seedpair_65k` holds teacher logits computed on end-to-END inputs. Its only
   identity check is `labels_sum`, and labels are RATINGS, which the interval does not touch -- so a
   champion re-run on e2s dbs would PASS the checksum while distilling toward predictions for a
   different input. Identical in shape to the augmentation/KD incompatibility already recorded: the
   checksum proves LABEL alignment and gets read as proving BATCH alignment. Regenerate the dump
   with the teacher forwarding the e2s batches.
3. **The `-id` datasets already do this automatically** -- `elapsed_end_to_start` is gated on the
   presence of `review_time`, not on a flag -- so a future `-id` rebuild needs no change, and the
   featB confound noted elsewhere in this file is resolved in the direction of e2s.

plus **`MAX_TRAIN_GLOBAL_LEN = 65536`** and **`NUM_FETCH_PROCESSES = 2`** in the toml.
Defaults stay OFF in code, so these must be set EXPLICITLY; old runs stay reproducible.
Result: WS 4h23m -> **2h37m**, decay 63 -> **40 min** (1.68x). Eval: use `--fetch-per-shard 2`.

⚠ **MAX=65536 COSTS ~0.0003 IN BOTH MODES at the OLD LR** (ahead -0.000264, imm -0.000307 vs
iter 31 rectified). That is a real-but-small systematic loss, not noise: both modes moved the
SAME direction, whereas the accuracy-neutral combo went +0.000064 / -0.000047 (one up, one down).
Mechanism: groups 22,346 -> **10,935**, i.e. HALF the optimizer steps per epoch at unchanged LR.
**Andrew accepted it ANYWAY and directed HP tuning to recover the 0.0003** — batch size is
structural and LR/warmup are tuned after it (methodology (f)). Do NOT treat the -0.0003 as
permanent; it is the tuner's target.

**HP TUNING IS CLOSED** -- the 2026-07-30 tuner run became **iter 34**, which recovered the
MAX=65536 cost and was the phase's largest gain; the champion HPs have since been confirmed
against 19 alternatives at full eval. The ~44-line LIVE description of that tuner (recipe, lever
order, val-prune reference, per-trial bar) is archived to `HISTORY.md`. **Four things from it stay
operative:**
* **⚠ Tuning `PEAK_LR` alone moves only ~10% of the weights.** Muon has its OWN base LR
  (`RWKV_MUON_LR`) and the schedulers scale it proportionally (`train_rwkv.py:188-196`), so the
  AdamW group is 57,412 params against Muon's 500,800. Any future LR work must move BOTH.
* **1.253 steps/s** is the reference rate for this trunk at MAX=65536 -- but it is the **PLAIN,
  no-KD** recipe. A KD run is ~0.92 steps/s (measured on iters 45 and 53). Do not use one as the
  other's baseline; that mistake read a free flag as 27% slower on 2026-08-17.
* Journal `optimization/tuner_5k_log.jsonl`; the d=32/QAT-era rows are archived to
  `tuner_5k_log_d32qat_era.jsonl` (different arch AND batch -- not comparable).
* Its three runner guards are now generalized in **`scratchpad/preflight_runner.py`** -- run that
  before arming ANY runner.

### ★ THE ORDER FROM HERE
The 2026-08-01 ordering (finish HP tuning -> seed pair -> PAVA lambda) is **COMPLETE** -- those became iters 34, 35 and 36. The ~340-line queue that tracked it, including the speedup phase's ranked list and the per-item DONE annotations, is archived to `HISTORY.md` (2026-08-10). What remains live:

1. **The algorithmic loop** (the endgame's step 1) -- **the ranked proposal queue now lives in `optimization/PROPOSALS.md`**, along with Andrew's 3-agent generation protocol (three subagents with DIFFERENT priors -- literature / domain / reject-log steelman -- each write 5 proposals from >=2 families; rank all 15; implement the top). ⚠ **WRITE THE RANKED LIST TO THAT FILE THE MOMENT IT IS PRODUCED:** the 2026-08-10 ranking lived only in the transcript and a compaction destroyed items 7-15 permanently.
2. **NEW INPUT FEATURES -- the long-lead item. ⚠ ONLY THE PREPROCESSING IS CPU-ONLY (corrected by Andrew 2026-08-12: "Pre-processing is CPU-only, sure, but training is obviously not").** This line used to claim features "do not compete with the GPU loop" -- WRONG, and it would have led to planning them as a free parallel track. Only the ~2-4 day LMDB rebuild overlaps the loop. Everything that makes features *count* -- re-basing the champion on the new inputs, then training + evaluating each candidate -- is GPU work on the same single 4070, and every pre-rebuild iteration is gated against a champion the rebuild invalidates. Features are a PHASE that largely DISPLACES the algorithmic loop, not a parallel one. Fully scoped in `optimization/FUTURE_FEATURES.md`: the four code sites, the F:-side-by-side disk plan (605 GB against 889 GB free -- no delete needed), the measured constants, the ~23 h build, the NaN-clamp landmine, and Andrew's directive that the rebuild DROP Anki's card-state input (dim 22). ⚠ It moves the `size` gate: the filter amplifies a 0.001% raw-row difference into ~30% of users getting a different equalized count, so gate #1 must be read as *within a rebuild generation*.
3. **THEN the 10x-budget run, ONCE, on the final champion** -- see THE ENDGAME above for the two arms (plain, then warm-started QAT), the ~4-day cost, and the three 1-epoch assumptions (warmup 200, augmentation off, wd/dropout) that must be reconsidered first.
4. **Rust port** (`rust/rwkv-infer/TRACK2_PORT_PLAN.md`) -- **★ GAPS 7 + 8 CLOSED AND PARITY-VERIFIED 2026-08-11** (`276f379`): both engines (candle + the default fast path) now run the interleaved schedule (`RWKV_INTERLEAVE=1`) and the reordered stream list (`RWKV_STREAM_ORDER=card,note,deck,preset,user`), against a fresh `reference_iter41/` trace that is self-contained at exactly 0.000e+00 -- interleaved PARITY PASS on both paths (max per-review 4.78e-06 / 1.25e-06) and the sequential path BIT-IDENTICAL to the green iter-31 preds on both. **Gap 8 was a LIVE cross-wiring bug the gate caught** (states were assembled positionally, so `_cnd` fed DECK's state into the NOTE module; `name_to_idx` likewise would have quantized card+DECK for `card,note` scopes). ⚠ Front-loaded placement only -- fine today (iter 44's spread was rejected), but a future spread adoption needs `interleave_schedule()`'s table, not `r < depth[m]`. Remaining measured items and the AGPL/SIMD note are in that plan.

**★★ THE 2026-08-17 HARD FREEZE HAS A DIAGNOSED CAUSE, AND IT IS THIS RULE FIRING (first time with
telemetry).** At 12:53 the box froze mid-eval; Andrew forced a dump with **RightCtrl + Space x2** and
Windows captured it -- bugcheck **0xE2 MANUALLY_INITIATED_CRASH** (all four params zero = the
keyboard-forced path, NOT the underlying fault) plus a 4.4 GB `MEMORY.DMP` and minidump
`081726-8234-01.dmp`. **The flight recorder caught the approach**, which the July hangs did not have:
VRAM pinned at **11,981 of 12,282 MiB (97.6%)** for the last three samples, then the recorder
**stopped dead at 12:53:01**.
⚠ **CORRECTION, measured the same day: low power with high util is NOT the tell.** This entry first
read the accompanying 42-51 W (at util 99%) as "stalled in paging, not computing", and the very next
giant user REFUTED that: on the resume, user 6104 ran at **11,864 MiB / util 97-100% / 51-53 W for
minutes and COMPLETED NORMALLY**, then VRAM fell to 6,756 MiB and power returned to 110 W. Low power
at high util is just what a memory-bound giant user looks like on this card, and it triggered a false
alarm within the hour. **The only reliable freeze signal is the ABSENCE OF FORWARD PROGRESS** -- the
flight recorder ceasing to log, or the shard log's byte count not growing. Check `stat` on
`shard_s0.log` over ~25 s before concluding anything from a power reading. ⚠ Do NOT generalize this to the July black-screens, which
are recorded as having zero telemetry precursor; same symptom, not yet the same proven cause.
**THE TRIGGER: giant user 6104, work 1,274,765** -- ~3.5x the 5002/5905/5995 trio this rule names --
hit inline in a process that had already run 1,103 users, with a heavy desktop (Anki, 2x Edge,
2x Chrome, Word, Excel, Telegram, Steam, Razer). Desktop VRAM was ~1.1 GB after reboot vs the 4.6 GB
this rule cites for its three failures.
**★ AND THE CONFIGURATION IS THE FIXABLE PART: the runners pass `--solo-threshold 0`, which DISABLES
the power-user solo phase** Andrew approved 2026-07-14 for exactly these users. A dry run shows
**25 of the 2500 VAL users are >= 1,000,000 work (11.2% of total work)**, 6104 among them; with the
solo phase on they run alone, first, in their own process. The live rule above says d=80 evals use
`--shards 1 --solo-threshold 0`, and those are INDEPENDENT axes -- `--shards` controls parallel
shards, `--solo-threshold` controls giant isolation. **Turning the solo phase back on is not free
mid-run**: `merge_jsonl` ASSERTS no duplicate users across phases, and 7 of the 25 giants
(5414/5626/5835/5859/5900/5991/6007) are already banked in `-s0.jsonl`, so they would be re-run and
then collide at merge. Fix it at LAUNCH time, not on a resume.
**RECOVERY, and it was cheap:** the 13 h decay phase survived entirely (`qtaxg_i45kd_d_10935.pth` +
both exported catalogs), and `get_result` skips users already present in the output jsonls, so a
plain relaunch resumes at 6104 **in a fresh process** -- most of what the solo phase would buy (clean
allocator, giant first). Runner `scratchpad/qat_tax/run_i45kd_evalresume.cmd` is phase-B-only, sliced
from the chain runner so the env is byte-identical, with asserts that no training/dump phase leaked
in and that nothing deletes the result jsonls.
⚠ **A dirty shutdown truncates the chain log's last write to NUL bytes** (76 of them here, from a
second incident the same afternoon when the PC was switched off). Harmless to the anchored
`findstr /B /C:"DONE_EXIT_"` waiters, but strip them before relaunching or the next append lands
after the padding.

**⚠⚠ POWER OUTAGE 2026-08-18 ~10:32 -- RECOVERED BY MID-EPOCH RESUME, and it exposed a latent bug
in `make_resume.py`.** The box lost power mid-WS; boot 10:33, all four waiters dead, GPU idle.
iter 54 had reached step **8110 of 10,935** with a clean checkpoint pair at 8000 and **zero NUL
bytes** in its log, so ~2.4 h of training was salvageable for ~110 lost steps. Recovery ran the
documented path (`make_resume.py` + `RWKV_RESUME_SKIP_GROUPS=1`) and is confirmed live:
`[resume-skip] epoch 0: skipping the first 8000 already-trained groups`.
**★ THE BUG, and it is the day's recurring shape: `make_resume.py` REPLACED ONLY KEYS THAT
ALREADY EXISTED.** Every runner's WS toml is cloned from a from-scratch config, which declares
`LOAD_MODEL` and `STEP_OFFSET` but NOT `LOAD_MODEL_FOLDER` / `LOAD_MODEL_NAME` -- so those two
were silently never written and the resume died on `AttributeError: 'Namespace' object has no
attribute 'LOAD_MODEL_FOLDER'`. **`train_rwkv` SWALLOWED it and exited 0**, so the runner logged
`WS OK` after 8 seconds and marched on to decay a half-trained model; it was killed within two
minutes and no artifact was lost. Fixed to REPLACE-OR-APPEND with an assert on the OUTPUT toml.
**TWO RULES:** (1) a config transformer must assert the keys are present in what it WROTE, not
trust that they were there to rewrite -- same family as the mk53/mk54 slice and the `endlocal`
marker; (2) **a runner phase must gate on the ARTIFACT, not the exit code** -- the resume runner
now refuses to decay unless `i54_ws_10935.pth` exists, because CLAUDE.md's own warning that
"train_rwkv can swallow fatal errors to exit 0" is exactly what happened.
⚠ Also: `wait_then_iter52.cmd` polls the **QAT#2** log, which already carries a terminal marker,
so re-arming it would have fired instantly and run iter 52 BESIDE iter 54. Replaced by
`wait_then_iter52_v2.cmd`, pointed at `iter54.log`. **After any outage, re-check what each
waiter polls before re-arming it** -- a waiter is only as correct as the log it watches.
⚠ The resumed tail's DROPOUT DRAWS differ from an uninterrupted run (weights/optimizer exact),
so iter 54's number is a fair measurement but the run is not bit-reproducible.

**⚠⚠ THE SINGLE-WITNESS RULE RECURRED 2026-08-20, IN A WATCHER WRITTEN THE SAME DAY -- SO STATE IT
SHARPLY: NEVER IDENTIFY A CHAINED RUNNER BY PID.** A featA watcher pinned the WS phase's pid. A
runner is a CHAIN (WS -> decay -> eval) and every phase is a NEW process, so the normal transition
two minutes after `featA WS_OK` read as `featA DOWN`. Nothing was wrong; the alert measured its own
witness. **Identify the runner by COMMAND LINE** (the `cmd.exe` wrapper spans all phases) **and
require TWO witnesses**: process gone AND no terminal marker in its log. A finished chain writes the
marker, so gone+marker is SUCCESS and only gone+no-marker is a death. Knowing the rule below was not
enough to avoid re-implementing the bug -- which is the argument for the two-witness pattern being
the DEFAULT shape of any monitor here, not a fix applied after a false alarm.

**⚠ THE FLIGHT-RECORDER HANG SIGNAL BREAKS AT MIDNIGHT (2026-08-18, one false alarm).** The
recorder writes `flight_YYYYMMDD.csv`, so at 23:59:47 it stops appending to yesterday's file and
starts today's. A monitor that resolves the filename ONCE at launch then watches a file nothing
will ever write to again and fires ~10 min later -- 23:59:47 + 624 s, exactly when the alert came,
while training was advancing at 0.967 steps/s. **Chains here run ~25 h, so every one crosses
midnight.** Fixed in `scratchpad/chain_monitor.sh`, which (1) re-resolves the newest `flight_*.csv`
each poll and (2) **never declares a hang from the recorder alone** -- a hang stops the BOX, so it
must also stop the training log; requiring BOTH witnesses means one signal failing costs a log
line instead of a false alarm. That second fix is the general one: **an alert built on a single
witness reports the witness's health, not the system's.**

**⚠ BIG-EVAL OPS RULE (learned 2026-07-29/30):** giant users (5002/5905/5995, 266k-367k reviews) OOM the 12 GB card **iff the DESKTOP holds several GB of VRAM** (4.6 GB during three failures vs ~0.5 GB when the same users cleared three evals overnight). `expandable_segments` does NOT help. **Never `del` the result jsonls between eval attempts** -- `eval_sharded` skips completed users, so a relaunch only re-risks the remainder. Check `nvidia-smi` before starting a big eval.

**⚠ CPU-INFERENCE REALITY CHECK:** in the PYTHON RNN path a 4.5x arithmetic cut buys only **1.24x** wall-clock and plateaus -- that path is overhead-bound, so cost tracks op count (layers x streams), not width. **1 thread beats 3 and 6 -> deploy single-threaded.** The Rust path DOES convert the cut: **2.39x** measured. Full numbers: `optimization/CPU_INFERENCE.md`.

#### FAMILY SCOREBOARD (conduct rule 5: 1-2 rejects = deprioritized, NOT closed)

**★★ THE CROSS-FAMILY PATTERN, and it is now THREE independent confirmations (2026-08-19): THE MODEL
USES ANY NEW DEGREE OF FREEDOM IT IS GIVEN, AND *USE IS NOT EVIDENCE OF NEED*.**

| iter | lever | the parameter demonstrably moved | held-out result |
|---|---|---|---|
| 48 | `rcouple_w`, R(t) into the 4 rating logits | learned, **sign-correct** (Again -0.0138) | exact tie, p=0.19/0.37 |
| 50 | deck-tree level embedding | zero-init trained to **L2=1.766**, ~2x a typical input-projection row | exact tie, p=0.52/0.86 |
| 57 | learnable channel-mixer exponent | all 4 live exponents moved **2.0 -> 1.26-1.86**, same direction | exact tie, both inside the floor |

Three different mechanisms -- an architectural coupling, a new scope, a functional form -- one
signature. **=> A "the parameter trained, so the lever engaged" check proves only that the lever is
NOT INERT. It says nothing about whether the loss had anything to gain, and the two questions need
separate evidence.** Report the engagement diagnostic in every such iteration (it is what makes a
null interpretable rather than ambiguous), but never read it as a partial success or as grounds to
retry the same lever harder. **The productive inference runs the other way: when a model moves
decisively into a new freedom and gains nothing, the constraint that freedom removed was not
binding** -- so look for a DIFFERENT constraint, not a bigger dose of the same one.

**expressiveness-vs-capacity 0/1 -- DEPRIORITIZED, NOT CLOSED** (iter 57, the learnable channel-mixer
exponent; the family Andrew opened 2026-08-17). ⚠ **The lever reached 4 of 13 channel mixers** -- 9
`cmix_pow` params get no gradient and the dead set is EXACTLY `RWKV_STRIP_CMIX` (verified as a set
equality). So the honest claim is "null on card:0, note:0, deck:0, deck:3", not "learnable exponents
do not help". **But it is a STRONGER null than that caveat suggests: at the four sites it reached the
lever was FULLY engaged (up to a 37% move), so this is not a too-weak-to-matter result** -- decisive
where tested, silent elsewhere. A second variant must target a richer form at a site that SURVIVES
`RWKV_STRIP_CMIX`, and must clear the redundancy test. ⚠ Do not close this family on one run: it was
opened precisely because "capacity-at-5k is 0/3" had been standing in for an argument it could not
support. Note the overlap with iter 49 from the opposite direction -- it ADDED the user/preset L0
mixers back and got nothing; iter 57 made the SURVIVING mixers richer and got nothing.

**LR-schedule shape 0/1 -- and the follow-up is CLOSED BY ARITHMETIC, not by a second run** (iter 56,
`RWKV_DECAY_SHAPE=linear`). Real but sub-bar vs iter 45: ahead +0.000057 (INSIDE the +/-7.5e-5 floor,
so its reality rests on rank consistency at p=6e-12, not magnitude) / imm +0.000104 (clears both).
Loses to iter 53 at -0.000117/-0.000080. **The obvious follow-up -- does it STACK on iter 53, since
the levers are orthogonal -- was priced BEFORE queueing: under PERFECT additivity the stacked run
sits at +0.000057 ahead, which FAILS the 0.0001 bar. Even the best case cannot clear the gate, so do
NOT spend 6.1 h on it.** ★ The iteration also refutes a general claim: the "same-capacity
rearrangement is indistinguishable" result of iters 41/43/44 is NOT a law about this trunk -- it held
for the curve head and FAILED for the rating head at p=3e-161.

**explicit regularisation at the 1-epoch budget -- SAM 0/1 (iter 65), DEPRIORITIZED AS A FAMILY with a
mechanism.** SAM (decay-only, rho 0.05) found a demonstrably flatter minimum (sharpness −42%) and
REGRESSED in both modes (ahead −0.000277, imm −0.000155) -- with the TRAIN loss rising alongside the
held-out loss. Train and held-out moving together means the flatter point is a worse fit, not a
better-generalising one: at 1.25 epochs the model is FIT-LIMITED and there is no generalisation gap
for a regulariser to close. Four independent readings now agree (the tuner cut dropout x0.5, wd 0.2
lost, lorawd tied engaged, SAM regressed engaged), and Muon's edge is therefore spectral COVERAGE
(fit per step), not flatness. Do not propose dropout/noise/flatness/norm regularisers for this phase;
re-screen them on the 10x endgame's checkpoints, where a gap can exist (its wd/dropout item).
Neighbours closed on CPU the same night: PCGrad (no gradient conflict), LAWA checkpoint averaging
(the WS-average is worse than the WS-final), the cold-grade probe (no grade beyond R in the trunk).
**label-side curve supervision 0/1 -- DEPRIORITIZED with a named mechanism (iter 64, ordinal one-cut).**
The next review's rating is a real label the ahead path never sees, and the lever used it exactly as
designed (AUC Good-vs-Hard on the curve's own R 0.737 -> 0.851) -- and lost 0.0032 ahead, because a
SHARED logit cannot serve the pass/fail and the Hard/Good boundaries when they are not shifted copies
(the Easy share is U-shaped in R). A second variant must give the ordinal branch its own SCALE
(`z_ord = s*z - a`) or feed the grade to a separate head; not a smaller dose of the same term.
**deploy-contract alignment 0/3 -- CLOSED ON MECHANISM (iters 18, 33, 63).** Three instruments aimed at
the current-duration half of the rectification penalty -- permanent removal (18: -0.0018/-0.0024),
probe-only withholding (33: -0.0028/-0.0008, confounded), stochastic withholding (63: -0.00013/-0.00037
at a measured -37% engagement) -- and all three lose on imm, because the duration input serves TWO
consumers: the curve head the metric rectifies and the rating head it does not. The penalty is a
property of the deploy contract (Andrew 2026-07-27: zeroed duration + PAVA), not a training target.
Do not propose a fourth withholding scheme; the 30% PAVA-pooling half is the rectifier's own price.
**capacity-at-5k 0/3** (iter 49 added the user/preset L0 channel mixers back, +4.7% params, for +0.000067 ahead at p=0.11 and +0.000087 imm -- both under the bar). Three placements now agree: this model is not capacity-limited at 5k. Do not propose a fourth width/depth add without a mechanism argument that distinguishes it from these three.
**ahead-vs-imm-gap exploitation 0/2 -- CLOSED ON MECHANISM** (iters 46, 48). Both attempts to route
the better-conditioned imm signal into the ahead/rating path returned exact nulls, by structurally
DIFFERENT routes (soft targets; an architectural coupling), and iter 48 showed the coupling was
learned yet negligible -- the trunk already carries the information. The 0.032 gap is intrinsic
difficulty, not a routing deficiency. Attack CONTENT (new input features), never routing again.
**low-rank-friendly regularization 0/1 -- CLOSED ON MECHANISM, and the single reject is enough**
(iter 47). Conduct rule 5 normally forbids closing a family on one result; this is the exception it
allows for, because the iteration did not merely fail, it **measured that the target term has almost
no logloss in it**: the exact rank-1 truncation error fell 43% card / 75% note (note median 0.0152 =
essentially exactly rank-1) and the deployed loss did not improve. Any other route to the same term
-- rank-2 states, softer/scheduled lambda, a different proxy -- is aimed at the same empty term, and
the step-50 check already proved engagement is not the bottleneck. **The QAT tax lives in the CODEBOOK
and NORM terms.**
**TOPOLOGY 1/4 -- and iter 50 CLOSES the remaining direction.** Iters 41-44 showed that the
EXISTENCE of a cross-scope information path pays (interleaving) while its CHOREOGRAPHY does not
(three arrangements indistinguishable at |delta| <= 7.5e-5). **Iter 50 (the deck tree) shows that
adding more SCOPES does not pay either** -- an exact tie at p=0.52/0.86 with the level embedding
demonstrably learned. The scope ladder card->note->deck->preset->user is SUFFICIENT: the
productive lever was moving information between the levels that exist, and it is banked.
Do not propose a new intermediate scope without a mechanism that distinguishes it from a level
the ladder already brackets.
**TOPOLOGY (the 41-44 detail), and both rejects are CONTROLS that changed what we believe** (iter 41 ACCEPTED
— interleave + reorder bundle, the phase's largest architectural gain; iter 42 REJECTED —
order-alone is a small NEGATIVE, so INTERLEAVING carries all of it; iter 43 REJECTED AS A TIE —
interleave at the original order equals the champion (p=0.42/0.098), so the reorder's cost
VANISHES under interleaving. **ORDER lever CLOSED; SCHEDULE is the productive one** and the
2×2 is complete) ·
**distillation 4/5** (external-teacher sub-family 4/4; SELF-distillation 0/1 -- iter 46, a clean null that explains why: a same-forward-pass teacher carries no independent information) (iter 32 ACCEPTED, the d=128 teacher; iter 35
the seed pair; iter 39 alpha 0.9; **iter 45 KD through DECAY, the current champion** — teacher signal
pays in BOTH phases, so the "anneal onto the true objective" intuition is wrong here. Open and cheap
on the same dump: alpha_decay 0.9 / 0.25. ⚠ iter 10 was mis-filed under early-training-intervention,
which is why this family once read as absent) ·
curve-shape constraints **2/3** (PAVA ACCEPTED iter 23; lambda=0.2 DIRECTED-ACCEPTED iter 36 on a
5.9:1 ahead-for-imm trade; lambda=0.3 rejected as the worse point of the same lever) ·
objective-alignment **0/1 mechanism-refuted** (iter 37 by-user weighting: worse in every size
quartile incl. its intended beneficiaries — do not retry milder doses) · **optimizer 2/4 -- iter 53 SPLIT the family in two, and iter 62 (`lorawd`, decoupled wd on the LoRA Muon group, REJECTED as a tie at a demonstrably engaged dose: LoRA norm ratio 0.811) put NORM CONTROL on the non-paying side**
(COVERAGE pays, DESCENT QUALITY does not) (Muon ACCEPTED iter 29, the phase's largest imm gain; cautious wd
REJECTED iter 30 — a pure trade; iter 51 PolarExpress FAILED structurally, p(1)<1 is load-bearing.
**Muon is a REGULARIZER here** — its train-loss edge decays to −0.00058/+0.00097 while eval holds at
+0.0019 — so descent-quality refinements incl. NorMuon target the half that stopped paying) · GRU-head N-sweep **peaks at
N=3** (N=4 worse, closed) · readout/xhead **0/3** with real signal but negative under the GRU
head (iter 28), closed pending new ideas · loss-reweighting **0/2** (pbin scale lever closed by
dose-response — a linear imm/ahead trade through zero) · early-training-intervention **0/2** ·
grade-representation **0/1** · capacity-at-5k **0/2** · state-size ladder **0/5 CLOSED** · HP
tuning **CLOSED** (champion HPs confirmed vs 19 alternatives at full eval).
All hooks stay in-repo, env-gated, default off.

#### LIVE RULES (both tracks)
- **★★ EVERY RUN WITH QAT IS EVALUATED WITH QUANTIZATION ON (Andrew 2026-09-10, verbatim: "Every run with QAT should be evaluated with quantization on").** A QAT run's number is a DEPLOY number or it is nothing: training under fake-quant and scoring in full precision measures a model nobody ships, and it silently understates the tax by the whole precision-degradation term. Budget the eval accordingly -- a quant-aware eval is ~10 h, not the ~2.9 h a plain one takes.
  **It holds by construction in these runners** (the `RWKV_QAT_*` block sits inside `setlocal` and is never cleared, so phase C inherits it) -- **and "by construction" is exactly what the 2026-08-12 inert-env bug looked like**, where the banner was truthful and the config object it mutated was discarded one line later, so an entire track-2 phase evaluated unquantized without a symptom. **So it is CHECKED, twice, at the right costs:** `assert_qat_live.py` re-run in a FRESH process immediately BEFORE phase C (seconds; fails before the 10 h eval, `DONE_EXIT_54`), plus a **non-fatal** post-check for `[QAT-LOWRANK] set:` in the **SHARD** logs (`eval_sharded`'s parent log never carries banners -- that mistake cost a spurious rc 41 on both qat_tax arms). The post-check is deliberately non-fatal: a guard that can destroy a finished 10 h eval on a banner-string mismatch is worse than the failure it guards.
  ⚠⚠ **AND IT MUST QUANTIZE WITH THE CATALOG THAT SHIPS (2026-09-10).** A run with `RWKV_QAT_PQ_LEARN=1` / `RWKV_QAT_SHIFT_PQ_LEARN=1` learns catalogs the checkpoint does NOT contain; its eval must set `RWKV_QAT_PQ` / `RWKV_QAT_SHIFT_PQ` to the exported `<prefix>_{wkv,shift}cb_<final step>.txt` and clear both LEARN flags (`run_cblearn.cmd`'s recipe), and must PROVE it on a small probe by grepping the loaded path. Arm 2's runner missed this and was fixed mid-decay -- see LIVE.
  ⚠ **NAMING TRAP, and it is live in every runner: `RWKV_QAT_COMPILE` IS NOT A QUANTIZATION FLAG.** It is the `torch.compile` speed flag from the 2026-07-30 stack. A run whose only `RWKV_QAT_*` variable is that one -- arm 1, realcyc, gen4base, the whole gen-5 lineage -- is **full precision**. Grep for `RWKV_QAT_LOWRANK_SCOPE`, not for `RWKV_QAT`.
- **⚠ VAL/TEST SPLIT (from iter 29 / post-A8):** candidates eval ONLY the VAL half = users
  **5001-7500** (n=2500); all verdicts + p-gates run there, pairing vs the champion's jsonls via
  `paired_pvalue --intersect`. **TEST = 7501-10000 is touched ONLY at each track's close** —
  never for decisions. Eval tomls: `write_eval_toml ... 5001 7500`.
- **`RWKV_NO_AHEAD_RESIDUAL=1` in EVERY run** (Andrew 2026-07-16): the piecewise-linear curve
  correction is disabled, so the curve is monotone in t by construction.
- **d=128/d=80 runs:** `RWKV_EMPTY_CACHE_EVERY=1` + `RWKV_EMPTY_CACHE_WINDOW=0` (allocator creep
  -> WDDM paging -> 4x slowdown otherwise). **MAX=32768 EVERYWHERE** incl. `write_decay_setup.py`
  arg 10 — pairing needs MAX identical across runs. Evals UNSHARDED (`--shards 1
  --solo-threshold 0`). d=32 evals use phased `eval_sharded.py`.
- **✓ FIXED 2026-09-09 -- the decay phase now CAN checkpoint: `RWKV_DECAY_VALIDATE_EVERY`**
  (`write_decay_setup.py`, default 100000 = unchanged, so every existing runner is byte-identical).
  All seven endgame branches set **2000**, giving ~11 resume points in a 21,870-step decay for ~4%
  of the phase. **It is trajectory-FREE, proven rather than assumed**: validation runs under
  `model.eval()` inside `no_grad` and draws no main-process RNG --
  `scratchpad/ws10/rng_neutrality.py` shows the state unchanged across an eval-mode pass AND
  changed across a train-mode one, so the check cannot pass vacuously. Every branch uses the SAME
  value, so they stay mutually comparable. The original entry follows.
- **⚠ THE DECAY PHASE HAS NO MID-RUN CHECKPOINTS (cost 3.5 h, 2026-09-02 PC restart).** `train_rwkv` saves only on `validate_iter`, and `write_decay_setup.py` writes `VALIDATE_EVERY = 100000`, so a decay checkpoints at step 50 and at the end -- gen4base's decay died at step 10681 of 10935 with nothing to resume from. The "crash recovery loses <=1000 steps" line below is TRUE OF WS ONLY. Not changed mid-lineage (realcyc must stay single-variable vs gen4base); the 10x endgame's decay (~19 h) MUST set a real VALIDATE_EVERY.
- **MID-EPOCH RESUME:** `RWKV_RESUME_SKIP_GROUPS=1` + `python scratchpad/make_resume.py
  <run_dir> <prefix> <ws_toml>`, then rerun the WS phase with the run's FULL env, WITHOUT
  deleting step-trace files. Crash recovery loses <=1000 steps. The resumed tail's dropout draws
  differ (weights/optim exact) — statistically equivalent, not bit-identical.
- **⚠ NO co-tenant GPU work during gate-critical runs** — cuBLAS algo selection under memory
  pressure breaks bit-replay (~1e-4 val drift), and at 11.6/12 GB two processes deadlocked in
  WDDM paging for 2.7 h. Smokes wait for a free GPU or run tiny/CPU.
- **Seed-pair doctrine:** any single-run margin < ~0.0005 needs the exact recipe re-run at
  `RWKV_AUGMENT_SEED=4321` first — cross-seed spread on the same recipe is ~0.0004 both modes;
  in-seed Wilcoxon p (even 1e-29) measures per-user consistency, NOT cross-seed robustness.
- **TorchScript hook rules** (cost 2 dead launches): `@torch.jit.ignore` bodies must NOT call
  submodules (scripted code sees the raw C++ ScriptModule -> 'not callable' -> the NaN-except
  turns the run HOLLOW) — use root Parameters + `F.linear`, names containing weight/bias for the
  wd groups; root-level Parameters are INVISIBLE to `selective_cast`'s module walk (cast them
  explicitly); ScriptModule forbids `persistent=False` buffers. Old-style ScriptModule bakes the
  FIRST construction's env flags into the compiled class — never two flag values in one process.
  Smoke tests MUST exercise the SCRIPTED forward. Gate every `.cmd` phase on exit codes AND
  artifacts (train_rwkv can swallow fatal errors to exit 0).
- **`RWKV_GRAD_STATS=<out.json>` on every ablation run** (Andrew 2026-07-16) — per-param
  mean|grad| + SNIP saliency, to rank targets. Report: `python
  optimization/grad_stats_report.py <json>`.
- FETCH WORKERS = 4 in every toml (RAM). Live loss plot: `detach.ps1 -Script
  scratchpad/liveplot/run_liveplot.cmd`.
- **⚠ Eval-path fetch-worker leak is SYSTEMATIC:** every eval leaves 1-2 orphan pythons, some
  spinning a full core for hours. **CHECK + KILL orphan pythons after every run** — but inspect
  command lines first: the spare `pythonw` are the bridge/controller, the ~80000s-CPU python is
  Andrew's FSRS benchmark, and he also runs a Reddit bot + liveplot. **Do not kill those.**
  **★ AND AN ORPHAN HOLDS ITS LMDB OPEN INDEFINITELY (2026-08-24).** A featA2 fetch worker from
  08-21 was still holding `F:/rwkv_lmdb/test_db_5k_fix` **three days later** (10 CPU-seconds
  total, parent long dead), which made the directory un-renameable with a bare `Access is
  denied` and no indication of why. Diagnose in this order, because the obvious suspect is
  usually wrong: rename a fresh dir on the same volume (rules out permissions), then
  `[IO.File]::Open(path,'Open','ReadWrite','None')` on `data.mdb` (proves a handle exists), then
  find the process by START TIME + near-zero CPU + dead ParentProcessId. My first two theories --
  my own verification handle, then F: permissions -- were both wrong.
- **★★ SEVERAL `F:/rwkv_lmdb/*` PATHS ARE JUNCTIONS TO `C:\rwkv_lmdb\`. DO NOT DELETE
  `C:\rwkv_lmdb` — it is not scratch, it is those databases.**
  **UPDATED 2026-08-30 (the 08-24 list is stale):** `test_db_5k_fix` was DELETED with the other
  two superseded dbs, so its junction is gone too. The current junctions are
  **`train_db_5k_h1_id3`**, **`train_db_5k_h1_e2s`** and **`test_db_5k_e2s`** — the last two moved
  after Andrew made e2s the default, because reading a db from F: costs **2.2x per step**
  (the C:-hosted teacher dump ran 1.40 steps/s, the same dump on F: 0.63, GPU utilisation 8% =
  starved on reads, not computing).
  **✓ CONFIRMED BY THE OUTCOME, not just the microbenchmark:** after the move, the e2s teacher
  dump ran **2 h 03 m** for 10,935 steps -- against 2 h 10 m for the original C:-hosted dump and a
  **4.7 h projection while it was on F:**. The WS phase then resumed the champion's normal
  **0.905 steps/s** (reference 0.92). So the penalty was I/O and the move removed it entirely.
  ⚠ **Deleting a junction is not deleting a copy, and the two paths need different tools.**
  `Remove-Item -Recurse` on the F: path deletes THROUGH the link and destroys the C: data. Safe
  order: delete the real C: directory first, THEN remove the now-dangling link with
  `[IO.Directory]::Delete(path, $false)`. (`cmd /c rmdir` also removes a link without following,
  but the harness blocks that invocation.)
- **★ DELETED 2026-08-30, with Andrew's authorization, to make room:** `train_db_5k_h1` (91.0 GB),
  `train_db_5k_h1_fix` (103.4 GB) and `C:\rwkv_lmdb\test_db_5k_fix` (103.0 GB) — **297.4 GB freed,
  C: 52.2 -> 328 GB.** All three are end-to-END and so superseded by the e2s default; the first two
  additionally carry Bug A / Bug C. **They are REBUILDABLE in ~80 min each** from the read-only
  `anki-revlogs-10k`, so what was given up is bit-reproducibility of iter 53 and featA2 *until* a
  rebuild, not data. Their results are already in the record.
  ⚠ Measure LMDB sizes with `GetCompressedFileSize` — these are SPARSE, and `Get-ChildItem |
  Measure Length` and `Scripting.FileSystemObject.Size` BOTH report the map_size RESERVATION
  (372.5 GB), which is fiction. Free-space before/after is the other reliable method. They were moved to the SSD for speed: measured random-read throughput went
  25.1 -> 643 MB/s and 8.1 -> 346 MB/s (**25x and 43x**); junction overhead is ~10% vs a direct
  C: path. The F: paths are junctions precisely so nothing had to be edited — the db paths are
  hardcoded absolute strings in runners *and inside their `findstr` guard assertions*, so a path
  edit is the clone-a-runner failure mode waiting to happen. `test_db_5k_id3` (37 MB/s) and
  `test_db_5k` (35 MB/s) are still real directories on F:.
  ⚠ **Move them with `lmdb.Environment.copy(dst, compact=True)`, never a file copy** — these are
  SPARSE, so a plain copy materialises the reservation (`test_db_5k_fix` would land as its
  232 GB apparent size, not 103 GB). And budget ~7% GROWTH, not a saving: compaction wrote
  110.6 GB from a 103.0 GB source, because the sparse source's allocated extents undercount what
  a densely-written copy needs. Tools: `scratchpad/workload/move_lmdb.py` (copy + verify) and
  `finalize_lmdb.py` (rename -> junction -> verify through it -> delete original). They are two
  scripts because verifying and renaming in one process fails: the verifier's own handle blocks
  the rename.
- **★★ THE CHEAP END-TO-END DRY RUN IS NOW A TOOL AND A RULE (Andrew 2026-09-09: *"Maybe we should
  have a rule like 'do a test run on 20 users every 5 iterations' or something, lol"*).** He is
  right, and his own 2026-08-18 bug-hunt directive already specified the shape (EPOCHS 0.05, 20 eval
  users); the audit beside it says every failure of that week would have been caught by one such
  run. It was used once and never made routine.
  **TWO TIERS, and the cheap one needs NO GPU:**
  1. **STATIC (seconds, zero GPU) -- `preflight_runner.py`** grew a **decay-writes-here-vs-
     eval-looks-there** check, which catches today's bug class by construction: it pairs each
     `write_decay_setup` (whose SOURCE folder is where the checkpoints land) against each
     `write_eval_toml` (which globs somewhere else), fails on a prefix mismatch, and where the
     folders differ requires the `decay.toml` fallback's precondition to hold. Proven non-vacuous
     against two deliberately broken copies; all seven endgame branches PASS with a note.
  2. **DYNAMIC (~10 min GPU) -- `scratchpad/mk_dryrun.py <runner>`** rewrites any branch runner to
     0.05 epochs / 20 eval users with its own tag, dir, log, **checkpoint prefix** and result
     jsonls, so it cannot touch a real branch's artifacts. It catches what static analysis cannot:
     the HOLLOW run (ordcut raised on every step with a correct banner and param count), a guard
     whose findstr disagrees with the value the runner SETS, `endlocal` before the marker, a phase
     that exits 0 without its artifact.
  **⚠ THE TRIGGER IS STRUCTURAL NOVELTY, NOT A COUNT.** Today's bug did not appear because five
  iterations had passed -- it appeared because these are the FIRST branches to decay from a SHARED
  WS. Fire a dry run on a new runner family, a new phase shape, or an edit to a shared tool
  (`write_decay_setup`, `write_eval_toml`, `eval_sharded`). Keep the periodic version as a BACKSTOP
  against slow drift.
  **⚠ IT PROVES PLUMBING, NEVER CAPACITY.** The 6701 OOM, the WDDM co-tenant deadlock and the
  giant-user freeze all need real users and real VRAM; 20 small users cannot see any of them.
  **⚠⚠ AND THE TOOL'S OWN GUARD MUST USE THE TRAINER'S ARITHMETIC, NOT ITS OWN. Cost 2 h of idle GPU, 2026-09-10.** `mk_dryrun.py` computed the decay's final step as `int(round(0.05 * 10935))` = **547** while `train_rwkv.py:967` uses `total_steps = int(config.EPOCHS * len(groups))` = **546**. So every dry run did its job perfectly -- 546 steps, zero tracebacks, checkpoint and both catalogs written -- and then failed its own artifact guard at `DONE_EXIT_28 DECAY_SHORT` asking for a checkpoint that could not exist. The arm-2 waiter saw a failed dry run and **correctly refused to launch**, so the GPU sat idle from 04:56 to 06:55. Fixed to truncate; all six dry runners regenerated. ⚠ `10935` is `len(groups)` for THIS lineage's train db at MAX=65536 and is hardcoded in the tool -- a different db moves it.
  **THE GENERAL LESSON, and it is about gated chains rather than about rounding: a gate converts a TOOLING false negative into idle GPU.** The existing rule says to check a detached waiter is still ALIVE an hour later; extend it -- **also check it has not REFUSED**. A waiter that exits 8x is doing its job and looks identical to a healthy one from the outside: gone, with a marker.
  ⚠ Two traps the tool itself hit, both already documented for other parsers and both re-hit here:
  the checkpoint prefix does NOT follow the tag (arm 2 is `w10qat` but writes `w10q_d`), and a REM
  line mentioning `write_decay_setup.py` in prose was matched first, capturing an English word as
  the prefix -- silently, because the bogus word then passed the stale-prefix assert VACUOUSLY.
- **⚠⚠ OPS -- CLONING A RUNNER MEANS UPDATING EVERY STRING THAT DEPENDS ON THE LEVER, NOT JUST
  THE LEVER. Three failures on 2026-08-18, same shape, one of them caught live.**
  * **iter 54 phase 2a: the ENV was wrong, the guard right.** The champion uses KD alpha **0.9
    for WS** (iter 39) and **0.5 for DECAY** (iter 45); the reset line sits INSIDE the WS phase,
    which a decay-only generator slices away. Phase 2a decayed 3.3 h at 0.9 -- iter 55's lever --
    and its own guard rejected it (`DONE_EXIT_WRONGALPHA_DECAY`). **The guard saved the
    iteration**: the number would otherwise have been a mixture of two experiments. A guard
    DETECTS, it cannot REPAIR.
  * **`decayshape`: the guard was wrong, the env right.** `mk57.py` set alpha to 0.5 but left
    `findstr /C:"alpha FIXED at 0.9"`, so a correct 3.3 h decay would have been rejected at the
    end. **Caught 90 s into the run** by reading the runner rather than trusting it; killed,
    fixed, relaunched for ~1 min of lost GPU. Both generators now assert the guard matches the
    value the runner SETS.
  * **`rgate`: the smoke's control inherited the lever.** `run_iter55.cmd` does
    `set RWKV_RGATE=card` BEFORE calling the smoke, and the smoke built arms with
    `dict(os.environ, **extra)` -- so the OFF arm was gated too. Its param check caught it
    (`rgate keys present with the flag OFF`), but note the inertness check had passed
    **VACUOUSLY at 0.000e+00 while comparing two gated models**. **A test that reads its
    CONTROL's configuration from the ambient environment is not a control.** Fixed by stripping
    the smoke's own vars before applying each arm's.
- **⚠⚠ AND THAT RULE HAD A LIVE BUG BEHIND IT, CAUGHT MID-RUN 2026-09-09 WITH 3.5 h TO SPARE.**
  Every earlier decay had source == destination (each run decayed from its OWN WS), so every
  runner passes its own directory to `write_eval_toml.py` and it always worked. **The endgame
  branches are the first to decay from a SHARED WS**: arm 1 writes `w10p_d_*.pth` into
  `scratchpad/ws10` while its runner asks `write_eval_toml` for `scratchpad/w10plain`. The eval
  would have died with `DONE_EXIT_24` **after the 6.2 h decay**, and taken the chain with it --
  the arm-2 waiter refuses on any non-zero marker. Found by checking where the checkpoints were
  actually landing, not by reading the runner.
  **FIXED in `write_eval_toml.py`, not in the running `.cmd`** (which must never be touched): an
  empty glob now falls back to the run directory's own `decay.toml` and reads `SAVE_MODEL_FOLDER`
  from it, which is where the checkpoints provably are. Inert for every existing runner, since
  their first glob is non-empty. **Proven by EXECUTING arm 1's exact phase-C command**, which
  found the 5 checkpoints and wrote a valid toml. It covers every endgame branch uniformly.
- **⚠ OPS -- A DECAY-ONLY RUN WRITES ITS CHECKPOINTS INTO THE *SOURCE* RUN'S DIRECTORY.**
  `write_decay_setup.py` takes the dir holding the WS-final checkpoint, so iter 52's decay landed
  in **`scratchpad/iter45_kddecay/i52_d_10935.pth`**, not in `scratchpad/iter52_kdalpha/`. The eval
  toml's `MODEL_PATH` points there and is correct. **Two consequences:** (1) `ls` in the run's own
  directory shows NO decay checkpoint, which looks like a failed decay and is not; (2) deleting an
  old champion's directory during housekeeping would silently take later runs' checkpoints with
  it. iter 57 (`decayshape`) will land there too, since it also decays from `i45_ws_10935`.
  Check `MODEL_PATH` in the eval toml before concluding anything about where a checkpoint is.
- **OPS gotcha:** PowerShell `Set-Content -Encoding utf8` writes a BOM -> `tomli` dies at line 1
  col 1. Write tomls with the Write tool or `UTF8Encoding($false)`. A crashed run's
  `DONE_EXIT_WSFAIL` satisfies downstream waitloop greps — relaunch upstream FIRST, then re-park
  dependents.
- ⚠ **WAITLOOP TRAP, cost one wrongly-started co-tenant eval (2026-07-26):** `findstr /C:"DONE_EXIT"`
  matches a log line that merely MENTIONS the token — including the waiter's own
  `=== WAIT for ... DONE_EXIT ===` message — so the loop fires instantly. **Anchor it:
  `findstr /B /C:"DONE_EXIT_"`** (terminal lines start with the token; prose never does) and do not
  write the token in non-terminal log lines. This is distinct from the known
  `DONE_EXIT_WSFAIL`-satisfies-the-grep gotcha.
- ⚠ **A GUARD'S VERDICT MUST BE THE CHAIN'S EXIT STATUS (2026-09-02, caught within the hour).**
  I armed `wait_then_rebuild5.cmd` after a preflight that had printed `PREFLIGHT_FAILED` --
  because the call was `preflight | tail -2 && detach`, and `tail` exits 0. The pipe swallowed the
  verdict and `&&` marched on. Never put a guard behind a pipe before `&&`; capture
  `${PIPESTATUS[0]}` or run the guard bare. The failure itself was a FALSE POSITIVE in
  `preflight_runner.py` -- it does not see a `set VAR=` inside `for /f ... do set VAR=%%R`, so it
  reports `%FREEMB% never set` on the RAM-check pattern that `wait_then_rebuild4.cmd` ran
  successfully with that morning ("RAM OK: 51250 MB free"). Fixed in the tool; but the order is
  fix the tool -> preflight PASS -> arm, never "arm because I know why the guard is wrong".
- ⚠ **`goto` CAN MISS A LABEL IN AN LF-ONLY `.cmd` (2026-09-06, killed the hord verdict waiter TWICE, silently).**
  "The system cannot find the batch label specified - waithord": cmd.exe scans for labels in
  512-byte blocks and an LF-only file can hide a label from that scan depending on where it falls;
  the SAME file works or fails by byte alignment, which is why sibling waiters written the same way
  looped for hours. A stubbed execution does not catch it when the stub's fake log already carries
  the marker (no `goto` runs). **Write `.cmd` files with CRLF line endings** (the Write tool emits
  LF: convert with python `newline=""` before arming) and run the real waiter in the FOREGROUND for
  longer than one poll interval, so its first `goto` executes, before detaching it. Never convert
  a RUNNING waiter's file (byte offsets).
- ⚠ **`detach.ps1`: pass the path as a LITERAL single-quoted Windows path from bash** (2026-09-03). Three waiters
  launched with the path assembled in a bash `for` loop variable died instantly (the third detach already saw no
  parent), while the identical files launched with a single-quoted C-drive literal ran. Verify every detach by pid
  AND by the runner's own first log line; a returned pid is not a running process.
- ⚠ **`detach.ps1` needs an ABSOLUTE path.** `Win32_Process.Create` starts in System32, so a
  relative script path exits instantly, silently, and still returns a pid.
- ⚠ **AN UNESCAPED `)` INSIDE A PARENTHESISED BLOCK ENDS THE BLOCK AND ABORTS THE BATCH (2026-09-03, rc 255,
  reproduced on a 6-line stub).** `if errorlevel 1 ( ... echo gate FAILED (a, b) ... )` dies at the inner `)`
  with "text was unexpected at this time", caller included. cmd parses a block only when it REACHES the `if`,
  so `wait_then_realcyc_v3.cmd` looped for 5 h in its first block looking alive, then died the instant gate 0
  opened and the second block was parsed -- twice, silently, no log line. A `(` that is not at command position
  is plain text; only the `)` is fatal; double-quoted or `^)` is safe. `preflight_runner.py` checks it (proven
  to FAIL the old file, PASS the fixed one); all 30 chain `.cmd` files swept clean.
- ⚠ **PERCENT-TILDE IN A `REM` LINE KILLS THE WHOLE BATCH AND ITS CALLER, SILENTLY (2026-09-03, measured).**
  cmd expands batch-parameter substitution before it honours `REM`: a comment reading "the %~N trap" is
  an invalid modifier and aborts with rc 255 -- `run_ablate.cmd` carried exactly that in its header and its
  waiter simply vanished after "gen4base reported" (no log, no marker, no process). Found by EXECUTING a
  stubbed copy (`%PY%` replaced by `cmd /c exit 0`, log redirected), which is now the way to validate any
  runner cmd.exe has never run. `preflight_runner.py` checks it. **Probed the rest of the folklore in
  cmd.exe, top level AND inside an if-block: `& | ^ < >`, arrows, `[label]`, parens, `%VAR%`, `100%`, `%%R`
  and a valid `%~1` are ALL harmless in a REM line.** The 08-14 incident below did not reproduce in either
  position, so its real trigger was something else; the bracket/arrow avoidance stays as a convention only.
- ⚠ **NO `< > & | ^` IN `REM` COMMENTS** — cmd.exe processes REDIRECTION *before* it honours `REM`,
  so a comment containing an arrow (`->`) or a usage line with placeholder brackets is parsed as a
  redirect. Symptom is baffling and points nowhere near the comment: `'M' is not recognized as an
  internal or external command` (cmd resumes mid-`REM`) plus `< was unexpected at this time`. Cost
  one dead launch 2026-08-14. Write `CKPT_PATH LABEL`, not bracketed placeholders, and `==` not
  `->`. Same family as the backslash-in-generated-content and BOM traps: **content destined for a
  shell needs escaping discipline even when it is "just a comment".**

### Ops
- **Compaction (ONLY sanctioned way):** run `claude-automation/request_compact.ps1 -Focus "<carry-through>"`
  then yield idle and STOP beating the heartbeat. `/compact <focus>` fires only from a FRESH (<=30 min) +
  focus-bearing flag (stale/empty = purged). Never hand-create `pending_compact.txt`. The injector is 24/7
  (ClaudeLoopController every 3 min; acts only on a stale heartbeat) and may inject EXACTLY `/compact <focus>`
  or a short `Continue` -- nothing else Claude-originated. **The Telegram bridge is RETIRED (Andrew 2026-08-30)** --
  superseded by Dispatch in the Claude app; task `ClaudeTelegramBridge` is Disabled at the scheduler
  and its processes stopped. ⚠ Removing its master switch had NOT stopped it: the task kept firing
  every 5 min and idling on the absent flag. Reversible (code + config untouched). So the injector is
  now the ONLY injection source, and the two-form limit governs all of it.
- **★★ NEVER TOUCH A RUNNING `.cmd` -- AND `git checkout` IS NOT A SAFE UNDO (cost iters 43 AND 46).**
  cmd.exe re-reads a batch file from a saved BYTE OFFSET every time a command returns, so any edit that
  shifts bytes past that offset makes it resume mid-garbage. Three things follow, learned the expensive way:
  (1) A chain's LATER phases are new processes that import whatever is on disk THEN -- so editing
  `rwkv/*.py` mid-chain silently changes the next phase too (found during iter 45; mitigate by gating new
  code on its env flag so it is inert when unset).
  (2) **Reverting an accidental edit with `git checkout --` DOES NOT RESTORE THE BYTES.** git normalizes
  line endings: a runner written LF (python `newline='\n'`) comes back CRLF, +1 byte per line. Iter 46's
  runner grew 222 bytes that way; cmd.exe resumed at the wrong offset, re-ran a fragment that re-opened the
  SAME ws log with `>`, and TRUNCATED the training log to 44 bytes. The WS checkpoint survived, so only the
  chain and the log were lost -- recovered with a phase-2 runner, same as iter 43.
  ⚠ The tell was visible and dismissed: `md5sum` of the file vs `git show HEAD:` differed, and it was waved
  off as "just line endings". It WAS just line endings, and that was exactly the failure.
  **If a running runner has already been touched, restore from a BYTE-EXACT copy (keep one before editing)
  or leave it alone and write a phase-2 runner. Never `git checkout` it.**
  (3) Annotate a running experiment in a SEPARATE file (e.g. `GATE.md`), never in its runner.
- **★★ A RUNNER GENERATOR THAT CAN DELETE NEEDS OUTPUT GUARDS, NOT JUST CONTAMINATION GUARDS
  (2026-08-17; caught iters 53 AND 54 armed and broken, ~12 h of GPU saved).** `mk53.py`/`mk54.py`
  build a runner as `HEADER + s[s.index("setlocal"):]`. In the iter-45 runner they copy from, `cd /d`
  and the whole `DIR/LOG/STAMP/DUMP/WSSTEPS/MAXSTEPS` block sit **before** `setlocal` -- so the slice
  silently threw all of it away. (iter 52, generated by a different script that puts them AFTER
  `setlocal`, was unaffected -- which is why this was invisible until all three were compared.)
  **Failure mode is maximally quiet:** `%LOG%` expands to empty, so phase 0's `>> "%LOG%"` is a
  syntax error, the guard fires, tries to log to `""` as well, and exits 37 **without ever writing a
  `DONE_EXIT_` line** -- so a downstream waiter hangs forever and nothing in any log explains it.
  Missing `cd /d` compounds it: `Win32_Process.Create` starts in System32, where
  `.venv\Scripts\python.exe` does not exist.
  **THE RULE:** every generator assert in those files checked that stale text did not leak **IN**
  (no `iter45`, no `i53_`, KD schedule preserved). None checked that required setup **SURVIVED**.
  **★ THE MIRROR-IMAGE GAP, hit 2026-09-01 in my own generator.** `mk_fixc_arm.py` asserted that
  every line it CHANGED carried a db/tag token -- a check against stale text leaking in. It cannot
  catch a line that SHOULD have changed and did not, because no substitution fires on it and the
  line reads as unmodified. Result: `run_fixc_arm.cmd` logs "PHASE 4: rectified VAL-half eval on
  the **e2s** test db" while correctly evaluating fixc. Harmless here only because the GUARD is
  derived from cfg (`findstr /C:"...test_db_5k_fixc"`) while the prose was hardcoded -- so the
  runner cannot actually act on the wrong db, it can only describe itself wrongly.
  **A substitution-based generator needs BOTH directions: no stale token survives, AND every line
  mentioning the old identity was visited.** Grep the output for the SOURCE arm's name and require
  zero hits outside deliberate provenance comments. Same asymmetry as the bullet below (assert
  what leaked IN vs assert what SURVIVED), which is why it was easy to repeat.
  Assert on the OUTPUT: every `%VAR%` the runner references must be declared before its first use,
  and `cd /d` must be present. Same family as the QAT env that was parsed-then-discarded -- the
  banner was truthful and the object it mutated was thrown away.
  **AND VERIFY BY EXECUTION:** the repair was confirmed by running phase-0-only copies (everything
  up to the first training call, log redirected) to exit 0 -- which is also what re-proved iter 54's
  558,225 param count. Reading a `.cmd` does not tell you cmd.exe agrees with you.
  ⚠ Patching an ARMED runner is safe **only** while its waiter is still looping: a `call`ed `.cmd`
  is not open until the call, so the byte-offset hazard above does not apply. Check the waiter log
  says only "waiter armed" first, and keep a byte-exact backup.
- **ESC-PROOF detached launches:** Esc / session teardown tree-kills Claude's Bash/PowerShell bg jobs INCLUDING
  training. Launch each training as a self-contained `.cmd` via `scratchpad/detach.ps1` (WMI Win32_Process ->
  parented to WmiPrvSE, survives); log to a STABLE repo path (`scratchpad/*.log`, NOT the rotating session
  temp); end the .cmd with `echo DONE_EXIT_%ERRORLEVEL%`. MONITOR via OS truth (poll the log / Get-Process /
  ckpt mtime) -- detached runs give NO tool-completion event. A Bash watcher gives notifications but is itself
  Esc-killable (re-arm it each turn; the training survives). Beat the heartbeat each working turn
  (`claude-automation/beat.ps1`). **Do NOT kill the FSRS benchmark PIDs (the ~80000s-CPU python procs).**
- **DATA FACT (SUPERSEDED 2026-07-26 -- read the next bullet before acting on it):** the PUBLISHED
  `anki-revlogs-10k` has NO absolute timestamp / review-id (anonymized; raw `revlogs` parquet = card_id,
  day_offset [integer DAY counter], rating, state, duration, elapsed_days, elapsed_seconds). Time-of-day is
  unrecoverable **from that set**. elapsed_seconds (time-since-last) is already an input.
- **★ THE REAL-TIMESTAMP DATASET EXISTS AND IS BUILT — `C:\Users\Andrew\anki-revlogs-10k-id`** (Andrew
  2026-07-26: *"we should have code for making it, so idk why CLAUDE.md doesn't mention it"* — it didn't;
  fixed). Built 2026-07-15/16 by **`scratchpad/dataset_id/`** (`run_build_id.cmd` -> `build_parquet_id.py`,
  adapted from the upstream anki-revlogs-dataset-builder), staging copy `anki-revlogs-10k-id-raw` (38.7 GB,
  keeps `revlogs.7z`). 15.8 GB; **10,000 user dirs in revlogs + decks, 9,934 in cards**; same layout and same
  1:1 user numbering as the published set, so results are comparable.
  - **IDs stay RAW Anki epoch-ms** = creation timestamps (`card_id`/`note_id`/`deck_id`/`parent_id`/
    `preset_id`), instead of upstream's per-user factorized small ints.
  - **`review_time` is CORRECTED to SHOW time** = `revlog.id - taken_millis` (the row is written on ANSWER),
    which is the right base for elapsed/time-of-day. Everything downstream (day_offset, elapsed_days,
    elapsed_seconds, sort order) is recomputed from it. Raw answer time = `review_time + duration`.
    ⚠ So day_offset can differ by one from the published set for reviews spanning the day rollover.
  - Spot-checked 2026-07-26 (user 1): `review_time` = 2021-05-22 15:31:47 UTC, `card_id` = 15:14:10 UTC —
    the card was created 17 min before its first review, i.e. "first review - card creation" reads directly.
  - **=> every HIGH-priority feature in `optimization/FUTURE_FEATURES.md` is derivable TODAY** (time-of-day
    + circular-mean deviation, true calendar phase, creation->first-review, seconds-resolution
    time-since-any-review, creation-batch size, tenure, note/deck/preset ages). No export is blocked.
  - **What IS still needed for them:** they are per-review FEATURE COLUMNS, so they need a preprocessing
    change + an **LMDB rebuild** sourced from `-id`. **Unlike the DECK TREE**, which needs NO rebuild at
    all — see the correction in `FUTURE_FEATURES.md`.
  - **★ THE DELETE IS PROBABLY UNNECESSARY — BUILD ON F: (measured 2026-07-27).** The "must delete
    first" conclusion assumed the rebuild lands on C:. It does not have to. `train_db_5k_h1` is a
    BARE RELATIVE path (`data_processing_train_5k_h1.toml:10`), i.e. repo root on C:, which is the
    only reason it is competing for C:'s 242 GB. Retarget it at F: and the new train (372.5 GB) +
    new test (232.8 GB) = **605 GB against F:'s 889.5 GB free** — both fit BESIDE the originals with
    ~284 GB spare, so a bad rebuild is `rm -rf` of the new dir instead of a 2-4 day re-run. Reclaim
    candidates if F: gets tight, both Andrew's call and neither needed to start: `train_db_5k_h2`
    (372.5 GB on F:, the swap half, referenced by NO live toml) and the closed-era `train_db_sc8k`
    + `train_db_sc8k_1500` + `test_db` (101 GB on C:). ⚠ **The TEST db must be rebuilt too** — eval
    feeds the same feature vector, so a train-only rebuild silently scores a mismatched layout.
    Full plan + the four code sites + the 100-user de-risk build: `optimization/FUTURE_FEATURES.md`
    "IMPLEMENTATION PLAN". Andrew's delete authorization stands as a fallback; prefer not to use it.
  - **DISK / DELETE-THE-OLD-DB — AUTHORIZED, WITH A SEQUENCING CONSTRAINT (Andrew 2026-07-26).**
    `train_db_5k_h1` is 372.5 GB and C: has 229 GB free, so a side-by-side rebuild does NOT fit ON C:. Andrew:
    *"We can delete the current copy, sure. It's strictly more data, not less, so nothing will be lost."*
    **Verified, and he is right:** published vs `-id` over 6 users (1/2/3/17/101/555, 363,598 reviews) —
    row counts IDENTICAL user-for-user, and `day_offset` differs on **4 of 363,598 reviews = 0.001%**
    (the show-time correction moving a review across a day rollover). So the rebuild is additive in
    columns and ~identical in rows.
    ⚠ **BUT DO NOT DELETE UNTIL THE REBUILD IS READY TO RUN.** The endgame order puts the algorithmic
    phase FIRST, and every run in it reads `train_db_5k_h1`; the rebuild is 2-4 days of CPU. Deleting
    early = a dead GPU and a killed run for zero gain. **The delete is step 1 of the FEATURES phase, not
    a preparatory step.** Do it when the preprocessing change is written and smoke-tested, not before.
    Three things to settle at that moment, none now: (1) re-run the champion on the new DB to re-base —
    at 0.001% it should be ~free, but cross-rebuild numbers are otherwise not comparable; (2) confirm
    the `size` gate still holds (row counts say yes, but the equalize filter is derived); (3) decide
    whether `label_filter_db` (37.3 GB, the "permanent deterministic cache") needs rebuilding too.
    `test_db_5k` (232.8 GB) lives on F: with 890 GB free, so it CAN be built side-by-side.
- Quant papers: `scratchpad/{rwkvquant,rwkvedge}.txt` (poppler installed; the Read tool handles PDFs). Use the
  CURRENT session's scratchpad dir for transient logs (it rotates on teardown -- check task-output paths).
