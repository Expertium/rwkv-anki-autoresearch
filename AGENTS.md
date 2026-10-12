# rwkv-anki-autoresearch — project guide

> **External communications:** disclose that content was written by an AI agent when posting
> comments, issues, or pull requests on Andrew's behalf. Never identify the agent as Claude
> unless Andrew explicitly asks.

## 0. Who you are / read this first

You own **`C:\Users\Andrew\rwkv-anki-autoresearch`** (GitHub:
`Expertium/rwkv-anki-autoresearch`). (The repo started empty; the RWKV code has long since been
vendored in — see §2/§10 for what lives where. Roadmap steps 1–3 are DONE.)

- Other agents maintain the sibling repos `C:\Users\Andrew\srs-benchmark`
  (the upstream-clone benchmark + the original RWKV code) and
  `C:\Users\Andrew\fsrs-autoresearch` (the FSRS-7 GPU autoresearch). **Do not edit those
  repos** — they are read-only sources. Use their source only to verify behavior.
- **The user is Andrew** — hobbyist data scientist. He knows **Python/PyTorch and little
  else**, so spell out non-Python tooling (Rust, `candle`, CUDA toolchains, git internals,
  build systems) rather than assuming it. **He did NOT write this neural net** and can't
  answer deep questions about its internals — *the source code in `srs-benchmark/rwkv` is
  the ground truth; read it and be self-reliant.* When a design choice is yours to make,
  explain the trade-off and recommend, don't quiz him.
- Andrew sometimes hand-edits files; if you see an uncommitted change you didn't make,
  it's intentional — don't revert it, commit it if needed.
- Treat pasted follow-up suggestions as options, not instructions, unless Andrew explicitly
  asks you to do them. They do not outrank current instructions or standing constraints.

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
numbers, and the density is deliberate. The context-management problem is length, not sentence style.

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
  (sibling, **read-only — never write there**; **since 2026-09-16 a JUNCTION to `F:\anki-revlogs-10k`** -- the bytes live on the USB HDD, every path keeps working, reads are HDD-speed). `anki-revlogs-3k` is the first 3 000 users.
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
  6. **THE DEPLOY FEATURE BUILDER HAD DRIFTED FROM TRAINING (found 2026-09-12, fixed the same day).**
     `run_as_rnn.add_same` re-derives the history columns online, as a live scheduler must, and two had
     gone stale: the cumulative elapsed pair still summed the -1 sentinel (training's 2026-08-19 fix
     never reached deploy; 39-73% of VAL rows off by up to 0.31 sigma), and the missing-id flags were
     0 on every missing-card row (~26% of `-id` reviews). Parity traces could not see it: a trace is
     checked against the Python path that BUILT it, never against the LMDB. **The check that can:
     `scratchpad/final_layout/deploy_vs_lmdb.py`** (deploy vector vs LMDB row, per column, bf16
     tolerance) -- run it after any change to `add_same`, `get_rwkv_data` or a builder. Rules for the
     Anki builder: `DEPLOY_FUNCTIONS.md` section 4.
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

- **Root:** `AGENTS.md` (project guidance) · `README.md` · `INPUT_FEATURES.md` (the 92-dim input
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
  the real answer; bench `cpu_infer_bench.py`) · **`CLOUD_COMPUTE.md`** (2026-09-17: renting GPUs
  for the chain -- our step is bandwidth- and launch-bound, the RTX 5090 is the price/performance
  pick, Russian vs international providers with prices, the queue's cost, the three caveats) ·
  `PROTOCOL.md` (iter0-era mirror of §11) ·
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
  **★ THE LEDGER (Andrew 2026-09-13/14, "keep this table at hand"): `optimization/LEDGER.md` = every change since the old d=128 model as
  telescoping measured pairs (recipe +0.0037/+0.0043, smaller model +0.0010/+0.0005, rectifier +0.0019/+0.0001, HP tuning
  -0.0013/-0.0010, features -0.0007/-0.0029, 10x budget -0.0005/-0.0015, leak fix +0.0002/+0.0021, QAT +0.0025/+0.0045 ...).
  Update it when a pending row lands (w10lf, decay length, grafts, leak-free QAT, phase 5).**
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
4./5. **★★ CURRENT RULE FROM 2026-09-24 (Andrew, RELAXED): "either a double improvement, or an improvement in one mode and null in the other".** Accept iff AT LEAST ONE mode improves by **raw >= 0.00005** ("Lower the absolute bar to 0.00005") with the paired one-sided Wilcoxon **p < 0.0001**, AND **neither mode is significantly worse** (worse = the by-user mean declines AND p_worse < 0.05 -- the curve-side harm test, now symmetric). Tool: `paired_pvalue.py --either-mode` (exit 0 = pass). This SUPERSEDES the both-modes rule and the curve-side exception below, which are kept as history. Re-scan of the rejected rows under it: 106 `s64hsib` (sibling) and 67 `muonscale` (Muon on the scale matrices) pass and still matter -> confirmed together at 10+1 as `s64hrs10` (control s64hrf10); 87 `lfwu20k`, 49 `iter49_cmix`, 38 `iter38_kda075` also pass but are superseded (warmup is in the WS tuning; old lineage; KD off). R2 (97, 105) fails: ahead significantly worse.
   ~~OLD:~~ 4./5. **★ CURRENT RULE (Andrew 2026-08-10, TIGHTENED): each mode's RAW improvement vs the CURRENT
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
**★ EVERY ROW GETS A NUMBER, NOT AN ACRONYM (Andrew 2026-09-19: "change the naming from arcane acronyms to either numbers (same as before) or numbers with a one letter prefix").** The front table's first column is `<number>` + the run tag in code font, e.g. `**70** \`w10plain\``; the number is what chat replies and summaries use. Plain numbers continue the old sequence in completion order (70 = w10plain ... 84 = lfq1; screens count too); **F1-F6** = the six features-phase baselines that finished between 61 and 62 (featA2, e2sc, fixc, featB, gen4base, realcyc), because 62-69 were taken. `research_log.jsonl` carries the same value in `number`; the run TAG stays the path / checkpoint identity (never put a number in a path). A record script takes the next free number at verdict time.
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
**⚠ 2026-09-12: the gen-5 corpus behind this was dumped MIS-WIRED (no `RWKV_STREAM_ORDER`); re-checked on a correct dump the stale verdict STANDS (old WKV 0.9392 vs random 0.9597, old shift > 1), but the replacement WKV catalog is ~25% worse than a correct refit (0.6046 vs 0.4845). `scratchpad/qat_gen5/FINDING.md` addendum.**
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
| 5 | **Final HP tuning, WITH QAT ON** | ⚠ NEW — all prior tuning was PLAIN. **SCOPE DECIDED (Andrew 2026-09-12: "let's only tune the decay/QAT phase"): trials are QAT decays branched off w10lf's shared WS** (~1.2 days each at L*=2, ~16 h at L*=1, ~5 h for a bad trial stopped early by the validation trace, which predicted the gate within 0.0004) -> ~1-1.5 weeks. WS HPs stay fixed (a WS trial is ~38 h; the ws10 gap screen says fit-limited). Candidate knobs: decay LR (Muon AND AdamW together), catalog LR, KD weight, QAT-phase wd. Decay length, KD and the b13 bits are already answered by the running chain. Starts after the chain (~09-20). **PLAN DRAFTED 2026-09-13: `optimization/PHASE5_PLAN.md`** (knobs + env names + grid, the validation-trace kill rule, the 1000-user subset eval with full-VAL confirmation, the four tools to build). |
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
**⟶ 2026-09-29 17:00 -- d128e10 IS TRAINING (started 16:55; dry twin 0 skipped batches under the 11 GiB cap; 0.72 steps/s -> ends ~10-04). Chain v4 (`scratchpad/wstune/chain_ws4.cmd`, detached) waits for its waiter's marker, then wtmuon0p5 + its decay/eval, wtmuon2 + its decay/eval (preflight after each WS), then wtwarmup2000 (WS-only). All regenerated with the cap + guard 61 (`mk_ws_trial.py`).**

**⟶ 2026-09-29 16:00 -- 124 `wtlr2` (WS peak LR x2) REJECTED: -0.000236 / -0.000157 vs s64hc10 (n=2486). With 123 the WS LR optimum is x1; WS knob 1 CLOSED.** Chain v3 wrote chain_ws2.log's marker, so the d128e10 waiter fires (dry twin, then the ~4-day run, both under the 11 GiB cap). The Muon-ratio and warmup trials follow d128e10 on s64hc10, LR-type ones judged after their decays.

**⟶ 2026-09-29 15:25 -- RAM hit 0 GB at ~15:00 (Andrew's PC unusable): wtlr2's eval (~30 GB on giant users) + the FSRS-rs session's 1000-user benchmark (~74 GB, 18 workers, started 14:57). I stopped my eval; the FSRS-rs session stopped its benchmark and now uses <= 4 threads. Andrew then asked for a VRAM cap below 12 GB: `RWKV_VRAM_CAP_GB=11` (LIVE RULES). wtlr2's eval resumed under the cap (`run_wtlr2_evalresume2.cmd` via `scratchpad/wstune/finish_chain_ws3b.cmd`, which then writes chain_ws2.log's marker for the d128e10 waiter); d128e10 + its dry twin now carry the cap and the skipped-batch guard (61). Interim wtlr2 (1804 users): -0.00019 ahead / -0.00015 imm vs s64hc10, i.e. LR x1 is the optimum.**

**⟶ 2026-09-29 08:25 -- 123 `wtlr0p5` (WS peak LR x0.5) REJECTED AFTER ITS DECAY: -0.000237 ahead / -0.000159 imm vs 104 s64hc10 (both significantly worse, n=2492, size 0).** Its WS endpoint had read -0.0011 ahead: the WSD bias predicted in the 01:50 entry is real, so LR-type trials are judged only after the decay. `wtlr2` (x2) decays now (peak 2e-3, Muon 0.005; started 08:20, eval ends ~14:30), then the d128e10 waiter fires. Also 09-29: disk check done (C: 140 GB, F: 1098 GB, nothing to delete); `gpu_cold_check.py` skips the box while `scratchpad/cloud/BOX_RETIRED.txt` exists.

**⟶ 2026-09-29 01:50 -- ★ LR KNOBS NEED THEIR DECAY BEFORE A VERDICT; CHAIN v3 BUYS THEM.** wtlr0p5 (LR x0.5, AdamW + Muon) ended its WS at -0.00110 ahead / +0.00024 imm vs s64hc10 (last-5 validation window; wtlr2 x2 was +0.00022 / +0.00061). Under WSD a lower LR looks better at the WS end, so an LR trial is judged only after its 1-epoch decay + full-VAL eval (either-mode gate vs 104 s64hc10); wtlr2's rejection is re-opened. `mk_ws_trial.py` also hardcoded the decay peak LR at 1e-3 (fixed). Chain v2 stopped 13 min into wtmuon0p5 (partial files in `scratchpad/wtmuon0p5/aborted_0929/`); **`scratchpad/wstune/chain_ws3.cmd`** = `run_wtlr0p5_de.cmd` -> `run_wtlr2_de.cmd` (`mk_decay_eval.py`, guards 77/78 on the decay LRs) -> writes chain_ws2.log's marker -> the d128e10 waiter. Muon/warmup trials are regenerated on the winning LR after d128e10. `optimization/WS_TUNE_PLAN.md` CORRECTION section.

**⟶ 2026-09-29 00:00 -- d128e10 (d=128 at 10+2, 2,575,766 params) IS QUEUED BEHIND THE WS TRIALS AS A POSSIBLE KD TEACHER (Andrew: option A = run it before the final QAT run; "make sure to preserve its weights"). The real run had NEVER run: it sat in the box's queueB and was dropped when the lanes merged into queue_all.txt on 09-20 (only the dry twin ran). `scratchpad/d128e10/wait_ws2_then_d128.cmd` (detached, log `wait_d128.log`) fires on chain_ws2's marker -> preflight -> the local 20-user dry twin (proves MAX 32768 fits 12 GB) -> the real run (~4 days); stub-tested on the success and dry-fail paths. KEEP every d128e10 checkpoint (`scratchpad/d128e10/KEEP.md`); the final model moves ~4 days, to ~14 October.**

**⟶ 2026-09-28 23:55 -- PHASE 5 KNOB 2 CLOSED (122 `s64hcq4dl2` decay LR x2: -0.00006 / -0.00013; 121 x0.5: -0.00015 / -0.00007): the 112 recipe is at its optimum on both LR knobs. BOX WORK IS FINISHED AND ARCHIVED: `F:/box_archive/box_0928_part1.tar` + `box_0928_part2_full.tar` hold every WS final + optim, decayed checkpoint, learned catalog, log and box result (328 key files, md5 verified against the box). Andrew: DESTROY the instance (he clicks it). PC: the 22:14 restart killed wtlr0p5 at step ~96,700; `scratchpad/wstune/chain_ws2.cmd` resumed it from 96,000 at 22:18, then runs wtmuon0p5, wtmuon2, wtwarmup2000.**

**⟶ 2026-09-28 11:40 -- PHASE 5 KNOB 1 CLOSED: the catalog LR optimum is x1** (119 x0.5 -0.00010 / -0.00007, 120 x2 -0.00018 / -0.00035, both vs 116 on the box). 118 `s64hsf` (self-feed on s64h, lever live) -0.00018 / -0.00018, rejected. **Knob 2 (decay LR, Muon + AdamW together; `mk_box_qat.py --declr`, guards 77/78) runs on the box: `s64hcq4dl0p5`, `s64hcq4dl2`, decays started 11:18/11:21, evals end ~03:00 on 09-29.** Box-only artifacts part 1 (all WS finals + optim, decayed checkpoints, learned catalogs, logs, box results; 549 MB) are in `F:/box_archive/box_0928_part1.tar`; part 2 = the two dl runs, before 06:00, then ask Andrew STOP vs DESTROY.

**⟶ 2026-09-27 23:00 -- 117 `s64hsibq4`: THE SIBLING COLUMNS RAISE THE QAT TAX (-0.00035 / -0.00106 vs `s64hq4s`), WHICH EXPLAINS 114; 110 is NOT the deploy base, the recipe stays 112 on s64hc10. RECTIFIER: Andrew keeps the current PAVA ("Let's keep the current PAVA").** Screen `scratchpad/rectscreen/`: p down to -4 buys 0.00006-0.00010 at eval; anchor clip on Good won on w10lf (-0.00024) and LOST on s64hc10 (+0.00062); every monotone rectifier costs ~0.0011 vs none. The Anki fork rectifies only the button intervals (arithmetic PAVA on margins) and schedules from the unrectified curve -- a known difference, not a research item.

**⟶ 2026-09-27 12:40 -- ★ BOX BUDGET: $23.01 LEFT at $0.493/h (Andrew) = ~46.7 h -> the balance ends ~2026-09-29 11:00. PLAN:** (1) running now: `s64hsf` (lane A), the sibling-isolation QAT pair `s64hq4s` (lane C) / `s64hsibq4` (lane D) -- results ~18:00-22:00; (2) queued behind them: phase 5's first knob on the box, the catalog LR x0.5 / x2 on the 112 recipe (`s64hcq4cb05` lane C, `s64hcq4cb2` lane D; `mk_box_qat.py --cblr`; control = 116 `s64hcq4`, same box) -- ends ~09-28 midday; (3) STOP launching new box work at ~09-28 20:00; (4) before ~09-29 06:00 fetch every box-only artifact worth keeping (results, logs, the s64hsm10 / s64hsg10 / s64hc10 WS-finals and optim, the learned catalogs), then ask Andrew STOP vs DESTROY (a stopped Vast instance still bills storage). **Also 09-27: 116 `s64hcq4` (112 on the box) = 112, so 114's QAT loss (-0.0005 / -0.0007) is 110's base change; 115 (2-bit norm) is a null; the deploy recipe stays 112 on s64hc10.** The PC keeps the WS trials (wtlr2 ends ~11:00 today).

**⟶ 2026-09-26 09:45 -- ★ 112 `s64hqr4b` IS ACCEPTED: 4 RESIDUAL WKV STAGES (card 403 b / note 214 b, inside Andrew's 450 / 235 budget) CUT THE d=64 QAT TAX BY ~1/3.** Deployed 0.299369 / 0.267496 = +0.000937 ahead / +0.001202 imm vs s64hq1 (p 7e-190 / 0, n=2492); tax vs s64hc10 +0.001846 / +0.002803 (was +0.002782 / +0.004005); vs the old d=128 model +0.0048 / +0.0039. This is now the deploy quant recipe for d=64 (`scratchpad/s64hrefine/FINDINGS.md` section 7, LEDGER row 16). Still above the < 0.0010 tax target; the card budget is used. Also: the six unused F: databases were DELETED (Andrew; F: 1,481 GB free); `write_eval_toml.py` / `write_decay_setup.py` still default to the deleted `test_db_5k`, so a runner must set its db. Box: `holdeval2` sits before `s64hsf` in `queue_all.txt` (its eval is done, so it releases on its next poll). PC: `wtlr2` WS since 04:49 (kill check at step 32,000). Owed: a QAT re-base of s64hsm10 (105 inputs) with this 4-stage recipe.

**⟶ 2026-09-25 18:05 -- ★ 111 `s64hqnr` IS REJECTED: THE FITTED WKV NORM RANGE DOUBLES THE QAT TAX (+0.0059 / +0.0079 vs s64hq1's +0.0028 / +0.0040; -0.0031 / -0.0039 vs s64hq1, p_worse ~0).** The local probe agrees, the box eval loaded the right range and catalogs. PTQ on the PLAIN model ranked the range best, but the narrow range caps the largest WKV state norm at 0.20 instead of 1.0 and QAT needs that level: **a PTQ ranking of a quantizer RANGE does not transfer to QAT** (`scratchpad/s64hrefine/FINDINGS.md` section 6). `s64hqr4` (4 stages + a narrow 2-bit range) was STOPPED at decay step ~1500 and replaced by **`s64hqr4b` = s64hq1 + 4 residual WKV stages only** (1-bit default norm; card 403 b, note 214 b; `mk_qr4b.py`; chain `chain_qr4b.cmd`: dry twin -> arm -> probe -> wtlr2). Also: `s64hsg10` (the sibling-gap flag `sibgap_none`, read-time, 10+1 vs s64hsm10) is queued on the box after holdeval; the calendar-cycle conventions are to be unified at the next rebuild (Andrew).

**⟶ 2026-09-25 15:40 -- ★ ANDREW LIFTS THE STATE BUDGET: CARD <= 450 BITS, NOTE <= 2.5x TODAY (235 b), "a 1000x reduction ... instead of 2250x if it significantly improves log loss" (=> card <= ~418 b).** d=64 today: card 163 b, note 94 b. The PTQ ladder (`scratchpad/s64hrefine/FINDINGS.md` section 5) picks **4 residual WKV stages of b10 + 2-bit norms on [-3.3,-2.0] = card 411 b / note 218 b** (+0.0018 / +0.0034 PTQ vs +0.0030 / +0.0058). Arm **`s64hqr4`** (control s64hqnr) runs on the PC via `scratchpad/s64hrefine/chain_qr4.cmd` (chain_nr2 was stopped in its probe wait): s64hqnr probe -> dry twin -> s64hqr4 -> wtlr2. Also today: **110 `s64hsm10` ACCEPTED** (sibling rating + Muon-on-scale, imm +0.000105 p 7e-11, ahead null) -> plain incumbent, 105 inputs; the deploy port of the sibling columns and a QAT re-base on it are owed. Shift m2b14 gave no PTQ gain. **The run board** (https://claude.ai/artifact/8KDuhUEdBhUqsUVXtv1b41) is live -- keep it current (LIVE RULES).

**⟶ 2026-09-25 11:50 -- ★ DRIVE C: FILLED TO ZERO AT 06:12 AND KILLED `s64hqnr`'S DECAY AT STEP ~6,676; RESUMED FROM STEP 6000.** The trainer printed no traceback, only "Killed processes." then a cut-off line; the System/Application event log shows SearchIndexer "low free disk space" at 06:12:54. Cause: `C:\Users\Andrew\clanki-worktrees` = **273.8 GB** (Clanki's parallel Anki builds overnight), plus clanki 48 GB and clanki-logs 50 GB; C: had 291 GB free on 09-20. Clanki moved ~63 GB to `F:\clanki-moved-aside` (C: 61 GB free), asked Andrew to run its worktree-cleanup script, and its agents now refuse a build that would leave < 60 GB free. **I deleted nothing.** Recovery: `scratchpad/s64hrefine/mk_resume_nr.py` (adapted from `lfq1rvq3b10real/mk_resume.py`; guards 101-106, the norm banner replaces the stage guard) -> `run_s64hqnr_resume.cmd`; verified live: step-6000 learned catalogs + the [-3.1,-2.3] norm range loaded, LR schedule continues at factor 0.2409, steps 6001-6004 match the crashed run within ~0.003. Chain `chain_nr2.cmd`: resume detached -> `probe_gate.py` (reads only the lines after the last RESUME banner, because the appended log still holds DONE_EXIT_23) -> **`wtlr2` fresh** (stopped again ~4.7 h in; partial files in `scratchpad/wtlr2/aborted_0925/`). Box: `s64hsm10` at WS step 104,605 of 109,350 at 11:45 local; `holdeval` then keeps the GPU for s64hqnr's eval. **Box balance: Andrew topped up 2026-09-25 from $22.55 by $25 -> ~$47.5.** Box GPU monitor window restarted on Andrew's request.

**⟶ 2026-09-25 00:10 -- ★ THE d=64 NORM RANGE IS A STALE d=32 CONSTANT; THE ZERO-BIT REPAIR ARM `s64hqnr` RUNS FIRST ON THE LOCAL GPU.** Full note: `scratchpad/s64hrefine/FINDINGS.md`. (1) The planned "int4 low-rank factors" refinement does not exist: with the WKV catalog on, the kernel ignores the factor level (`!c_pq_active` branch); the 2-bit norm stays closed (measured 08-13). (2) **PTQ ladder, CPU, Rust engine, plain s64hc10, 40 users** (`ptq_ladder.py`): full recipe +0.00589 / +0.00863; the largest term is the 1-bit NORM; shift m4b12 buys only ~0.0002 for +24 bits/vector. (3) The WKV norm grid is log2-uniform over [-3, 0] octaves, fitted in the d=32 era: at d=80 33% of heads sit above the 1-bit threshold, at d=64 only 1.7%, so the bit is nearly a constant. Refitted range [-3.1, -2.3]: PTQ +0.00303 / +0.00580 at ZERO extra bits (the 8-bit floor is +0.00275 / +0.00510). A refitted SHIFT range made PTQ worse -> unchanged. New settings (default = old range, engine checked byte-identical): `RWKV_QAT_NORM_RANGE_WKV` / `_SHIFT` (QAT), `RWKV_PQ_NORM_RANGE_WKV` / `_SHIFT` (Rust). (4) **`s64hqnr` = 108 s64hq1 + the fitted WKV range, one variable**; PREREG in FINDINGS.md (P1 tax falls 0.0003-0.0012 per mode; P2 null = the learned catalog absorbs it). Local chain `scratchpad/s64hrefine/chain_nr.cmd`: dry twin (from s64hq1dry, local eval) -> arm detached (eval on the box) -> **`wtlr2` restarts FRESH on the arm's PROBE_OK** (it was stopped ~25 min in; partial files in `scratchpad/wtlr2/aborted_0924/`). Box: `holdeval` inserted after `s64hsm10` holds the GPU until `PRIORITY_EVAL_DONE` is newer than `/workspace/HOLD_FOR_EVAL` (30 h cap), so the arm's eval runs before `s64hsf`. If adopted, the range must go into the safetensors metadata (Anki sets no env). Also: INPUT_FEATURES rows marked dropped; DEPLOY_FUNCTIONS 7 now says the missing-id placeholder rule depends on the model version (old: one shared note; new: per-card note).

**⟶ 2026-09-24 23:50 -- ★ ANDREW DROPPED THE NINE COLUMNS ("Drop nine columns") after 109 failed the either-mode gate. The incumbent is `s64hc10` (332,922 params, 103 inputs). Consequences done: (1) the box arm `s64hrs10` (on s64hrf10) was STOPPED at 19:41 box time and replaced by **`s64hsm10` = s64hc10 + sib_grade,sib_none + RWKV_MUON_INCLUDE_SCALE=1** (333,178 params MEASURED, control s64hc10; `scratchpad/shrink/mk_sm10.py`), running; queue s64hsm10 -> s64hsf. (2) The WS trials are rebuilt from `scratchpad/s64hc10/run_s64hc10.cmd` with reference `scratchpad/wstune/ref_s64hc10.json` (111 val points); the old s64hrf10-based trials moved to `scratchpad/wstune/old_rf10/`. `wtlr2` restarts from scratch after Andrew's reboot. INPUT_FEATURES.md rows 44-55 (the nine columns) must be marked dropped.

**⟶ 2026-09-24 23:30 -- ★ ANDREW: THE d=64 ADOPTION STANDS, AND CARDS GET MORE BITS ("adoption of the smaller model stands. We'll have to give cards more bits").** This is a grant of card-state budget on d=64. 107 showed WKV index bits hit a floor, so the bits go where the floor is: the refinement experiment gives the SHIFT catalog, the int4 LOW-RANK factors and the 1-bit NORMS more precision one at a time (WKV held at one b10 stage), on the d=64 QAT recipe of 108 `s64hq1`, measured as tax vs `s64hc10`. Still open for Andrew: keep or drop the nine columns (109).

**⟶ 2026-09-24 23:10 -- 109 `s64hrf10` FAILS THE NEW GATE: imm +0.000150, ahead -0.000209 significantly worse.** 10+1 CONFIRMATION of the nine directed-accepted read-time columns (cum_new_cards + the eight rowfeat columns; 112 inputs, 334,074 params) on the adopted d=64 trunk vs s64hc10 (same runner, same box): ahead -0.000209 (WORSE, p_worse 1.4e-03) / imm +0.000150 (p 3.9e-18), n=2492, size 0, 0 tracebacks. FAILS the 2026-09-24 either-mode gate: imm improves, but ahead is significantly worse. At 1+1 on d=80 the same columns gave rowfeat +0.000087 / +0.000171 and newc1 +0.000112 / +0.000104; at the full budget on d=64 the ahead half turns negative. Keeping the columns is Andrew's call (they were directed accepts). s64hrs10 (on the box now) uses s64hrf10 as its control, and the WS trials use its WS trace as reference.

**⟶ 2026-09-24 23:05 -- ★ 108 `s64hq1`: THE d=64 QAT TAX IS +0.002782 / +0.004005 (1.29x / 1.15x the d=80 tax).** THE QAT TAX ON THE ADOPTED d=64 TRUNK: lfq1's exact recipe (1-epoch leak-free QAT decay, one b10 WKV joint stage at --h 4, m2b12 shift at c64, learned catalogs, 1-bit norms, int4 low-rank) from s64hc10's WS-final, evaluated quantized on the box. Deployed 0.300306 / 0.268698; tax vs its plain twin s64hc10 +0.002782 ahead / +0.004005 imm (p_worse 1e-300 / 0), against +0.002153 / +0.003468 on d=80 (84 lfq1) = 1.29x / 1.15x. n=2492, size 0, 0 tracebacks. PREREG (scratchpad/s64hq1/PREREG.md): P1 MISSED on ahead (band +0.0018..+0.0026), HELD on imm; P2's falsifier (>1.3x in either mode) did NOT fire, by 0.01 on ahead. CONSEQUENCE: the 1.69x cut costs MORE once quantized -- plain gap vs lfd1 +0.000028 / +0.000402 (104), deployed gap vs 84 lfq1 +0.000650 / +0.000940. The smaller model leans harder on each quantized state value. Card state at b10 = 88 WKV bits + shift (d=80: 110 + shift).

**⟶ 2026-09-24 22:34 -- ★★ 107 `lfq1rvq5b10` IS IN: THE QAT TAX AT +200% OF STATE BITS IS +0.001302 ahead / +0.002109 imm (NOT ENOUGH; Andrew's target is < 0.0010 in both modes).** 5 residual stages of b10 = 30 index bits per head, card 585 b / note 305 b; deployed 0.298649 / 0.266250, vs 99 +0.000303 / +0.000536 (p 8.3e-68 / 1.7e-219), n=2487, size 0, 0 tracebacks. At b10 the tax was +0.002153 / +0.003468. PREREG: both still over 0.0010 -- +200% not enough either; P4's reading applies; P1 band HELD; imm nearer model B (a floor; imm never clears 0.0010 by WKV bits) (|imm - A| 0.000641, |imm - B| 0.000057); tax - 99: -0.000303 / -0.000536. P3 (catalog stage norms still decreasing): HELD. NEXT: no further stages until the tax models are read (BITS.md; PREREG outcomes).

**⟶ 2026-09-24 18:26 -- CPU INFERENCE WORK IS POSTPONED UNTIL THE MODEL IS 100% FINISHED (Andrew: "let's put off CPU work until we have a 100% finished model").** This covers the fork's engine (rslib/src/rwkv: Total Knowledge warm-up ~125 s + curve sums ~44 s on his collection after Clanki PR #277) and our rust/rwkv-infer speed work alike. It is plan phase 7 anyway. Clanki was told; its B-028 (history build must drop ignored reviews BEFORE finding the replay start, as the fingerprint does) waits on Andrew separately.

**⟶ 2026-09-24 13:03 -- ★ THE ACCEPTANCE GATE IS RELAXED (Andrew): one mode >= 0.00005 at p<1e-4, the other not significantly worse (`paired_pvalue.py --either-mode`).** Newly eligible and relevant: 106 sibling rating and 67 Muon-on-scale; both are confirmed together at 10+1 as `s64hrs10` (s64hrf10 + sib_grade,sib_none + RWKV_MUON_INCLUDE_SCALE=1, 334,330 params MEASURED, control s64hrf10), queued on the box right after s64hrf10. Also 2026-09-24: the WS-trial Windows runner now honours RWKV_WS_ONLY and rc 42 (the first local dry twin ran a decay after its WS); round 1 regenerated from s64hrf10.

**⟶ 2026-09-24 08:19 -- 106 `s64hsib` IS IN: REJECTED BY 0.000002 ON IMM, NULL ON AHEAD (Andrew's call whether to direct-accept, like 96 and 102).** Sibling rating (sib_grade + sib_none, read-time columns) on the s64h trunk at 1+1 vs 94 s64h: ahead -0.000003 (p_better 0.14, a null) / imm +0.000098 (p_better 1.3e-09), n=2493, size 0, 0 tracebacks. Rejected under the both-modes rule: imm misses the raw 0.0001 bar by 0.000002 and ahead is a null. PREREG held on every point: P1 (+0.00003..+0.00015 imm, +0..+0.00008 ahead), P2 (imm is the mode that moves), P4 (no harm); P3 (imm > +0.0003) did not fire, so the note stream already carries most of the sibling signal. Same shape as 96 newc1 and 102 rowfeat, which Andrew directed-accepted: imm resolved, ahead null. At 1+1 effects run ~0.65 of full size.

**⟶ 2026-09-23 19:57 -- ★ RUST PARITY PASSES ON THE ADOPTED d=64 TRUNK.** `reference_s64h/` (exported from `s64hc10_d_10935.pth` by `scratchpad/s64hpar/run_export_s64h.cmd`): imm 0.000000 / ahead 0.000000 on users 107/136/156, max per-review 2.50e-06, with NO recipe variable set -- the safetensors metadata carried stream order, interleave and clamp, and the engine read the d=64 shapes (4 heads x K=16) and the stripped channel mixers from the weights unaided. Unquantized replay ~2,450 rev/s (R/s ~5,000, states/s ~5,050) vs ~2,160 on d=80. Only ref_metrics.json is tracked.

**⟶ 2026-09-23 19:46 -- ★★ THE SHIPPED TRUNK IS NOW `s64h` (332,922 params, d=64, 1.69x smaller).** Andrew accepted it on 09-22 conditional on the 10+1 check; 104 `s64hc10` passed (+0.000028 / +0.000402 vs lfd1). Consequences: the WS tuning trials regenerate from `s64hrf10` (s64h + the nine accepted columns, on the box) once it confirms; the QAT / bits chain moves to d=64 (catalogs refit on the d=64 model, card-bit budget redefined); the queued d=80 `selffeed2` is replaced by `s64hsf` (selffeed on s64h, 333,882 params, control 94). The running 5-stage arm stays on d=80: its question is about the quantizer.

**⟶ 2026-09-23 17:58 -- 105 `r2s64h` IS IN: REJECTED, THE SAME TRADE AS 97 AND LARGER ON AHEAD.** The same TRADE as 97 r2s80, now on the d=64 trunk and LARGER on ahead: ahead -0.000573 (worse at p=3.5e-29) against imm +0.000171 (better at p=1.0e-24) vs 94 s64h, n=2499, size 0/2499, 0 tracebacks. Rejected under the both-modes rule. Same caveat as 97: R2 evaluates PLAIN (monotone by construction, no PAVA probes) while s64h evaluates RECTIFIED, so this compares the two deploy quantities, not a single variable. Two trunks, same sign both times: the button-ordered head buys a little imm and costs more ahead. Its only remaining case is deploy speed (one forward pass instead of four probes), now priced at +0.00057 ahead on the trunk that may ship. The R2 route is closed for accuracy.

**⟶ 2026-09-23 12:34 -- ★ 104 `s64hc10` (THE 10+1 SHRINK CONFIRMATION) IS IN: PASS.** 10+1 confirmation of the directed-accepted 94 s64h (332,922 params, 1.69x smaller) against lfd1 (561,732, same budget): cost +0.000028 ahead (p_better 1.0e+00) / +0.000402 imm (p_better 1.0e+00), i.e. 0.000012 / 0.000176 per 100k removed against the 0.0002 rule (allowance +0.000458) -> PASS. At 1+1 it cost +0.000304 / +0.000493, so the cost scaled x0.09 / x0.82 with the 10x budget. n=2492, size 0, 0 tracebacks. Adoption would re-base phases 5-6 (d=64 catalogs, card-bit budget): Andrew's call.

**⟶ 2026-09-23 09:45 -- ★★ 103 `lfq1rvq3b10real` IS IN: THE QAT TAX AT +100% OF STATE BITS IS +0.001534 ahead / +0.002468 imm (NOT ENOUGH; Andrew's target is < 0.0010 in both modes).** 3 residual stages of b10 = 30 index bits per head, card 385 b / note 205 b; deployed 0.298881 / 0.266609, vs 99 +0.000071 / +0.000178 (p 7.6e-11 / 1.8e-42), n=2487, size 0, 0 tracebacks. At b10 the tax was +0.002153 / +0.003468. PREREG: P2 HELD (the null): tax within 0.0003 of 99's in both modes -- provenance is immaterial when the catalogs learn (tax - 99: -0.000071 / -0.000177). P4 held: all 3x1024 rows moved, stage norms stay decreasing. NEXT: +200% (5 stages).

**⟶ 2026-09-23 12:30 -- THE SIBLING RATING IS BUILT, VERIFIED ON ALL THREE PATHS THAT EXIST, AND QUEUED AS `s64hsib`.** Andrew asked for the screen on 200 users; it REVERSED the 100-user verdict (9 of 10 held-out-user splits positive on the sibling rows vs 6 of 10; +0.000778 over a simple baseline) -- the 100-user sample held too few of the ~45 heavy users to judge. **Build:** `scratchpad/rowtables/build_sibtable.py` precomputes `sib_flat.npy` (int8, 0 = none else the sibling's grade; 538.6 MB, 3 min) because raw per-row card + note ids would be ~8.6 GB; vectorised version == plain loop EXACTLY on 8 users. Two columns `sib_grade` (z-scored over the 4.28% of train rows that have one, NaN -> 0) and `sib_none`, APPENDED LAST in `extra_features.SPECS` so the 112-input layout is unchanged (checked: the old columns are an exact prefix). Deploy (`run_as_rnn`) computes it online per note and sitting; `deploy_vs_lmdb.py` 0 disagreeing columns on 5100/5076/5033, real and query rows, and a table SHIFTED BY ONE ROW makes both columns disagree on 20-31% of rows -- the check can fail. `smoke_id_features_width.py` gains 105 and 114. Box: code pushed (39/39 agree), table built THERE, **all six fingerprints byte-identical across machines**. **Arm `s64hsib` = s64h + the two columns, 333,178 params MEASURED, control 94 `s64h` (same runner, budget, box); PREREG leans small (+0.00003..+0.00015 imm) because the note stream already sees sibling grades.** Queue: s64hc10 (decay/eval) -> r2s64h -> **s64hsib** -> s64hrf10 -> selffeed2. Placed before s64hrf10 on purpose: if it wins, it can join the confirmation run before that starts. ⚠ An ssh launch timed out (the known never-returns trap) and its traceback printed the box address into tool output only -- nothing committed.

**⟶ 2026-09-23 09:30 -- ★ ANDREW ACCEPTED `cum_new_cards` TOO ("Let's accept cum_new_cards even if it failed the p-value test"); THE FINAL LAYOUT IS 112 INPUTS; THE SIBLING RATING IS SCREENED AND NOT RECOMMENDED.** (1) 96 `newc1` is now `accepted (directed)` (+0.000112 / +0.000104, both over the raw bar; ahead p 3.2e-03 missed 1e-4). **It joins the eight rowfeat columns in `s64hrf10`**, which had not started: regenerated with all nine (**334,074 params MEASURED**, 112 inputs), re-uploaded after asserting the box had no `runs/s64hrf10` yet; the `cum_new_cards` side table is already on the box. So ONE run confirms all nine accepted columns. (2) `INPUT_FEATURES.md`: the final table is now 112 inputs (rows 44-52 marked ⏳ until `s64hrf10` confirms and the Rust + Anki builders compute them), with the exact definitions from `rwkv/extra_features.py`. (3) **Sibling rating (Andrew's kept item D)**, screened on CPU before any build (`scratchpad/siblingrating/`): a same-sitting sibling exists on **3.37%** of scored reviews (median user **0.27%**); where it exists a failed sibling means a 0.33 fail rate vs 0.19, but the held-out-USER gain over a simple baseline is positive on the sibling rows in only **6 of 10** splits, carried by ~23 heavy users -- and the note stream already sees every sibling review with its grade. **Recommended: do not build yet** (a third per-row table + a deploy port + an arm for an expected gain below the bar). Andrew's call.

**⟶ 2026-09-23 03:45 -- THE 3-STAGE DECAY FINISHED (PROBE_OK 02:45) AND ITS BOX EVAL HIT TWO BUGS OF MINE BEFORE IT COULD LAUNCH; BOTH FIXED, IT IS QUEUED ON THE BOX.** (1) **`remote_eval.py`'s eval.sh template carried a bare `%` in a comment I added 09-22 05:00 ("+24% measured")**, so every box-eval launch since died at `ValueError: unsupported format character` before touching the box (DONE_EXIT_26 -- clean, nothing mis-scored). Escaped; `scratchpad/cloud/smoke_remote_eval_template.py` renders the template with typed dummies and was proven to FAIL on the old file. **Run it after any edit to that template: it is formatted only at launch, so nothing else checks it.** (2) **The relaunch then hit the stale-kernel guard, correctly**: the box's live `.so` predates its `.cu` (the swap owed since 09-22 01:10), and a 3-stage eval needs `c_pq_stages`. Both the 3-stage eval-only runner (`run_lfq1rvq3b10real_evalonly.cmd`) and the ARMED 5-stage runner (`run_lfq1rvq5b10.cmd`, edited before its waiter calls it, byte-exact backup kept, a one-line diff, preflight PASS) now pass `--kernel /workspace/kernel_build/rwkv/model/RWKV_CUDA.cpython-312-x86_64-linux-gnu.so` -- the isolated build 99's re-eval used. ⚠ Git Bash rewrote that `/workspace/...` into `C:/Program Files/Git/workspace/...` on the first attempt (the 09-22 trap again; `MSYS_NO_PATHCONV=1`). **OWED, now blocking every REMOTE eval (not the box runners' own phase C, which never calls `remote_eval`): swap the box's live `.so` (`swap_kernel.sh`) or keep passing `--kernel`.** The 3-stage eval is uploaded and waits in the box's GPU gate behind `s64hc10`. **Local: the 5-stage waiter fired at 02:46 (its gate saw PROBE_OK) and its dry twin is running.**

**⟶ 2026-09-22 21:00 -- ★ ANDREW ACCEPTED 102 `rowfeat` ("rowfeat - accept"), A DIRECTED ACCEPT; ITS CONFIRMATION `s64hrf10` IS QUEUED.** Same shape as iter 67 and as his s64h accept: imm +0.000171 resolved (p 3e-32), ahead +0.000087 a sub-bar positive. Recorded as `accepted (directed)` in `research_log.jsonl` / `research_5k.md`. **It was measured on the 561k s80 trunk at 1+1, and the trunk that ships is s64h**, so the columns are re-confirmed there at 10+1: **`s64hrf10` = `s64hc10`'s runner + `RWKV_EXTRA_FEATURES` (the eight columns), 333,946 params MEASURED** (`scratchpad/shrink/measure_arm.py`, whose control reproduces 332,922; +1,024 = 8 inputs x 128), control `s64hc10` on the same box. Generator `scratchpad/shrink/mk_rowfeat10.py`; preflight PASS. **Box queue: s64hc10 (WS ~27k) -> r2s64h -> s64hrf10 -> selffeed2.** If the columns hold, s64hrf10's WS trace becomes the reference the WS tuning ranks against (the trials are generated from s64hc10 today and must be regenerated from the adopted incumbent). ⚠ **OWED before the columns ship: the deploy builder** -- the eight columns exist in training and the Python RNN path (`deploy_vs_lmdb.py` passed on 09-21), but NOT in `rust/rwkv-infer` or the Anki fork. ⚠ Ops: the queue was edited with `scratchpad/cloud/queue_insert.py`, which writes through the SAME inode -- a `sed -i` would replace the file and the running supervisor, which reads through an fd opened at start, would never see the change. Also restarted on Andrew's request: the box GPU monitor window (`box_monitor.py`).

**⟶ 2026-09-22 20:40 -- THE PC RESTART KILLED THE LOCAL 3-STAGE DECAY; IT RESUMED FROM STEP 6000. AND 102 `rowfeat` IS IN.** (1) `lfq1rvq3b10real`'s decay died at step ~7,449 at 20:17 with `DECAY_FAILED_1073807364` = 0x40010004, the code Windows gives a process it kills at shutdown (Andrew restarted to update Claude Code). **Resumed from the step-6000 checkpoint + LEARNED catalogs** by `scratchpad/lfq1rvq3b10real/mk_resume.py` (the lfqkd1 recipe: `make_resume`, `RWKV_RESUME_SKIP_GROUPS=1`, `RWKV_DECAY_RESUME_SCHED=1`) with ONE new guard, die 99: the resumed log must show `stages=3 (98304 floats)`, because a resume that reloaded one stage would be iteration 95's error inside training. Verified live: the step-6000 catalog loaded with all three stages, the LR curve continues at factor 0.2409, the first 6000 groups are skipped, and **steps 6001-6006 match the pre-crash losses to ~0.001** (only dropout draws differ). Ends ~04:30, then its eval goes to the box. `ClaudeRamGuard` was NOT running after the boot (`Ready`) and was started by hand. (2) **102 `rowfeat`** (the eight per-ROW read-time columns, 564,292 params) vs s80: **+0.000087 ahead (p 3.8e-03) / +0.000171 imm (p 3.1e-32)**, n=2485, size 0, 0 tracebacks -- REJECTED on ahead, but the imm gain is strongly resolved and the PREREG had registered a firmly NULL expectation. It is not a dead lever (unlike `selffeed` it differs from s80 per user). **The read-time route now reads imm-positive, ahead-null** (newc1 +0.000112/+0.000104, newc2 null, rowfeat +0.000087/+0.000171) -- the same shape as iter 67, which is the still-open Pareto half-accept question for Andrew. At the 1+1 budget, where effects run ~0.65 of full size. (3) **THE LOCAL GPU WILL NOT IDLE AFTER THE RESUME: `lfq1rvq5b10` IS ARMED BEHIND ITS PROBE, not behind its DONE_EXIT.** After the probe the 3-stage run only waits on the box eval, and the box's eval gate needs zero compute apps while `s64hc10` holds the card until ~tomorrow afternoon -- so gating on DONE_EXIT would idle the 4070 ~11 h. `scratchpad/rvq3gpufree/gate.py` (pid-detached via `run_gate.cmd`) writes `DONE_EXIT_0` to its OWN fresh log when `lfq1rvq3b10real PROBE_OK` appears AFTER the last `RESUME at step` banner, or `DONE_EXIT_190` if the resume fails first. ⚠ **The run log is POISONED for a plain marker gate: it is appended and already holds `DONE_EXIT_23` from the killed decay**, which any `findstr DONE_EXIT_` waiter would have read as an immediate failure. `smoke_gate.py` checks four synthetic cases plus the real log (None while decaying). `mk_wait_rvq5.py` takes the upstream as argv 4 and the stub driver reads it out of the waiter; two generator bugs fell out: the stage-count prose assert (`"5-stage" not in CMD`) had been written while generating the 3-stage waiter and REFUSED the 5-stage arm itself, and the stub hardcoded `s80ddh`. `wait_lfq1rvq5b10.cmd` passes all six stub paths and was alive past one 300 s poll. The 5-stage arm is justified whatever the 3-stage number says unless 3real clears 0.0010 in both modes (its PREREG predicts it will not): per `BITS.md` it is the arm that DISCRIMINATES the two tax models.

**⟶ 2026-09-22 16:30 -- ★★ `selffeed` IS WITHDRAWN: THE ARM RAN 9.5 h AND THE LEVER NEVER EXECUTED, AND EVERY GUARD PASSED.** The box arm ended `DONE_EXIT_0` with the `[selffeed] ON` banner, `Trainable parameters: 562932` (the measured guard), 0 skipped batches -- and its model is **BIT-IDENTICAL to its control `s80` on all 2,485 shared eval users in BOTH modes** (0 users differ, max |d| 0.0). **THE TELL IS ONE NUMBER: `fb2h.weight` AND `fb2h.bias` are EXACTLY 0.0 after 21,870 steps.** A zero WEIGHT has an innocent reading (`dL/dW` vanishes if the features are all zero); **a zero BIAS does not** -- its gradient is the plain sum of `dL/dout` over the second head pass, so a bias still at its init says that pass never ran. And because `fb2h` never left zero it never perturbed anything, which is why the whole model trained bit-identically to the control. **THE CAUSE: `PreparedBatch.to(device)` rebuilt the dataclass BY NAMING EACH FIELD BY HAND, and the two fields added 2026-09-21 -- `selffeed_prev` / `selffeed_srcq` -- were not in the list.** They carry `None` defaults, so the call compiled, the batch reached the GPU empty, and `forward` took its `selffeed is None` branch, which is the legitimate flag-OFF path. Everything upstream was correct: `prepare()` built the indices in the fetch workers, the env reached every phase, and `box_sync.py diff` reported 39 files 0 differ. **The batch crossed one function on its way to the GPU and arrived empty.** ⚠ **WHY THE SMOKE MISSED IT, and this is the transferable half: `smoke_selffeed.py` part 3 explicitly checks that a gradient reaches `fb2h`, and it PASSES** -- it builds its batch in-process and hands it straight to the model. It covered the component; the lever died in the ~5 lines between `prepare()` and the model. **A component smoke and a pipeline are different tests.** **FIX:** `to()` now derives its field list from `dataclasses.fields`, so it cannot drop a field that exists; numerically identical for every other run (same `.to(device)` on the same tensors), and **checked rather than assumed** -- `scratchpad/parity3/smoke_batch_to.py` fills all 17 declared fields with distinct values and PROVES IT CAN FAIL twice (a rebuild dropping the last field loses exactly one; the pre-fix file loses exactly `selffeed_prev` and `selffeed_srcq`). **THE RULE, now paid for three times** (`make_resume.py` replacing only pre-existing keys, the QAT env parsed into a discarded object, this): **a function that REBUILDS a structure must derive its field list from the structure, or assert that it covered it -- adding a field to a dataclass must never require remembering a second place.** **THE RE-RUN `selffeed2` is generated, preflighted and carries an ARTIFACT check on the WS-final** (`assert_fb2h_live.py`, die 61, validated on all three cases) -- and that check is emitted by **`mk_cloud_runner.py`**, not transcribed from the `.cmd`, because that generator builds the `.sh` from a TEMPLATE and a guard added only to the Windows runner would never reach the box, which is where these arms run. It is queued LAST, behind `s64hc10` and `r2s64h`, which gate the tuning phase; it needs `box_sync.py push` of the fixed `srs_model.py` first. Cost: ~9.5 h, ~$4.70; the PREREG is UNANSWERED and stands. Full account: `scratchpad/selffeed/WITHDRAWN.md`.

**⟶ 2026-09-22 17:30 -- ★ THE WS TRIAL MACHINERY IS BUILT, AND ONE OF THE FIVE KNOBS TURNS OUT NOT TO BE A TRIAL AT ALL.** `scratchpad/wstune/mk_ws_trial.py <incumbent runner> <knob> <value>`; round 1 is generated and **PREFLIGHT_ALL_PASS**: `wtlr0p5`, `wtlr2`, `wtmuon0p5`, `wtmuon2`, `wtwarmup2000`. (1) **A trial is WS-ONLY** (`RWKV_WS_ONLY=1`, new and default off): no decay, no eval, because it is ranked by its 10-user validation endpoint read beside the incumbent's at the same steps -- zero GPU -- and a decay + full eval is bought only for a trial about to be ADOPTED. That is ~a third off every survivor. (2) **The kill rule rides in the TRAINER** (`RWKV_VPRUNE_WINDOW_MEAN=5`, `MIN_STEP == MAX_STEP == 32000`, 0.0010 both modes, persist 1) rather than in a watcher, and **a pruned WS is now a RESULT**: exit 42 stops the runner with no retry, no decay and no eval, writing `WS_PRUNED` + `DONE_EXIT_0` so the supervisor walks on -- a tuning phase in which every pruned trial halts the queue for a human is not a tuning phase. (3) **★ `epochs 10 -> 15` IS NOT A TRIAL, AND THAT IS A SAVING: the WS schedule is warmup then CONSTANT LR, so `EPOCHS` sets only how many steps are taken and a 15-epoch WS IS the 10-epoch WS CONTINUED.** It costs the 5 extra epochs as a resume from the incumbent's WS-final (~15.5 h local / ~7.5 h box), not a 46 h retrain, and the kill rule cannot apply to it because through step 32,805 it IS the reference. The generator REFUSES the knob rather than emitting a from-scratch trial that would spend 31 h reproducing a checkpoint we already have. It also refuses `max` without `--groups N`: groups per epoch is a property of the train db at that MAX (10,935 at 65536, unknown at 98304) and a wrong `WSSTEPS` silently breaks the resume loop's completion test. (4) ⚠ **`mk_cloud_runner.py` NOW CARRIES ACROSS ANY WS-LOG GUARD THE `.cmd` DECLARES.** It builds the `.sh` from a TEMPLATE, so a guard added only to the Windows runner did not exist on the box -- the same gap the `fb2h` check hit hours earlier, and worse here, because **a trial's entire safety net is a findstr on the trainer's own echo of the value it was given**, and the recurring tuner failure is a lever that silently does nothing and measures as a perfect null. Verified inert: the three queued runners regenerate byte-identically.

**⟶ 2026-09-22 16:00 -- ★ THE EPOCH-3 KILL RULE IS CALIBRATED (the item owed before the first WS trial), AND THE OBVIOUS SHAPE OF IT IS REFUTED.** `scratchpad/wstune/`; the tool is `kill_check.py <candidate ws log> <reference ws log>` (0 LIVE / 1 KILL / 2 PENDING). **THE RULE: KILL iff, at step 32,805, the mean of the last 5 validation checkpoints is worse than the incumbent by >= 0.0010 in BOTH modes -- one decision, at that step, nowhere earlier.** (1) **The inherited 0.004 / 0.006 deltas are vacuous here by a factor of a hundred**: they were set where the curves fall ~0.01 per 1000 steps, and these four traces fall **0.0001 per 1000 steps at epoch 3** (0.00003 by epoch 5). (2) **A single checkpoint cannot carry the decision** -- the NULL pair (`cw10lf` vs `w10lf`, same recipe, different hardware) reaches |d| = 0.0017 ahead / 0.0011 imm on individual checkpoints, which OVERLAPS the near-tie arm's range. A 5-checkpoint mean collapses it: over **107 sliding windows the null's both-modes-worse floor never exceeds +0.00012**, giving **0 false kills at every threshold from 0.0005 up**. The both-modes conjunction does most of that work -- the null "candidate" is slightly BETTER on imm throughout. (3) **★ "FIRE AT ANY CHECKPOINT UP TO EPOCH 3" IS REFUTED, AND IT IS THE SHAPE I WOULD HAVE WRITTEN:** `lfwu20k` fires at step 1000 with a **+0.1492 ahead** gap at EVERY threshold tested, because its one knob IS the warmup (400 -> 20,000) so its LR is still ramping; by epoch 3 the gap is +0.0008 / +0.0011 and by epoch 10 the ahead gap has **flipped sign to -0.0006**. **Knob 4 of the grid is warmup**, so an any-checkpoint rule would kill, by construction, the knob it is most needed for -- the 2026-07-08 late-bloomer failure at 10-epoch scale. **0.0010 rather than the 0.0005 the null alone allows**, exactly because `lfwu20k` crosses over after the decision point; that costs pruning rate and is the trade taken deliberately. `smoke_kill.py` proves the rule can FIRE (KILL at +0.0015 both modes), that the conjunction is load-bearing (LIVE at +0.0030 in ONE mode), that the threshold is where it says (LIVE at +0.0009), and that a truncated log is PENDING. ⚠ LIMITS: n = 1 null pair and it is cross-HARDWARE; and the calibration traces are 561k while the trials are 332k -- each trial is paired against its OWN 332k reference so a scale shift cancels, but the VARIANCE of a 10-user validation at 332k is unmeasured. Re-read the first trial's window spread against these numbers.

**⟶ 2026-09-22 12:00 -- ★★ ANDREW ADDS A WS TUNING PHASE, CORRECTS THE BUDGET, AND PUTS R2 FIRST.** Verbatim: *"WS three with full trials and early pruning of bad runs at 3 epochs, got it"*; *"I'm currently at 57$ at vast.ai. I could throw in another 20, but that's it"*; *"I'll leave the details of which hyperparameters to tune and in which order to you"*; *"don't forget to also run monotone-by-construction curve before all this"*. **PLAN: `optimization/WS_TUNE_PLAN.md`.** (1) **BUDGET IS A REMAINING BALANCE, not a spend: $57 now, $77 ceiling** = 116-156 h of box at $0.493/h. My earlier "~$41 left" was an estimate and was low. Committed: `selffeed` tail, `rowfeat`, `s64hc10`, `r2s64h` ~ 34 h, leaving ~82 h. (2) **DESIGN: full 10-epoch trials, losers killed at epoch 3** -- NOT shortened trials, because under WSD a 3-epoch run has a different LR trajectory and LR is the main knob, so a short trial is unreliable exactly where it would be used most. 31.1 h a trial at 332k, 9.3 h if killed, **~20 h averaged**; ranking is the 10-user validation endpoint at ZERO GPU cost (it predicted arm 1's gate deltas within 0.0004). **~12 local + ~8 box = ~20 trials against a grid that needs 11.** **SHRINKING IS WHAT MADE THIS AFFORDABLE: a WS trial was ~38 h at 561k, which is exactly why phase 5 was scoped to the decay.** (3) **KNOB ORDER (mine): LR scale with AdamW AND Muon together -> the Muon-vs-AdamW RATIO (the largest untried knob; every previous tune moved them together or moved AdamW alone) -> `MAX` 65536->98304 (+14.3% rev/s but -34% optimizer steps, and the right LR depends on how many steps there are) -> warmup 400->2000 -> epochs 10->15 last. wd/dropout/SAM are CLOSED: the model is fit-limited and a 1.69x smaller one is more so.** (4) **⚠ THE EPOCH-3 KILL THRESHOLDS MUST BE CALIBRATED FIRST and the data is already on disk** -- four 10-epoch WS traces (`ws10`, `w10lf`, `cw10lf`, `lfwu20k`), where cw10lf-vs-w10lf is the same recipe on different hardware (the null spread) and lfwu20k-vs-cw10lf is one knob different. The existing val-prune deltas (0.004 / 0.006) were set for step 1000-2000 of a ONE-epoch run where curves fall ~0.01/1000 steps; at epoch 3 of ten they are far flatter and would kill everything. (5) **`r2s64h` IS BUILT AND QUEUED** (334,080 params MEASURED = s64h + 1,158; R2 head, PAVA off, probes off, plain eval; control `s64h`). It goes before the tuning because **if adopted it removes the four probe rows per scored review from training AND eval, so every later trial is cheaper** -- deciding it afterwards would mean tuning the wrong recipe. ⚠ 97 `r2s80` already rejected it at d=80 (-0.000378 ahead / +0.000249 imm); this asks on the architecture that ships. Box queue: `selffeed` -> `rowfeat` -> `s64hc10` -> `r2s64h`.

**⟶ 2026-09-22 10:40 -- `lfq1rvq3b10real` IS LIVE, AND THE BITS QUESTION IS NOW "WHICH TAX CURVE IS TRUE", NOT "WHICH RUNG NEXT".** The dry twin ended DONE_EXIT_0 (PROBE_OK, EVAL_OK) and the arm started 10:37. Its banner is the check that matters: **`[QAT-PQ] uploaded ... reference/pq_cb_wkv_real_rvq3b10.txt: ... stages=3 (98304 floats)`** -- the correct-provenance catalog, 3 stages, on the grown-capacity kernel. ~13 h, then its eval goes to the box. **THE PROJECTION (`BITS.md`): the two measured tax points admit two readings that disagree about whether Andrew's < 0.0010 is reachable at all.** Linear in BITS gives -0.000274 ahead / -0.000411 imm per +10 bits, so the target arrives at **~70 bits/head = card 785 b (+324%)**. Linear in RECONSTRUCTION ERROR gives **a floor of +0.001124 ahead / +0.001923 imm as error -> 0**, i.e. imm NEVER clears 0.0010 by buying WKV index bits. **The two differ by 0.00048 imm at 50 bits, six times the noise floor, so `lfq1rvq5b10` DISCRIMINATES them -- that is now the strongest reason to run it, ahead of its value as the next rung.** ⚠ n=2, and the two points also differ in catalog provenance, which is exactly what the running arm measures: its number is a third point at 30 bits and sharpens both fits. **IF THE FLOOR READING WINS, the floor is not in the WKV catalog**, and the components never separately priced on this trunk are the shift catalog (+0.000365 / +0.000720 measured 2026-08-12 under PTQ, ~27% of today's imm tax), the int4 low-rank factors, and the 1-bit norms. Phase D closed decomposition by REMOVAL (deleting a quantizer throws the model off distribution, +0.035 / +0.043); **decomposition by REFINEMENT -- one component given more bits at a time -- is a different experiment and is the successor.** **AND THE NON-BITS LEVER IS PHASE 5**, which has never run: every QAT number uses PLAIN-tuned HPs, and `RWKV_CB_LR_MULT` is the sharpest knob in it (learnable catalogs cut the tax 45%, and their LR is a d=32-era 1x never tuned under QAT). Its six tools are built and waiting on an adopted arm.

**⟶ 2026-09-22 10:00 -- ★★ ANDREW ACCEPTED `s64h` ("Accept s64h"): THE SHRINK TRACK IS CLOSED AND ITS CONFIRMATION IS QUEUED.** The track ends at **332,922 parameters, 1.69x smaller, for +0.000304 ahead / +0.000493 imm at the 1+1 budget** -- a **directed accept**, the same shape as iter 36, because it misses his own 0.0002-per-100k rule on imm by 0.000035, under half the noise floor and a tenth of the cross-seed spread. Recorded in `research_5k.md` (94's row now reads **ACCEPTED (directed)**) and in `research_log.jsonl`. **ADOPTION REMAINS CONDITIONAL and the condition is queued as `s64hc10`** -- `w10lf`'s own runner with exactly four changes (architecture, `RWKV_STRIP_CMIX`, the measured 332,922 guard, a 1-epoch decay), i.e. **10 WS epochs + 1 decay epoch against `lfd1` at 0.297231 / 0.264004**; on the box behind `selffeed` and `rowfeat`, ~17 h and ~$8.5. Generator `scratchpad/shrink/mk_confirm.py`, which asserts the WS budget survived AND that the parameter count moved in the PROSE as well as in the guard -- that assert caught a comment still describing the 561k model. **THE CONFIRMATION IS NOT A FORMALITY: every arm ran at ONE WS epoch, and the 2026-08-11 calibration says capacity binds HARDER once there is enough training to fill the larger model, so the honest expectation is that 1.69x costs MORE at 10 epochs than it did here.** ⚠ **AND WHAT ADOPTION WOULD COST DOWNSTREAM, said before it surprises anyone: `s64h` is d=64, so the per-card WKV state, the catalog shape (`sub_dim = 2K`) and the card-bit budget all change, and every catalog needs refitting. Switching the shipped architecture is a re-base of phases 5 and 6, not a drop-in -- Andrew's call, and nothing here assumes it.** The running bits arms stay on the 561k trunk, where their question is about the quantizer rather than the architecture. `s80dh` is marked cancelled in `arms.json`.

**⟶ 2026-09-22 09:50 -- ★ THE SHRINK TRACK IS AT A DECISION THAT IS ANDREW'S, AND THE QUEUE BELOW IT IS EMPTY.** Eight arms in, the frontier is settled and what remains is a judgement about the bar, not a measurement. **`s64h` (332,922 = 1.69x) is the answer if there is one: +0.000304 ahead / +0.000493 imm, i.e. 0.000133 / 0.000216 per 100k against his 0.0002 rule -- ahead passes with 34% margin, imm is over by 8%.** That 8% is **0.000035 per 100k, under half the 7.5e-5 noise floor**, and the cross-seed spread on one recipe is ~0.0004 = **ten times the miss**, so a seed pair would add noise rather than settle it. **THE QUESTION, and the rate rule and the 1.5-1.7x target are both his: accept `s64h` as the shrink result, or keep cutting?** If "keep cutting", the DIRECTION has changed and `FRONTIER.md` records why -- depth at fixed width is what costs (101), the rating head is what it costs (100 and every arm above), and narrowing to d=64 is NOT the overspend -- so a next arm targets rating-head capacity at d=64 rather than removing layers. If "accept", the track closes and the 1.69x model needs its confirmation at the full 10+1 budget first, as its own PREREG said. `s80dh` is marked **cancelled** in `arms.json` with the reason, so no future session generates it. Nothing is queued on this track; the box's remaining work is `selffeed` (decaying) then `rowfeat`.

**⟶ 2026-09-22 09:45 -- ★★ 101 `s80ddh` FALSIFIES MY OWN 00:20 READING: THE WIDTH STEP IS NOT THE OVERSPEND, DEPTH IS.** Pre-registered before it ran: allowance +0.000501, **projected +0.000476 ahead / +0.000466 imm, "a pass with 5-7% margin"**. **Measured +0.000709 / +0.001074** = rates 0.000283 / 0.000429 per 100k -- 49% low on ahead and **130% low on imm**, and the arm fails both modes (311,327 params, n=2485, size 0, 0 tracebacks). **The direct pairing is what inverts the conclusion: `s80ddh` is 21,595 parameters SMALLER than `s64h` and -0.000416 ahead / -0.000582 imm WORSE than it (n=2499).** So at the same size class, NARROWING beats staying wide and cutting depth -- the exact opposite of the claim that motivated the arm. **WHERE THE REASONING FAILED:** I composed three marginal rates, each measured over ONE small step, as if cost were linear in parameters removed -- and `FRONTIER.md` had ALREADY found the cost convex on the d=64 branch (the knee after `s64dh`, where the last 51k parameters cost 3.5-4x the bar). I pre-registered convexity as "the risk" and then projected as though it were absent. **A marginal rate measured on a 100k step does not price a 250k one, and this track has now shown that twice on two different branches.** **CORRECTED READING: `s64h` (1.69x, 332,922) remains the track's best arm and the width step d=80 -> d=64 is fine; DEPTH AT FIXED WIDTH is what costs, and the rating head is what it costs -- the same asymmetry 100 `s64dhc` hit.** **=> `s80dh` (414,167), generated as the safer d=80 fallback, is CANCELLED: a shallower cut on the branch that just proved expensive, at a size nobody asked for.** Also in: **100 `s64dhc`** -- `s64dh` with one layer moved DECK -> CARD at identical 273,747 params, so a pure rearrangement: paired directly it is **+0.000092 ahead (p 6.3e-03) / -0.000055 imm**, rejected on the imm rate (0.000279 vs the 0.0002 bar) but a second refutation of "same-capacity rearrangement is indistinguishable" (iters 41/43/44), after iter 56's. **BOX: `selffeed` is LIVE and in its decay** -- `[selffeed] ON: per-card feedback, 14 features -> d_model 80, detach=True`, `Trainable parameters: 562932` (the guard value, so the lever cannot have been silently inert), 0 exceptions in the WS. `rowfeat` is behind it.

**⟶ 2026-09-22 07:20 -- ★ THE CORRECT-PROVENANCE LADDER CONFIRMS THE RATE-DISTORTION LAW THIS FILE PREDICTED, AND CORRECTS A NUMBER THE RECORD HAS BEEN CARRYING.** Four catalogs fitted on the real-checkpoint corpus (40 users, 169,311 states), scored on user 156:

| stages | bits/head | rel. error | ratio per +10 bits |
|---|---|---|---|
| 1 | 10 | 0.1502 | -- |
| 2 | 20 | 0.0984 | 0.655 |
| 3 | 30 | 0.0702 | 0.713 |
| 5 | 50 | 0.0369 | 0.725 |

**`BITS.md` predicted 0.665** from the (u,v) pair living in ~17 effective dimensions (participation ratio 8.5 and 9.3 of 16), so error should fall as 2^(-bits/17). A prediction made from the state geometry, confirmed by four independent fits -- and it means the ladder is NOT near a fitting ceiling at 50 bits. **⚠ IT ALSO CORRECTS `pq_cb_wkv_w10lf_b10.txt`'s 0.1042**, which the record has quoted since 09-17: that is BETTER at 10 bits than the wide 1-stage fit's 0.1502, which is impossible for a better catalog and obvious for an OVERFITTED one -- it was fitted on the NARROW 3-user corpus, so the scored user is a third of its training data. **The honest single-stage number on this trunk is 0.1502**, and iteration 84 `lfq1` started from ~0.15, not ~0.10. **THE SHARPEST STATEMENT OF THE PROVENANCE ERROR: iteration 99 spent 20 EXTRA index bits per head and still started from a catalog 3.3x WORSE than a correct 10-bit one (0.5001 vs 0.1502) -- and won anyway by +0.000551 / +0.000814. That difference is the measure of what the learned catalogs repair**, which is why `lfq1rvq3b10real`'s PREREG predicts only ~0.0002 of tax relief rather than a win.

**⟶ 2026-09-22 07:10 -- ★ THE ARM ORDER IS CHANGED BEFORE ANYTHING RAN: THE CHEAP RUNG GOES FIRST, BECAUSE THE ESCALATION RULE'S PREMISE WAS NEVER TESTED.** Andrew's rule picked +200% from "+100% is not enough", and +100% (99) was measured with a catalog fitted on the **stand-in** model -- 0.5001 reconstruction error on the real w10lf states against **0.0395** for a correct-provenance fit at 50 bits (honest: fitted on 39 users, scored on the 40th; the in-sample number is 0.0369, so the optimism is 6.6%). **A correct-provenance 3-stage arm has never run.** `lfq1rvq3b10real` is now first and `lfq1rvq5b10` is queued behind it:

| | `lfq1rvq3b10real` | `lfq1rvq5b10` |
|---|---|---|
| decay | ~13 h | ~23 h |
| vs 99 | differs in **exactly one** variable, the catalog's provenance | bits AND provenance, confounded |
| if it clears 0.0010 | deploy stays **385 card bits**; the 5-stage arm is unnecessary | 585 bits, and P5 demands the 3-stage run anyway |

Going straight to 5 stages needed the 3-stage run afterwards regardless, so it was the same 36 h in the worse order. **Nothing in Andrew's rule is skipped -- this establishes the input the rule needs**, and he can overrule it from the record. PREREG `scratchpad/lfq1rvq3b10real/PREREG.md`, and it does NOT predict a win: **P1 expects the tax to fall only ~0.0002 per mode and NOT clear 0.0010**, because `RWKV_QAT_PQ_LEARN=1` repairs most of a stale start -- the record already credits that learning with ~45% of the tax. **P2 is the falsifier and is the useful null: if the tax does not move, a stale start catalog costs nothing once the catalogs learn, the 8.0x reconstruction gap has no consequence, and this repo's own "refit on the model that deploys" rule is not worth its 40 minutes when LEARN is on.** P3's surprise case (tax falls > 0.0006) would make the NEXT arm ONE stage, i.e. today's 185 card bits kept. **OPS:** `mk_lfqat.py` gained `--cbreal`, which puts the catalog's provenance in the TAG and not only in the path -- two arms differing only in which model their catalog came from are different experiments, and a shared identity is what the 32,768-vs-98,304 withdrawal cost. The waiter generator and its stub driver are now parameterised by (tag, stages, catalog) and read the catalog path OUT OF the waiter, so a stub cannot test a different arm than the file it was handed; both waiters pass all six terminal paths. The 5-stage waiter was stopped in gate 0 with a note and **no marker** (verified: 0 `DONE_EXIT_` lines in its log), so nothing downstream can fire.

**⟶ 2026-09-22 07:00 -- FITTING ON THE RIGHT MODEL CUTS THE 5-STAGE ERROR 8.6x, AND I COMMITTED 3.1 GB OF CORPUS DOING IT.** (1) **The corpus is re-dumped from `scratchpad/w10lf/w10lf_ws_109350.pth`** (40 users, 169,311 states, CORPUS ORDER PASS on all 160 files, 92 min) and the 5-stage catalog refitted on it. Held out at user 156 on the REAL states: **the new catalog 0.0369, its stand-in-fitted twin at IDENTICAL bits 0.3179, iteration 99's start catalog 0.5001.** The residual RMS falls 0.5611 / 0.4426 / 0.3645 / 0.3028 across the stages, which is what residual coding should look like. ⚠ The 0.0369 is IN-SAMPLE (user 156 is in the 40-user fit), so a 39-user refit scored on the 40th is running for an honest both-sides-out-of-sample number -- do not quote the 8.6x until it lands. **THE LADDER IS RE-FRAMED, and this is the part that matters more than the number: the +200% arm starts from a catalog ~8.6x better on its own model than 99's did, so its step over 99 mixes bits with provenance.** Accepted rather than engineered away, because Andrew's target is on the TAX and the arm answers that directly -- but the follow-up INVERTS: **if a correct-catalog arm clears 0.0010 at 5 stages, the next experiment is 3 stages, or even 1, with a correct catalog, not 7.** The provenance fix may buy what we were about to spend 400 extra card bits on, and card state is the budget Andrew grants in ones and twos. PREREG: `scratchpad/lfq1rvq5b10/PREREG.md` (P1 predicts ahead PASSES and imm FAILS; P5 makes the downward question mandatory). (2) ⚠⚠ **I COMMITTED `scratchpad/lfqat/corpus_real` -- 160 files, 3.1 GB, 458k lines -- WITHOUT CHECKING ITS SIZE, which is the 385 MB side-table mistake of 09-21 repeated four days later.** Undone before anything was pushed. Checking then showed the condition is OLDER than my commit: `corpus/` (12 files) and `corpus_wide/` (160) were already tracked by earlier sessions and `.git` had reached **3.9 GB**; `.gitignore` ignored `scratchpad/qat_tax/corpus*` for exactly this reason and never covered the `lfqat` ones. All three are now ignored, the two tracked ones untracked with `--cached` (files stay on disk, nothing queued loses its input), and a `gc --prune=now` brought `.git` to **2.1 GB**. The corpora are redumpable in ~40 min from a checkpoint plus the trace exporter; what belongs in git is the fitted catalog and the runner. Purging the older blobs still needs a rewrite of the unpushed commits -- Andrew's call, already flagged for the side table. **THE RULE I KEEP RE-LEARNING: `du -sh` the directory BEFORE `git add`, and when a sibling directory is already in `.gitignore` for a size reason, check whether its cousins are too.**

**⟶ 2026-09-22 05:15 -- ★★ EVERY RESIDUAL-VQ CATALOG WAS FITTED ON THE STAND-IN MODEL, NOT ON w10lf -- AND THE RUNNER SAID SO IN A PARENTHESIS NOBODY CHECKED.** `run_wide_corpus.cmd` sets `RWKV_CHAMP_CKPT=scratchpad/final_layout/standin/w10lf_ws_109350.pth` and its phase 2 comment reads *"merge the traces beside the existing three (the stand-in weights are identical)"*. They are not, and **`make_standin.py`'s own docstring is the refutation**: the stand-in is **ws10's WS-final -- the LEAKED 10-epoch WS** -- with the six FINAL_LAYOUT columns sliced out of the input layer, and it ends *"It is a TEST INPUT for graft.py / smoke_graft.py, never a model to evaluate."* **MEASURED on the REAL w10lf states (`scratchpad/lfqat/corpus`, held out at user 156): `b10` 0.1042 (fitted on real w10lf, IN-SAMPLE), `rvq3b10` 0.5001, `rvq5b10` 0.3179** -- so iteration 99's start catalog reconstructs its own model's states several times worse than the single-stage catalog it was compared against, with three times the bits. ⚠ The 0.1042 is in-sample, so the gap is not a clean 5x; that is precisely why the corpus is being re-dumped from the right model and measured honestly. **WHICH WAY IT CUTS, and it is the reassuring way: 99's 3-stage arm started from a catalog that was FAR staler on its own model than the 1-stage arm it beat by +0.000551 / +0.000814, and the learned catalogs (`RWKV_QAT_PQ_LEARN=1`) closed that gap and then some. The bits effect is a FLOOR, not an artefact.** The two corpora were compared numerically before any of this was claimed: same user, same 1,934 states, same shape, **max abs difference 1.01 and RMS 0.0342 vs 0.0290** -- different models, not float formatting. **THE NAME IS WHAT HID IT:** `make_standin.py` writes its output under the REAL checkpoint's FILENAME and `export_rnn_trace.py` labels `ref_metrics.json` with `os.path.basename(MODEL_PATH)`, so `reference_final_inputs/ref_metrics.json` reads `"model": "w10lf_ws_109350.pth"` while every file in that directory is dated 09-12 and the real checkpoint was written 09-14. **A label taken from a path describes the path, not the file.** RUNNING: `run_refit_rvq5_real.cmd` -- the corpus re-dumped from `scratchpad/w10lf/w10lf_ws_109350.pth` (40 users, 4 processes, ~40 min; the three already-traced users are re-exported because their traces are the stand-in's too), then the 5-stage refit into the arm's catalog path, then a held-out score of old-vs-new on the REAL states. One runner, because the arm's waiter is already armed and checks only the header -- a refit landing after the arm starts would be a refit of nothing. `score_cb.py` also reads the stage count now; until today the tool the record points at for "is this catalog any good" could not read the catalogs in use.

**⟶ 2026-09-22 05:00 -- ★ THE +200% RUNG IS BEING BUILT, AND THE TWO BOX GATES DISAGREED WITH EACH OTHER.** (1) **Andrew's escalation rule fires for the first time on a TESTED premise**: 99's tax is +0.001605 / +0.002645, both over the 0.0010 target, so the next rung is **+200% = 5 residual stages** (50 index bits per head; card 185 -> 585 b, note 105 -> 305 b). The 5-stage b10 catalog is fitting on the 160-file wide corpus now (CPU, BelowNormal). **`mk_lfqat.py`'s stage cap was 4 and would have REFUSED the very rung the rule selected** -- raised to 8, which is the real constraint and a mechanical one: the learnable path writes one `rec_idx` slot per stage and the slot stride is 8 ints, which `rwkv_ops.py` asserts and the kernel TORCH_CHECKs. The old 4 had no mechanism behind it. Decay cost scales with stages x ncent searched per head per step, so expect ~0.13 steps/s and ~23 h locally, after `s80ddh`. (2) **⚠ `s64dhc` SAT IDLE FOR 3 h 09 m WAITING FOR THE RVQ EVAL, AND THAT WAS CORRECT** -- the runners' `gpu_gate` serialises evals against trainers since 2026-09-20, when `s64dh`'s decay died of CUDA OOM (19 MiB free) beside `s80`'s eval on a giant user; a full-VAL eval peaks near 23 GB of the 24 and Linux has no WDDM spill. **What was wrong is that `remote_eval.py` never got that fix, so the two gates disagreed: a RUNNER refuses to train beside an eval, while `remote_eval` would happily start an eval beside a running trainer** -- the exact arrangement that caused the OOM, and it only failed to bite this time because no trainer had started yet. Tightened to match: an eval now waits for zero compute apps. Trainers still share with trainers (+24% measured); only evals are exclusive. **The lesson is the pair, not the rule: when two scripts implement the same policy, fixing one of them leaves the policy broken in the direction nobody tested.**

**⟶ 2026-09-22 04:49 -- ★★ 99 `lfq1rvq3b10v2` IS IN: THE QAT TAX AT +100% OF STATE BITS IS +0.001605 ahead / +0.002645 imm (NOT ENOUGH; Andrew's target is < 0.0010 in both modes).** 3 residual stages of b10 = 30 index bits per head, card 385 b / note 205 b; deployed 0.298953 / 0.266786, vs 84 `lfq1` +0.000551 / +0.000814 (p 3.2e-116 / 8.7e-287), n=2487, size 0, 0 tracebacks. At b10 the tax was +0.002153 / +0.003468. NEXT: +200% (5 stages).

**⟶ 2026-09-22 02:00 -- THE RVQ RE-EVAL IS RUNNING AND ITS BANNER IS THE PROOF OF THE WITHDRAWAL.** `lfq1rvq3b10v2`, same checkpoint, same catalog FILE, on the repaired box: `[QAT-PQ] uploaded ... joint=1 **stages=3 (98304 floats)**` against the withdrawn run's 32,768. Three times the floats from a byte-identical catalog is the whole finding in one line -- the header's 6th field was always there and the box's `rwkv_ops.py` could not read it. The shard log also carries `[RWKV_CUDA] ALTERNATE build loaded: /workspace/kernel_build/...`, so the isolated kernel is in use and the live `.so` is untouched. 237 of 2,499 users at 18 min; ~3 h. It runs BESIDE `s64dhc` (started 22:36 box time), which is what the one-trainer-plus-one-eval gate is for. **And the selffeed lever is now smoked ON THE BOX, where it will train** -- SMOKE PASS, all seven parts, index statistics identical to this machine (13,998/15,191 = 92.1% with a predecessor; the shifted-source control at 1.0000 against the real source's 0.6756) and the shared-weight fingerprint agreeing to 1.2e-8, which is CPU summation order. PART 7 is new and proves the P4 instrument rather than reading it: with `fb2h` perturbed, `RWKV_SELFFEED_BASE_ONLY=1` returns the zero-init output EXACTLY (3555.937933549039) while the full pass moves (3556.343761134456).

**⟶ 2026-09-22 01:50 -- ★ 98 `newc2` IS IN, AND IT TAKES BACK WHAT `newc1` SEEMED TO GIVE.** `cum_new_cards` + `review_th` + `new_card_share` (562,692 params, input 106) vs s80: **+0.000018 ahead (p 0.62) / -0.000008 imm (p 0.88)**, n=2485, size 0, 0 tracebacks -- a clean null in both modes, both deltas well inside the 7.5e-5 floor. **`newc1` with `cum_new_cards` ALONE gave +0.000112 / +0.000104, so adding two more read-time columns did not merely fail to add, it ERASED the single-column gain.** Two readings and no arm here can separate them: (a) newc1's effect was never solid -- its ahead p was 3.2e-03 against a 1e-4 gate, i.e. already unresolved, and a second draw of a noise-sized effect looks exactly like this; (b) the extra columns actively cost, `review_th` being a raw counter with no scale and `new_card_share` a ratio of two accumulations. **=> the read-time feature route is 0 for 2, and `rowfeat` bundles EIGHT columns -- the very configuration newc2 shows is not better than one.** It stays queued, because its columns are different in KIND (session structure, phase coherence, streak, grade mean) rather than more counters, and because one arm closes the question either way -- but **its PREREG now registers a firmly NULL expectation, and a null closes the family rather than motivating a ninth column.** ⚠ This also re-reads the 00:05 entry: "the read-time feature route earns its next arm" was written on newc1 alone and the sentence should be read as "earns ONE more arm", not as a positive finding.

**⟶ 2026-09-22 01:30 -- ★ THE FRONT TABLE'S HEADLINE COLUMN IS NOT ON ONE BASIS, AND THE BIAS IS THE SIZE OF THE ACCEPT BAR.** Every shrink arm's recorded `ahead`/`imm` is the mean over ITS OWN result file, and the box OOM-skips a DIFFERENT set of giant users per arm -- so the arms are averaged over different user sets, and the skipped users are the giants, which score worse than average. **Measured across nine arms the bias is +0.000116 .. +0.000265, i.e. at or above the 0.0001 bar** (`s64h` and `s48` kept all 2,499; `s80` has 2,485). **The PAIRED deltas and therefore every RATE are unaffected** -- those are computed on the intersection and always were -- so no verdict moves. It bites when a headline is compared with ANOTHER LINEAGE's number or with the stop criterion. **⚠ And the table already MIXES the two conventions: `newc1` was recorded by hand on the paired basis (0.298021, not the 0.298137 its own file gives), every shrink arm on its own file.** The corrected all-arms-on-one-basis table is at the end of `scratchpad/shrink/FRONTIER.md`; rank with that. **HOW IT SURFACED: `s80ddh` is the first shrink arm evaluated LOCALLY**, where the 4070 survives the giants through the WDDM spill, so its file holds all 2,499 and its headline would have carried the full penalty against arms that had 14 skipped. The bias was always there; a local eval is where it stops being nearly constant. All three record scripts now write the paired-basis mean and keep the own-file one beside it; the rows already written are left as measured.

**⟶ 2026-09-22 01:10 -- TWO ARMS ARE BUILT AND QUEUED, AND BOTH PARAMETER COUNTS WERE MEASURED RATHER THAN DERIVED.** (1) **`selffeed`** (`scratchpad/selffeed/mk_selffeed_arm.py`) = `s80` + `RWKV_SELFFEED=1` + `RWKV_SELFFEED_DETACH=1`, one variable, control `s80`. **562,932 params, i.e. +1,200 -- the PREREG said +960**, from counting the feature list; `Linear(14, 80)` is 1,120 weights plus 80 biases. The runner's guard names the measured number, so a derived count would have been rejected at launch -- the intended outcome, but only after the queue had moved. PREREG corrected. (2) **`rowfeat`** = `s80` + all eight per-ROW read-time columns (111 inputs, **564,292 params**), the arm `newc1` earned. (3) **The 2.69 GB row tables are BUILT ON THE BOX, not uploaded** (19 s there, against ~8 min of a shared link), which needed two hardcoded Windows paths removed from `build_rowtables.py` -- the dataset root is now `RWKV_ID_REVLOGS` and the repo root comes from `__file__`. **⚠ A table each machine builds for itself is not obviously the same table, so it is CHECKED: `scratchpad/rowtables/fingerprint.py` hashes both arrays in blocks and all five hashes are byte-identical across the two machines** (538,569,153 rows, 2692.8 MB). (4) Box queue appended: ... `s64dhc` -> `selffeed` -> `rowfeat`. `supervisor.sh` reads its queue through a file redirect, so an append reaches the RUNNING loop. Budget ~$43 of $100. (5) ⚠ **The launch ssh of a box eval DOES NOT RETURN and that is not a failure**: the detached `eval.sh` holds the session's channel open, so `remote_eval.py` sat the full 600 s ceiling while the eval was already running and had written its START line. The launch is now fire-and-VERIFY -- send it with a 60 s ceiling, then prove it started by reading the box's own log. Waiting on a control channel proves nothing either way.

**⟶ 2026-09-22 01:10 -- ★ THE BOX AND THIS MACHINE NOW AGREE ON ALL 39 MODEL FILES, AND MY OWN DIAGNOSIS OF THE KERNEL HALF WAS WRONG.** (1) **CORRECTION, made by reading the box rather than assuming: the box's `PQ_CB_MAX_FLOATS` is 262,144, already larger than the 3 x 1024 x 32 = 98,304 floats the RVQ arm uploads.** The 00:40 entry says the rebuild is needed because `g_pq_cb` holds 32,768 floats; it does not, and capacity was never the binding constraint. What the box's kernel actually lacks is **`c_pq_stages` -- the residual-VQ stage loop, absent from its source entirely** (`grep -c` returns 0). The 32,768 figure was the number of floats `rwkv_ops.py` UPLOADED when it read only stage 0, i.e. a symptom of the python drift, and I promoted it to a cause. (2) **The drift is repaired: 8 python files plus `rwkv7_cuda.cu` pushed, and `box_sync.py diff` now reports `39 files compared, 0 differ`.** Each pushed file was diffed against the box's copy FIRST and every local addition is flag-gated (selffeed, the row-table features, `RWKV_DECAY_RESUME_SCHED`, the `_PQ_META` 7th field behind `len(...) > 6`), so the box's running eval and its next arm are unaffected -- which is the only reason a push beside a live run was acceptable. (3) **`hash_rwkv.py` now hashes the KERNEL SOURCE too** (`.cu/.cuh/.h/.cpp`, 32 files -> 39). The RVQ incident was a python drift, but the kernel behind it had drifted the same way and nothing was watching that half. (4) **⚠ THAT CREATED A NEW BLIND SPOT AND IT IS NOW GUARDED: a box whose SOURCE matches can still be RUNNING a `.so` built from the old source.** `remote_eval.py` refuses when the newest of `csrc/**` + `rwkv/model/*.so` is a source file -- proven live, the box's `.cu` is newer than its `.so` right now. (5) **The kernel is rebuilt in ISOLATION (`/workspace/kernel_build`, BUILD_RC=0, 9,418,472 bytes) and the RVQ eval points at it through `RWKV_CUDA_PYD`, which `rwkv/model/__init__.py` honours for a `.so` as well as a `.pyd`.** No `.so` swap under a running trainer, so no gap is needed; the isolated build carries the `stages out of range` string and the live one does not. **OWED: swap the live `.so` at the next GPU gap** so the box stops being stale -- the stages==1 path is deliberately untouched in that source, so the swap is numerically inert. (6) The re-eval runs as tag **`lfq1rvq3b10v2`** (a suffix, so the withdrawn jsonls are not clobbered). ⚠ Two latent bugs surfaced on the way, both in code written 2026-09-22 and never executed since: the generated `eval.sh` is written as ASCII and the OOM comment I added carried a `⚠`, which killed every launch; and Git Bash rewrites a bare `/workspace/...` argument into a Windows path (`MSYS_NO_PATHCONV=1`). **A fix that has not been run is not a fix.**

**⟶ 2026-09-22 00:40 -- ★★ A BOX EVAL SCORED A DIFFERENT MODEL THAN THE ONE THAT TRAINED, AND NOTHING FAILED. ITERATION 95 IS WITHDRAWN.** (1) **`lfq1rvq3b10` (residual VQ, 3 stages x b10, card 385 b = +108%) trained LOCALLY and was evaluated ON THE BOX with a ONE-stage quantizer.** The box's `rwkv/model/rwkv_ops.py` predates the RVQ work -- its `_PQ_META` is the old 6-tuple with no `stages` -- so it ignored the catalog header's 6th field (=3) and uploaded 32,768 floats = 1024x32 = stage 0 only, exactly as its own banner records. The eval ran 3 h, produced 2,487 clean per-user numbers, and the verdict read **"more bits made it WORSE" (-0.000825 ahead / -0.001178 imm)** -- which is a 3-stage-trained model crippled by a 1-stage deploy quantizer, not a bits result. **This is the section-9 train/eval divergence, occurring across MACHINES rather than across code paths, where no smoke was watching.** Andrew's escalation rule ("not enough -> +200%") is NOT triggered: its premise was never tested. **THE GUARD (`remote_eval.py`): hash every `rwkv/**/*.py` on both sides and REFUSE to launch on any difference** (`scratchpad/cloud/hash_rwkv.py`, run on both machines). ⚠ It had to normalise BOTH sides to be usable -- raw hashing flagged 23 of 32 files and local-only normalisation flagged 30, because the two copies carry different line endings per file; normalised it reports **8 of 32**, which is exactly today's selffeed work plus `rwkv_ops.py`. A guard that cries wolf gets overridden. **OWED: upload the RVQ loader AND rebuild the box kernel (its `g_pq_cb` is 32,768 floats, too small for 3 x 1024 x 32 = 98,304), then re-run the eval.** Do it when newc2's decay ends, or locally after s80ddh. (2) **A second stale guard cost a day on the same run:** `remote_eval.py`'s eval.sh compared the merged user count against 2499 **without adding the OOM-skipped giants**, so a complete eval (`COMPLETE: 2487 merged + 0 nan-skipped + 12 oom-skipped = 2499 rostered`) died at DONE_EXIT_37. `mk_cloud_runner.py` got that fix on 09-20 and this script did not. Fixed; and `record_rvq.py` gained `--remote-complete`, an EVIDENCE-BASED override that re-proves completeness from artifacts (merged + oom-skipped == 2499) rather than a blanket `--force`.

**⟶ 2026-09-22 00:20 -- ★ THE SHRINK OVERSPEND IS THE WIDTH STEP, AND `s80ddh` IS RUNNING LOCALLY TO AVOID IT.** 94 `s64h` (332,922 = 1.69x) costs +0.000304 ahead / +0.000493 imm = **0.000133 / 0.000216 per 100k**: ahead passes the bar with 34% margin, imm is over by **0.000035** -- less than half the 7.5e-5 noise floor, and below what a seed pair could resolve (cross-seed spread on one recipe is ~0.0004). So the answer is not more measurement of that arm. **Decomposing the MARGINAL rates on imm says where the overspend is bought:** s80 -> s80d (depth at d=80) **0.000186**, s64 -> s64h (the mult halving) **0.000180**, s80 -> s64 (NARROWING to d=64) **0.000223**. Only the width step is over the bar, and every d=64 arm inherits it. **=> `s80ddh` = d=80, depths (2,1,2,1,1), mults 2/2 = 311,327 params (1.80x), reaching s64h's size class with no narrowing at all.** Allowance +0.000501; at the measured rates it projects +0.000476 / +0.000466, a pass with 5-7% margin. The risk is convexity (250,405 removed is 2.8x the only measured d=80 depth step). Started 00:17 locally, param guard green at 311,327, 0 tracebacks. `s80dh` (414,167 = 1.36x) is generated as the safer fallback. `s64h` also DOMINATES `s64` outright (40,064 smaller, cheaper in both modes), so s64 is off the frontier.

**⟶ 2026-09-22 00:05 -- 96 `newc1` AND 97 `r2s80` ARE IN (both rejected, both informative).** **`newc1`** (`cum_new_cards`, input 104, no rebuild): ahead **+0.000112** (p 3.2e-03) / imm **+0.000104** (p 7.9e-13) vs s80. BOTH raw deltas clear the 0.0001 bar and imm is strongly resolved; ahead's p misses the 1e-4 gate, so the both-modes rule rejects it. **Its PREREG leaned NULL** -- this is the one candidate a linear recurrence could rebuild by accumulating `diff_new_cards` -- and it came back positive in both modes at the 1+1 budget, where effects run ~0.65 of full size. **=> the read-time feature route earns its next arm (`rowfeat`, the eight per-ROW columns already built and smoked).** **`r2s80`** (button-ordered curve head): ahead **-0.000378** / imm **+0.000249** (p 2.4e-50) -- a clean trade, rejected. ⚠ r2s80 evaluates PLAIN and s80 RECTIFIED, so it is a deploy-quantity comparison, not a single-variable A/B. R2's case was always deploy speed (one forward pass instead of four probes); that case stands and now carries a price.

**⟶ 2026-09-22 00:00 -- ITERATIONS ARE KEPT IN ORDER (Andrew: "Please keep iterations in order").** The front table read 87, 94, 93, 92, 91, 90, 89, 88: the NUMBERS were right (assigned at verdict time) but ten record scripts each carried a hardcoded startswith() anchor on one fixed row and inserted directly beneath it, so a block of consecutively recorded arms grew DOWNWARD in reverse. Block re-sorted; every record script now calls **`optimization/front_table.py:insert_row`**, which appends after the HIGHEST-numbered row and then ASSERTS the whole column is ascending. The assert is the load-bearing half -- an anchor drifts silently, an ordering check cannot.

**⟶ 2026-09-22 00:26 -- ★★ 95 `lfq1rvq3b10` IS IN: THE QAT TAX AT +100% OF STATE BITS IS +0.002980 ahead / +0.004638 imm (NOT ENOUGH; Andrew's target is < 0.0010 in both modes).** 3 residual stages of b10 = 30 index bits per head, card 385 b / note 205 b; deployed 0.300328 / 0.268778, vs 84 `lfq1` -0.000825 / -0.001178 (p 1.0e+00 / 1.0e+00), n=2487, size 0, 0 tracebacks. At b10 the tax was +0.002153 / +0.003468. NEXT: +200% (5 stages).

**⟶ 2026-09-21 09:00 -- ★★ ANDREW RELAXED THE SHRINK TARGET, WHICH RE-RANKS THE WHOLE TRACK: "a cut by 1.5-1.7x is good too, so even if we don't reduce parameter count to 250k, something like 330k is fine too".** (1) **THE WANTED RANGE IS NOW 330,000-375,000 PARAMETERS**, and exactly two arms land in it: `s64` (372,986 = 1.51x, MEASURED +0.000399 ahead / +0.000421 imm against an allowance of +0.000377 -- misses by 6% / 12%) and **`s64h` (332,922 = 1.69x, allowance +0.000458, pre-registered to PASS** because the mult halving measured NEGATIVE cost at the deeper cut). So `s64h` is now the most important arm on the track, not the speculative extra it was queued as three hours earlier. (2) **THE RATE RULE STILL COMPOSES AND NEEDS NO RE-DECIDING:** Andrew set "<= 0.0002 per 100k removed, both modes" on 2026-09-19 when the goal was 2x, and he has changed the SIZE target, not the price; at 1.69x that rule gives +0.000458, a real bar rather than a vacuous one. Flagged, not assumed. (3) **CANCELLED AS A CONSEQUENCE: `s64ddh` (2.72x) and `s64c3h` (2.49x)** -- both built, both queued, neither started, now stubbed on the box (`run_<tag>.sh.real_bak` holds the real runner; restoring is one `cp` plus a queue line). They aim PAST the wanted range and `FRONTIER.md` had pre-registered both as failures, so two 5 h slots of a rented GPU would buy a confirmed failure at a size nobody wants; the same slots get `s64h` ~10 h sooner. ⚠ **THE PRE-REGISTERED PREDICTION THEREFORE GOES UNTESTED, DELIBERATELY** -- it is not evidence for the convexity reading, which now rests on s64 / s64dh / s48 alone. (4) Box queue after the change: s80d (evaluating) -> r2s80 -> newc1 -> newc2 -> **s64h**. `newc1` keeps its slot ahead of `s64h` because it is the gate for the ENTIRE read-time feature route (eight more columns built and waiting on it), not merely one arm. (5) The local 4070 is deliberately NOT used for `s64h`: it is reserved for the bits chain, which is on the critical path for the shipped model, while the shrink track is Andrew's stated side goal.

**⟶ 2026-09-21 07:05 -- ★ 92 s64 IS THE BEST RATE YET, IT WITHDRAWS MY OWN "MULT" READING FROM TWO HOURS EARLIER, AND THE FRONTIER TURNS OUT TO BE CONVEX WITH A KNEE RIGHT AFTER s64dh.** Full analysis + the pre-registered prediction: **`scratchpad/shrink/FRONTIER.md`**. (1) **92 `s64` (372,986 = 1.51x) costs +0.000399 ahead / +0.000421 imm = 0.000211 / 0.000223 per 100k -- the best of the four and still a FAIL**, ~10% over the 0.0002 bar in both modes. n=2485 (7 users OOM-skipped), size 0, 0 tracebacks. (2) ⚠⚠ **THE 04:10 ENTRY'S "IT IS A MULT STORY" CLAIM IS WITHDRAWN.** I read s64d and s48 (both mults 4, both ~0.00025-0.00034) against s64dh (mults 2, 0.000203/0.000260) and called the mults the culprit. `s64` is a **mults-4** arm with the best imm rate of all four, so the pattern was two points of a coincidence. **The mult halving is a real win at FIXED DEPTH -- that is the s64d-vs-s64dh pair, 40,064 FEWER parameters and better in both modes at p=6.0e-14 -- but it is invisible in a CUMULATIVE rate, which is dominated by how deep the cut is. Two arms that differ in depth cannot be compared on their mults.** Same shape as every "the parameter moved, so the lever is the cause" error in the family scoreboard: a consistent pair is not a mechanism. (3) **THE REAL STRUCTURE IS THE MARGINAL RATE, and the cumulative rate hides it.** Along the undominated frontier: s80 -> s64 (188,746 params) 0.000211 / 0.000223; s64 -> s64dh (99,239) **0.000186** / 0.000332; **s64dh -> s48 (51,299) 0.000692 / 0.000795**. So the cost is CONVEX and the knee sits immediately after `s64dh`: the last 51k parameters cost 3.5-4x the bar against 0.9-1.7x for everything before them. s48's mild-looking 0.000277 cumulative is a cheap 2.05x cut with a very expensive tail bolted on. (4) **`s64dh` ALL BUT PASSES ON AHEAD AND FAILS ON IMM**: allowance 0.000576 at 287,985 removed, spent 0.000584 (1.4% over) on ahead and 0.000750 (30% over) on imm. **The binding constraint on this track is the RATING head, the same asymmetry the QAT tax shows** -- so a pass, if one exists, comes from attacking s64dh's imm rather than from cutting further. (5) **PRE-REGISTERED, before they run: `s64c3h` (225,918, 2.49x) and `s64ddh` (206,193, 2.72x) both sit PAST the knee and are predicted to FAIL** at ~0.00026-0.00031 / 0.00032-0.00039 and ~0.00029-0.00035 / 0.00035-0.00043. They stay queued (~$2.50 each, ~$28 of $100 spent) because s64c3h is within 3,470 parameters of s48 by a DIFFERENT route and therefore prices depth-at-d=64 against narrowing-to-d=48 at equal size -- and because "both come in at or below 0.00022" is the falsifier for the convexity reading. (6) ⚠ **A RATE TRAP WORTH THE LINE: do NOT recompute these from the headline `ahead`/`imm` fields in `research_log.jsonl`.** Each arm's headline mean is over its OWN user set (2485 / 2492 / 2499 -- the box OOM-skips different giants per arm), so mixing them inflates every rate by ~1.7x and reorders the arms. `record_shrink.py`'s paired-on-the-intersection numbers are the only comparable ones. I made exactly this mistake while writing FRONTIER.md and caught it because s64d disagreed with its own recorded note.

**⟶ 2026-09-21 04:10 -- ★ 91 s48 FAILS AND NAMES THE MULTS AS THE CULPRIT; THE REST OF ANDREW'S KEPT FEATURES ARE BUILT ON ONE PER-ROW TABLE; grade_mean's LEAK CHECK NEEDED THREE VERDICT RULES BEFORE IT MEANT ANYTHING.** (1) **91 `s48` (222,448 = 2.53x) costs +0.000939 ahead / +0.001158 imm = +0.000277 / +0.000341 per 100k removed -- the 0.0002 bar FAILS**, and worse than the shallower `s64dh` did. Read beside the other two arms it is a MULT story, not a depth story: s64dh (mults **2**) 0.000203/0.000260, s64d (mults 4) 0.000245/0.000345, s48 (mults 4) 0.000277/0.000341 -- the two mult-4 arms agree with each other and the one mult-2 arm is cheaper, which is what s64d said directly by losing to s64dh with 40k MORE parameters. So s48's number is an UPPER BOUND on what d=48 costs, not a measurement of d=48; `s64c3h` (225,918, mults 2) reaches the same size by the other route and prices the difference. n=2485, size 0. (2) **THE PHASE-COHERENCE PICTURE ANDREW ASKED FOR IS DONE** (`scratchpad/phase_coherence/phase_coherence.png`, built from REAL users rather than from the formula: `scan_R.py` scans 400 users and picks user 330 R=0.94 a metronome, user 138 R=0.03/R2=0.03 genuinely spread, user 229 R=0.05 but **R2=0.46** two sittings 12 h apart -- the case R alone calls random, which is the whole argument for the second moment. Median R over those 400 users is 0.427). ⚠ The file did NOT reach his phone (upload socket error); it is desktop-only until re-sent. (3) **THE OTHER SIX KEPT FEATURES ARE BUILT, on a per-ROW table** (`scratchpad/rowtables/`: `secs_flat.npy` int32 epoch SHOW seconds + `rating_flat.npy` int8, 538,569,153 rows for users 1-7500, 2.69 GB, 48 s to build, **gitignored before the first write this time**). Eight columns behind the same flag (111 inputs with all on): `sess_pos`/`sess_secs`/`roll10`/`roll60` (session structure at the 30-min cut + the rolling counts), `phase_r`/`phase_r2`, `streak`, `grade_mean`. Two RAW columns rather than eight precomputed ones, because a prefix sum per feature is ~8.6 GB for `phase_r` alone and these two serve every feature including unwritten ones; the cost is one O(n) pass per chunk **from row 0**, which is the section-9 chunking hazard handled rather than hoped about. Constants over the train half's 357,460,182 rows; the three averaging features read exactly 5,000 fewer rows, one first review per user. (4) ⚠ **`review_time` IS DROPPED IN `get_rwkv_data` ON PURPOSE and the deploy builder needs it** -- so the drop is now gated on exactly the features that need it and `review_time` becomes a KEEP column for the query row. The easy alternative (let deploy read training's side table) would have made `deploy_vs_lmdb.py` compare the table with itself. (5) **CHECKS, each made to fail first:** `smoke_rowfeat.py` PASS -- alignment proved against the LMDB ITSELF (the stored `tod_sin`/`tod_cos` ARE this feature set's angle, `cyc30`/`cyc100` are the UTC day index mod 30/100), values at EXACT bf16 equality against a plain-loop reimplementation, and **a row table shifted by one row fails 9 of 9 chunks**; `deploy_vs_lmdb.py` 0 columns disagree on users 5030/5033 real AND query rows, while the shifted table makes all 8 disagree on up to 96.9% of rows; `smoke_id_features_width.py` gains a 111 case and the old 92/114/109/103/104/106 still pass. (6) **★ THE LEAK CHECK IS THE LESSON. Its verdict rule was wrong twice.** (a) "fail if the shipped column correlates > 0.35 with the current rating" failed the shipped column at 0.42 -- and 0.42 is HONEST, because with a one-review history the feature IS the previous rating and grades are that autocorrelated; a bar on the LEVEL must be calibrated against the thing under test. (b) So the rule became algebraic and threshold-free -- the leak is exactly invertible, `feature*(n+1) - sum(before)` is identically the rating for the unshifted copy (**100.000000%** of 2,552,081 rows) -- but the shipped copy still "recovered" 0.025%, and a 0.01% bar failed it again; those are CONSTANT users whose past mean equals the button they then press, i.e. a right prediction, not a recovery. (c) The floor is therefore a **within-user permutation** of the ratings, this project's own idiom from the duration leak probe: shipped sits **1.5x** above its floor (0.025% vs 0.017%), the leak sits at 100%. **LEAK_CHECK_PASS.** **THE RULE: when the honest signal and the leak differ only in DEGREE, the statistic needs a floor that shares the coincidence structure -- never a number chosen in advance.** (7) **NOTHING NEW IS QUEUED.** PREREG `scratchpad/rowtables/PREREG.md` registers ONE arm (`rowfeat` = s80 + all eight, control s80) and leans null, because all eight are ACCUMULATIONS over inputs the model already sees -- the same bet `newc1` makes in its simplest form. **`newc1` decides whether this arm is worth GPU time**; P4 is the falsifier for the whole read-time route. The sibling RATING (the last kept item) is deliberately NOT built yet: it needs a third table and it is the one candidate that is structurally different, so it waits for newc1.

**⟶ 2026-09-21 00:40 -- ★ ANDREW'S FEATURE IS WIRED AND QUEUED, 90 s64d KILLED MY RETRY DESIGN, AND THE 64-REVIEW CAP TURNS OUT NOT TO EXIST IN TRAINING.** (1) **`cum_new_cards` needs NO rebuild** and is input 104 behind `RWKV_EXTRA_FEATURES` (default off, `rwkv/extra_features.py`): the value is a pure function of `(user_id, review_th)`, so `prepare_batch.get_data` rebuilds it from a side table as it decodes a stored row -- no new `size` baseline, gated against the current lineage directly. Deploy keeps one integer (`DEPLOY_FUNCTIONS.md` 4b). Checks, each made to FAIL first: `smoke_extra_features.py` (128,670 rows, the 63 stored columns bit-identical with the flag on and off, every value EXACTLY bf16 of a parquet recomputation, 57,088 query rows inheriting their real row's value; a side table shifted by one row fails all 9 chunks), `deploy_vs_lmdb.py` with the flag on (0 columns disagree; ⚠ it was VACUOUS at first because its env reset dropped `RWKV_EXTRA_TABLE_DIR`), and two new width cases in `smoke_id_features_width.py` (103 and 104). **Arm `newc1` = s80 + the flag, 562,052 params, control = s80 itself; queued last on the box.** PREREG `scratchpad/newcards/PREREG.md` leans NULL, on purpose: this is the one candidate the recurrence could rebuild from `diff_new_cards`, so it is the control that makes session structure / phase coherence / the grade mean interpretable. Box readiness proven by hashing a decoded chunk BEFORE and AFTER uploading the five changed `rwkv/*.py` (identical -- so s48's later phases are safe) and by the feature column hashing identically on both machines. (2) **90 `s64d` (313,811, 1.79x) FAILS at +0.000245 / +0.000345 per 100k -- and it refutes the design of the two retries.** It is `s64dh` with `head_fc_mult`/`features_fc_mult` back at 4, i.e. 40,064 MORE parameters, and it is WORSE in both modes (paired n=2,492: s64dh +0.000024 ahead p 0.075, **+0.000107 imm p 6.0e-14**). So the mult halving is free-to-positive at d=64 and `architecture.py:253`'s "both REQUIRED -- cutting either fails imm" (iters 33/34) does NOT carry here. The retries were re-aimed BEFORE starting, which is why they were queued behind s64d: **`s64dd`/`s64c3` cancelled, `s64ddh` (206,193 = 2.72x) and `s64c3h` (225,918 = 2.49x) queued**, both at mults 2/2, both buying the rest from depth -- the cumulative rate keeps improving as the cut deepens (0.000245 -> 0.000203). (3) **Andrew's "lift the >64 cut-off for training" idea is already the case, and the cap is 128 not 64**: `features/base.py:104` drops `i > max_seq_len*2` inside `create_features`, whose only caller is the SCORED-set builder; training masks on `has_label`. Measured on the dbs (`check_seq_cap.py`): train user 66 has a 718-review card, 15,152 rows past i=128, **14,992 of them labelled**, 0 scored. ⚠ The first run sampled users 1/2/17 (longest card 112) and read as "truncated" -- a short sample is not evidence. (4) **The disk check is no longer a routine** (Andrew: "you can't communicate with agents in 'routines'. Which defeats the whole point"): `scratchpad/disk_check/disk_check.py` + `LAST_CHECK.txt`, run in-session every 3 days. First run: C: 291 GB, F: 787 GB free. (5) ⚠ **I committed the 385 MB side table into git** (00789cc, 8d67c39) before checking its size; now untracked and gitignored, but the blobs sit in LOCAL history (`.git` 843 MB) and purging them needs a rewrite of 153 unpushed commits -- **Andrew's call**.

**⟶ 2026-09-20 19:15 -- ★ THE +100% BITS ARM IS RUNNING LOCALLY, AND THE BOX NOW RUNS ONE TRAINER AT A TIME.** (1) **`lfq1rvq3b10` started 17:06** after its dry twin ended DONE_EXIT_0 with 0 skipped batches: 3 residual stages of b10 = 30 index bits per head, **card 385 b (+108%), note 205 b (+95%)**, 0.218 steps/s -> the decay ends ~07:00, then the quantized eval goes to the box. `record_rvq.py` is written and rehearsed; it applies Andrew's rule automatically (tax >= 0.0010 either mode -> +200%, below -> keep, <= 0.0003 both -> +50%). The Rust deploy path for stages > 1 landed first, so a win is shippable. (2) **THE BOX HAD BEEN IDLE 3.7 h** -- both lanes stopped 'for a human' and NEITHER failure was real. **s64dh died 37 on a COMPLETE eval**: the user-count guard added scored + OOM-skipped (2,492 + 10 = 2,502 vs 2,499) without noticing that 3 skipped users had been RETRIED and completed later; the union is exactly 2,499. **s64d died 44 WS_HOLLOW with 155 skipped batches of 10,935** -- two lane trainers on one 4090 OOM each other, which the earlier pair (20 and 14 skips) had already hinted at. **DECISION (mine, ops): ONE TRAINER AT A TIME.** The two lanes are merged into `queue_all.txt` = s64d, s48, s64, s80d, r2s80 under a single supervisor; ~24% less throughput against a 5 h re-run every time a WS goes hollow. s64d's hollow checkpoints are quarantined in `runs/s64d/contaminated_ws` and it restarted from scratch at 15:57 box time (box clock = local minus 3 h). (3) **89 `s64dh` IS IN AND IS THE TRACK'S FIRST REAL DATA POINT: 273,747 params (2.05x smaller) costs +0.000584 ahead / +0.000750 imm vs s80, i.e. +0.000203 / +0.000260 per 100k removed -- the 0.0002 bar FAILS**, narrowly on ahead and clearly on imm. (4) Two guard fixes came out of it: `mk_cloud_runner.py` counts only skipped users absent from the results, and `record_shrink.py` reads tracebacks from the decay that PRODUCED the checkpoint (s64dh's quarantined first decay held 185, and counting a quarantined attempt undoes the self-healing fix). ⚠ A near-miss the same afternoon: an ad-hoc stub copy of a waiter kept the real runner's call line and STARTED this arm beside its own dry twin for 4 minutes. Nothing was lost; the rule is in LIVE RULES.

**⟶ 2026-09-20 14:10 -- ★★ RESIDUAL VQ IS BUILT AND VERIFIED; THE +100% ARM'S CATALOG IS FITTING (Andrew: 'start with a 100% increase and go from there' -- 100% not enough -> 200%, just right -> keep, near zero -> try 50%).** Full analysis: **`scratchpad/lfqat/BITS.md`**. (1) **THE HONEST LADDER on the 21x corpus** (832,370 training pairs, 39 users, held-out 156): joint b10 = today 0.1697, joint b13 0.1181, role-split 2xb10 0.1467, **RVQ 2xb10 0.1100, RVQ 2xb13 0.0708**, RVQ 3xb10 0.0779. Widening the corpus cut the error **41% at zero bits**; RVQ beats role-split by 25% at equal bits (so there is no no-kernel shortcut); fewer, bigger stages win. ⚠ Every pre-today catalog number was fitted on the users it was scored on -- b10 reads 0.1042 that way and 0.2875 honestly. (2) **WHY THE TAX IS STUBBORN: the (u,v) pair lives in ~17 effective dimensions** (participation ratio 8.5 and 9.3 of 16), so VQ error falls as 2^(-bits/17) -- about 28% per 10 bits, which the ladder confirms. Hence the projection: **<0.0010 is a 2-2.5x state, not +50%**. (3) **KERNEL: `c_pq_stages` runs the existing search once per stage against the residual and sums the picks; the single-stage block is UNTOUCHED (restructuring its distance expression would change FMA contraction and break bit-identity with the record); the stage count is DERIVED from the upload size, so no op schema or caller changed; `rwkv_ops.py` reads an optional 6th header field; `pq_train_juv.py --stages N` fits each stage on the residual.** Proofs: `check_pqcap.py` PASS (34/34 bit-identical at b10, grown region read at 8,192 AND 32,768) and **`check_rvq.py` PASS** (a ZERO second stage reproduces one stage bit for bit; a real one changes the output and moves it closer to the unquantized state; gradients reach both stage blocks). **✓ THE RUST DEPLOY PATH FOR stages > 1 IS WRITTEN AND CHECKED (16:00, `eb19490`)**: the loader reads the 6th header field and keeps `stages` blocks, the joint encode codes each stage against the earlier stages' residual in the kernel's accumulation order; on held-out user 156 ONE stage is BYTE-IDENTICAL to the pre-patch binary (so no recorded deploy number moves) and THREE stages pull the predictions toward the unquantized ones (RMS imm 0.077929 -> 0.065407, ahead 0.076168 -> 0.060893). Checker `scratchpad/pqcap/check_rvq_rust.py`. ⚠ Engine-vs-kernel PICK parity for stages > 1 is still owed and needs a trace from a model trained with stages > 1, i.e. after the arm. ⚠ The engine reads `RWKV_TRACE_DIR` / `RWKV_WEIGHTS`; `RWKV_REF_DIR` is the EXPORTER's variable and leaves the engine on `reference/` = the OLD d=128 weights, which panics inside the catalog search with a shape error that reads like a residual-VQ bug. (4) **RUNNING: the +100% catalog (the arm is 3 stages x b10 = 30 bits/head -> card 385 b (+108%), note 205 b (+95%); the 2 x b14 variant it replaced was card 365 b / note 195 b)**; `mk_lfqat.py <L> rvq2 b14` generates the arm. ⚠ Decide from the DRY RUN's steps/s whether to keep b14 or drop to b13: the kernel searches stages x ncent per head per step (32x today at 2xb14, 16x at 2xb13) and only stage 0 is warm-started, so the decay may run 2-3x slower than lfq1's 13 h. (5) The b13 arm and its chain stay cancelled; the local GPU is free for the RVQ arm.

**⟶ 2026-09-20 11:15 -- ★★ THE BITS QUESTION IS RE-AIMED BY MEASUREMENT (Andrew: cancel the b13 arm, sweep TOP-DOWN from +50%, target QAT tax < 0.0010 both modes; and 'why is the corpus only 16k states?').** (1) **STOPPED: the b13 arm at step ~10,200 of 10,935** (cmd first, then its python; checkpoints and step-10,000 learned catalogs survive in `scratchpad/w10lf/lfq1b13_d_*`), plus the budget-curve and params x epochs waiters (notes in their logs, no markers). Three orphaned fetch workers of that run were still holding the pqcap `.pyd` and 18 GB -- they blocked a rebuild and were killed. (2) **WHERE THE BITS MUST GO, measured on a held-out user (`scratchpad/lfqat/score_cb.py`, relative error of the rank-1 (u,v) pair, 1.0 = encode to zero): joint b10 (today) 0.1042, joint b13 0.0317, ROLE-SPLIT u and v at 20 bits/head 0.0922.** So the only other way to add bits inside the fused rank-1 kernel -- splitting the pair into two codes -- buys 12% for DOUBLE the bits, while +3 bits of joint index buys 3.3x. Rank 2 is not an option either: only rank 1 is fused (`rwkv_ops.py`), rank 2 falls back to the Python per-timestep loop. **=> +50% of bits is not spendable today; the productive spend is finer JOINT indices, and their ceiling is the CORPUS.** (3) **THE CORPUS WAS 8,042 STATES BECAUSE OF A TOOLING ASSUMPTION, NOT THE DATA**: it is one state per reviewed entity from the three users we had parity traces for, and the trace export (Python deploy RNN, ~25 rows/s) was written off on 09-08 using a GIANT user (112: 790k rows = 8.8 h). A MODERATE user is 2-9k rows = 2-6 min. `scratchpad/lfqat/run_wide_corpus.cmd` is exporting 37 more (4 processes, ~40 min) -> ~5x the states; `pq_train_juv.py` grew a mini-batch path (full k-means is hopeless above ~2k centroids). (4) **KERNEL CAP RAISED TO b15** (32768 x 32 floats; `check_pqcap.py` PASS -- 34/34 tensors bit-identical at b10 and the grown region proven read at BOTH 8192 and 32768); `mk_lfqat.py` accepts b10..b15. (5) NEXT: fit b13/b14/b15 on the wide corpus, score them held-out, launch the largest FITTABLE one (card 225-235 b = +22-27%, not +50%) as the top rung. **The honest projection: the sibling's best density (0.125 bits per state float, tax +0.0013/+0.0002) is ~2x our card budget, so <0.0010 probably needs a SECOND quantization stage (residual VQ), which is new kernel work -- costed before proposing.** Lane B on the box was released (its priority-eval hold was waiting for the cancelled arm).

**⟶ 2026-09-20 10:00 -- ★ R2 RUNS END TO END ON A GPU (r2dry DONE_EXIT_0), THE SHRINK LANES ARE LIVE, AND d128e10 IS QUEUED.** (1) **`r2dry`: WS 218 steps with 0 skipped batches, decay, 20-user eval, DONE_EXIT_0** -- the button-ordered curve head (`RWKV_R2_HEAD`, 564,618 params, synthetic first-review query rows) trains, evaluates and scripts on real GPU batches, not only in the CPU smoke. `r2s80` is LAST in lane A (control `s80`). (2) **Lane A started 06:52 box time with `s80`**; lane B (s64dh, s64, s80d, then d128e10) waits for the chain's eval to have its gate slot. (3) The d=128 dry twin exposed a bug of MINE: the generator's eval user-count guard was hardcoded to 2499 while the eval RANGE now comes from the `.cmd`, so a 20-user dry twin died 37 after a clean WS + decay. Fixed (the count follows the range, minus 6701 when in range), all runners regenerated and uploaded except the running `run_s80.sh`; the dry twin re-ran to DONE_EXIT_0 and `d128_arm.sh` appended `d128e10` to lane B. ⚠ Two re-run traps met again, both already in this file and both caught by guards rather than by me: a canonical jsonl blocks its own re-merge (`merge refuses to clobber` -- move it aside), and an orphaned `sleep` child of a killed runner holds `run.lock` (eval_box.sh refuses and names the holder). **BUDGET: ~$18 of $100 spent by 10:00 on 09-20; lane A (4 arms) ~36 h, lane B (3 arms + d128e10 at ~4 days) ~123 h -> the box runs out ~09-27, which is about when d128e10 would finish. If it slips, d128e10 is the thing to drop.**

**⟶ 2026-09-20 09:35 -- THE MEMFILL FLAG IS MEASURED AND DROPPED: bit-identical, 184 s vs 165 s (i.e. SLOWER).** `RWKV_QAT_NO_MEMFILL=1` (the d=32-era flag that fell out of the d=80 runners) is numerically free -- both `cmp_traces.py` comparisons are BIT-IDENTICAL over 218 steps, including the warm-vs-off control that proves the box replays bit-exactly -- but it does NOT pay: flag off 165 s, flag on 184 s per 218 steps, each arm alone at the gate. `after_pair.sh` adopts only identical-AND-faster, so the eight queued runners were left untouched, which is the outcome the rule exists for. **The d=32-era argument (`aten::fill_` is ~25% of the launch storm) does not survive contact with a d=80 step on torch 2.12: launch COUNT is not wall-clock cost.** `optimization/DISPATCH_PLAN.md` 2026-09-20. So speed plan (c) ends with no training-side win: the mixers were already inductor-fused and the one dropped flag does not pay. The wins this round were all on the CPU deploy engine (replay 1,395 -> ~2,160 rev/s unquantized, ~990 -> ~1,475 quantized, byte-identical).

**⟶ 2026-09-20 09:25 -- ★ THE BOX AND THE PC AGREE TO WITHIN 2e-5, AND THE 20,000-STEP WARMUP IS REJECTED (86 `cw10lf`, 87 `lfwu20k`).** (1) **CROSS-HARDWARE OFFSET, the number every later box verdict needed: cw10lf (w10lf's recipe from scratch on the rented 4090) minus the local lfd1 = -0.000023 ahead / -0.000021 imm (p 0.73 / 0.90, n=2487)** -- a third of the 7.5e-5 noise floor, and an UPPER bound at that (cw10lf lost 20 WS steps to my co-tenant OOM, lfd1 lost none). **So a box number can be compared with a PC number directly**, which unblocks the shrink lanes, link D and any future rented box. (2) `lfwu20k` (WARMUP_STEPS 400 -> 20,000, one variable, trained beside its control): ahead +0.000236 at p 1.4e-02, imm +0.000053 at p 7.0e-05 -- the both-modes rule needs raw >= +0.0001 AND p < 1e-4 in each, and each mode fails a DIFFERENT half, so **rejected**. The ahead effect is real in size but not resolved (p 0.014 on 2,487 users); the imm effect is resolved but inside the noise floor. The LEDGER's unrecovered recipe cost (+0.0019 / +0.0018 at 10x) is NOT the warmup. (3) Both numbers are over 2,487 users: **12 giant users are OOM-skipped on the box** (see the 06:35 entry) -- [5626, 5859, 5991, 6198, 6431, 6630, 6714, 6720, 6810, 7033, 7221, 7238]; every box-vs-box and box-vs-PC pairing intersects on them, which the tools do by default. (4) `after_pair.sh` re-armed and running: the memfill bench alone, then r2dry + d128e10dry, then lane A, then lane B when the chain's eval has had its slot.

**⟶ 2026-09-20 09:11 -- 86 `cw10lf` + 87 `lfwu20k` ARE IN: THE 20,000-STEP WARMUP IS REJECTED (ahead +0.000236 p 1.41e-02 / imm +0.000053 p 7.00e-05 vs the control, n=2487, size 0).** Both are w10lf's recipe from scratch on the rented 4090, trained side by side; `cw10lf` also gives the CROSS-HARDWARE OFFSET against the local lfd1: -0.000023 / -0.000021 (an upper bound -- 20 of its WS steps were lost to CUDA OOM from my co-tenant tests). Recorded: `research_log.jsonl`, `research_5k.md`, verbose, PREREG verdict.

**⟶ 2026-09-20 06:35 -- ★★ THE BOX CANNOT EVALUATE THE GIANT USERS THE WAY THE PC CAN, AND NOW IT SKIPS THEM INSTEAD OF DYING.** Both pair evals died at EXACTLY 625 users, on the same user's 1,048,576-token mega-chunk: `torch.OutOfMemoryError ... Tried to allocate 1.38 GiB, 23.49 GiB capacity, 21.99 GiB held by this process`. **The 12 GB 4070 survives those users only because WINDOWS SPILLS GPU MEMORY INTO SYSTEM RAM (the WDDM shared pool, which is also why a giant-user eval needs ~40 GB free locally). LINUX HAS NO SUCH SPILL, so 24 GB is a hard wall** -- a cross-hardware difference that is NOT a speed or a numerics question and that nothing in the record anticipated. **FIX (`rwkv/get_result.py`): a CUDA OOM now skips the WHOLE USER, exactly like the NaN guard** (partial batches would silently change that user's equalized `size`, and a wrong size is worse than a missing user, which every verdict tool refuses to pair), records it to `result/RWKV-<tag>.oomskip.jsonl`, and the eval CONTINUES; the shard used to exit 1 and lose the rest of the run. Proven by fault injection (`RWKV_EVAL_FAKE_OOM=<user>`, default off): on a local CPU eval user 5001 was skipped, recorded, and the eval moved on to 5002. The box runners' user-count guard now counts the skipped users (`$N + $SK = 2499`), all runners regenerated and uploaded, and both pair evals restarted under the guard (banked users are kept). **CONSEQUENCE for every box verdict: n will be ~2,499 minus the giants; pair on the intersection (the tools already do) and note it.** If a giant user's own number is ever needed, evaluate that user on the PC and merge. ⚠ after_pair.sh EXITED at 03:07 on cw10lf's DONE_EXIT_25 -- correct behaviour, a failure marker stops the track -- so it must be re-armed after the evals land.

**⟶ 2026-09-20 06:00 -- ★ A BOX EVAL HANGS ON THE 1024 FILE-DESCRIPTOR LIMIT, AND IT WOULD HAVE HIT EVERY LANE.** cw10lf's eval banked 46 users, then a fetch worker died with `RuntimeError: unable to open shared memory object ... Too many open files (24)`: `torch.multiprocessing` hands every batch tensor to the workers as a shared-memory FILE DESCRIPTOR, and the container's soft limit is 1024 (hard 1,048,576). **The parent does not notice** -- `get_result` waits forever on an empty batch queue, GPU idle, log frozen, no marker. It sat like that for 1 h 22 min. Windows uses a different sharing strategy, so no local eval in this project's history could have shown it. **FIX: `ulimit -n 65535`, now emitted by `mk_cloud_runner.py` and by `remote_eval.py`'s eval.sh; all ten box runners regenerated and uploaded** (cw10lf / lfwu20k were NOT regenerated -- they were executing, and bash re-reads a script from a byte offset like cmd.exe). **`scratchpad/cloud/eval_box.sh <tag>...` re-runs a runner's phase C under the raised limit** by simply calling the runner again: its phases A and B sit inside `if [ ! -f <decay checkpoint> ]`, so an existing checkpoint skips them and the run keeps its own env, guards and markers -- the script ASSERTS that checkpoint first, because without it the runner would start a 30 h WS. cw10lf and lfwu20k are re-evaluating through it now (banked users are kept). ⚠ Two traps met on the way, both worth the line: a dead runner's sleep child still held `run.lock`, which makes the next launch exit 0 in silence (`fuser` the lock before relaunching -- the script now does); and `pkill -f eval_sharded.py` matched MY OWN ssh command line and killed the session -- kill by PID on a shared box.

**⟶ 2026-09-20 03:15 -- LANE B NOW WAITS FOR THE CHAIN'S EVAL, AND A FLOCK TRAP COST THREE SILENT RELAUNCHES.** (1) Two lane trainers fill the GPU gate (`apps < 2` fails), so `remote_eval.py`'s eval -- lfq1b13's, and every later chain arm's -- would have starved for as long as a lane ran. One lane PLUS that eval fits. `remote_eval.py` now writes `/workspace/PRIORITY_EVAL_PENDING` at launch and `PRIORITY_EVAL_DONE` after EVAL_OK; `after_pair.sh` starts lane A at once and holds lane B until DONE appears or 12 h pass (a chain eval that never arrives then costs throughput, never the track). (2) ⚠ **KILLING A SHELL DOES NOT RELEASE ITS `flock`: the `sleep` child inherited fd 9 and held `after_pair.lock`, so three relaunches hit `flock -n` and exited 0 -- silently, because that is the legitimate 'another instance runs' path.** Same shape as the orphaned `flock 8` children of the jobs stopped at 00:30. **When stopping a flock-guarded script, kill its sleep/flock CHILDREN too, and check `fuser` on the lock file before relaunching.** (3) A second cmd/bash trap: `A && B && setsid C &` backgrounds the WHOLE list, so the setsid never detaches it and ssh's exit kills it -- launch the script directly with `setsid <script> &`, never behind a `&&` chain. Tools built while waiting, all stub-tested: `scratchpad/shrink/verdict_shrink.py` (the 0.0002-per-100k rate rule, both modes; pass / fail / unusable / unknown paths), `scratchpad/lfwu20k/verdict_wu.py` (the warmup pair under the both-modes rule + the box-vs-PC offset, reject AND adopt paths), `scratchpad/lfqat/record_lfqb13.py` (the byte grant's record; adopt / reject / unclean paths; stub tags cannot write the record) and `scratchpad/cloud/fetch_box.py` (jsonls, logs, checkpoint; never overwrites without --force).

**⟶ 2026-09-20 00:30 -- THE BOX QUEUE IS NOW ONE ORCHESTRATOR: `scratchpad/cloud/after_pair.sh` (pid on the box in `/workspace/runs/after_pair.log`).** Found a race: cw10lf and lfwu20k started BEFORE the GPU gate existed, so their decay and eval phases are UNGATED; when a WS ends, the GPU shows one app for the seconds before the decay starts, and the three gated jobs queued beside them (r2dry, the memfill bench, d128e10dry, each polling every 60 s) could take that slot = three jobs on the card = the 09-19 OOM incident. Stopped those three and `start_shrink.sh` (mine, no GPU work started, the orphaned `flock 8` children gone, gate lock free). The orchestrator: (1) waits until BOTH pair logs end `DONE_EXIT_0` (a failure stops all); (2) the memfill bench ALONE (clean wall-clock); (3) r2dry + d128e10dry side by side (gated); (4) if the bench shows both comparisons BIT-IDENTICAL and the flag faster, `sed` adds `RWKV_QAT_NO_MEMFILL=1` to the eight unstarted box runners (s80, s80d, s64, s64d, s64dh, s48, r2s80, d128e10; logged per file, `bash -n` checked) -- the local copies do NOT get it, copy them back from the box for the record; (5) starts the two shrink lanes exactly as start_shrink.sh did. `r2_arm.sh` / `d128_arm.sh` still append r2s80 / d128e10 (the latter only on d128e10dry `DONE_EXIT_0`). ⚠ r2_arm does NOT check r2dry: read `/workspace/runs/r2dry/r2dry.log` before r2s80 (last in lane A, days away) starts. ⚠ BUDGET: $100 at $0.493/h runs out ~09-27; d128e10 (~2.6+ days WS beside a second trainer) queued last in lane B may not finish -- decide when the lanes' pace is known.

**⟶ 2026-09-19 21:35 -- RUST ENGINE, SECOND PASS (all byte-identical preds on users 107 / 136 / 156): replay 1,395 -> ~2,160 rev/s unquantized, ~990 -> ~1,475 quantized; state pass ~1,470 -> ~2,350 states/s, query pass ~2,780 -> ~4,300 R/s.** (1) Every `linear` / norm call built 1-2 Strings and SipHashed them (~20% of the pass): weight maps are now keyed by FNV-1a, a layer's prefix is hashed once (`fast.rs`). (2) 4-row matvec in the serial add order (AVX2 elementwise, no FMA). (3) INTEGER prescreen (`model.rs::I16Blk`: 13-bit catalog + query, `madd_epi16`, a rigorous error bound, 1.05 exact rechecks per search; integer sums are CPU-independent; `RWKV_PQ_SCAN=f32|exact` = the older scans). A native-CPU build gives +2% only and an AVX2 WKV step gives nothing: the pass is not SIMD-width-bound. Left: libm exp / tanh / sigmoid = ~12% (an approximation breaks byte-identity; not done). **Also measured for the Clanki session (`CPU_INFERENCE.md` 2026-09-19): a 1e-6 float-order difference flips quantizer picks -> mean |dp| ~1e-4, 1-3% of predictions move > 1e-3, growing along a user's history, signed mean ~0; advice = one canonical float order (same lanes + fused multiply-add in the scalar fallback) + a kernel-version tag on the state cache; rounding before the quantizer does not help.** `RWKV_DEBUG_JITTER` is the instrument (inert unset).

**⟶ 2026-09-19 21:05 -- LINK E (LOCAL) STOPPED; d128e10 GOES TO THE BOX BEHIND A DRY TWIN.** (1) The local link E waiter (12296) was stopped in gate 0 with a note in `lfwu20k/wait_wu.log` (no marker): lfwu20k already runs on the box. (2) `mk_cloud_runner.py` now takes the eval user range from the `.cmd` (a dry twin evaluates 5001-5020; a real runner regenerates byte-identical). `run_d128e10.sh` + `run_d128e10dry.sh` are on the box; **the dry twin waits at the GPU gate (behind r2dry and the memfill bench); `/workspace/d128_arm.sh` appends `d128e10` LAST to lane B only if `/workspace/runs/d128e10dry/d128e10dry.log` ends `DONE_EXIT_0`** (else it logs the refusal in `/workspace/d128_arm.log`). ⚠ **The LOCAL link D waiter (27796) is still armed and would run d128e10 again on the 4070 (~4 days) after link C.** When the box dry twin has passed: stop 27796 in gate 0 with a note and run only `lfb1` locally (then `d128b1` = a decay from `d128e10_ws_22000`, on the box, needs a decay-only cloud runner). If the dry twin fails, the local link D is the fallback. Box disk 212 GB free; budget: ~$12/day.

**⟶ 2026-09-19 21:00 -- SPEED PLAN ITEM (c) IS RE-SCOPED: THE MIXERS ARE ALREADY FUSED, AND A DROPPED FLAG IS THE CHEAP WIN.** `RWKV_QAT_COMPILE=1` (in every runner since 2026-07-30) already runs inductor-fused Triton kernels over all 26 mixer forwards + backwards (`triton_*_fused_*` in the 08-31 profile; 3,105 `cuLaunchKernel`/step are those), so a hand-written fused block kernel would replace fusions that exist; my 1.3-1.5x estimate was wrong. The other ~12,700 launches/step sit OUTSIDE the mixers, and **4,879 of them are `aten::fill_` = deterministic mode's NaN-fill of every fresh tensor (`torch.utils.deterministic.fill_uninitialized_memory`)**. `RWKV_QAT_NO_MEMFILL=1` turns it off (train_rwkv.py:57; verified in the d=32 era, in every champ5k runner) and **fell out of the runners when the d=80 lineage began** -- no runner since track 2 sets it. Expected ~5% (launch-bound share), numerics-neutral for correct code (a read of unfilled memory would have shown as NaN while the fill was on: zero exceptions in the lineage). **MEASURING, not assuming: `scratchpad/cloud/bench_memfill.sh` waits at the box's GPU gate** (s80 env, 218 steps x three arms warm / off / on; `cmp_traces.py` warm-vs-off = the bit-replay control, off-vs-on = the test; wall seconds per arm) -> `/workspace/runs/memfill/memfill.log`. If identical and faster: add the flag to the STANDARD TRAINING ENV for NEW runs only (armed shrink runners stay as they are). Still open after it: `aten::mm` 1,580 calls/step (about twice the model's linears; source unknown, needs a GPU profile) and CUDA graphs on the box (24 GB may not need `empty_cache`, which is what blocks graphs locally).

**⟶ 2026-09-19 20:50 -- R2 IS BUILT (flag `RWKV_R2_HEAD=1`, default off = the old paths) AND QUEUED ON THE BOX; THE QUANTIZED RUST ENGINE IS AT ~990 rev/s.** (1) **R2** (`scratchpad/r2/DESIGN.md`): `gru_s` has 4N rows = log S of Again + three softplus increments -> the four button curves are ordered for every t by construction; `prepare_batch.insert_r2_queries` pairs EVERY labelled real row with the query row of the same review (no probes, no RNG) and inserts ONE synthetic query row before a card's first review (the LMDB has none: real row with duration + grade cleared, is_query = 1 -- exactly the columns in which 59,714 stored pairs differ; 13% extra rows, moved by the clock fix like a probe); `srs_model._r2_substitute` writes the pressed-button curve of the query row over the real row's own curve value WITH gradient, so the ordinary ahead loss, validation, statistics and get_result all read one quantity (no `RWKV_EVAL_PAVA`, asserted). `scratchpad/r2/smoke_r2.py` PASS (12 rows, 2,891 of 2,891 labelled rows paired, 966 synthetic, max order violation 0.0 on random weights at five horizons, gradient reaches gru_s, scripts). Params 564,618 (+2,889). **Arm `r2s80`** (`mk_shrink.py`: s80 + R2, PAVA 0, probe density 0, plain eval; control s80) is appended LAST to lane A by `/workspace/r2_arm.sh`; **a dry twin `r2dry` (218 WS steps + 109 decay steps + 20-user eval) waits at the GPU gate on the box** -- check `/workspace/runs/r2dry/r2dry.log` for DONE_EXIT_0 and its ws log for `Exception caught` BEFORE r2s80 starts. ⚠ While r2dry waits it HOLDS the launch lock; nothing else is gated until the pair ends. The box has the new `srs_model.py` / `prepare_batch.py` (backups `/workspace/*.bak_r2`); the box runners now die 44 WS_HOLLOW if > 1% of WS steps were skipped. NOT built yet: the RNN deploy path + Rust for R2 (only if adopted). (2) **Rust:** `review_query` (query / probe passes skip the state compression), stack-matrix power iteration, pick-identical prescreen: quantized replay 233 -> ~990 rev/s, preds byte-identical; tables S / R / Q in `DEPLOY_SPEED_LOG.md`. (3) NEXT: the fused Triton block kernel (speed plan c); record r2dry; pair verdicts when the box runs end (~12:00 on 09-20).

**⟶ 2026-09-19 19:30 -- SPEED PLAN ITEM (a) DONE + A BOX INCIDENT.** (1) **The deploy-QUANTIZED Rust engine was 6x slower than the unquantized one every speed row had measured (233 vs 1,395 rev/s), ~85% of it the catalog search; `model.rs::nearest` (transposed-catalog prescreen + exact recheck, pick-identical by construction, preds BYTE-IDENTICAL on 3 users; `RWKV_PQ_SCAN=exact` = the old scan) gives ~757 rev/s.** `optimization/CPU_INFERENCE.md` 2026-09-19. (2) ⚠ **MY CO-TENANT TESTS ON THE BOX (box_dry + the 8-user remote eval, 09-19 afternoon) MADE BOTH WS RUNS SKIP BATCHES ON CUDA OOM: cw10lf 20, lfwu20k 14 of 109,350 (`Exception caught`, steps ~57k-70k); both arms hit alike, the pair stays comparable, cw10lf's cross-hardware reading carries the caveat.** The WS log is not traceback-gated on the box, so nothing failed. **Fix = a GPU GATE in `mk_cloud_runner.py` (`gpu_gate train|eval`: one launch at a time under `/workspace/gpu_launch.lock`, open only when < 2 compute apps and, for an eval, no other eval) and the same queue in `remote_eval.py`; tested on the box (waits with 2 trainers, opens with 1); the six shrink runners were regenerated and uploaded (none was running); the runners now log the WS skipped-batch count.** cw10lf / lfwu20k run ungated until they end (~12:00 on 09-20). Never start anything on the box outside the gate.

**⟶ 2026-09-19 19:57 -- 85 `lfqkd1` IS IN: LIVE QAT-KD IS REJECTED (ahead -0.000061 p 8.84e-01 / imm +0.000034 p 3.89e-03 vs 84 `lfq1`, n=2499, size 0, 0 tracebacks; deployed 0.299692 / 0.267677).** The tax with KD is +0.002215 / +0.003434, unchanged. A same-architecture full-precision teacher gives the quantized student nothing: the tax is a state-precision cost. `KD.txt` = nokd; **link B2 fired by itself: `lfq1b13` (b13 WKV catalog, the byte grant) started 19:10 on 09-19, its REAL eval goes to the box (`remote_eval.py`)**; then C, D, E. Recorded: `research_log.jsonl`, `research_5k.md`, verbose, LEDGER row 15, PREREG verdict.

**⟶ 2026-09-19 17:50 -- SPEED PLAN APPROVED BY ANDREW ("👍 on the plan"; on the fused kernel: "we should've done it a while ago").** While both GPUs are busy, in this order: **(a) PROFILE `rust/rwkv-infer` on a real replay** -- my untested estimates: the PQ encode search on a real review (WKV 1024 x 32 per head per layer + shift 4096 x 40 x 2 chunks ~ 2 M MACs) may cost MORE than the network (~0.56 M MACs) -> a two-stage / SIMD search, no retraining; ~2,500 exp / sigmoid / tanh calls per review may be ~1/3 of the time -> SIMD approximations at ~1e-6 (inside parity). **(b) R2 DESIGN PASS (monotone-by-construction curve head):** accuracy ceiling ~0 (r2probe), but 1 forward pass instead of 5 per ahead prediction at deploy AND no probe rows in training / eval (rectified evals several times shorter, less VRAM); if the design holds, add it as one more from-scratch arm in the box's shrink lanes (control `s80`, same budget, ~$3). **(c) A FUSED TRITON KERNEL for each block's elementwise section, fwd + bwd** (the step has ~18,700 launches; two runs side by side gave +24% = the idle gap; estimate 1.3-1.5x), with a bit-exactness / golden check like the QAT kernels'. Also recorded today: Clanki PR 160 in `CPU_INFERENCE.md`, the AnkiHub content extrapolation in `CONTENT_EMBEDDINGS.md`, `scratchpad/cloud/box_monitor.py` (Afterburner-style window for the box GPU; `run_box_monitor.cmd`), front-table rows renumbered (70-84, F1-F6; lfqkd1 becomes 85).

**⟶ 2026-09-19 14:40 -- ★ TWO DIRECTIVES FROM ANDREW. (1) "move evals to the box"** (a quantized giant-user eval fills the 4070's 12 GB and stalls his desktop): `scratchpad/cloud/remote_eval.py launch|collect <runner.cmd>` runs a local runner's phase C on the 4090 (uploads the decayed checkpoint + learned catalogs, parses the runner's env AT phase C, downloads the jsonls + the shard log); proven on 8 users of lfqkd1 (sizes equal, box-local +0.000019 / -0.000017, QAT banners present); **`mk_lfqat.py` now routes every REAL arm's eval there (`LFQAT_EVAL_ON_BOX=0` restores local; the dry twin stays local)**; the remote eval waits while another eval runs or > 1 GPU job exists (two trainers + an eval = CUDA OOM on 24 GB, measured). Found on the way: `eval_sharded.py` had a Windows-only interpreter path (fixed; it would have failed the box runners' phase C), and the box had never run a decay or an eval -- `scratchpad/cloud/box_dry.sh` now PASSES (make_resume, a 433-step decay, an 8-user eval). **(2) THE SHRINK TRACK RE-OPENS SIZE REDUCTION AS A SIDE GOAL: "try more ablations to see if we can reduce model size further by 2x", bar relaxed to <= 0.0002 LogLoss per 100k params removed, both modes (= +0.0005 for -250k).** `scratchpad/shrink/` (PREREG.md, `mk_arch.py`, `mk_shrink.py`): six from-scratch arms at the track-2 budget 1+1 on the box, base `s80` 561,732; `s80d` depth 471,927; `s64` 372,986; `s64d` 313,811; **`s64dh` 273,747 = 2.05x**; `s48` 222,448 (params MEASURED; s48 + s64dh smoke-trained 173 steps on the box, 0 tracebacks). Bundles because one d=80 layer's allowance (+0.00006) is under the noise floor. `start_shrink.sh` is armed on the box: when cw10lf AND lfwu20k end DONE_EXIT_0 it starts two lanes (`queueA.txt` s80, s64d, s48; `queueB.txt` s64dh, s64, s80d; `supervisor.sh` takes `QUEUE=`). A pass is a CANDIDATE: confirm at 10+1 before it replaces the 561k model (capacity binds harder at the full budget). Verdicts: download `result/RWKV[-P]-<arm>.jsonl` from the box and pair each arm with s80.

**⟶ 2026-09-19 12:05 -- ★ THE BLACK-SCREEN HANG HAS A READABLE CAUSE (second hang of the day at 09:48; Andrew forced a dump).** `cdb` on `C:\Windows\MEMORY.DMP`: all 32 CPUs idle, RAM fine; **dwm.exe holds the adapter core resource inside `DxgkDestroyAllocation2 -> DisablePrimaryAllocation -> VidSchiFlushPendingHWSubmittedFlips` and waits ~115 s for a hardware flip the NVIDIA driver never completes, while CUDA saturates the GPU**; every other GPU client (python's submit included) queues behind it; no TDR because it is a flip wait. HAGS was ON (`HwSchMode=2`, driver 610.62). **Andrew turned HAGS OFF and rebooted at ~11:55 (`HwSchMode=1`)**; the second lever if it returns = MPO off (`OverlayTestMode=5` DWORD under `HKLM\SOFTWARE\Microsoft\Windows\Dwm`, his action). Commands + detail: memory `black-screen-hangs`. **Recovery both times:** lfqkd1's eval resumes from its banked users (`scratchpad/lfqkd1/run_lfqkd1_evalresume.cmd` from `mk_evalresume2.py` = the runner's env + the EVAL ENV block + phase C; `finish_linkB2.cmd` then writes link B's marker); 1,736 of 2,499 banked at 12:00; waiters B2/C/D/E re-armed (pids 31448 / 31276 / 27796 / 12296, finisher 31432). The box is unaffected (cw10lf step 52,869, lfwu20k 45,209 of 109,350).

**⟶ 2026-09-19 01:40 -- THE PC WAS REBOOTED AT 01:04 (black screen with a cursor, Andrew); RECOVERED.** lfqkd1's QAT-KD decay died at step 8,234; it RESUMED AT STEP 8,000 (`scratchpad/lfqkd1/run_lfqkd1_resume.cmd` from `mk_resume_runner.py`: `make_resume.py` on the decay toml, the step-8000 LEARNED catalogs as start catalogs, `RWKV_RESUME_SKIP_GROUPS=1`, and the new gated `RWKV_DECAY_RESUME_SCHED=1` in `train_rwkv.py` -- without it a resumed decay restarts its LR curve at 1.0; unset = byte-identical). Proof: the loss at step 8,008 is 1.614233 vs 1.614112 before the crash. `finish_linkB.cmd` (pid 29196) runs it and then writes link B's DONE_EXIT_0 into `lfqat/wait_lfqatkd.log`; waiters re-armed from their unchanged files: B2 29404, C 22620, D 28912, E 29720; ClaudeRamGuard running. The resumed tail's dropout draws differ. **THE BOX: cw10lf and lfwu20k now run SIDE BY SIDE (Andrew's idea): 1.16 + 1.16 steps/s vs 1.87 solo = +24%** (`CLOUD_COMPUTE.md` 5f); lfwu20k was started by hand at 01:12 with a per-tag lock, so the supervisor's later launch of it leaves quietly; evals never overlap. Both WS end ~03:30 on 09-20.

**⟶ 2026-09-19 00:05 -- THE BOX IS BOOTSTRAPPED AND RUNNING ITS QUEUE.** Kernel built (CUDA 13.0 by apt), dbs built ON the box (train 57 min, val, VAL-half test 83 GB in 6 min; params 561,732 and 10,935 groups = the PC's), **benchmark 1.900 steps/s at MAX 65536 = 2.08x the PC**; MAX stays 65536 by the budget rule (`optimization/CLOUD_COMPUTE.md` 5f). **`/workspace/queue.txt` = cw10lf then lfwu20k, `supervisor.sh` started 00:01 local (21:01 UTC on the box)**; each runner = WS (~16 h) + 1-epoch decay (~1.6 h) + phase C, the rectified VAL-half eval ON the box (`--exclude 6701`, 2,499-user gate, resumable) -> `result/RWKV-<tag>.jsonl` there; download results + the decayed checkpoint + logs, then gate lfwu20k vs cw10lf with `paired_pvalue.py`. Logs: `/workspace/runs/<tag>/<tag>.log`, `/workspace/runs/supervisor.log`. cw10lf also measures the cross-hardware offset vs local lfd1 (same recipe). Link D's d=128 cells are box candidates (2.08x >= 1.5x): generate their cloud runners next (mk_cloud_runner handles WS+decay+eval; d128e10 needs MAX 32768 and its own param guard), then stop the local link D waiter in gate 0 with a note. Wandb must be installed on the box (import only).

**⟶ 2026-09-18 21:50 -- ★ A RENTED BOX IS LIVE (Andrew, $100): Vast.ai RTX 4090 + Ryzen 9 7950X (32 threads), 62 GB RAM, 500 GB, $0.493/h.** Address, ids and layout: `scratchpad/cloud/BOX.local.md` (gitignored) + memory `rented-box`. Andrew: "which runs to keep on my RTX 4070 and which runs to do on this one is up to you"; MAX / batch changes are allowed there under the budget rule (`optimization/CLOUD_COMPUTE.md` 5d). **MY PLACEMENT: the local 4070 keeps the chain (lfqkd1 running since 09-18 14:43 -> b13 -> budget curve -> link D -> link E); the box takes the WS-level work AND its own evals: first the pair `cw10lf` (control = w10lf's WS + the 1-epoch decay) and `lfwu20k` (warmup 20,000), then link D's d=128 cells if the benchmark says the box is >= 1.5x (then stop the local link D / E waiters in gate 0 with a note, never a marker).** **BOOTSTRAP STATE at 21:50 (resume here):** (1) upload from `C:/Temp/cloud_upload` by scp in a session background job: `repo.tar` DONE and extracted to `/workspace/rwkv-anki-autoresearch`; `label_filter_db_id_e2s.tar` (~4 GB) in flight, then `anki-revlogs-10k-id.tar` (16 GB), ~6 MB/s -> ~55 min; if the job died, re-run scp for the missing/short file (compare sizes with the local ones). (2) The image is a CUDA **12.8** tag by accident: `apt-get install cuda-nvcc-13-0 ...` runs on the box (log `/workspace/apt_cuda13.log`, ends with `APT_RC=`); the kernel build then needs `CUDA_HOME=/usr/local/cuda-13.0` and `PATH` with its `bin`. (3) `uv venv .venv` exists; torch 2.12.1+cu130 + the pinned deps install (log `/workspace/pip_setup.log`, ends with `PIP_RC=`; the first try timed out on a shared link, the retry uses `UV_HTTP_TIMEOUT=900`). ⚠ An ssh call that starts a background job on the box HANGS unless it uses `setsid ... < /dev/null > log 2>&1 &`. (4) NEXT: `python setup.py build_ext --inplace` (then `import rwkv.model.RWKV_CUDA`), untar the data into `/workspace` and `/workspace/rwkv_lmdb`, `bash scratchpad/cloud/rebuild.sh` (val db then train db, NPROC up to 8; **ADD the TEST db 5001-7500 there -- evals run on the box**, since the local GPU is busy and gate-critical runs forbid co-tenancy), the 15-minute benchmark (steps/s at MAX 65536 vs the PC's 0.913, plus MAX 98304 / 131072 / 196608 for the budget rule), then `queue.txt` + `supervisor.sh`. The kit is UNTESTED on Linux: expect first-contact fixes (the `.sh` files were committed with LF; `git archive` may have converted nothing, check `file run_*.sh`). An eval runner for the box does not exist yet (port `write_eval_toml.py` + `eval_sharded.py` calls from lfd1's phase C).

**⟶ 2026-09-18 21:14 -- ★ lfq1 IS IN: THE LEAK-FREE DEPLOY NUMBER IS ahead 0.299630 / imm 0.267711; THE LEAK-FREE QAT TAX IS +0.002153 / +0.003468** (vs lfd1, n=2499, size 0, 0 tracebacks, quantized eval on the learned catalogs, probe-proven). P1 held in both modes. vs arm 2's leaked tax (+0.002529 / +0.004466) the imm tax is 0.0010 smaller -- ~a fifth of it was the quantizer stripping the precision the clock leak needed; the ahead tax is ~+0.002 whatever the budget or the leak. Misses the stop criterion by +0.0046 / +0.0037. Recorded: `research_log.jsonl` `lfq1`, `research_5k.md`, verbose "ENDGAME: lfq1", LEDGER row 15, PREREG verdict. **Link B fired by itself: lfqkd1 (live QAT-KD, teacher = lfd1) started 14:43 on 09-18** (~13 h decay + ~13 h eval), then B2 (b13), C, D, E. **Also 09-18: Andrew's weekly limit reset; the rented-GPU work starts (`optimization/CLOUD_COMPUTE.md`, kit `scratchpad/cloud/`).**

**⟶ 2026-09-17 11:22 -- R1 IS IN: THE IMPUTED PROBE DURATION IS NOT ADOPTED.** lfr1e1 (the imm query row + the four probe rows carry a shrinkage median of PAST durations, k=3, instead of 0.0; a plain 1-epoch decay from w10lf's WS) vs lfd1: ahead +0.000076 (p better 2.2e-05) / imm -0.000004 (p better 9.6e-01, p worse 3.8e-02), n=2499, size 0, 0 tracebacks; 100.0% of query rows had a history to impute from; decay 3.45 h vs 3.62 h. Curve-side gate -> R1.txt = off. PREREG P1 missed, P2 held. The QAT arms run without it (twin lfd1); the idea is closed. Recorded: `research_log.jsonl` `lfr1e1`, `research_5k.md`, verbose "ENDGAME: lfr1e1", PROPOSALS R1. **Link A v3 fired next: the leak-free QAT arm lfq1.**

**⟶ 2026-09-17 10:10 -- LINK E ARMED: THE WARMUP EXPERIMENT `lfwu20k` (Andrew: "keep augmentation off, otherwise sure, try that"), AND A CLOUD-COMPUTE MEMO.** `scratchpad/lfwu20k/` (`mk_lfwu20k.py`): w10lf's leak-free 10-epoch WS with **`WARMUP_STEPS` 400 -> 20,000** (upstream's; ours was tuned at 1.25 epochs and never re-tuned for 10), then the L* = 1 decay and the rectified full-VAL eval; ONE variable vs w10lf's WS, control = lfd1; guard 60 = `Warmup steps: 20000` in the WS log; preflight PASS (runner + dry twin). It aims at the LEDGER's unrecovered recipe cost (+0.0019 / +0.0018 at 10x); PREREG `scratchpad/lfwu20k/PREREG.md` (P1 ahead -0.0002..+0.0008; a large win says the WS schedule, not size, holds it; the peak-LR variant follows only on a win). **Waiter `wait_pxe_then_wu.cmd`** (`mk_wait_wu.py`, log `lfwu20k/wait_wu.log`) fires on link D's DONE_EXIT_0 (refuses 230 otherwise; preflight -> dry twin -> real run; a failure SKIPS with a log line), two stub paths executed, 75 s in the foreground, then detached. ~46 h of GPU, ~10 days out behind the QAT arms, the budget curve and link D; on a rented GPU it would run first with a twin of lfd1 as its control. ⚠ Found on the way: **a WS resume re-runs the WHOLE warmup** (`train_rwkv` builds the LinearLR fresh and the LOAD_MODEL path resets lr to PEAK_LR), harmless at 400 steps, a different experiment at 20,000 -- `make_resume.py` now honours `RWKV_RESUME_KEEP_WARMUP=1` (the remaining warmup, max(1, W - n), goes into the resume toml; unset = byte-identical), and this runner sets it. **The cloud memo: `optimization/CLOUD_COMPUTE.md`** -- our step is bandwidth- and launch-bound, so the RTX 5090 is the price/performance pick (~2.1x a 4070 for ~$0.40-0.65/h on Vast.ai, ~131 RUB/h on immers.cloud; A100/H100/H200 cost 2-5x more per unit of our work); the whole remaining queue is ~4.5 days of one 5090 (~$60-70 / ~15,000 RUB); evals stay local (64 GB RAM); controls must run on the same box (cross-GPU numerics ~ the noise floor); the runners need a bash port first. Nothing rented; Andrew decides. lfr1e1's eval: 1,808 of 2,499 users at 10:07.

**⟶ 2026-09-17 03:11 -- lflora16 IS IN: A NULL, AND ARCHITECTURE GRAFTS ARE CLOSED AT 0/3.** The card LoRA-rank graft (4/4/2/4 -> 16/16/8/16, +12,480 params, state unchanged) on w10lf's WS + w10lf's decay: ahead +0.000059 (p better 0.03) / imm +0.000006 (p better 0.01) vs w10lf, n=2499, size 0, 0 tracebacks. Engagement: L0 new gate-B cols |new| 1.2517 vs |old| 23.9245; L0 new decay-A rows |new| 2.5666 vs |old| 8.0172; L1 new gate-B cols |new| 1.3989 vs |old| 32.4165; L1 new decay-A rows |new| 2.7828 vs |old| 6.4400. With cmixgraft and lfcmf2 that is three grafts, three nulls at low engagement (Andrew closed the family on 09-16; PREREG P4: a null bounds capacity from below only; link D's from-scratch d=128 answers capacity). Recorded: `research_log.jsonl` `lflora16`, `research_5k.md`, verbose "ENDGAME: lflora16", scoreboard 0/3. Link G runs `verdict_cardgraft.py` on the pair and hands over to the R1 link (`lfr1e1` at L* = 1, `scratchpad/r1/PREREG.md`).

**⟶ 2026-09-16 22:30 -- CLANKI CROSS-SESSION WORK (a day of peer messages from the Clanki session, now PAUSED for a few days -- Andrew: no more Clanki work, usage limits).** Reviewed/approved on the Anki fork's RWKV side (its own repo, nothing edited here): PR 112 (replay start row for cards with no Learning row: latest learning start, else first rated row after the last Forget; the training builder DROPS such cards, so it is an inference-only divergence, queued for the next rebuild in FUTURE_FEATURES.md), R2 (per-review RWKV-Curve recording in the warm-up from the previous review's stored curve, with a cross-call fallback), the lazy state-cache load (one SQLite row per entity; 3.5 GB / 4 s start-up -> 95 MB / 171 ms), a query-only Stats retrievability call, and the rules for the model-quality graphs (held-out rows only for FSRS-7, any raw row for frozen RWKV, paired sets). THE AUC QUESTION IS CLOSED: Andrew's collection scores RWKV-Instant 0.717 on the benchmark's own scorer (equalized set, CPU fallback = fp32 state like the CUDA kernel) vs Clanki's 0.726 -- the fork's replay is faithful; his collection sits at ~p1 of unseen users (our old-model reference on 4,979 unseen users: mean 0.845, median 0.840, p5 0.750, p1 0.713; his earliest era is his most rankable). Ten candidate mechanisms were eliminated by code reading; a README section now explains scoring one foreign collection. **The injector (nudges + compaction) is back in normal use from 22:30; the 2-hourly cron keep-alive is deleted.**

**⟶ 2026-09-16 16:50 -- THE PUBLISHED DATASET MOVED TO F: (Andrew: \"Move it\").** `C:\Users\Andrew\anki-revlogs-10k` (29,934 files, 14.593 GB; used by nothing in the -id lineage, read by srs-benchmark's runs and the only source for a published-lineage db rebuild) was robocopied to `F:\anki-revlogs-10k` (paused during lfcmf2's eval, since both read the F: HDD; 07:46-16:43 wall-clock, ~3 h of copying), verified (file count, every file's size, 300 sampled sha256: 0 mismatches), the C: copy renamed, **the C: path is now a JUNCTION to the F: directory** (read through it verified: 29,934 files, sample hash equal), the old copy deleted. C: 313 GB free. `anki-revlogs-10k-id` (the live -id source) stays on C:.

**⟶ 2026-09-16 15:20 -- DECIDED (Andrew, after lfcmf2: "Seems like grafting in general doesn't work very well" ... "Yeah, let's not do grafts anymore"): NO MORE ARCHITECTURE GRAFTS.** Two function-preserving grafts on a trained WS, two nulls (cmixgraft engaged at 20-31% of trained norms: +0.000085 / -0.000039; lfcmf2 at ~2%: +0.000013 / -0.000009). Mechanism: zero-output-weight growth bootstraps only through the weights that are zero, inside a decay whose LR is already annealing, so new parameters get the smallest budget any parameter ever gets; a null under it bounds the capacity from below only, and there is no cheap capacity win. lflora16 runs to completion (cost sunk; a WIN would still be decisive) and then the family is closed. The clean capacity answer is link D's from-scratch d=128 at 10+2. A mid-WS graft (branch at w10lf_ws_55000, ~5 stable epochs for the new parameters) was offered as the only sensible graft variant and is NOT scheduled unless link D says size binds AND Andrew asks. Scope: this closes ARCHITECTURE grafts (capacity grown on a trained checkpoint); PROPOSALS' "graft policy" (combining measured sub-bar LEVERS in one run) is a different mechanism and is untouched.

**⟶ 2026-09-16 14:48 -- lfcmf2 IS IN: A WEAK NULL, BARELY ENGAGED.** The card channel-mixer graft (factor 1.0 -> 2.0, +12,802 params, state unchanged) on w10lf's WS + w10lf's decay: ahead +0.000013 (p better 0.25) / imm -0.000009 (p better 0.87) vs w10lf, n=2499, size 0, 0 tracebacks -- both inside the noise floor. The new W_v columns ended at norm 0.39 vs 20.4 for the old ones (~2%): the grown units were hardly used in a 2-epoch decay, so this bounds card channel-mixer capacity from below only (PREREG P4). Recorded: `research_log.jsonl` `lfcmf2`, `research_5k.md`, verbose "ENDGAME: lfcmf2". **Link G moved on by itself: lflora16 grafted, smoked, dry-run clean, decay started 14:09** (~8 h + ~8.5 h eval -> ~06:30 on 09-17), then `verdict_cardgraft.py` on the pair, then the R1 link. The chain watcher (74920) died with the crash; the 2-hourly cron tick covers it.

**⟶ 2026-09-16 12:15 -- DISK (Andrew: \"Used -> move, Unused -> delete\"): C: was at 19 GB free (it was 71 GB the evening before; not this session's files -- Program Files and other sessions' temp grew). Deleted `C:\rwkv_lmdb\test_db_5k_e2s` and `train_db_5k_h1_e2s` (103 GB real each; exclusive-open check first, then the dangling F: junctions) -> C: 229 GB free. Only `train_db_5k_h1_id5` remains on C: (the live train db; F: would cost 2.2x per step). Also in flight: `C:\Users\Andrew\anki-revlogs-10k` (14.6 GB, the published dataset, unused by the -id lineage) is being moved to F: with a junction left at the old path (robocopy pid 115716, suspended until lfcmf2's eval ends because both read the F: HDD). The C:-to-F: LMDB move tool was NOT written (the delete made it moot).**

**⟶ 2026-09-15 22:50 -- DISK CLEANUP (Andrew: "Delete the KD dump, stale checkpoints and old temporary folders"): C: 27 -> 55 GB free.** Deleted: `C:
wkv_kd_dump` (15 GB; the end-to-end teacher dump, unusable since KD was skipped and the dbs moved to e2s -- regenerate from a teacher if KD ever returns); every `.pth` in 25 CLOSED run dirs (2,049 files: `track2_a1`..`a17`, `track2_reanchor`, `baseline_gru`, `baseline_lstm`, `iter33_dur`, `speedval`, `budgetcal`, `seedpair65k`, `tuner65k/*`, `qat_tax` -- runners, logs, tomls and catalogs kept; the numbers are in the record); Claude session temp dirs older than 2026-09-12 and my smoke temp dirs. KEPT: every champion checkpoint (`iter53_muonlora`, `champ5k_b1`, `iter29_muon`), `track2_a18` (the trunk), the whole lineage `iter31..iter69`, `realcyc`, `gen4_base`, `featB`, and the endgame (`ws10`, `w10lf` incl. the lfd*/graft decays, `w10plain`, `w10qat`, `w10clk`). ⚠ Not mine and not touched: `C:\pagefile.sys` (36.5 GB, system-managed, grew at the 20:02 boot; only a reboot shrinks it), `C:\Temp\Rar$DRa23768.3457.rartemp` (18 GB, a WinRAR temp folder), `C:\Temp\claude\C--Users-Andrew\9d2a39df...` (13 GB, another Claude project's session from today), `C:\Windows\MEMORY.DMP` (4.2 GB, the 08-17 dump). `du` over C: is minutes -- use the targeted listing in this entry's commit, not `du -sh /c/*`.

**⟶ 2026-09-15 20:48 -- lfd1 IS IN: L* = 1.** lfd1 - w10lf: ahead +0.000278 (p better 7.3e-08 / worse 1.0e+00) / imm -0.000049 (p better 1.0e+00 / worse 2.9e-24), n=2499, size 0, 0 tracebacks. With lfd3 (+0.000248 ahead worse) the decay-length curve after the 10-epoch WS is flat-to-falling (1 ~ 2 > 3). **L* = 1** (rule 2; the chain's pick_lstar.py applies the same rule with an ambiguity grid -- see lfqat/LSTAR.txt). The chain moves on: card grafts (link G) -> R1 at L* -> the QAT arms at L*. Recorded: `research_log.jsonl` `lfd1`, `research_5k.md`, verbose "ENDGAME: lfd1".

**⟶ 2026-09-15 20:20 -- THE PC WAS REBOOTED AT ~20:02 (Andrew: "absolutely dying for some reason"); RECOVERED.** lfd1's eval had banked 2,481 of 2,499 users in `result/RWKV-lfd1-s0.jsonl` (last user logged 7483 at 20:00); every waiter and the chain watcher died. The flight recorder shows the GPU going idle at 19:52 (VRAM 10.1 -> 2.2 GB, util 0) with RAM at 36 of 64 GB used and CPU spiking to 100% at 19:50 -- the eval was on the giant users at the end of the range with ~28 GB free system RAM, i.e. the WDDM-pool shape of the 2026-08-17 freeze (the rule says a giant-user eval wants ~40 GB free), not RAM exhaustion; cause not proven. **Recovery (WMI-free where needed):** after the boot every CIM call hung for ~15 min (36 stuck powershell.exe, `Invoke-CimMethod` and `Get-Process` alike -- the WMI repository re-verifying after a dirty shutdown), so `detach.ps1` calls stalled; the Task Scheduler was verified as a WMI-free launcher (`schtasks /create ... /sc once` + `/run`, a probe task wrote its file in 5 s) but WMI recovered before it was needed. **Chain re-armed 20:17-20:19:** links G (pid 31632), R1 (31296), A v3 (30688), B (29540), B2 (29200), C (28256), D (118256) detached from their unchanged files (all logged "armed"); **`scratchpad/lfd1/run_lfd1_evalresume.cmd`** (`mk_evalresume.py`: run_lfd1.cmd's env + phase C only; eval_sharded skips the banked users; lfd1.log appended, not truncated) runs under **`finish_decaylen.cmd`** (pid 54088), which then writes into `lfd3/wait_lfdecay.log` the two lines the dead decay-length waiter would have written (or DONE_EXIT_230/231 on failure, so link G refuses). Started 20:19:50; ~18 users, the giants, remain. Then `record_lfd1.py` (ready) and the chain continues as before. Disk after the boot: C: 75 GB free, F: 159 GB.

**⟶ 2026-09-15 14:35 -- DECIDED (Andrew: "let's do R1 only. R2 will be postponed until you think it's time").** R1 stays armed as the link before the QAT arms; R2 (monotone-by-design head) is POSTPONED at my discretion -- its only remaining case is deploy speed (one forward pass instead of four probes), so the time for it is after phase 5 at the earliest, and only if the Rust deploy path needs the speed. PROPOSALS R2 updated.

**⟶ 2026-09-15 14:20 -- R2 IS SIZED BEFORE IT IS BUILT, AND ITS ACCURACY CASE IS GONE.** `scratchpad/r2/probe_buttons.py` captured the four button-probe curve values per scored review (before and after PAVA) on w10lf's decayed checkpoint, CPU bf16, 40 moderate-size VAL users, 321,332 scored rows (the rebuilt per-user LogLoss matches the recorded jsonl within 1.7e-4). PAVA's pooling cost on these users is **+0.0010 by-user ahead** and it is CONCENTRATED: 2.9% of rows (pressed value moved by > 0.05) carry 83% of it, and 61% of it sits on HARD-pressed rows where the model put the Hard curve ABOVE the Good curve (10.6% of rows, mean excess 0.09) and was right -- pooling it down raised the loss. Pooling helped on 8% of rows, hurt on 27%. **So the cost is information the monotone constraint forbids, not a pooling artefact; a head monotone by construction cannot express it either, and R2's accuracy ceiling is ~0.** What R2 still buys is deploy speed (one forward pass instead of four probes). Recommendation: not before the QAT arms; Andrew decides whether it runs after phase 5 for the speed win. Recorded: `research_log.jsonl` `r2probe` (screen), `research_5k.md`, verbose "SCREEN: r2probe", `scratchpad/r2/FINDINGS.md`, PROPOSALS R2. lfd1's decay ended 14:10 (DECAY_OK), its eval runs (ends ~18:15).

**⟶ 2026-09-15 12:48 -- R1 (THE IMPUTED PROBE DURATION) IS BUILT, SMOKED AND ARMED AS A LINK BETWEEN THE CARD GRAFTS AND THE QAT ARMS (Andrew: "R1, imputed probe duration -> slot it before the leak-free QAT; R2, monotone by construction -> after that").** `rwkv/probe_impute.py` (`RWKV_PROBE_IMPUTE_K=3`; unset = byte-identical): the imm query row and the four PAVA probe rows of a review take `w * median(this card's PAST durations) + (1 - w) * median(this user's PAST durations)`, `w = n_card / (n_card + 3)`, both RUNNING medians over the real rows strictly earlier in the frame (no whole-table statistic; a card with no history takes the user median, a user with none keeps 0.0; medians restart at a training chunk boundary, like the RNN state). Hooked in `prepare_batch.prepare` (train / validation / eval, one scan per chunk, `ProbeMeta.pick` maps a probe target back to its pre-insertion row) and in the deploy path `run_as_rnn.imm_predict` / `process_row` (`OnlineImputer`: value before the review, push after it); the Anki builder's rule is `DEPLOY_FUNCTIONS.md` section 6 (two running medians per collection). **Smoke `scratchpad/r1/smoke_probe_impute.py` PASS**, six parts: inert = bit-identical to the pre-hook prepare_batch on three gen-5 chunks x three densities, clock fix off AND on; the query values equal an independent numpy recomputation at 0.0 deviation; the four probes carry exactly their query row's value; prefix invariance (truncate after m rows, nothing at <= m moves); end-to-end only query/probe rows change and only in scaled_duration; deploy = LMDB on 6,313 of 6,313 query rows with bf16 inputs (exact inputs differ by one bf16 ulp of the medians' magnitude, max 7e-3); cost 0.03 s per 16k-row chunk. **The run:** `mk_lf_branch.py r1 <L>` -> `lfr1e<L>`, a plain L-epoch decay from w10lf's WS with the flag in the decay, its validation and the rectified full-VAL eval (guards 58/59 = the ON banner + the chunk-count line; a non-fatal eval-banner post-check whose marker file makes the verdict REFUSE); L = L*, so the control is the plain twin at L* and the R1 model can be the QAT arms' probe base and KD teacher. **Gate = the curve-side rule** (`scratchpad/r1/verdict_r1.py`, 8 synthetic cases: ahead >= +0.0001 at p < 1e-4 AND imm not significantly worse) -> `scratchpad/r1/R1.txt` on/off; **`mk_lfqat.py` now REFUSES without R1.txt**, and ON puts the flag into every QAT arm (decay, probe, eval; guards 86/87/88, absent-guard 89 when off) with `lfr1e<L*>` as the twin. **Chain surgery:** link A v2 (16924) stopped in gate 0 (a note in its log, no marker); **the R1 link = `scratchpad/r1/wait_grafts_then_r1.cmd` (pid 39912, log `r1/wait_r1.log`, `mk_wait_r1.py`, 16 stub scenarios PASS)** fires on link G's DONE_EXIT_0, refuses on a pre-existing R1.txt (194), picks L* (LSTAR.txt or pick_lstar; AMBIGUOUS -> L = 2), regenerates, preflights, dry-runs, runs, judges; any failure AFTER the gates SKIPS the rest and writes R1.txt = off with the reason, so an unmeasurable R1 never blocks the deploy arms; **link A v3 = `lfqat/wait_r1_then_lfqat.cmd` (pid 23420, same log)** follows it. ⚠ Found on the way, independent of R1: the b13 KD arm's checkpoint gate sat 43 lines after its training call, so link B2's preflight would have REFUSED it at fire time; the live-KD / pqcap / R1 guards now sit after the gate and all ten arm runners PASS. ⚠ Two cmd traps the stub driver caught: `echo on>`/`echo off>` toggle echo instead of printing (use `echo(on`), and `%L%>>` with L=2 is `2>>` = a stderr redirect (space before the redirect; the generator now asserts it). PREREG `scratchpad/r1/PREREG.md` (P1 ahead +0.0003..+0.0010, P2 imm -0.0002..+0.0006, cost ~14 h at L = 2). R2 follows R1 (a CPU design pass first). Chain: lfd1 (step ~6,800 of 10,935 at 12:45) -> grafts -> R1 -> QAT arms -> bits -> budget curve -> grid.

**⟶ 2026-09-15 10:40 -- lfd3 IS IN: A 3-EPOCH DECAY HURTS AHEAD (+0.000248 vs w10lf, worse on 55% of users) AND IS NULL ON IMM (-0.000018).** PREREG P1 fails on the wrong side, so L* is 1 or 2; lfd1 (1-epoch decay, started 10:33, first decay on 2 fetch workers) decides by rule 2 at ~18:30. The direction matters beyond the length: more identical epochs now make ahead WORSE at 558k -- the shape the params x epochs grid was armed to explain. Recorded: `research_log.jsonl` `lfd3` (rejected, pair pending), `research_5k.md`, verbose "ENDGAME: lfd3".

**⟶ 2026-09-15 02:55 -- RAM (Andrew: "RAM usage is very high again").** 0.4 GB available of 64: lfd3's trainer (10.6 GB private) + 4 decay fetch workers (3.2 GB private + ~9 GB mapped each) + Andrew's FSRS-7 (12.7 GB) + desktop (~10 GB) + an ORPHAN fetch worker from 2026-09-12 (pid 41080, parent dead, 8.2 GB private fully paged out, 20 CPU-hours spinning) -- killed. `write_decay_setup.py` now writes NUM_FETCH_PROCESSES = 2 for every future decay (was 4; override `RWKV_DECAY_FETCH_PROCESSES`; the override = 4 reproduces the old file byte-for-byte; bit-identical numerics). lfd3 keeps its 4 workers until it ends (~06:00); lfd1 and everything after run on 2.

**⟶ 2026-09-14 21:25 -- ARMED as LINK D behind the budget curve (Andrew: "Behind the current chain, I guess"): `scratchpad/pxe/wait_budget_then_pxe.cmd` (log `pxe/wait_pxe.log`, `mk_wait_pxe.py`) fires on `lfqat/wait_lfbudget2.log`'s DONE_EXIT_0 (else refuses 220), then per cell preflight -> dry twin -> real run (lfb1, d128e10, d128b1; a failed dry or preflight SKIPS the cell, never a marker), then `verdict_pxe.py`. Stub paths: refuse, all-skip. Phase 5 is launched by hand after link D. THE GRID (20:45):** THE PARAMS x EPOCHS GRID IS BUILT (Andrew: "Knowing how n params and n epochs interact would be nice"; no augmentation; no KD from a d=128 teacher).** The ledger cannot split rows 1/2/13 (+0.0019 ahead / +0.0018 imm of the recipe's cost is unrecovered at 10x). `scratchpad/pxe/PREREG.md`: a 2x2 on ONE basis (gen 5, final layout, clock fix, KD off): 558k at 1+2 (`lfb1`, decay from `w10lf_ws_11000`, 300 users) and 10+2 (w10lf, done) vs d=128 (2,575,766 params, MAX 32768) at 1+2 (`d128b1`, decay from `d128e10_ws_22000`) and 10+2 (`d128e10`, full VAL). Runners `scratchpad/lfb1`, `d128e10`, `d128b1` + dry twins (`mk_pxe.py`, `mk_lf_branch.py budget 11000`); d128e10 + dry preflight PASS, d128b1 fails only on the not-yet-existing WS checkpoint; `mk_dryrun.py` grew `MKDRY_GROUPS` / `MKDRY_MAX` (old == new byte-identical on w10lf). Cost ~5 GPU days (d128e10 ~4 d); placement is Andrew's call. Predictions in the PREREG (P2: if size binds, d=128 beats w10lf by 0.0020-0.0035 ahead at the full budget). **Rectifier levers queued in `optimization/PROPOSALS.md`** (R1 imputed probe duration = running medians of PAST durations only, R2 monotone-by-design head).

**⟶ 2026-09-14 19:06 -- THE SECOND-REVIEW ABLATION IS IN: creation -> first review is worth 3.8x / 2.5x MORE on a card's second review than on all rows.** w10clk's checkpoint, users 5001-5300, per-user mean over each card's second review inside the equalized set (1,907,863 rows): zeroing the feature costs **+0.000716 ahead (62% of users worse, p=4e-10) / +0.000580 imm (69%, p=7e-17)** against +0.000189 / +0.000229 on all rows (ablcf). Both row dumps reproduce the recorded per-user numbers at max |d| 1e-6 (the harness is exact). Reliance, not value; the feature stays. Recorded: `research_log.jsonl` `ablfr` (screen), `research_5k.md`, verbose "SCREEN: ablfr", `INPUT_FEATURES.md` row 23. **The decay-length waiter v2 fired at 19:07** (preflight PASS, dry runs lfd3dry / lfd1dry, then lfd3 ~12 h + eval, then lfd1).

**⟶ 2026-09-14 18:05 -- ★★ w10lf IS IN: THE HONEST LEAK-FREE 10+2 NUMBER ON THE SHIPPED BASE IS ahead 0.297755 / imm 0.264195** (n=2,499, size 0/2,499, nan 0, 0 tracebacks; WS 0.913 steps/s, 33.3 h; params 561,732, final 103-input layout). vs arm 1 (leaked): +0.000178 / +0.002129, inside the PREREG bands. **vs w10clk (leak-free DECAY only): a TIE (-0.000046 p 0.42 / +0.000034 p 0.02)** -- training the WS without the leak recovers NOTHING beyond the decay-only fix, so the leak's value after full adaptation is +0.0021 imm / +0.0002 ahead, and the layout change is ~0 as predicted. Neither stop criterion is met (imm short by 0.0002); worse than the old d=128 model by +0.0031 / +0.0006. Recorded: `research_log.jsonl` (`w10lf`, baseline), `research_5k.md` row + THE LEDGER row 14 / bottom line, verbose "ENDGAME: w10lf", PREREG verdict. **The chain continues on w10lf's WS: the second-review ablation started 18:07 (~1 h), then the decay-length waiter v2 fires (lfd3dry, lfd1dry, lfd3, lfd1), then the card grafts, then the leak-free QAT arms.**

**⟶ 2026-09-13 07:20 -- THE SECOND-REVIEW ABLATION IS SLOTTED IN RIGHT AFTER w10lf (Andrew: "do the 'zero out the card creation date -> first review feature' eval again, but calculate log loss only on first reviews").** A FIRST review is never predicted: `data_processing.add_queries` builds no query row for `is_first_review` (so no imm), no earlier row of the card exists for an ahead target, and the benchmark drops it anyway (`delta_t > 0`). The closest event is each card's SECOND review -- the first the model predicts, where creation->first-review is the freshest fact about the card. That needs per-ROW predictions, which `get_result` discards, so **`scratchpad/ablate_cf/eval_rows.py`** (the same forward, every labelled row kept per user as `.npz`; its equalized per-user LogLoss reproduces the recorded numbers: CPU smoke on user 5001 vs w10clk / ablcf at max |d| 2e-6 / 1.3e-5) and **`score_second.py`** (second reviews from the -id parquet: `elapsed_days == -1` then the card's next row; `review_th` = parquet row + 1, verified against the LMDB's end_th; reports EQ = inside the equalized set, rectified, PRIMARY, and ALL; per-user paired Wilcoxon + row-pooled). **Runner `run_ablate_fr.cmd`** (`mk_ablate_fr.py`): w10clk's checkpoint and env, users 5001-5300, control arm then the ablated arm (guards 11-24: banners, clock fix, eval_rows exits 3 on a missing user), ~1 h of GPU. **Placement, by waiter surgery:** `wait_lf_then_fr.cmd` (pid 16860, log `ablate_cf/wait_fr.log`) polls the w10lf waiter's log; on `DONE_EXIT_0` it runs the ablation then writes its own `DONE_EXIT_0`, any other code is forwarded verbatim without running; the decay-length waiter v1 (37544) was stopped in gate 0 (cmd first, note in its log, no marker) and **v2 (pid 33764, `mk_wait_lf_decaylen.py <prev log>`, SAME log `lfd3/wait_lfdecay.log`) polls `wait_fr.log`**; link G downstream untouched. 13 stub paths executed (one caught a real bug: a `find /c /v "" < file` user count BLOCKS on a console -- moved into eval_rows.py), preflight PASS, 70 s foreground each, detached. The chain is delayed ~1 h. Verdict: `ablate_cf/fr_verdict.txt` when `ablate_fr.log` has `DONE_EXIT_0`; record it (research_log screen row `ablfr`). **Also queued for the next LMDB rebuild (Andrew): the count of distinct cards reviewed at least once at review time** -- `FUTURE_FEATURES.md` top entry (it is `cum_new_cards`, already computed at `data_processing.py:364` and shipped only as differences).

**⟶ 2026-09-13 00:50 -- PHASE 5 TOOLING, built on CPU while w10lf trains (`optimization/PHASE5_PLAN.md`).** `scratchpad/phase5/mk_trial.py <incumbent runner> <knob> <value>` = one-knob QAT-decay trials (lr = Muon AND AdamW together, cblr, kd, wd) off the adopted arm, with run-time guards on the trainer's own echo of the value (exit 90-94), the 1000-user tune-eval subset, and the validation-trace early stop (deltas 0.0015 both modes, from step 6000, persist 2; exit 42 -> `DONE_EXIT_42` + `pruned.json`). **`rwkv/train_rwkv.py` changed by ONE gated line: `RWKV_VPRUNE_DECAY=1` lets the existing val-prune run in a decay** (unset = byte-identical, py_compile PASS; the running chain never sets it). `mk_vprune_ref.py` builds the reference from the incumbent's decay log. `mk_dryrun.py` now matches the runner's peak LR (was hardcoded `1e-3`; old == new on w10qat). 16 stand-in runners (`p5*`, from lfq2/lfqkd2) pass preflight; regenerate from the adopted arm at phase start. **01:00: the descent itself is built too** -- `tuner.py` (init / next / record / last / status; coordinate descent, promotion at raw >= 0.0001 both modes on the subset, tested on synthetic journals) and the loop `run_phase5.cmd` (`mk_run_phase5.py`; three terminal paths executed). **It is launched BY HAND after the chain's verdicts** (`tuner.py init <adopted arm runner>` then detach `run_phase5.cmd`), because choosing the adopted arm is the verdict step. Plus `mk_confirm.py gen|merge <winner>` (01:05): the confirmation evaluates the winner's existing checkpoint on the REST of the VAL half (6001-7500, ~6 h) and merges with its subset rows for `paired_pvalue --intersect`. Everything for phase 5 is now on disk except the adopted arm.

**⟶ 2026-09-12 21:30 -- THE CREATION -> FIRST-REVIEW ABLATION IS IN: A SMALL, REAL RELIANCE.** w10clk's checkpoint on users 5001-5300 with `scaled_creation_to_first_review` zeroed (= its training mean), same eval env: **+0.000189 ahead (worse on 61% of users, p=7e-06) / +0.000229 imm (worse on 71%, p=2e-18)**, size 0/300, 0 exceptions. ~1/7 (ahead) and ~1/22 (imm) of `t_since_any_review`'s reliance on featB. RELIANCE, not value: a retrained model recovers part of it, so ~+0.0002 is the CEILING of what removing it costs. The feature stays in the final layout. Recorded: `research_log.jsonl` `ablcf` (screen), `research_5k.md`, verbose "SCREEN: ablcf". **The w10lf waiter v4 then fired at 21:29: final-layout smoke PASS, runners regenerated, preflight PASS, dry run 21:33-21:58 clean (WS_OK / DECAY_OK / EVAL_OK, 0 exceptions), and w10lf STARTED 21:58** -- params 561,732, clock fix ON in every worker; step 8,226 at 00:27 = 0.92 steps/s (ws10 ran 0.80), so the 109,350-step WS ends ~2026-09-14 07:00 and the decay + eval ~2026-09-14 16:00. Chain watcher re-armed after a session restart; all seven downstream waiters alive.

**⟶ 2026-09-12 21:05 -- cmixgraft IS IN: A WEAK NULL.** ws10 + the six user/preset channel mixers restored (zero-init graft) + arm 1's decay: **ahead 0.297492 (+0.000085) / imm 0.262105 (-0.000039) vs arm 1**, n=2,499, size 0, Wilcoxon p 0.95 / 1.0 (47% / 39% of users better). Engaged (restored W_v at 20-31% of trained norms), P1 fails, P4 fires; the ahead mean sits in the smallest users, the largest quartile gains nothing. Bounds coarse-stream capacity from below only; the CARD stream (where the d=128 gap grows) is `lfcmf2`/`lflora16`'s question. Recorded: `research_log.jsonl` `cmixgraft` (rejected), `research_5k.md`, verbose "ENDGAME: cmixgraft", PREREG verdict. The chain moved on: ablation eval (21:03) -> w10lf waiter v4.

**⟶ 2026-09-12 20:30 -- ★ THE RUST ENGINE NEEDS `RWKV_STREAM_ORDER=card,note,deck,preset,user`, AND TWO CATALOG CORPORA WERE DUMPED WITHOUT IT.** The engine does not read the stream order from the weights; it defaults to the ORIGINAL card,deck,note order, and the `_cnd` trunk runs card,note,deck. Without the variable: Rust parity FAILS even on the certified realcyc trace (max 0.52; with it, 7.187e-07 = the certified run exactly), the note corpus is another stream (3,612 states for user 156 instead of 903), card states 15.5% off. **The 2026-09-08 gen-5 and 2026-09-10 ws10 corpora were dumped that way** (the 2026-08-12 dumps set it). Re-checked on a correct dump: the gen-5 stale verdict on the old catalogs STANDS; the replacement WKV catalogs work but are ~20-25% worse than a correct refit (gen5 0.6046 vs 0.4845; ws10's on the stand-in 0.6359 vs 0.5105) -- **so arm 2 STARTED from a sub-optimal WKV catalog** (it learns during QAT, so the effect on its tax is unmeasured). Fixed for the leak-free line: `mk_lf_fit_catalogs.py` sets the variable and refuses (68) unless `scratchpad/lfqat/check_corpus_order.py` finds the engine's stream-order line, in the Python model's order, atop every corpus file; a temp copy of the whole refit ran end to end on the stand-in (CORPUS ORDER PASS, `DONE_EXIT_0`). **Rust parity on the 103-input layout: PASS** (imm 0.000000 / ahead 0.000000, max 3.53e-06; stand-in weights, clock fix on). Addenda: `scratchpad/qat_gen5/FINDING.md`, `scratchpad/lfqat/PREREG.md` (lfq minus arm 2 is now confounded by three changes). **✓ DURABLE FIX DONE 20:15: the order now TRAVELS WITH THE WEIGHTS.** `export_rnn_trace.export_weights` writes `stream_order` (the exported model's own `stream_names`) into the safetensors METADATA; `rust/rwkv-infer` `Model::load` reads it (`safetensors_metadata`): metadata only -> used; env + metadata -> must AGREE or the engine refuses (exit 1); env only -> today's behaviour; neither -> the historical default plus a WARNING on stderr (new messages go to stderr only: stdout is captured into corpus files). Verified: old files byte-identical to the pre-change binary with and without the variable (realcyc 107 still 7.187e-07); a metadata export (tensors bit-identical to the old one) gives PARITY PASS on `reference_final_inputs` WITHOUT the variable (0.000000 / 0.000000, max 3.53e-06); a conflicting variable is refused; a corpus dump still passes `check_corpus_order.py` (note 903 states). Live binary rebuilt in place (backup of the old exe in the session scratchpad). **★ 20:25, closed the rest of the class:** `RWKV_INTERLEAVE` is the same failure (order set, flag missing: realcyc 107 misses by up to 0.66 per review), so the export now ALSO writes `interleave`, `interleave_placement` and `state_clamp_tau` (read off the exported model object), with the same precedence in `detect_arch`; a `spread` placement is refused (never ported). Verified: old files byte-identical to the pre-change binary in all four env combinations; a new export gives PARITY PASS with NO recipe variable at all; conflicting interleave / clamp values are refused (exit 1); the chain's corpus dump + `check_corpus_order.py` pass with identical state rows. **So a weights file exported from now on describes its own recipe -- which the Anki port needs, since Anki sets no environment.** Files exported before 2026-09-12 (every `reference*/` dir) still need the variables.

**⟶ 2026-09-12 19:50 -- ★ w10lf NOW TRAINS ON THE FINAL INPUT LAYOUT: 103 inputs, params 561,732 (Andrew: "both ❌ and ⚠️ should be removed ... the timing of the rebuild is up to you").** NO rebuild: a side-by-side one needs ~300 GB (C: 109 / F: 185 GB free). Instead `RWKV_FINAL_LAYOUT=1` (`99ad334`) drops 6 duplicate card-feature columns as `prepare_batch.get_data` reads a gen-5 row; every consumer resolves by name. **Every audit claim was re-measured first, and one failed: `is_default_deck` is NOT dead** (the audit's "always zero" was 17 users; on 100 users it is set on 14, up to 100% of one user's rows, and it is the only thing that tells the default deck from a new deck) -- **KEPT**. Dropped: `deck_id_is_nan` (= note flag by construction), `preset_id_is_nan` (differs on 0.0015% of 727.7 M reviews, `preset_join_scan.py`), the decade/century `_first` cycle halves (r 0.99). `smoke_final_layout.py` 50/50 PASS (inert when off, exact slice when on, clock fix on and off, 103-input model, deploy vector); param guards MEASURED (`param_counts.py`: base 561,732, cmf2 574,534, lora16 574,212); graft path PASS on a final-layout stand-in (`make_standin.py`). Every w10lf-line runner regenerated with the flag (lfd3/lfd1/lfb2/lfb5, lfq*/lfqkd*, lfcmf2/lflora16; `run_fit_catalogs_lf.cmd` now uses `reference_final_inputs/` and refuses without its check, 67). **w10lf waiter v4** `scratchpad/w10lf/wait_ablate_then_lf_v4.cmd` (pid 26648, log `wait_lf2.log`): smoke at fire time (157), regenerate + preflight, runners must carry the flag and the 561732 guard (158), dry run, then w10lf; 7 stub paths, 85 s foreground. **While slicing the traces I found §9 case 6 (the deploy feature builder had drifted; fixed in `run_as_rnn.py`, `DEPLOY_FUNCTIONS.md` section 4).** `INPUT_FEATURES.md`: final table = 103 inputs, removed columns listed with evidence.

**⟶ 2026-09-12 18:40 -- CREATION -> FIRST-REVIEW ABLATION SLOTTED IN BEFORE w10lf (Andrew: "zero out the Creation -> first review interval feature and do an eval").** `scratchpad/ablate_cf/` (`mk_ablate_cf.py`): w10clk's checkpoint on users 5001-5300 in w10clk's exact eval env + `RWKV_ABLATE_FEATURES=scaled_creation_to_first_review` (0 = its training mean); control = w10clk's own per-user results (phase L showed a subset eval reproduces them exactly); guards 11-17 incl. the `[feat-mask]` banner; `verdict_ablate_cf.py`. RELIANCE, not value. **Waiters:** `wait_cmixgraft_then_ablate.cmd` (pid 30456, log `ablate_cf/wait_ablate.log`; forwards a cmixgraft-waiter 90 unchanged) -> **w10lf waiter v3** `w10lf/wait_ablate_then_lf.cmd` (pid 16568, SAME log `wait_lf2.log`; v2 = 16676 stopped in gate 0). ~30 min added before w10lf. 7 stub paths. **Also: `INPUT_FEATURES.md` now marks ❌ what the final model does not use and ⚠️ what is dead or a duplicate, and has the final 109-input table (checked against the code).** Andrew asked whether a rebuild is needed because a useless feature is zeroed instead of removed: it is NOT (gen 5 removed `state` at the source; `RWKV_ZERO_FEATURES` is empty); the audit's dead/duplicate columns would need one, and I recommended against it before w10lf (answer pending).

**⟶ 2026-09-12 15:35 -- ★ TWO CARD-STREAM CAPACITY TESTS ARE ARMED (Andrew: "Yep, do 2 tests"; "Make sure to do a small test run first").** Why: asked whether architecture is still worth pursuing, a per-user check (w10clk vs the old d=128 model, 2,482 VAL users) showed d=128 better on ahead for 77.8% of users, a gap nearly flat in user size but GROWING with reviews per card (ahead +0.0027 Q1 -> +0.0042 Q4, imm +0.0002 -> +0.0011) -> the card stream. Both tests add capacity that is NOT stored per card: **`lfcmf2`** = card channel-mixer factor 1.0 -> 2.0 (+12,802 params) and **`lflora16`** = card LoRA ranks 4/4/2/4 -> 16/16/8/16, the d=128 original's (+12,480; I had proposed 8, the larger dose makes a null more decisive). **Both are function-preserving GRAFTS on w10lf's WS** (`scratchpad/cardgraft/`: `graft.py` builds the grown model with the real code, copies weights and optimizer moments by name -- grown tensors get the old moments in their slice; ⚠ the GATE LoRA needs its new OUTPUT columns zeroed because sigmoid(0) = 0.5; `smoke_graft.py` on the deploy path: cmf2 bit-identical, lora16 2.4e-07 = float32 summation order, perturbed controls 1.8e-07 / 5.0e-06), so each costs a ~10 h decay, control = w10lf. Tested on ws10's WS: both graft PASS (the verification child loads the state and steps). **LINK G = `scratchpad/cardgraft/wait_decaylen_then_grafts.cmd`** (pid 27132, log `cardgraft/wait_grafts.log`): after the decay-length pair -> per graft: graft, smoke, preflight, the dry twin (0.05 ep, 20 users), the arm; a failed graft is SKIPPED (a log line, never a marker) and the chain goes on; then `verdict_cardgraft.py` (WIN = both modes, AHEAD-ONLY = Andrew's call, NULL/HARM; engagement = norm of the grown region). **Link A was stopped in gate 0 and replaced by link A v2 `scratchpad/lfqat/wait_grafts_then_lfqat.cmd`** (pid 16924, same log `wait_lfqat.log`, which held no marker), which follows link G. 11 link stub paths + 4 runner stub paths, preflight PASS, 80 s foreground. PREREG `scratchpad/cardgraft/PREREG.md` (a WIN is decisive, a NULL weak). If a graft wins, the QAT arms need one extra arm on the grafted architecture. **Phase 5 scope decided the same day (plan table).**

**⟶ 2026-09-12 09:55 -- ★ w10clk IS IN: THE HONEST LEAK-FREE 10+2 NUMBER (leaked WS, leak-free decay + eval) IS ahead 0.297801 / imm 0.264160** (n=2,499, size 0/2,499). vs arm 1 (leaked): +0.000224 / +0.002095, both inside the pre-registered bands; on phase L's users the leak-free DECAY recovered **30%** of the no-adaptation leak cost (ratio 0.70). **Misses both stop criteria** (imm by 0.00016) and is **WORSE than the old d=128 model on imm by 0.000575** -- the "beats d=128 on imm" claim is withdrawn for good. Deploy estimate if additive: 0.300330 / 0.268626. All five PREREG predictions held. Recorded: `research_log.jsonl` (`w10clk`, baseline), `research_5k.md`, verbose "ENDGAME: w10clk", PREREG verdict. **w10lf (the WS trained without the leak) is the shipped base and must beat +0.0021 imm vs arm 1 to show the WS mattered.** Chain continues: cmixgraft now.

**⟶ 2026-09-12 00:45 -- ★ THE BYTE GRANT IS SETTLED AND ITS ARM IS ARMED. Andrew: "I meant the first one [1-3 bits in total], but since it buys nothing, let's go with the second one" = +1-3 bits PER WKV CATALOG INDEX. Chosen: b13, once** (card 185 -> 215 bits, note 105 -> 120; the largest rung, so a null is decisive). **The kernel is PROVEN: `check_pqcap.py` PASS at 00:22** (a tiny co-tenant check beside w10clk, 5.5 GB VRAM free): 34/34 tensors bit-identical to the live build at b10, and b13 PAD/SHIFT catalogs built from the b10 one reproduce it exactly. **LINK B2 = `scratchpad/lfqat/wait_kd_then_bits.cmd`** (`mk_wait_lfbits.py`, pid 39396, log `wait_lfbits.log`): after link B's `DONE_EXIT_0` -> `pick_kd.py` applies the KD adoption rule (writes `KD.txt`; the bits arm's BASE is lfq or lfqkd, so the arm is one variable) -> `check_pqcap` gen/check again with the GPU free (212/213) -> `run_fit_wkv_lf_b13.cmd` (`mk_lf_fit_wkv_bits.py`: b13 k-means on link A's w10lf corpus, never touches the b10 refit; 287 s on ws10's corpus) -> `mk_lfqat.py L* [kd] b13` -> dry -> the arm. **The old link C (38728) was stopped in gate 0 and replaced by `wait_bits_then_budget.cmd`** (pid 9308, FRESH log `wait_lfbudget2.log`, follows B2). `mk_wait_lfqat.py` now REFUSES to run without `--fresh-chain`, because A and B are armed. 20 stub paths (incl. the check's env: gen with the override empty, check with it set), preflight PASS, 80 s foreground. PREREG P6-P9 + adoption rule appended to `scratchpad/lfqat/PREREG.md`; verdict `verdict_lfqat.py bits <L> <base>` (P8 counts moved centroids).

**⟶ 2026-09-11 23:50 -- THE LEAK-FREE QAT CHAIN IS BUILT AND ARMED (links A/B/C behind the decay-length pair); ANDREW GRANTED "1-3 MORE BITS" and its unit is being confirmed.** `scratchpad/w10lf/mk_lfqat.py <L> [kd] [b<N>]` = arm 2's decay runner + arm 2's learned-catalog eval runner (phases L and D dropped), re-based on w10lf's WS at decay length L, start catalogs `reference/pq_cb_*_w10lf_*` (refit by `scratchpad/lfqat/run_fit_catalogs_lf.cmd`), the clock fix in the decay, the probe and the eval (guards 56/57/79 + a non-fatal shard-log check), live KD ON (teacher = the plain twin at L; guards 80/81) or explicitly CLEARED (guard 82). Env asserted = arm 2's + exactly those changes, and = w10lf's on every shared key; stub-tested on 13 paths (`lfq2`, `lfqkd2`, both dry twins, every guard). **Chain** (`mk_wait_lfqat.py`; pids 16424 / 36188 / 38728; logs `lfqat/wait_lfqat.log`, `wait_lfqatkd.log`, `wait_lfbudget.log`): **A** fires on `lfd3/wait_lfdecay.log` -> `pick_lstar.py` applies `lfd3/PREREG.md`'s rule (AMBIGUOUS, measured by moving every bar to a small margin, runs the plain budget curve and stops at 173 for a human; tested on 4 synthetic outcomes) -> catalog refit -> dry run -> `lfq<L*>`; **B** = `lfqkd<L*>` (only after A's `DONE_EXIT_0`); **C** = the budget curve `lfb2`/`lfb5` whatever B says (skips if A's ambiguous branch ran it). 26 stub paths, preflight PASS, 80 s in the foreground, then detached. PREREG + KD adoption rule: `scratchpad/lfqat/PREREG.md` (P1-imm is the informative half: a leak-free imm tax near +0.0045 = the leak played no part in arm 2's tax). **THE BYTE GRANT, verbatim: "feel free to add 1-3 more bits to card/note state".** Card = 185 bits, note = 105 (`research_5k_notes.md` "DEPLOY STATE BUDGET"). Per catalog index (WKV b10 -> b11..b13) it is +10..+30 card / +5..+15 note bits; as a total it is +1..+3 bits, for which no lever is known (the norm levers are CLOSED). **Asked him which reading** before spending it. ⚠ **b > 10 needs a kernel change:** `g_pq_cb[32768]` / `g_pq_cb_grad[32768]` in `rwkv7_cuda.cu` hold 1024 x 32 floats = exactly b10, and the upload TORCH_CHECKs it. **PREPARED 00:05 (`3bc2f53`), NOT ARMED:** both arrays grown to `PQ_CB_MAX_FLOATS` 262144; built in isolation into `scratchpad/pqcap/lib` (BUILD_EXIT_0; the live `.pyd` hash 07412c40... is unchanged); `RWKV_CUDA_PYD=<pyd>` in `rwkv/model/__init__.py` loads it for one run (unset = the old import; verified on CPU, CUDA not initialised); `mk_lfqat.py <L> [kd] b13` sets it and guards it (85 / 83 / 84; 18 stub paths). ⚠ **OWED before any b>10 run: `scratchpad/pqcap/check_pqcap.py gen` (live build) then `check` (new build) in a GPU gap** -- b10 bit-identity plus b13 PAD/SHIFT catalogs that must reproduce b10. Also owed: a b13 WKV fit on w10lf's corpus that does not overwrite `run_fit_catalogs_lf.cmd` (link A calls it for b10). The bits arm = the adopted arm (lfq or lfqkd) + b<N>, as a link between B and C. Andrew restarted his FSRS-7 run with `CUDA_VISIBLE_DEVICES=-1` (23:12; no CUDA contexts now).

**⟶ 2026-09-11 21:14 -- ★★ ARM 2 IS IN: THE QAT TAX AT 10 EPOCHS IS +0.002529 ahead / +0.004466 imm** (deployed 0.300107 / 0.266532 vs arm 1's 0.297577 / 0.262066; n=2,499, size 0/2,499; 84.7% / 97.9% of users worse). Single variable vs arm 1, evaluated with the LEARNED step-21870 catalogs (probe-proven, no post-check NOTE). **Ahead's tax is budget-independent (1.25 ep: +0.002286); imm's GREW 28% (+0.003486).** All four PREREG questions held. **The deployed 10-epoch model is WORSE than the plain 1.25-epoch realcyc in both modes** -- the tax is ~5x the budget gain on ahead, ~3x on imm -- and misses the stop criterion by 0.0051 / 0.0025. **=> THE QAT TAX IS NOW THE PROJECT'S LARGEST COST.** Levers, in order: the phase-D bit split, QAT-KD on the leak-free base, then (Andrew: only if it is the only way) more card/note bytes. ⚠ Trained and scored WITH the clock leak, so part of the imm tax may be leak loss; the leak-free QAT arm separates it. Recorded: `research_log.jsonl` (`w10qat`, baseline), `research_5k.md`, verbose "ENDGAME ARM 2", PREREG verdict. **PHASE D (23:09): THE TAX DOES NOT DECOMPOSE BY REMOVAL** -- removing the shift quantizer throws the model OFF DISTRIBUTION (+0.035 / +0.043 on the probe, arm skipped); removing the WKV codebook makes it slightly WORSE (C = -0.000135 / -0.000246, p 1e-4 on imm): the learned quantizers CO-ADAPTED with the weights. The pre-registered falsifier fires (C < 30% of T): **zero-byte catalog routes exhausted, the bit-split A/B is DEAD, the leak-free QAT arms keep b10 / m2b12; remaining routes = QAT-KD (leak-free base), then Andrew's byte grant (rank-2)**. `scratchpad/qat_decomp/PREREG.md` verdict. **w10clk's dry run started 23:12:21** (T=1800, clock-fix ON in both workers).

**⟶ 2026-09-11 08:47 -- ★★ PHASE L: THE CLOCK LEAK IS MATERIAL -- imm +0.0032 / ahead +0.0003 on arm 1's checkpoint** (users 5001-5300: lkc30m - lkc0 = **+0.003152 +/- 0.000454 imm**, worse on 273/300, p=2.2e-47; **+0.000256 ahead**, worse on 198/300; the 3 h arm adds only +0.000129, so **T = 1800**). Harness exact (control = arm 1 at max |d| 0); both modes inside the pre-registered bands (`scratchpad/leak/leak_report.txt`). **ARM 1's imm CLAIMS ARE WITHDRAWN** (~0.265 at deploy-faithful inputs: misses 0.2640 and trails the d=128 model's 0.263586); its ahead claims move ~0.0003 and stand. It is the cost for a model TRAINED with the leak; a leak-free model recovers part of it. The features' whole imm gain (-0.0024) is SMALLER than this. Recorded: `research_log.jsonl` (`leakcf`, status screen), `research_5k.md` row, verbose "THE QUERY-ROW CLOCK LEAK", section 9 case 5.
**THE QUEUE WAS RE-PLANNED BY HAND at 08:55:** stopped the four leaked-base waiters -- QAT-KD (28684), budget curve (29148), decay length (36528) and the end-of-queue w10lf (3724) -- cmd first, no markers (a note in each log). **The chain is now: arm 2's eval (running since 08:50; probe OK on the learned catalogs; then phase D) -> `w10clk` at T=1800 (3644, automatic) -> cmixgraft (36904) -> `w10lf` (NEW: `scratchpad/w10lf/wait_cmixgraft_then_lf.cmd`, pid 16676, log `wait_lf2.log`, T fixed at 1800, refuses 150 on cmixgraft's 90; stub-tested on nine paths and run 75 s in the foreground).** cmixgraft stays ahead of w10lf because its question is ahead-side (the leak barely touches ahead), and a capacity win would change which architecture the leak-free base should have. **AFTER w10lf, on its WS, all with the flag -- ORDER (my call): the decay-length pair FIRST, then pick the decay length L*, then the catalog refit + a leak-free QAT arm at L* (the deploy QAT decay AND QAT-KD's control), then QAT-KD at L* (teacher = the leak-free decayed final), then the budget curve LAST** (it cannot change the shipped model). Decay length goes first so the QAT arms need not be re-run at a new length. **BUILT (`e8ebc8a`):** `scratchpad/w10lf/mk_lf_branch.py` -> `lfd3`, `lfd1` (decay length) and `lfb2`, `lfb5` (budget points from `w10lf_ws_22000` / `_55000`, 300-user eval, `verdict_budget.py lf`), each with a dry twin; each differs from w10lf only in decay length or branch step. Their preflight fails today ONLY because w10lf's WS checkpoints do not exist yet -- the waiter must preflight again at fire time. **ARMED 2026-09-11 11:40: `scratchpad/lfd3/wait_lf_then_decaylen.cmd` (pid 37544, log `lfd3/wait_lfdecay.log`, generated by `mk_wait_lf_decaylen.py`)** -- fires on `wait_lf2.log`; refuses 160 on its 150 (arm 2's eval failed), 161 if w10lf never ran, 162 unless `w10lf.log` has its own `DONE_EXIT_0` AND both final checkpoints exist; then preflights the four runners (163), both dry runs (164-169), lfd3, lfd1. Stub-tested on eleven paths, 75 s live. **(BUILT AND ARMED 23:42 -- see the 23:50 entry.)** Why re-base rather than keep the ws10 versions: quantizing card/note state can strip the precision the leak needs, so a leaked model's imm QAT tax can be too large; and part of the budget's +0.0015 imm may be the model learning the leak better. ⚠ **Read arm 2's imm tax with that caveat.**

**⟶ ARCHIVED 2026-09-12: the LIVE entries dated 2026-09-08 .. 2026-09-10 moved VERBATIM to `optimization/HISTORY.md` "CLAUDE.md LIVE archive (moved out 2026-09-12)"** -- ws10 and arm 1, arm 2's build and its incidents, the leaked QAT-KD chain (superseded by `lfqkd`), the leak discovery, the fix and w10lf's build, phase D's design, the feature audit, the architecture screens. **What they established that is still OPERATIVE:**
* **THE RUNS.** ws10 = the shared 10-epoch WS (`scratchpad/ws10/w10_ws_109350.pth`, 109,350 steps, 38 h, zero tracebacks). Arm 1 `w10plain` = plain 2-epoch decay + eval: 0.297577 / 0.262066 (FULL PRECISION -- see the naming trap in LIVE RULES); vs realcyc +0.000506 ahead / +0.001526 imm, i.e. **budget pays the rating head and barely the curve head** (12% of the projected +0.0042 on ahead). Arm 2 `w10qat` = the same decay + the QAT stack (21:14 entry). **w10lf** = ws10's WS REPEATED from scratch + arm 1's decay + eval, with `RWKV_CLOCK_AT_PREV_ANSWER=1800` in all three phases -- the base of the shipped model; ~47 h; PREREG `scratchpad/w10lf/PREREG.md` (imm +0.0005..+0.0030 / ahead +0.0000..+0.0006 worse than arm 1); dry run `scratchpad/w10lfdry/`. **w10clk** = arm 1 + the flag (leaked WS, leak-free decay + eval); PREREG + `verdict_w10clk.py`.
* **THE CLOCK FIX** (`rwkv/clock_fix.py`, `8914d1a`): query + probe rows moved to the previous answer for in-session gaps, a labelled real row's ahead t minus its target's gap; train/val/eval via `prepare_batch.prepare`, Python deploy via `run_as_rnn.imm_predict` + `clock_fix.ahead_t`, the trace exporter for Rust; Anki-fork rule `DEPLOY_FUNCTIONS.md` section 5. Default off = bit-identical (`scratchpad/leak/smoke_clock_fix.py`, five parts). The UTC-midnight channel is fixed; the Anki-day and creation-batch channels (~3e-7 nats per query row) are ACCEPTED (Andrew: "no LMDB rebuild"). delta comes from the bf16-stored gap (~1%, up to ~18 s at T=1800).
* **Q2, STILL ANDREW'S CALL: CAPACITY BINDS AT THE REAL BUDGET.** At 12 epochs the 4.95x reduction is free on imm and costs ~0.003 on ahead vs the old d=128 model; every earlier capacity reject was measured at ~1.25 epochs. Recovering it grows the per-card state. The one zero-byte direction is deck/preset/global capacity (gate #3 lets those grow): `stream_rank.py` gave a WEAK positive; **`cmixgraft`** (ws10 + the six user/preset channel mixers grown with zero-init output projections, bit-identical to ws10 at step 0, `expand_ckpt.py` + `expand_optim.py` which maps the optimizer state BY NAME; 641,862 params) **came back a WEAK NULL on 2026-09-12 (+0.000085 ahead / -0.000039 imm vs arm 1, engaged; see LIVE)**. ⚠ A WIN is decisive, a NULL is weak (2 epochs for new parameters); never quote a null as "capacity does not bind".
* **QAT FACTS.** QAT decays run ~0.25-0.33 steps/s (3-4x slower than plain), and the tax closes by ~step 2,000 and is flat after (arm 2's preview), so ~90% of a 2-epoch QAT phase buys no closure. QAT + `RWKV_QAT_COMPILE=1` needs `cl.exe` (runners `call vcvars64.bat` + a `where cl.exe` guard; without it inductor fails per batch and the run is HOLLOW); under QAT the compile flag looks like a net LOSS (0.25 vs 0.33 steps/s) -- worth a standalone A/B outside a gate-critical run. **A PQ catalog does not survive a budget change** (the realcyc-fitted WKV catalog scored 0.76 on ws10's states vs 0.39 on its own): refit on the model that deploys. ⚠ Both corpora behind those two numbers were dumped MIS-WIRED (no `RWKV_STREAM_ORDER`, found 2026-09-12), so re-measure before building on it; the advice to refit on the deploying model stands either way. The fused rank-1 kernel needs a FINITE factor level (`card:1` falls to a Python loop). The live QAT-KD path mirrors `_get_loss` under the GRU head (`qat_kd_targets`, fixed 2026-09-10); **the reusable test: a distillation target is correct iff, with teacher == student, the KD gradient vanishes** (`scratchpad/qatkd/smoke_qatkd.py`: 2.9e-07 vs 0.79 for the old formula).
* **QAT-TAX ROUTES CLOSED ON MECHANISM -- do not re-run:** the rank-1-friendly regulariser (iter 47), 2-bit norm / per-stream norm ranges / learnable norm levels / per-stream catalogs (the learned catalog absorbs all four), the teacher swap (iter 54), QAT from scratch (iter 40), a longer QAT phase, the WKV/shift bit split (phase D), refit-on-the-deploying-model (the catalogs already learn), WKV chunk structure (already m=1 per head). Left: QAT-KD (`lfqkd`) and the b13 grant (`lfq*b13`).
* **INSTRUMENTS.** A decay branch's 10-user validation endpoint, read beside the reference run's endpoint on the SAME 10 users, predicted arm 1's gate deltas within 0.0004 at zero GPU cost (`scratchpad/ws10/plot_losses.py`). ⚠ Units trap: training `imm_loss` is the 4-way CE, validation `imm` is the binary logloss of 1-P(Again); only AHEAD compares between the two logs. The regularisation screen is SETTLED at the full budget: still FIT-limited (ahead train/val gap +0.0065 -> +0.0083 at 10 ep while validation improves), so SAM, wd and dropout stay out. The shared WS BAKES IN wd, dropout and `WARMUP_STEPS` 400; no decay branch can re-open them.
* **ARCHITECTURE IS ANSWERED** (`scratchpad/arch_2026-09-07/FINDINGS.md`): the curve head's hidden is ~2.4-dimensional and a sufficient statistic of its input; realcyc vs the d=128 model residual correlation 0.9957; FSRS-7 adds nothing (fitted blend weight 0.02, gain +0.00001). **Method: residual correlation overstates decorrelation when the models differ in quality -- price a disagreement with a FITTED blend weight.** Do NOT open a new architecture family on "there must be low-hanging fruit"; untested and cheap-to-fit: cross-stream state fusion, layer reuse, multi-step delta on the coarse streams, a richer feature encoder.
* **FEATURE AUDIT** (`scratchpad/feataudit/FINDINGS.md`): nothing is discarded or leaked; six build-time coding warts (sharpest: `scaled_sibling_gap` writes 0.0 on 97.4% of rows, inside its defined range, with no NaN flag; the three `*_id_is_nan` columns are bit-identical duplicates). Fixing any costs a ~7 h rebuild and a lineage re-base -- not scheduled.
* **OPS LESSONS FROM THOSE DAYS** (each also in LIVE RULES or a tool): stop a waiter's cmd.exe FIRST, then its children (a dying cmd takes a wrong branch); a decay-only run writes checkpoints into the SOURCE folder, and every consumer of that path must be fixed, not one (preflight's decay-writes-here check); a branch from a TRANSFORMED checkpoint must transform everything the resume path loads (weights, optimizer state, side-car catalogs); a gate converts a tooling false negative into idle GPU, so check a waiter has not REFUSED as well as that it is alive.

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
**ARCHITECTURE GRAFTS 0/3 -- CLOSED BY ANDREW 2026-09-16 ("let's not do grafts anymore")**: cmixgraft (coarse mixers restored on ws10), lfcmf2 (card mixer doubled on w10lf) and lflora16 (card LoRA ranks raised on w10lf, +0.000059 / +0.000006, landed 2026-09-17) are all nulls at low engagement; graft-and-decay under-trains new parameters by construction. Capacity is answered by from-scratch runs (link D's d=128 at 10+2), not by growth on a trained checkpoint.
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
- **★ VRAM CAP 11 GiB IN EVERY NEW RUN (Andrew 2026-09-29: "see if you can cap VRAM for your runs a little below 12 GB"; a full card makes his desktop sluggish).** `set RWKV_VRAM_CAP_GB=11` (`rwkv/vram_cap.py`, called at the start of `train_rwkv.main_loop` and `get_result.main`; unset = unchanged). It caps PyTorch's allocator only (the CUDA context, ~0.4 GB, is extra). Consequences: in TRAINING an over-cap batch is SKIPPED by the per-batch exception handler, so every capped runner guards its WS log with `scratchpad/count_skips.py <log> <steps>` (> 1% = hollow, exit 61); in EVAL a giant user that used to survive through the Windows sysmem fallback is OOM-skipped and listed in `result/RWKV-<tag>.oomskip.jsonl`, so verdicts pair on the intersection (as with the box). Tested: 10.5 GiB allocates, 11.5 GiB is refused. ⚠ **VALIDATION is a third consequence (found 2026-09-30): at d=128 one of the 10 validation users needs > 11 GiB, so every WS validation of d128e10 failed with OOM ("Exception in validate", 30 of 30; training unaffected, and count_skips did not count it). Fix: `vram_cap.run_uncapped_on_oom` retries an OOM validation user ONCE with the cap lifted (logged `[vram-cap] validation user X over the cap`), so the 10-user mean stays comparable; WS-trial runners call `count_skips.py ... --max-val-fail 2`. d128e10's running WS has the old code (no WS validation trace); its decay gets the fix.**
- **★ THE RUN BOARD MUST STAY CURRENT (Andrew 2026-09-25: "as long as you don't forget to update it").** https://claude.ai/artifact/8KDuhUEdBhUqsUVXtv1b41 shows running / queued / planned / finished runs from its `db` (collection `runs`, doc `meta/board`; schema in memory `run-board`; page copy `scratchpad/runboard/runboard.html`). **Every time a run starts, ends, is queued, re-planned or recorded, update its document and `meta/board.updated` with ArtifactData in the same turn.** No republish is needed for data.
- **★★ THE DISK CHECK IS MINE TO DO, EVERY 3 DAYS, IN-SESSION -- NOT A ROUTINE (Andrew 2026-09-20).** He first asked for a daily reminder, I built it as a scheduled LLM task, and he deleted it with the reason that settles the design: **"you can't communicate with agents in 'routines'. Which defeats the whole point."** A scheduled task runs in its own session, so it cannot ask this one for approval and get an answer -- and the approval step was the point (his earlier instruction: *"make sure the agent always reports to you and asks you for permission to delete. The cleanup should be automatic, without needing me, but it should still go through both of you Claudes"*). A routine could therefore only do the useless half (report into the void) or the forbidden half (delete unattended).
  **THE RULE NOW:** check `scratchpad/disk_check/LAST_CHECK.txt` at the start of a session; **if it is more than 3 days old, run the check as ordinary work** (it is CPU-only and costs minutes), then write the report to `scratchpad/disk_check/report_<date>.md` and stamp the file. Delegate to a subagent only if the session is otherwise occupied; the budget rule still applies, and this is usually cheaper done directly.
  **WHAT THE CHECK IS:** free space on C: and F:; every LMDB under both `rwkv_lmdb` directories with its REAL size (`GetCompressedFileSize` -- `Get-ChildItem | Measure Length` reports the ~372 GB map_size RESERVATION, which is fiction); for each, whether a live runner, a queued runner or an open handle still needs it (⚠ an ORPHAN fetch worker holds an LMDB open for days with a dead parent); then old `.pth` in CLOSED run directories, KD dumps and stale `C:\Temp\claude` session directories. Andrew's rule is **"used -> move, unused -> delete"**, and a move is `lmdb.Environment.copy(dst, compact=True)` with ~7% GROWTH budgeted, never a file copy.
  **NEVER DELETE:** champion checkpoints; any run directory that is queued, running or awaiting a verdict; the read-only datasets; `label_filter_db*`; another Claude session's temp directory; the sibling repos; `pagefile.sys` and `MEMORY.DMP`. ⚠ A junction deletes THROUGH the link -- delete the real C: directory first, then `[IO.Directory]::Delete(link, $false)`.
  **Deleting anything still needs Andrew's word**, as it always has; the 3-day cadence is only about noticing in time.
- **A COLD GPU NOW REPORTS ITSELF (Andrew 2026-09-20, after the box sat idle 3.7 h; and his correction: "you don't need an LLM for a check like this").** The Windows task **`ClaudeGpuColdWatch`** runs `scratchpad/gpu_cold_watch.cmd` every 30 minutes, hidden, and it is silent unless `scratchpad/gpu_cold_check.py` exits 2. COLD = no compute process AND no run log written for 30 minutes AND work still outstanding; all three, per machine. **Progress, not utilization, is the signal** -- a memory-bound giant user sits at 50 W and 100%, a stalled job sits at 0%. On an alarm it shows Andrew a balloon and writes **`scratchpad/GPU_COLD.txt`**; it never restarts anything. **CHECK FOR THAT MARKER FILE AT THE START OF A SESSION** -- it is how an alarm that fired while no session was running reaches the next one. Delete it once the cause is handled. Log: `scratchpad/gpu_cold_watch.log`, last verdict `gpu_cold_last.txt`.
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
- **★★ KEEP AT LEAST 10 GB OF RAM AVAILABLE, ALWAYS (Andrew 2026-09-15: "for real though, I REALLY need to always keep at least 10 GB of RAM free").** The thief is the fetch workers' memory-mapped LMDB pages: Windows keeps every touched page in the worker's working set, so two decay/eval workers reach ~23 GB each and "available" falls to 0.4-4 GB although the pages are reclaimable. **`scratchpad/ram_guard.py` = the LOGON TASK `ClaudeRamGuard`** (`schtasks`, interactive, hidden via `run_ram_guard_hidden.vbs` -> `run_ram_guard.cmd`, log `ram_guard.log`, singleton) **caps every rwkv fetch worker's working set** (`SetProcessWorkingSetSizeEx` hard max; the trainer/eval process is never touched) and `EmptyWorkingSet()`s them once a minute if available is still under 10 GB. Measured 23:15: available 2.8 -> 33.8 GB in seconds, worker RSS 22 -> 5.9 GB, fetch waits unchanged at 4 ms. **The cap is DYNAMIC (23:42): a worker starts at 6 GB; while available > 16 GB its cap = RSS + its share of (available - 14 GB), up to 16 GB; under 11.5 GB the caps drop by the deficit at once.** Why: a fixed 6 GB cap made EVERY 2000-step validation cold -- the validation users are re-read from the F: HDD at ~3 MB/s (~7 min per validation instead of ~1 min; the step-50 one took 21:20-21:27 uncapped and cold, so the cap does not slow the reads, it only stops the pages staying resident: the train stream is ~0.2 GB/min per worker, so 43 min of it evicts them under 6 GB, ~10-12 GB keeps them). With Andrew's programs at ~20 GB the caps settle near 7-8 GB and the validations stay cold: ~11 x 6 min per decay, the rule's price. **ACCEPTED EXCEPTION (Andrew 2026-09-16, "Accept it"): a full-VAL eval on its GIANT users breaks the rule and the guard cannot help** -- the eval PROCESS itself holds ~40 GB private (26 GB resident) on a ~1M-row user (measured 08:14 on lfcmf2's eval: available 1.9 GB with the workers already paged out to 0). It lasts ~1-2 h per full eval; do not add a second RAM consumer (a rebuild, a copy, a second training) in that window. **After every reboot check the task ran** (`schtasks //query //tn ClaudeRamGuard`; `tasklist` for ram_guard.py; re-run with `schtasks //run //tn ClaudeRamGuard`) and size every new run so trainer + eval private memory + 10 GB fits in 64 GB; a giant-user eval never overlaps a rebuild or a second training.
- FETCH WORKERS: 2 in the WS tomls, and since 2026-09-15 also 2 in every decay toml (`write_decay_setup.py`, was a hardcoded 4; `RWKV_DECAY_FETCH_PROCESSES` overrides). RAM: a decay worker holds ~3.2 GB private + a ~9 GB LMDB mapping, the trainer ~10.6 GB, so lfd3 with 4 workers left 0.4 GB available beside Andrew's FSRS-7 (~13 GB). Bit-identical: `DataFetcher.get` reassembles batches by group index, so the worker count never touches numbers; the WS ran at the same steps/s on 2. Also: kill orphan fetch workers after every run -- one from 09-12 held 8.2 GB of commit for two days. Live loss plot: `detach.ps1 -Script
  scratchpad/liveplot/run_liveplot.cmd`.
- **⚠ Eval-path fetch-worker leak is SYSTEMATIC:** every eval leaves 1-2 orphan pythons, some
  spinning a full core for hours. **CHECK + KILL orphan pythons after every run** — but inspect
  command lines first: the spare `pythonw` are the bridge/controller, the ~80000s-CPU python is
  Andrew's FSRS benchmark, and he also runs a Reddit bot + liveplot. **Do not kill those.**
  **From 2026-09-11 he also runs srs-benchmark FSRS-7 `--raw` on 2 CPU threads** (to compare with
  RWKV later; CPU-only, `--processes 2`). Keep 2 threads free: my side-work (screens, refits,
  k-means) stays at <= 4 threads and BelowNormal priority while it runs.
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
  **CURRENT LIST (2026-09-16): `train_db_5k_h1_id5` ONLY.** `train_db_5k_h1_e2s` and `test_db_5k_e2s` (103 GB
  real each, the published-lineage e2s dbs, used by nothing queued) were DELETED 2026-09-16 with Andrew's rule
  "used -> move, unused -> delete" (C: 19 -> 229 GB free); rebuildable in ~80 min each from the published
  dataset if that lineage ever returns. The live train db stays on C: because reading it from F: costs 2.2x per
  step. (Previous list, 2026-09-11: `train_db_5k_h1_id5`, `train_db_5k_h1_e2s` and `test_db_5k_e2s`.) `train_db_5k_h1_id3` was deleted 2026-09-03 (Andrew) and gen 5's train db took
  its place on C:. Everything else on F: is a real directory, including `test_db_5k_id5`.
  **(2026-08-30 list, superseded):** `test_db_5k_fix` was DELETED with the other
  two superseded dbs, so its junction is gone too. The junctions then were
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
- **⚠⚠ A STUB HARNESS THAT DOES NOT ASSERT ITS SUBSTITUTIONS IS A LAUNCHER (2026-09-20, near-miss, 4 minutes of GPU).**
  I stub-tested a new waiter by copying it and replacing the real runner call with `call cmd /c exit 0`.
  The replacement silently did not apply -- the copy on disk still held
  `call scratchpad\lfq1rvq3b10
un_lfq1rvq3b10.cmd` at line 30 -- so running the "stub" STARTED THE REAL
  ARM, beside a dry twin that was already on the card. It ran 4 minutes before I stopped the harness; no
  checkpoint had been written, the dry twin logged 0 `Exception caught`, and nothing was lost. It could as
  easily have been a 13 h run on a busy GPU.
  **THE RULE: a stub must assert every substitution it depends on (`assert s.count(old) == 1` before
  `replace`), and the copy must be greped for the real runner's name and refuse to run if it is still
  there.** The generators in this repo already assert; the ad-hoc one written in the moment did not, which
  is the whole lesson -- the dangerous script is the one you write for a single use. Same family as the
  mk53/mk54 slice: assert on the OUTPUT, never trust that the edit landed.
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
- **DATA FACT (2026-09-20): THE PER-CARD REVIEW CAP IS 128, NOT 64, AND IT NEVER REACHED TRAINING.** Andrew proposed lifting srs-benchmark's cut-off for training only ("more data in the very-hard-cards and very-old-cards regions"). Measured, not read: `features/base.py:104` drops `i > max_seq_len * 2` = **128** (`max_seq_len` defaults to 64 and nothing overrides it), and it lives inside `create_features`, which **only `find_equalize_test_reviews.py` calls** -- the SCORED-set builder. `data_processing.py` computes `i` at :372 and never filters on it, and the training loss masks on `has_label`, not `label_is_equalize` (`srs_model.py:1624/1640`; the equalize masks feed reporting only). **Proof on the databases themselves (`scratchpad/newcards/check_seq_cap.py`): train user 66 owns a 718-review card, 15,152 of its rows (13.1%) sit past i=128, and 14,992 of those are LABELLED -- training already uses them; the scored set contains 0 of them.** So the convention is already exactly where Andrew wants it, and lifting anything would move the EVAL set away from srs-benchmark. ⚠ The sample matters: the first run used users 1/2/17, whose longest card is 112, and would have read as "truncated" -- pick a user with a long card before concluding anything. What does still shorten long histories in training is CHUNKING (a median of 4 chunks per user, state reset at each boundary), which is the shelved chunk-continuous item, not the cap.
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
