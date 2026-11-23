# Iter 55 (`l2i8s4e1c`) - a 3,584-bit recurrent state, with the tail-energy penalty on

This bundle is self-contained. It holds the weights, the code and the evidence for ONE recipe from a
state-quantization research arm: **the RWKV recurrent state of the card and note streams is stored at
rank-2 / rank-1 int8 factors plus int4 token-shift vectors, 3,584 bits per user instead of
139,264 - and it is trained with a penalty that lowers the state
energy the truncation throws away.**

* `README.md` (this file) - what the recipe is, the numbers, the mechanism, the layout.
* `ANSWERS.md` - point-by-point answers to the questions and concerns raised by another team in
  "Recurrent State Compression Notes" (answer-count buckets, per-card movement, the trace-vs-eval
  gap, relative bars, per-write solver cost, rank-and-precision-by-age).
* `verification.md` - the exact commands that reproduce every number in this file.
* `MANIFEST.md` - every file with its size and SHA-256.
* `verify.cmd` - builds the Rust engine and runs the checks of `verification.md` (Windows).

Nothing here needs the training data. The Rust engine runs on the included traces alone.

---

## 1. The recipe in one table

| | card stream (1 layer) | note stream (3 layers) | total |
|---|---:|---:|---:|
| rank of the factor form | 2 | 1 | |
| factor precision | int8 | int8 | |
| token-shift vectors | int4 | int4 | |
| **stored bits** | **1,280** | **2,304** | **3,584** |
| gate bar (half of naive int4) | <= 2,176 | <= 6,528 | <= 8,704 |
| naive int4 state (all streams quantized at 4 bits) | 4,352 | 13,056 | 17,408 |
| unquantized state | 34,816 | 104,448 | 139,264 |

3,584 bits = 448 bytes per user, **2.6% of the unquantized state**,
4.9x smaller than the naive all-int4 layout and
38.9x smaller than fp32. The deck, preset and user streams
(4 + 3 + 3 layers) stay fp32, as in every row of this arm.

**Read this bit carefully: iter 55 uses EXACTLY row 17's bit layout (1,280 / 2,304 =
3,584 bits).** Its size claim is against the naive int4 bars and against a dense per-matrix code,
NOT against row 17 - the two are the same size. What iter 55 changes is the LOSS: it reaches the same
sizes with no measurable tax, and its immediate head is BETTER than the row-17 control
(-0.000045). Rows that go BELOW 3,584 bits in this arm are the fixed-basis "codec" rows
(2,912 / 2,432 / 2,696 bits), which ship a fitted basis with the model; they are summarised in
`ANSWERS.md` section 7.

Bit arithmetic, as the arm's own accounting function computes it (`rust_grid.bits_for`):

```
card = 1 layer * (2 * rank * K * factor_bits + 64 * shift_bits) = 1*(2*2*32*8 + 64*4) = 1,280
note = 3 layers * (2 * rank * K * factor_bits + 64 * shift_bits) = 3*(2*1*32*8 + 64*4) = 2,304
K = 32 (the WKV matrix is K x K); 64 = two token-shift vectors of K
```

One state element is the WKV matrix (`K x K` = 1,024 floats) plus two token-shift vectors (`2 x K` =
64 floats) = 1,088 floats per layer. The accounting counts the int-N CODES only; the two per-factor
f32 scales that the engine stores add 64 bits per compressed matrix (under 2% of the totals). The
gate bars use the same convention, so the comparison is fair.

### The environment variables

| purpose | training side (Python, QAT) | deploy side (Rust) |
|---|---|---|
| low-rank scope | `RWKV_QAT_LOWRANK_SCOPE=card:2:int8,note:1:int8` | `RWKV_STATE_LOWRANK_SCOPE=card:2:int8,note:1:int8` |
| shift level | `RWKV_QAT_SHIFT_SCOPE=card:int4,note:int4` | `RWKV_QUANT_SHIFTS=1` + `RWKV_STATE_SHIFT_LEVEL=card:int4,note:int4` |
| E1 tail-energy penalty | `RWKV_QAT_TAIL_PENALTY=1.0` + `RWKV_QAT_TAIL_STRIDE=8` | *(training only; not used at deploy)* |
| which QAT loop | `RWKV_QAT_FUSED=0` (the Python per-step loop) | *(not applicable)* |
| everything else | default | default |

The first two rows are ITER 55 == ROW 17. The last two are what make iter 55 a new point.

---

## 2. The E1 tail-energy penalty - the only change against row 17

`S ~ U_r diag(s_r) V_r^T` throws away every singular direction above `r`. The fraction of the
state's own energy in those directions is the **tail energy**

```
tail = sum_{i > r} sigma_i^2 / sum_i sigma_i^2          (per (batch, head), from the state's spectrum)
```

and `RWKV_QAT_TAIL_PENALTY=lambda` adds `lambda * (mean over the four compressed layers of tail)` to
the training loss. The lambda was chosen from the measured PRE-RUN bracket (mean tail about 0.0245 over
those four layers, i.e. `lambda=1.0` about 2.4% of the ~1.02 training loss). The run's own log shows the
lever doing its job - it does not merely stay in the loss untouched:

| 4-layer mean tail | value |
|---|---:|
| first logged window | 0.02052 |
| mean over the first 12 windows | 0.01033 |
| mean over the LAST 12 windows | 0.00125 |
| last logged window | 0.00045 |
| mean over all 124 windows | 0.00265 |

So the truncated energy falls by roughly an order of magnitude over the 124 steps (about
45x from the first window to the last), and the loss term
averaged about 0.26% of the loss. The lever costs ZERO state bits and zero deploy compute: it
only changes what the 124-step QAT decay optimises, and the deploy artifact is unchanged in FORMAT.

Two implementation facts that matter for reproduction:

* **`RWKV_QAT_TAIL_STRIDE=8`** accumulates the penalty on one step in eight, per stream tag. A
  differentiable `svdvals` per timestep per layer keeps every step's graph alive until backward, so
  the every-step version OOM'd at 10.34 GiB of the 11.00 GiB cap before its first step; the sampled
  version is an unbiased estimate of the window's mean tail at 1/8 of the graph memory. The counter is
  per TAG, so the 3-layer note stream is not sampled three times as often as the 1-layer card.
* **`RWKV_QAT_FUSED=0`** is required: the fused rank-1/2 CUDA kernels truncate INSIDE the kernel, so
  the pre-truncation state whose spectrum the penalty needs is invisible there. The Python per-step
  loop is the documented bit-parity twin of the fused kernels, and row 44 of the research log
  (`l2i8s4pf`, the same layout with the penalty OFF, trained on this same loop) reads
  +0.000050/+0.000325 against row 17's fused +0.000052/+0.000327 - a 2e-6 difference - so the
  Python-loop/FUSED difference does not move the result.

The training log prints one line per accumulated window, e.g. `[tail-penalty] lambda=1.0 stride=8
card=0.006707(n=7) note=0.025130(n=23)`; `evidence/run/train.log` holds all of them. The penalty is
OFF by default: with `RWKV_QAT_TAIL_PENALTY` unset the module is byte-identical to the earlier runs.

---

## 3. Results

### 3.1 The gate (750 held-out users)

The gate compares the quantized QAT run against an **otherwise identical run with no QAT and no
quantization** (`ctl_fp32_124`, the same 124-step decay with QAT switched off), roster seed
20261119. A head passes when the one-sided 95% upper bound of the paired mean log-loss difference is
at most +0.0005.

| head | iter 55 | fp32 control (no QAT) | delta | ub95 | p(worse) | verdict |
|---|---:|---:|---:|---:|---:|---|
| ahead | 0.333830 | 0.333802 | **+0.000028** | +0.000150 | 0.0086 | PASS |
| immediate | 0.306166 | 0.306211 | **-0.000045** | +0.000059 | 0.8384 | PASS |

The immediate head is BETTER than the control (-0.000045, i.e. the quantized state costs
nothing there), and the ahead head is inside its ceiling. Row 17, the penalty-OFF twin at the same
bits, read +0.000052 / +0.000327; **iter 55 removes the immediate-head tax entirely at the same
size**, which is why it is the row worth shipping.

### 3.1a Two different pairings - do not mix them

The table above is the ACCEPTANCE GATE: the QAT + quantized checkpoint against a control trained
WITHOUT QAT and scored WITHOUT quantization, over 750 users in the Python evaluation path. It answers
"is this recipe good enough to ship", and it contains BOTH effects at once - the training gain and the
quantization tax. On the immediate head the training gain wins, which is why that delta is negative.

The deploy tax in 3.2 is a different measurement: the SAME weights with the state quantizer switched on
vs off, in the Rust deploy engine, on the two shipped traces. It answers "what does the quantizer cost
at deploy", and it is about ten times larger (+0.000329 / +0.000338
against -0.000045 / +0.000028). The two are not comparable: different pairing,
different path, different user set. Read the gate for "should we ship" and the engine tax for "what the
quantizer costs" - never the two against each other.

### 3.2 The other checks

| check | result |
|---|---|
| deploy speed (Rust engine, rev/s) | **2,358.7 = 0.800x** of the unquantized engine (2,948.3); floor 0.75x |
| corruption / finiteness, 50 longest-history users | **PASS** - 158,986 state observations and 150 p(recall) tensors, zero non-finite, histories NOT truncated (4.5 min) |
| finiteness, deploy engine, strict, shipped weights | **PASS** - 43,474,944 state values and 19,476 p(recall) values on the two short traces, plus 119,463,936 state values on the long trace; zero non-finite |
| deploy log loss, label-scored (2 users) | `plain` 0.243431 ahead / 0.198319 imm; `l2i8s4` 0.243760 / 0.198657 -> tax **+0.000329 / +0.000338** |
| Python <-> Rust parity | the engine reproduces the Python RNN to at most **5.9e-07 per review** on the shipped traces (PARITY: PASS) |
| answer-count bucket tax, 40 longest-history users | 128+ bucket resolves -0.000865 immediate (-0.160% relative), i.e. the quantized state is BETTER than the unquantized one on the oldest cards |
| trace-vs-eval input difference (40 users, both heads) | mean abs difference 0.0031 imm / 0.0032 ahead - the pair of paths agree, see `ANSWERS.md` section 4 |

### 3.3 What the numbers do NOT say

* The gate numbers come from the Python evaluation path with the QAT quantization active. The Rust
  engine is the deploy path; it is checked separately (finiteness, label log loss, parity), not
  scored on 750 users.
* A "no tax" immediate head is a statement about the MEAN over 750 users. Section 5 of `ANSWERS.md`
  gives the per-bucket and per-card split, where young cards still move by a few tenths of a percent.
* The speed ratio depends on the machine; re-measure it. The ratio is the meaningful part.

---

## 4. What is in the folder

```
weights/
  rwkv_iter36_558.pth             the STARTING point: iter-36 champion, before QAT
  qat_l2i8s4e1c_124.pth           THE RECIPE: after the 124-step QAT decay with the E1 penalty
  ctl_fp32_124.pth                the control: the SAME 124-step decay, QAT off
  deploy_iter55_l2i8s4e1c.safetensors   the recipe in deploy format (444 tensors)
  deploy_ctl_fp32_124.safetensors       the control in deploy format
  deploy_pre_qat_iter36.safetensors     the starting point in deploy format
  weight_names.json               deploy tensor name -> shape (all 444)
traces/                           the deploy engine's input for users 107 and 136: 92-dim per-review
                                 features + route ids + labels + the Python RNN's own predictions,
                                 exported for THIS checkpoint
code/
  rwkv/                           the model, the training loop, the QAT implementation
    model/rwkv_ops.py             QAT: fake_lowrank_state, fake_quant_shift, quant_aware_rwkv7,
                                  accum_tail_penalty (the E1 lever), take_tail_penalty
    model/rwkv_model.py           where the quantized state is applied in the forward pass
    model/srs_model.py            the loss, incl. the rank-1 and tail-energy penalty drains
    model/csrc/cuda/              CUDA source of the fused kernels (+ the built .pyd for win/py3.12)
  rust/rwkv-infer/                the deploy engine (Rust, candle)
    src/model.rs                  state compression + quantization (the deploy quantizer)
    src/finite.rs                 the non-finiteness observer (RWKV_FINITE_CHECK)
  export_rnn_trace.py             the trace exporter (Python RNN -> trace files)
  scratchpad/quant_plan/          the harness: qat_decay.py, quant_sweep.py, rust_grid.py,
                                  rust_speed.py, rust_finite.py, rust_score.py, py_rust_parity.py,
                                  bucket_tax.py, bucket_mix.py, bucket_percard.py, trace_eval_gap.py
  scratchpad/quant_plan/corruption/  the corruption gate + the trace exporters
  scratchpad/adaptive_wd/adaptive_train.py   the training entry point the harness calls
evidence/
  sweep_iter55_l2i8s4e1c.json     the 750-user gate result for this recipe
  eval/                           the eval config, the 750-user roster, the per-user log losses
  corruption_iter55_l2i8s4e1c.json  the 50-user corruption result (per-user detail included)
  corruption/                     the 50 users' per-user log losses
  run/                            this run's train.toml, qat_env.json, train.log, validation users
                                  + the 0.04-epoch smoke (0.04 is the arm's smoke rule)
  bucket/                         the answer-count diagnostics (see ANSWERS.md) + the roster's mix
  parity/                         the Python <-> Rust parity results
  rust_grid_row55.json, rust_finite_row55*.json, rust_speed_row55.json
                                  the deploy-engine checks with the shipped weights
MANIFEST.md, ANSWERS.md, verification.md, verify.cmd
```

The `code/` tree keeps the repository layout instead of flattening it, because the harness scripts
resolve paths from their own location and import each other by module name (`quant_sweep` imports
`adaptive_train` from a sibling directory). `code/` is therefore a small repository root: run the
tools from inside it, and pass the weights and traces with `../weights`, `../traces`.

---

## 5. Quick start: run the deploy engine on the included traces

Needs a Rust toolchain (1.95 or newer) and Python with `numpy` and `scikit-learn`. The first
`cargo build` needs network access for the crate dependencies.

```
cd code
cargo build --release --manifest-path rust/rwkv-infer/Cargo.toml
```

Then, with the recipe's quantization switched on:

```
set RWKV_WEIGHTS=..\weights\deploy_iter55_l2i8s4e1c.safetensors
set RWKV_TRACE_DIR=..\traces
set RWKV_PRED_DIR=preds_l2i8s4
set RWKV_STATE_LOWRANK_SCOPE=card:2:int8,note:1:int8
set RWKV_QUANT_SHIFTS=1
set RWKV_STATE_SHIFT_LEVEL=card:int4,note:int4
rust\rwkv-infer\target\release\rwkv-infer.exe 107 136
```

The engine prints one line per user (`<n> reviews in <t>s (<rate> rev/s)`) and writes
`preds_l2i8s4\rust_pred_<user>.json`. `verify.cmd` in the bundle root does this and the finiteness
check for you. `verification.md` lists every check with its expected output.

---

## 6. How the quantization works

**What is stored.** The recurrent state of one layer is the WKV matrix `S` (`K x K`, K = 32) plus two
token-shift vectors (`2 x K`). The card stream has 1 such layer, the note stream 3.

**Low rank, then int8.** `S` is truncated to its top `r` singular components: `S ~ U_r diag(s_r)
V_r^T`, with the singular values folded into the two factors, and the two `K x r` factors are stored
as symmetric int8 codes (one scale per factor; `qmax = 127`). Rank 2 on the card, rank 1 on the note.

**The shift vectors go to int4.** They carry much less information than the matrix, and int4 costs
nothing measurable at these ranks.

**Why rank, not precision.** The arm settled this with the `cn8` diagnostic: full-matrix int8 on both
streams with NO rank truncation costs only **+0.000033 / +0.000022** log loss. An 8-bit state is
essentially lossless when it is not truncated, so the whole tax of the rank-1 family is the RANK
TRUNCATION, and no finer level can buy it back. That is why this recipe spends its bits on rank (2 on
the card) rather than on precision, and why 4-bit factors fail. It is also why the E1 penalty - which
attacks the truncated energy directly, at zero deploy cost - is the right lever to try against it.

**Training with quantization (QAT).** The quantized state is applied inside the forward pass during a
short decay (124 steps, LR 0.001, 2 epochs of users 1-100, `TRAIN_MODE=D`, bf16, batch length 66,000,
peak reserved 10.301 GiB of the 11 GiB cap) through a straight-through estimator, so the
model learns against the same truncation it will deploy with. The checkpoint before it is included
(`rwkv_iter36_558.pth`). The decay is short because the model is already trained.

**Where the code is.** Training side: `code/rwkv/model/rwkv_ops.py` (`fake_lowrank_state`,
`fake_quant_shift`, `quant_aware_rwkv7`, `accum_tail_penalty`). Deploy side:
`code/rust/rwkv-infer/src/model.rs` (`compress_wkv_state`, `quant_roundtrip`, `quant_codes`,
`quant_factor_inplace`). Both implement the same symmetric quantization: `scale = max|x| / qmax`,
`code = round(x/scale)`, `x' = code * scale`.

**Non-finiteness.** Quantization can, in principle, produce a non-finite state or a non-finite
p(recall) on a very long history. Both paths observe this: the Python probe
(`code/scratchpad/quant_plan/corruption/finite_probe.py`) watches the RAW state before the `sanitize`
cap, and the Rust engine (`code/rust/rwkv-infer/src/finite.rs`, `RWKV_FINITE_CHECK=1` or `strict`)
counts every non-finite value it sees and can exit with code 3. Neither changes any number: they only
observe.

---

## 7. Versions the numbers were produced with

| | |
|---|---|
| OS | Windows 10 (10.0.19045), 32 logical cores |
| Python | 3.12.10 |
| torch | 2.12.1+cu130 (CUDA 13.0) |
| numpy / scipy | 2.5.0 / 1.18.0 |
| scikit-learn | 1.9.0 |
| safetensors / pyarrow / tomli | 0.8.0 / 24.0.0 / 2.4.1 |
| Rust | cargo 1.95.0, rustc 1.95.0 |
| CUDA toolkit (for the .pyd) | 13.2 |
| GPU | one NVIDIA GPU, 11 GiB VRAM cap for training runs |

The fused CUDA extension `code/rwkv/model/RWKV_CUDA.cp312-win_amd64.pyd` is included for
convenience; it is Windows + Python 3.12 only. Its source is in `code/rwkv/model/csrc/` and it can be
rebuilt with `nvcc`. **It is not needed for anything in this bundle**: the Rust engine, the parity
check and the corruption probe all run on CPU, and the QAT path can be run in pure Python with
`RWKV_QAT_FUSED=0` (which is exactly what iter 55 used).

---

## 8. Not included, and why

* **The datasets.** `train_db_5k_h1_adaptive92`, `test_db_5k_adaptive92` and
  `label_filter_db_id_e2s` are multi-GB LMDBs (Anki review logs). Re-running the 124-step QAT decay
  or the 750-user evaluation needs them. Running and scoring the Rust engine does not.
* **Optimizer states** (`qat_optim_124.pth`, ~1.9 MB). Only needed to resume a run.
* **The 29,166-review long-history trace** (23 MB, the largest single file this bundle used to carry). Its
  feature tensors hold no weight information, and its readings are already in
  `evidence/rust_finite_row55_long.json` and in the `per_user` block of
  `evidence/bucket/bucket_tax_row55.json`. Re-export it with
  `code/scratchpad/quant_plan/corruption/export_corrupt50_traces.py` if a long-history run is needed.
* **The Python environment.** See section 7.
* **The rest of the research arm** (other rows, other checkpoints, the run board, the supervisor).
  This bundle is one row on purpose.

## 9. Limitations

* One recipe, one lineage. The model is an RWKV-7 with d=32, H=1, K=32, 192,800 parameters, 92 input
  features, 9 streams; the numbers do not transfer to another architecture without re-measurement.
* The gate is a 750-user paired comparison on one roster (seed 20261119). The corruption gate uses 50
  users chosen by the longest average card histories, with users too large for VRAM excluded (the
  excluded list is in `code/scratchpad/quant_plan/corruption/corrupt50.json`).
* The corruption gate for this recipe ran on the Python QAT path (`RWKV_QAT_FUSED=0`, the bit-parity
  twin of the fused kernels) because the raw pre-truncation state is only visible there; the deploy
  engine's own observer is the separate Rust check.
* The E1 penalty's lambda was fixed at 1.0 from the pre-registered bracket (the tail measures about
  0.0245 before training, i.e. lambda=1.0 is about 2.4% of the loss at that point; the run's own log
  shows the tail then falls about an order of magnitude, so the term averaged about 0.26% over the
  run). lambda = 3.0 and 10.0 are the remaining points of that bracket.
* The AHEAD head of the trace path and the AHEAD head of the evaluation path are different input
  frames; compare ahead numbers only WITHIN one path. See `ANSWERS.md` section 4 for the measured
  size of that difference (about 0.003 mean absolute, on BOTH heads).
* `traces/` covers users 107 and 136. It is the same user set as the row-17 bundle (user 156 cannot be
  exported: the current label DB puts review 1142 of that user in the equalize set and the exporter's
  own per-prediction dictionary has no entry for it).
