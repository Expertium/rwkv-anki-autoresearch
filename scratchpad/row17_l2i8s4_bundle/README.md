# Row 17 (`l2i8s4`) - quantized recurrent state for an RWKV recall model

This bundle is self-contained. It holds the weights, the code and the evidence for ONE recipe from a
state-quantization research arm: **the RWKV recurrent state of the card and note streams is stored at
rank-2 / rank-1 int8 factors plus int4 token-shift vectors, 3,584 bits per user instead of 139,264.**

* `README.md` (this file) - what the recipe is, the numbers, the mechanism, the layout.
* `verification.md` - the exact commands that reproduce every number in this file.
* `MANIFEST.md` - every file with its size and SHA-256.
* `verify.cmd` - builds the Rust engine and runs the two fastest checks (Windows).

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
| unquantized state | 34,816 | 104,448 | 139,264 |

3,584 bits = 448 bytes per user, **2.6% of the unquantized state**. The deck, preset and user streams
(4 + 3 + 3 layers) stay fp32, as in every row of this arm.

Bit arithmetic, as the arm's own accounting function computes it (`rust_grid.bits_for`):

```
card = 1 layer * (2 * rank * K * factor_bits + 64 * shift_bits) = 1*(2*2*32*8 + 64*4) = 1,280
note = 3 layers * (2 * rank * K * factor_bits + 64 * shift_bits) = 3*(2*1*32*8 + 64*4) = 2,304
K = 32 (the WKV matrix is K x K); 64 = two token-shift vectors of K
```

One state element is the WKV matrix (`K x K` = 1,024 floats) plus two token-shift vectors (`2 x K` =
64 floats) = 1,088 floats per layer. The accounting counts the int8 CODES only; the two per-factor
f32 scales that the engine stores add 64 bits per rank-r matrix (+64 bits for the card, +192 for the
note, under 2% of the totals). The gate bars use the same convention, so the comparison is fair.

### The environment variables

| purpose | training side (Python, QAT) | deploy side (Rust) |
|---|---|---|
| low-rank scope | `RWKV_QAT_LOWRANK_SCOPE=card:2:int8,note:1:int8` | `RWKV_STATE_LOWRANK_SCOPE=card:2:int8,note:1:int8` |
| shift level | `RWKV_QAT_SHIFT_SCOPE=card:int4,note:int4` | `RWKV_QUANT_SHIFTS=1` + `RWKV_STATE_SHIFT_LEVEL=int4` |
| everything else | default | default |

The two names differ only because the training path and the deploy engine were written at different
times; the quantization they describe is the same.

---

## 2. Results

The gate compares the quantized QAT run against an **otherwise identical run with no QAT and no
quantization** (`ctl_fp32_124`, the same 124-step decay with QAT switched off - its training config
differs from the QAT run only in the output folder name, and its log has zero `[QAT]` banners).
750 held-out users, roster seed 20261119. A head passes when the one-sided 95% upper bound of the
paired mean log-loss difference is at most +0.0005.

| head | row 17 | reference (fp32, no QAT) | delta | ub95 | p(worse) | verdict |
|---|---:|---:|---:|---:|---:|---|
| ahead | 0.333853 | 0.333802 | **+0.000052** | +0.000118 | 0.00017 | PASS |
| immediate | 0.306539 | 0.306211 | **+0.000327** | +0.000390 | 1.6e-23 | PASS |

| check | result |
|---|---|
| deploy speed (Rust engine, rev/s) | **2,357.8 = 0.809x** of the unquantized engine (2,915.1); floor 0.75x. Median per-user ratio 0.789x. Under load the same row read 2,290 rev/s = 0.800x. |
| corruption / finiteness, 50 longest-history users | **PASS** - 158,986 state observations and 150 p(recall) tensors, zero non-finite values, histories NOT truncated (max 96,732 reviews). |
| finiteness, deploy engine, shipped weights, strict mode | **PASS** - 43,474,944 state values and 19,476 p(recall) values on the two shipped short traces, zero non-finite; the engine exits 3 if it finds one, and it exited 0. |
| deploy log loss, label-scored (2 users) | `plain` 0.243849 ahead / 0.197007 imm; `l2i8s4` 0.244184 / 0.197323 -> tax **+0.000335 / +0.000316**, both inside +0.0005. A check of the deploy path, NOT the 750-user gate. |
| Python <-> Rust parity | the engine reproduces the Python RNN to **at most 8.5e-07 per review** and 0.000000 log loss on 3,465 / 3,865 / 12,915 / 23,455-review histories. For the QUANTIZED state, the immediate head agrees with the Python evaluation to 8e-05 on the long user; the ahead head is not comparable across the two input paths - `verification.md` section 4b shows the same gap on the unquantized control. |

Row 17 is the **axis-1 (size) winner** of the arm: it costs the row-15 tax for 22% fewer bits
(3,584 vs 4,608). Its immediate head carries the largest tax of the passing rows, so it is not the
loss winner; rows 16 and 18 (rank-3 card) are the low-tax rows at more bits.

### What the numbers do NOT say

* The gate numbers come from the Python evaluation path with the QAT quantization active. The Rust
  engine is the deploy path; it is checked separately (finiteness, label log loss, parity), not
  scored on 750 users.
* `+0.000327` on the immediate head is a real, measurable tax (p = 1.6e-23). It is inside the gate's
  ceiling, not zero.
* The speed ratio depends on the machine. Re-measure it; the ratio is the meaningful part.

---

## 3. What is in the folder

```
weights/
  rwkv_iter36_558.pth             the STARTING point: iter-36 champion, before QAT (956 KB)
  qat_l2i8s4_124.pth              the recipe: after the 124-step QAT decay (952 KB)
  ctl_fp32_124.pth                the control: the SAME 124-step decay, QAT off (952 KB)
  deploy_iter17_l2i8s4.safetensors      the recipe in deploy format (444 tensors)
  deploy_ctl_fp32_124.safetensors       the control in deploy format
  deploy_pre_qat_iter36.safetensors     the starting point in deploy format
  weight_names.json               deploy tensor name -> shape (all 444)
traces/                           the deploy engine's input for users 107 and 136: 92-dim per-review
                                 features + route ids + labels + the Python RNN's own
                                 predictions, re-exported for this lineage (see section 6)
traces_long/                      user 5098, a 29,166-review history from the corruption roster,
                                 so the checks also run at a realistic history length
code/
  rwkv/                           the model, the training loop, the QAT implementation
    model/rwkv_ops.py             QAT: fake_lowrank_state, fake_quant_shift, quant_aware_rwkv7
    model/rwkv_model.py           where the quantized state is applied in the forward pass
    model/srs_model.py            the loss, incl. the rank-1 and tail-energy penalty drains
    model/csrc/cuda/              CUDA source of the fused kernels (+ the built .pyd for win/py3.12)
    train_rwkv.py, get_result.py, data_processing.py, ... the training and evaluation paths
  rust/rwkv-infer/                the deploy engine (Rust, candle)
    src/model.rs                  state compression + quantization (the deploy quantizer)
    src/finite.rs                 the non-finiteness observer (RWKV_FINITE_CHECK)
    src/fast.rs                   the fast rank-r solver (RWKV_LOWRANK_FAST)
  export_rnn_trace.py             the trace exporter (Python RNN -> trace files)
  optimization/DEPLOY_FUNCTIONS.md, PROTOCOL.md   the deploy contract and the experiment protocol
  scratchpad/quant_plan/          the harness: qat_decay.py (the QAT run), quant_sweep.py (the gate),
                                  rust_grid.py / rust_speed.py / rust_finite.py / rust_score.py,
                                  py_rust_parity.py, it36ladder_lib.py
  scratchpad/quant_plan/corruption/  the corruption gate: finite_probe.py, check_corruption.py,
                                  export_corrupt50_traces.py, export_parity_trace.py, corrupt50.json
  scratchpad/adaptive_wd/adaptive_train.py   the training entry point the harness calls
evidence/
  sweep_iter17_l2i8s4.json        the 750-user gate result for this recipe
  eval/                           the eval config, the 750-user roster, the per-user log losses
  corruption_iter17_l2i8s4.json   the 50-user corruption result (per-user detail included)
  corruption/                     the 50 users' per-user log losses
  run/                            this run's train.toml, qat_env.json, train.log + the 0.04-epoch smoke
  parity/                         the Python <-> Rust parity results (section 4 of verification.md)
  rust_speed_idle.json            the idle-machine speed measurement
  rust_speed_bundle.json          the speed measurement on the bundled traces
  rust_finite_iter17_l2i8s4.json  the strict finiteness check with the shipped weights
  rust_grid_iter17_l2i8s4.json    the deploy log loss with the shipped weights
  rust_finite_probe.json, rust_score.json   earlier 3-user checks (June trace set, see section 6)
  py_rust_parity_iter17.json      the Python <-> Rust parity check
  rust_finite_probe.json, rust_score.json   earlier 3-user checks (different checkpoint, see 6)
MANIFEST.md, verification.md, verify.cmd
```

The `code/` tree keeps the repository layout instead of flattening it, because the harness scripts
resolve paths from their own location and import each other by module name (`quant_sweep` imports
`adaptive_train` from a sibling directory). `code/` is therefore a small repository root: run the
tools from inside it, and pass the weights and traces with `../weights`, `../traces`.

---

## 4. Quick start: run the deploy engine on the included traces

Needs a Rust toolchain (1.95 or newer) and Python with `numpy` and `scikit-learn`. The first
`cargo build` needs network access for the crate dependencies.

```
cd code
cargo build --release --manifest-path rust/rwkv-infer/Cargo.toml
```

Then, with the recipe's quantization switched on:

```
set RWKV_WEIGHTS=..\weights\deploy_iter17_l2i8s4.safetensors
set RWKV_TRACE_DIR=..\traces
set RWKV_PRED_DIR=preds_l2i8s4
set RWKV_STATE_LOWRANK_SCOPE=card:2:int8,note:1:int8
set RWKV_QUANT_SHIFTS=1
set RWKV_STATE_SHIFT_LEVEL=int4
rust\rwkv-infer\target\release\rwkv-infer.exe 107 136
```

The engine prints one line per user (`<n> reviews in <t>s (<rate> rev/s)`) and writes
`preds_l2i8s4\rust_pred_<user>.json`. `verify.cmd` in the bundle root does this and the finiteness
check for you. `verification.md` lists every check with its expected output.

---

## 5. How the quantization works

**What is stored.** The recurrent state of one layer is the WKV matrix `S` (`K x K`, K = 32) plus two
token-shift vectors (`2 x K`). The card stream has 1 such layer, the note stream 3.

**Low rank, then int8.** `S` is truncated to its top `r` singular components: `S ~ U_r * diag(s_r) *
V_r^T`, with the singular values folded into the two factors, and the two `K x r` factors are stored
as symmetric int8 codes (one scale per factor; `qmax = 127`). Rank 2 on the card, rank 1 on the note.

**The shift vectors go to int4.** They carry much less information than the matrix, and int4 costs
nothing measurable: row 17 and row 15 have the same tax (the int8-shift version of row 17 is row 15)
while row 17 is 1,024 bits smaller.

**Why rank, not precision.** The arm settled this with the `cn8` diagnostic: full-matrix int8 on both
streams with NO rank truncation costs only **+0.000033 / +0.000022** log loss. An 8-bit state is
essentially lossless when it is not truncated, so the whole tax of the rank-1 family
(+0.000503/+0.001009) is the RANK TRUNCATION, and no finer level can buy it back. That is why this
recipe spends its bits on rank (2 on the card) rather than on precision, and why 4-bit factors
(row 23, `l2i4s4`) fail.

**Training with quantization (QAT).** The quantized state is applied inside the forward pass during a
short decay (124 steps, LR 0.001, 2 epochs of users 1-100, `TRAIN_MODE=D`, bf16, batch length 66,000)
through a straight-through estimator, so the model learns against the same truncation it will deploy
with. The decay is short because the model is already trained; the checkpoint before it is included
(`rwkv_iter36_558.pth`).

**Where the code is.** Training side: `code/rwkv/model/rwkv_ops.py`
(`fake_lowrank_state`, `fake_quant_shift`, `quant_aware_rwkv7`). Deploy side:
`code/rust/rwkv-infer/src/model.rs` (`compress_wkv_state`, `quant_roundtrip`, `quant_codes`,
`quant_factor_inplace`). Both implement the same symmetric quantization: `scale = max|x| / qmax`,
`code = round(x/scale)`, `x' = code * scale`.

**Non-finiteness.** Quantization can, in principle, produce a non-finite state or a non-finite
p(recall) on a very long history. Both paths observe this: the Python probe
(`code/scratchpad/quant_plan/corruption/finite_probe.py`) watches the RAW state before the
`sanitize` cap, and the Rust engine (`code/rust/rwkv-infer/src/finite.rs`, `RWKV_FINITE_CHECK=1` or
`strict`) counts every non-finite value it sees and can exit with code 3. Neither changes any number:
they only observe.

---

## 6. About the traces

* `traces/` holds users 107 and 136, and `traces_long/` holds user 5098 (29,166 reviews), all
  exported by `export_rnn_trace.export_user()` - the same function the arm's parity traces have always
  come from. Each trace carries the 92-dim per-review features, the route ids, the labels
  (`rating`, `label_rating`, `equalize_review_ths`) and the Python RNN's OWN per-review predictions
  (`py_pred_imm`, `py_pred_ahead`) for the recipe's weights. The features are inputs, so they do not
  depend on the weights; the `py_pred_*` do.
* Those `py_pred_*` are the UNQUANTIZED Python RNN (the RNN inference path does not apply the state
  quantization - it runs `rwkv_ops.single_timestep`, not `quant_aware_rwkv7`). They are therefore an
  exact parity target for the engine's `plain` config (section 4a of `verification.md`, agreement to
  1e-07) and NOT a target for the quantized config. For the quantized comparison see section 4b
  there, which also explains why the AHEAD head cannot be compared between the trace path and the
  evaluation path.
* The June-2026 `reference/trace_user_*` files are NOT shipped. Their feature tensors come from an
  older feature recipe (the same review indices, but feature values differing by up to 3.0), so
  scoring the current weights on them is meaningless - it was the reason an earlier version of this
  bundle reported a deploy tax of +0.0027 with the wrong sign. The two evidence files produced on
  that set (`rust_finite_probe.json`, `rust_score.json`, both with the `it36l2i8` deploy export) are
  kept for provenance only.
* The idle speed measurement (`rust_speed_idle.json`) also used the June set and the `it36l2i8`
  deploy export. The engine's cost is set by the config (rank, bit width), not by the weight values,
  and the bundle's own measurement (section 5) reproduces the same ratio.

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
`RWKV_QAT_FUSED=0` (which is what the corruption probe and the parity export use).

---

## 8. Not included, and why

* **The datasets.** `train_db_5k_h1_adaptive92`, `test_db_5k_adaptive92` and
  `label_filter_db_id_e2s` are multi-GB LMDBs (Anki review logs). Re-running the 124-step QAT decay or
  the 750-user evaluation needs them. Running and scoring the Rust engine does not.
* **Optimizer states** (`qat_optim_124.pth`, ~1.9 MB). Only needed to resume a run.
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
  twin of the fused kernels), because the raw pre-truncation state is only visible there. The deploy
  engine's own observer is the separate Rust check.
* Both the training-side and deploy-side quantization use a per-matrix (or, optionally, per-column)
  scale. Per-column scaling (`RWKV_LOWRANK_PERCOL`) is implemented but not used by this recipe.
* The AHEAD head cannot be compared between the evaluation path (LMDB) and the trace path: their
  time inputs differ, by about 0.005 log loss on a long user, WITH or WITHOUT quantization (the
  unquantized control shows the same gap). Compare ahead numbers only within one path.
* User 156 is not in `traces/`: the current label database puts review 1142 of that user in the
  equalize set and the exporter's per-review prediction dictionary has no entry for it, so the export
  aborts on its own assertion. Users 107 and 136 export cleanly.
