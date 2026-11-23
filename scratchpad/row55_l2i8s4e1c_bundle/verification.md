# Verification

Every command below runs from the bundle alone, except the ones in section 8, which need the training
datasets (multi-GB LMDBs that are not shipped). The numbers in **bold** are what we measured on the
machine described in `README.md` section 7. Commands are Windows-style; on Linux/macOS replace
`python` with `python3`, `\` with `/` and `set VAR=x` with `export VAR=x`.

## 0. Setup

```
cd code
cargo build --release --manifest-path rust/rwkv-infer/Cargo.toml      # first build needs network
python -m pip install numpy scikit-learn                             # only for the scoring scripts
```

`verify.cmd` in the bundle root runs sections 2, 3 and 4 in one go, plus two informational steps (4c
and 5). It was run end-to-end on a FRESH EXTRACT of this bundle before it was shipped.

## 1. The shipped deploy weights ARE the shipped `.pth`

`weights/deploy_iter55_l2i8s4e1c.safetensors` was exported from `weights/qat_l2i8s4e1c_124.pth` by
the repository's own exporter (`export_weights_only.py`, which reuses `export_rnn_trace.export_weights`
so the recipe metadata is read off the model object). Re-export it and compare:

```
python scratchpad\quant_plan\export_weights_only.py ..\weights\qat_l2i8s4e1c_124.pth tmp_repro deploy_repro.safetensors
python -c "from safetensors import safe_open; import numpy as np; a=safe_open('tmp_repro/deploy_repro.safetensors','pt'); b=safe_open('../weights/deploy_iter55_l2i8s4e1c.safetensors','pt'); ka=sorted(a.keys()); print(len(ka), sorted(b.keys())==ka, max(float(np.abs(a.get_tensor(k).float().numpy()-b.get_tensor(k).float().numpy()).max()) for k in ka))"
```

Expected: **444 True 0.0** - the same 444 tensors with a maximum absolute value difference of exactly
zero. (The two FILES are not byte-identical: the safetensors header lists the tensor keys in a
different order. The values and the recipe metadata - `stream_order`, `interleave`,
`interleave_placement`, `state_clamp_tau` - are identical.)

## 2. Deploy log loss: the quantization tax inside the Rust engine

`rust_grid.py` runs the engine twice on the same trace - once with no state quantization (`plain`) and
once with the recipe - and scores both against the labels with sklearn's log loss, exactly the metric
the Python gate uses. This isolates the DEPLOY quantization tax; it is not the 750-user gate.

```
python scratchpad\quant_plan\rust_grid.py --weights ..\weights\deploy_iter55_l2i8s4e1c.safetensors ^
    --trace-dir ..\traces --users 107,136 --configs plain,l2i8s4 --tag verify
```

Expected (**`evidence/rust_grid_row55.json`**):

| config | card/note bits | ahead | imm |
|---|---:|---:|---:|
| plain | 34,816 / 104,448 | 0.243431 | 0.198319 |
| l2i8s4 | **1,280 / 2,304** | 0.243760 | 0.198657 |
| tax | | **+0.000329** | **+0.000338** |

Both taxes are inside the +0.0005 ceiling. On the long-history user (5098, 29,166 reviews - measured at
build time; that trace is not shipped, its readings are in `evidence/rust_finite_row55_long.json`) the quantized state scores BETTER than the
unquantized one on the immediate head (-0.001749), which is expected rather than
contradictory: this checkpoint was trained WITH the quantized state in its forward pass, so the
unquantized state is the out-of-distribution case for it.

## 3. Non-finiteness, strict

```
python scratchpad\quant_plan\rust_finite.py --weights ..\weights\deploy_iter55_l2i8s4e1c.safetensors ^
    --configs plain,l2i8s4 --trace-dir ..\traces --users 107,136 --tag verify --strict
```

`--strict` makes the engine exit with code 3 if it sees ONE non-finite value, so a zero exit code is
an independent witness. Expected (**`evidence/rust_finite_row55.json`**):
**43,474,944 state values, 2,717,184 token-shift values and
19,476 p(recall) values on the two short traces, plus 119,463,936
state values on the 29,166-review long history, measured at build time
(**`evidence/rust_finite_row55_long.json`**) - zero non-finite anywhere**, with the largest state entry
0.2154 and the largest token-shift entry 5.1102. The
`plain` config reports `state_vals=0`, which is correct: with no quantization there is no state
compression step to observe.

## 4. Python <-> Rust parity

### 4a. The engine reproduces the Python RNN exactly (no quantization)

Each trace file carries the Python RNN's own per-review predictions (`py_pred_imm`, `py_pred_ahead`)
for the SAME weights, exported by `export_rnn_trace.export_user()` for the checkpoint in this bundle.
`py_rust_parity.py` compares the engine's output with them on the trace's equalize set:

```
python scratchpad\quant_plan\py_rust_parity.py --weights ..\weights\deploy_iter55_l2i8s4e1c.safetensors ^
    --trace-dir ..\traces --config plain --users 107,136 --tag verify_plain
```

Expected (**`evidence/parity/107_136_plain_rust_vs_python_rnn.json`**): **PARITY: PASS**, with a
maximum per-review difference of **5.9e-07** and a log-loss difference of
5.3e-09. This is float32 rounding noise: the Rust port and the Python RNN produce the
same predictions, on a 3.5k-review and a 3.9k-review history.

### 4b. Why the quantized path has no Python witness in this bundle

The trace's embedded `py_pred_*` is the Python RNN path, and that path does NOT apply the state
quantization (it runs `rwkv_ops.single_timestep`, not `quant_aware_rwkv7`). It is therefore an exact
target for the engine's `plain` config (section 4a) and NOT a target for the quantized one: running
the quantized config against it fails by construction, and the size of the printed difference only
measures HOW MUCH the quantization moves the predictions, not whether the port is correct. This
harness is a loose witness - it passes whenever the quantization is light - because the Python trace
exporter uses `export_rnn_trace.SrsRWKVRnn`, which has its own mixers and applies no state or shift
fake-quant at all. That is a known limitation of the exporter, not of the recipe.

The quantized Python reference lives in the EVALUATION path (`rwkv/get_result.py` with the QAT
environment), which needs the LMDB datasets and cannot be re-run from the bundle. The deploy-side
evidence for the quantized path is section 2 (label log loss), section 3 (finiteness) and the
answer-count tables in `ANSWERS.md`; the 750-user sweep in `evidence/` is the evaluation path's own
result, and the gate compares that path with itself.

The long-history reading (user 5098, 29,166 reviews) is in `evidence/rust_finite_row55_long.json` and in
the `per_user` block of `evidence/bucket/bucket_tax_row55.json`. The trace file itself (23 MB) is NOT
shipped; its feature tensors carry no weight information, so it can be re-exported with
`code/scratchpad/quant_plan/corruption/export_corrupt50_traces.py`.

## 5. Speed

```
python scratchpad\quant_plan\rust_speed.py --weights ..\weights\deploy_iter55_l2i8s4e1c.safetensors ^
    --configs plain,l2i8s4 --trace-dir ..\traces --users 107,136 --reps 2 --tag verify
```

The absolute rates depend on the machine and the load; the ratio is the number that matters
(**`evidence/rust_speed_row55.json`**):

| config | rev/s | x rev/s | states/s | x states/s | floor |
|---|---:|---:|---:|---:|---:|
| plain | 2,948.3 | 1.000 | 6,100.1 | 1.000 | |
| l2i8s4 | 2,358.7 | **0.800** | 3,979.5 | 0.652 | >= 0.75 PASS |

## 6. The answer-count bucket tables (the other team's questions)

```
python scratchpad\quant_plan\bucket_tax.py --weights ..\weights\deploy_iter55_l2i8s4e1c.safetensors ^
    --configs plain,l2i8s4 --trace-dir <a trace dir> --tag verify_bucket
```

Needs a trace directory for the users of interest (the 40-user set is not shipped: it is 40 x ~4 MB
and it is regenerated by `corruption/export_corrupt50_traces.py` from the label DB). The recorded
results are `evidence/bucket/bucket_tax_row55.json` and the three comparison files next to it; the
tables and their reading are in `ANSWERS.md`.

## 7. The gate itself (750 users) and the corruption gate

Both need the LMDBs. The recorded results are `evidence/sweep_iter55_l2i8s4e1c.json` and
`evidence/corruption_iter55_l2i8s4e1c.json`; the commands were:

```
# the QAT decay that produced the recipe (124 steps, the E1 penalty on, the Python QAT loop)
python scratchpad/quant_plan/qat_decay.py --candidate l2i8s4e1c --train-users 1 100 ^
    --max-train-len 66000 --start-folder pretrain/rwkv/opt_iter36 --start-name rwkv_iter36_558 ^
    --epochs 2 --max-steps 124 --prefix qat --tag it36l2i8s4e1c

# the fp32 control: the SAME decay with QAT switched off (its train.toml differs from the above
# only in the output folder/prefix, and its log has zero [QAT] banners)
python scratchpad/quant_plan/qat_decay.py --candidate c2_nl1i8 --no-quant --train-users 1 100 ^
    --max-train-len 66000 --start-folder pretrain/rwkv/opt_iter36 --start-name rwkv_iter36_558 ^
    --epochs 2 --max-steps 124 --prefix ctl --tag it36ctl

# the 750-user paired gate against that control
python scratchpad/quant_plan/quant_sweep.py --model scratchpad/quant_plan/qat_it36l2i8s4e1c/qat_124.pth ^
    --ref qnj_it36ctl --candidates l2i8s4e1c --shards 2 --solo 1000000 --stamp it36l2i8s4e1c ^
    --users-file scratchpad/quant_plan/roster750_seed20261119.json

# the corruption (finiteness) gate on the 50 longest-history users
python scratchpad/quant_plan/corruption/check_corruption.py --tag l2i8s4e1c
```

`qat_decay.py` takes ALL its quantization environment from `quant_sweep.CANDIDATES[<candidate>]`, so
the candidate entry IS the recipe; its environment is copied into `evidence/run/qat_env.json` and is

```
{
  "RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
  "RWKV_QAT_FUSED": "0",
  "RWKV_QAT_TAIL_PENALTY": "1.0",
  "RWKV_QAT_TAIL_STRIDE": "8"
}
```

A 0.04-epoch smoke of the same code path (different tag, folder and prefix) ran first: its config and
log are in `evidence/run/smoke_train.toml` and `smoke_train.log`, and it produced 6 steps with the
same `[tail-penalty]` lines in its log. The 750-user evaluation of the recipe took
36.0 min and reported n=750 with
`bits_ok` = True against the bars card <= 2,176 /
note <= 6,528; `evidence/run/train.log` holds the decay's own 124 steps and its
bench block (peak reserved 10.301 GiB of the 11 GiB cap).

## 8. What this does NOT settle

* Sections 2-5 use 2 or 3 users, not the 750-user roster. They are checks of the DEPLOY path, not the
  acceptance gate. The acceptance numbers are the 750-user sweep in `evidence/`.
* The corruption gate ran on the Python QAT path (`RWKV_QAT_FUSED=0`), not on the Rust engine; the
  Rust observer (section 3) covers the engine, on 3 users (2 short + 1 long).
* The forward pass of the fused CUDA kernels and the Python per-step loop are documented bit-parity
  twins (row 44 of the research log measures a 2e-6 difference between the two at this layout), but
  this bundle's own 124-step decay was run on the Python loop, and only that loop is exercised here.
* The speed numbers depend on the machine, the CPU governor and the load. Re-measure.
