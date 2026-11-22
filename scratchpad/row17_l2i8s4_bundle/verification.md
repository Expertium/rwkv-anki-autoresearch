# Verification

Every command below runs from the bundle alone, except the two in section 6, which need the training
datasets (multi-GB LMDBs that are not shipped). The numbers in **bold** are what we measured on the
machine described in `README.md` section 7. Commands are Windows-style; on Linux/macOS replace
`python` with `python3`, `\` with `/` and `set VAR=x` with `export VAR=x`.

## 0. Setup

```
cd code
cargo build --release --manifest-path rust/rwkv-infer/Cargo.toml      # first build needs network
python -m pip install numpy scikit-learn                             # only for the scoring scripts
```

`verify.cmd` in the bundle root runs sections 2, 3 and 4 in one go.

## 1. The shipped deploy weights ARE the shipped `.pth`

`weights/deploy_iter17_l2i8s4.safetensors` was exported from `weights/qat_l2i8s4_124.pth` by the
repo's own exporter. Re-export it and compare:

```
python scratchpad\quant_plan\export_weights_only.py ..\weights\qat_l2i8s4_124.pth tmp_repro deploy_repro.safetensors
python -c "from safetensors import safe_open; import numpy as np; a=safe_open('tmp_repro/deploy_repro.safetensors','pt'); b=safe_open('../weights/deploy_iter17_l2i8s4.safetensors','pt'); ka=sorted(a.keys()); print(len(ka), sorted(b.keys())==ka, max(float(np.abs(a.get_tensor(k).float().numpy()-b.get_tensor(k).float().numpy()).max()) for k in ka))"
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
python scratchpad\quant_plan\rust_grid.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors ^
    --trace-dir ..\traces --users 107,136 --configs plain,l2i8s4 --tag verify
```

Expected (**`evidence/rust_grid_iter17_l2i8s4.json`**):

| config | card/note bits | ahead | imm |
|---|---:|---:|---:|
| plain | 34,816 / 104,448 | 0.243849 | 0.197007 |
| l2i8s4 | **1,280 / 2,304** | 0.244184 | 0.197323 |
| tax | | **+0.000335** | **+0.000316** |

Both taxes are inside the +0.0005 ceiling. The immediate-head tax (+0.000316) is close to the
750-user Python gate's +0.000327 for this recipe.

On the long-history user (`traces_long/`, 29,166 reviews, 23,455 scored) the quantized state scores
BETTER than the unquantized one on both heads (imm -0.003950, ahead -0.003295), which is expected and
not a contradiction: this checkpoint was trained WITH the quantized state in its forward pass, so the
unquantized state is the out-of-distribution case for it.

## 3. Non-finiteness, strict

```
python scratchpad\quant_plan\rust_finite.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors ^
    --configs plain,l2i8s4 --trace-dir ..\traces --users 107,136 --tag verify --strict
```

`--strict` makes the engine exit with code 3 if it sees ONE non-finite value, so a zero exit code is
an independent witness. Expected (**`evidence/rust_finite_iter17_l2i8s4.json`**):

```
plain        PASS  rc=0 state_vals=0         ... p_vals=19476 p_bad=0
l2i8s4       PASS  rc=0 state_vals=43474944  state_raw_bad=0 state_quant_bad=0 vec_raw_bad=0
                     vec_quant_bad=0 p_vals=19476 p_bad=0 state_maxabs=2.1532e-01 vec_maxabs=5.1538e+00
```

**43,474,944 state values and 19,476 p(recall) values, zero non-finite**, with the largest state entry
0.215 and the largest token-shift entry 5.15. `state_vals=0` for `plain` is correct: with no
quantization there is no state compression step to observe.

The same check on the long-history trace is `evidence/parity/long_user_*.json` plus
`rust_finite.py --trace-dir ..\traces_long --users 5098`.

## 4. Python <-> Rust parity

### 4a. The engine reproduces the Python RNN exactly (no quantization)

Each trace file carries the Python RNN's own per-review predictions (`py_pred_imm`,
`py_pred_ahead`) for the same weights. `py_rust_parity.py` compares the engine's output with them on
the trace's equalize set:

```
python scratchpad\quant_plan\py_rust_parity.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors ^
    --trace-dir ..\traces --config plain --users 107,136 --tag verify_plain
python scratchpad\quant_plan\py_rust_parity.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors ^
    --trace-dir ..\traces_long --config plain --users 5098 --tag verify_plain_long
```

Expected (**`evidence/parity/107_plain_rust_vs_python_rnn.json`**,
**`evidence/parity/long_user_plain_rust_vs_python_rnn.json`**):

| user | equalized reviews | max per-review diff (imm) | max per-review diff (ahead) | log-loss diff |
|---|---:|---:|---:|---:|
| 107 | 3,465 | 4.15e-07 | 5.41e-07 | 0.000000 |
| 136 | 3,865 | 1.90e-07 | 5.61e-07 | 0.000000 |
| 5098 | 23,455 | 3.47e-07 | 6.79e-07 | 0.000000 |
| 6251 | 12,915 | 4.90e-07 | 8.52e-07 | 0.000000 |

This is float32 rounding noise: the Rust port and the Python RNN produce the same predictions, on a
short history and on a 29k-review history.

### 4b. The quantized path

The trace's embedded `py_pred_*` is the UNQUANTIZED Python RNN, so it cannot serve as the target for
the quantized run. The Python quantized reference comes from the evaluation path
(`rwkv/get_result.py` with the QAT environment), which needs the LMDB datasets and therefore cannot
be re-run from the bundle. We measured it and compared:

| user | head | Python eval (quantized) | Rust engine (quantized) | difference |
|---|---|---:|---:|---:|
| 5098 | immediate | 0.195803 | 0.195720 | **-0.000083** |
| 5098 | ahead | 0.195928 | 0.201681 | +0.005753 |
| 6251 | immediate | 0.494987 | 0.497969 | +0.002982 |
| 6251 | ahead | 0.528551 | 0.535332 | +0.006781 |

The immediate head agrees to 8e-05 on the long user. The ahead head does not agree - and the reason
is NOT the quantization: the two paths build the ahead head's time input differently. Running the
same comparison on the UNQUANTIZED control checkpoint shows the same gap, on both heads
(`evidence/corruption/` and `result/adw*_corrupt_ctl_fp32.jsonl`):

| user | head | Python eval (control, no quantization) | Rust engine (control, no quantization) | difference |
|---|---|---:|---:|---:|
| 5098 | immediate | 0.198314 | 0.198640 | +0.000326 |
| 5098 | ahead | 0.197319 | 0.203667 | **+0.006348** |
| 6251 | immediate | 0.487134 | 0.489148 | +0.002014 |
| 6251 | ahead | 0.527618 | 0.531727 | +0.004109 |

So: the immediate head is comparable between the two paths and agrees to a few 1e-04 (the Python
numbers here were produced in bfloat16 on a GPU; a float32 re-run would tighten them), while the
ahead head differs by ~0.005 REGARDLESS of quantization, i.e. the difference is an INPUT difference
between the evaluation path (LMDB) and the trace exporter's frame path - `export_rnn_trace.py`'s own
docstring warns that its frame replica and the shared `data_processing` path have not converged yet.

Practical consequence: compare ahead numbers only WITHIN one path (the engine against itself, or the
evaluation against itself). The deploy tax in section 2 is a within-path comparison and is valid.

## 5. Speed

```
python scratchpad\quant_plan\rust_speed.py --weights ..\weights\deploy_iter17_l2i8s4.safetensors ^
    --configs plain,l2i8s4 --trace-dir ..\traces --users 107,136 --reps 2 --tag verify
```

Measured on a LOADED machine (other experiments were running), so the absolute rates are low; the
ratio is the number that matters (**`evidence/rust_speed_bundle.json`**):

| config | rev/s | x rev/s | states/s | x states/s | median per-user x | floor |
|---|---:|---:|---:|---:|---:|---|
| plain | 1,965.6 | 1.000 | 4,054.7 | 1.000 | 1.000 | |
| l2i8s4 | 1,608.2 | **0.818** | 2,768.1 | 0.683 | **0.821** | >= 0.75 PASS |

The idle-machine measurement of the same recipe (**`evidence/rust_speed_idle.json`**, taken with no
other work running) reads plain 2,915.1 rev/s and l2i8s4 2,357.8 rev/s = **0.809x**, median per-user
0.789x. The two agree, so the load shifted the absolute rates by about 33% but not the ratio.

## 6. The training-side numbers (need the datasets)

The 750-user gate and the 50-user corruption gate both need the LMDBs. The recorded results are
`evidence/sweep_iter17_l2i8s4.json` and `evidence/corruption_iter17_l2i8s4.json`; the commands were:

```
# the QAT decay that produced the recipe (124 steps, ~3 min, 6.6 GB peak)
python scratchpad/quant_plan/qat_decay.py --candidate l2i8s4 --train-users 1 100 --max-train-len 66000 ^
    --start-folder pretrain/rwkv/opt_iter36 --start-name rwkv_iter36_558 --epochs 2 --max-steps 124 ^
    --prefix qat --tag it36l2i8s4

# the fp32 control: the SAME decay with QAT switched off (its train.toml differs from the above
# only in the output folder/prefix, and its log has zero [QAT] banners)
python scratchpad/quant_plan/qat_decay.py --candidate c2_nl1i8 --no-quant --train-users 1 100 ^
    --max-train-len 66000 --start-folder pretrain/rwkv/opt_iter36 --start-name rwkv_iter36_558 ^
    --epochs 2 --max-steps 124 --prefix ctl --tag it36ctl

# the 750-user paired gate against that control
python scratchpad/quant_plan/quant_sweep.py --model scratchpad/quant_plan/qat_it36l2i8s4/qat_124.pth ^
    --ref qnj_it36ctl --candidates l2i8s4 --shards 2 --solo 1000000 --stamp it36l2i8s4 ^
    --users-file scratchpad/quant_plan/roster750_seed20261119.json

# the corruption (finiteness) gate on the 50 longest-history users
python scratchpad/quant_plan/corruption/check_corruption.py --tag l2i8s4
```

A 0.04-epoch smoke of the same code path (different tag, folder and prefix) ran first; its log and
config are in `evidence/run/smoke_train.log` and `smoke_train.toml`.

## 7. What this does NOT settle

* Sections 2-5 use 2 or 3 users, not the 750-user roster. They are checks of the DEPLOY path, not the
  acceptance gate. The acceptance numbers are the 750-user sweep in `evidence/`.
* The corruption gate ran on the Python QAT path (`RWKV_QAT_FUSED=0`), not on the Rust engine. The
  Rust observer (section 3) covers the engine, on 2-3 users. A Rust finiteness run over all 50
  corruption-roster users is possible with `export_corrupt50_traces.py` +
  `rust_finite.py --trace-dir <those traces>`; it was started and had not finished when this bundle
  was built.
* `traces/` covers users 107 and 136. User 156 could not be re-exported for this lineage: the current
  label database puts review 1142 of user 156 in the equalize set, and the exporter's own per-review
  prediction dictionary has no entry for it (the evaluation path's vectorised gather does not raise
  on that; the exporter's slow path asserts). Nothing about the recipe depends on that user.
* The speed numbers depend on the machine, the CPU governor and the load. Re-measure.
