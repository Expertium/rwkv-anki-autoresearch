# Rename the "GRU head" → power-curve mixture (Andrew, 2026-08-19)

> we call the power forgetting curve mixture "GRU head" (or "heads"), but that's kinda confusing.
> Sure, they were used in GRU in srs-benchmark, but it's not like having a mixture of power
> forgetting curves has anything to do with GRU fundamentally, so let's rename that

## Why the current name is wrong

The head computes

```
R(t) = Σ_i  w_i · (1 + t / S_i)^(−d_i)      w = softmax(·),  S, d = exp(clamp(·, −25, 25)) > 0
```

Three tiny linears off the shared `head_w` trunk predict `w`, `S`, `d` per row. Each component is
**exactly FSRS's power forgetting curve** with its own predicted stability and decay; the mixture is
monotone decreasing in `t` by construction because every `d_i > 0`.

There is **no gate and no recurrence** in it. The name comes only from srs-benchmark's
`models/gru.py`, where this head shape first appeared — it describes the file it was borrowed from,
not the function it computes.

## The chosen name: `power_mix`

* env flag `RWKV_POWER_MIX=N` (was `RWKV_GRU_HEAD=N`)
* params `pmix_w_weight/bias`, `pmix_s_weight/bias`, `pmix_d_weight/bias` (were `gru_*`)
* method `_power_mix_heads` (was `_gru_heads`), field `power_mix_n` / `power_mix_on`
* prose: "power-curve mixture head", N components

**Why not "curve mixture".** The head this one *replaced* was also a mixture over curves — the
128 fixed-stability basis curves of `w_head`. Calling the new one "curve mix" collides with its own
predecessor in every doc that compares them. `power_mix` says the distinguishing thing: the
components are power curves with *predicted* `(S, d)` rather than a fixed basis.

**Why not `pfc_*`.** "Power forgetting curve" is the right FSRS term and matches Andrew's wording,
but `pfc` is an opaque acronym in code. `power_mix` reads without expansion.

## Footprint

| surface | mentions |
|---|---|
| `srs_model.py` + `srs_model_rnn.py` | 53 |
| `rust/rwkv-infer` (`model.rs`, `fast.rs`) | 48 |
| runners setting the flag | **122** |
| docs (CLAUDE.md, HISTORY, log.md, …) | ~77 |
| **checkpoint parameter keys** | **6** (`gru_{w,s,d}_{weight,bias}`) |

## Backward compatibility is mandatory, not optional

Two things break under a naive rename, and both are load-bearing:

1. **Every existing checkpoint.** The champion `i53_d_10935.pth` carries `gru_*` keys and is the gate
   reference for every future run. → add a **load-time key remap** `gru_* → pmix_*` so old
   checkpoints keep loading (strict-load must still pass).
2. **122 historical runners** set `RWKV_GRU_HEAD=3`, including the queued `rgate` runner. → keep
   `RWKV_GRU_HEAD` as a **deprecated alias** that still works, so history stays reproducible. Do not
   rewrite the historical runners.

Also: `reference/weight_names.json` and the Rust weight map must accept both names, or the deploy
parity gate breaks.

## ⚠ WHY THIS WAITS FOR A FREE GPU — a mandatory guard needs one

This edits `srs_model.py`, and CLAUDE.md's rule after touching that file is to run
**`scratchpad/parity3/smoke_scripted_eval.sh`** before any launch: a plain eval is the *only* path
that scripts the model, so this bug class is invisible to training and to QAT evals. That guard is
GPU-gated, and the "no co-tenant GPU work during gate-critical runs" rule forbids running it beside
a live eval. **So the rename cannot be verified while the chain is running — that, not caution, is
the reason to sequence it.**

It is also exactly the change the diagnostic harness was built for: a full train → decay → eval cycle
at `EPOCHS=0.05` / 20 users takes ~25 min and would prove, end to end, that an old checkpoint loads
under the new names, that the deprecated flag still works, and that the eval path still scripts.

**Sequence:** chain drains → bug hunt `base` diagnostic (proves the harness) → land this rename →
re-run the diagnostic to verify it → resume the research loop.
