# Optimization protocol (roadmap steps 4–5–7, merged loop)

Authoritative rules for the autoresearch loop that shrinks/speeds the RWKV model.
Every iteration is measured the same way and logged append-only. Agreed with Andrew
2026-06-27.

## Scope of allowed changes

Both **exact** (bit/float-noise) and **inexact** (accuracy-affecting) changes are allowed:
training pipeline, hyperparameters, and **model architecture** may all change. Start with
the **biggest wins first** — parameter-count reduction (target: 2–3× first, plausibly
10–15× with tuning, à la LSTM→GRU in srs-benchmark).

**Two hard invariants (never change):**
1. **Hierarchy** — information flows `card → note → deck → preset → global`. The 5-stream
   chained structure in that order stays.
2. **Inputs** — the model must still consume the *same preprocessed data* (the 92-dim
   feature vectors / existing LMDBs). No new/changed inputs; the new RWKV must run on the
   old preprocessed data.

## The 5 gates (a change is kept only if ALL pass)

| # | Gate | Rule |
|---|------|------|
| 1 | **LogLoss (both modes)** | ahead AND imm by-user-mean LogLoss must not worsen by **> +0.0015** vs **iteration 0** (the frozen Rust baseline). Exact changes ≈0; a real rise is a red flag, not "budget to spend". |
| 2 | **Review count ("size")** | per-user equalized review count must stay **identical** (it's a property of the data + filters, not the model — any change here means a pipeline bug). |
| 3 | **State size** | per-card RNN state (card_id stream) must stay the **same or shrink**, never grow. Baseline 13,056 floats = 51.0 KiB f32. |
| 4 | **Hierarchy** | card→note→deck→preset→global preserved. |
| 5 | **Inputs** | runs on the same preprocessed data. |

GPU training speed is **untimed** (point 3 of Andrew's spec) — prefer it not balloon, but
it does not gate.

## Eval recipe (fixed for every iteration)

- **Train** on users **1–100**, **evaluate** on the held-out **101–200** (all 100).
- LogLoss via the CUDA parallel path: `python -m rwkv.get_result --config
  rwkv/get_result_config_iter0.toml` (adapt MODEL_PATH per iteration), **bfloat16**
  (the data pipeline feeds bf16 batches; tolerance is *relative* so bf16-vs-bf16 across
  iterations is consistent). By-user mean over the 100 users, both `RWKV` (ahead) and
  `RWKV-P` (imm).
- **Rust↔Python parity invariant:** every iteration must still pass `verify_rust.py`
  (3-user, float32) so the deployable Rust engine stays bit-exact with the trained model.
  (Decided with Andrew: LogLoss tolerance is checked on GPU bf16 for speed; Rust parity is
  the separate correctness check.)
- Fixed training recipe = `rwkv/train_rwkv_config_ref100.toml` (WS, 18 epochs, peak LR
  7e-4) unless an iteration deliberately tunes training (then note it).

## Speed = batch throughput (Wilcoxon)

Anki serves via **batching** (cf. github.com/JSchoreels/anki), so the metric is **batch
throughput** (reviews/s), measured on CPU with the **locked frequency** (base 3400 MHz; see
below). Per-card streams are inherently sequential — batching parallelizes across
*independent* card-streams, which does **not** change outputs (so batching is a free,
exact speedup; iteration 0 is batch size 1, the honest baseline).

**Measurement = simultaneous, fixed-duration, paired trials + one-sided Wilcoxon:**
- One **trial** = run *before* (champion) and *after* (candidate) **at the same time**,
  each pinned to **3 threads**, each looping over the **same frozen pre-chosen batch set**,
  for a fixed wall-clock **T ≈ 20–30 s**; count reviews each finishes → one paired point
  `(thru_before, thru_after)` from the identical time window (external load cancels).
- Repeat **K ≈ 10** trials (discard 1–2 warm-ups). One-sided **Wilcoxon signed-rank** on
  `after − before`; accept the speedup only if **p < 0.01**.
- Pairing the *trial* (not the batch) keeps pairs independent and avoids the faster process
  racing ahead / tail bias. Cost is bounded by K·T (~3–5 min) regardless of model speed.

### Locking CPU frequency (run as admin, once per session)
```
powercfg -attributes SUB_PROCESSOR 75b0ae3f-bce0-45a7-8c89-c9611c25e100 -ATTRIB_HIDE
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCFREQMAX 3400
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100
powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
powercfg /setactive SCHEME_CURRENT
```
(`PROCFREQMIN` is not a valid alias — pin the perf state to 100% instead. Restore with
`PROCFREQMAX 0`, `PROCTHROTTLEMIN 5`.)

## Logging (append-only; never rewrite history)

Two files, one row per iteration:
- `optimization/log.jsonl` — machine-readable, full record incl. `comment`.
- `optimization/log.md` — human-readable table, **excludes** `comment`.

Per-iteration fields:
`number`, `timestamp` (ISO), `logloss` `{ahead, imm}`, `state_size_floats`,
`throughput` (median reviews/s), `wilcoxon_p` (one-sided Wilcoxon p of the speedup vs the
comparison model — decimal when large, scientific when tiny), `review_count_check`
(pass/fail), `logloss_tolerance_check` (pass/fail vs iter-0 +0.0015 both modes),
`state_size_check` (pass/fail ≤ iter-0), `summary` (≤15 words, written **before** running),
`comment` (any length, written **after**; jsonl only).

Keep dead ends in the log so they aren't re-tried (CLAUDE.md §8).

## Iteration 0 (frozen baseline)

Model `pretrain/rwkv/ref_100/rwkv_ref_558.pth` (current arch, trained 1–100). Params
2,762,884; per-card state 13,056 floats (51.0 KiB). LogLoss filled from the 101–200 eval.
