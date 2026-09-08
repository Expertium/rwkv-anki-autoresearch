# Both deploy QAT catalogs are STALE for the gen-5 trunk (2026-09-08)

**Found before the endgame's QAT arm spent a single GPU-hour on it, and it would have been silent.**

## The check

A fitted codebook is validated by SHAPE and used on CONTENT. `d_model`, `H` and `K` are unchanged
from the trunk the live catalogs were fitted on (2026-08-12, published 92-dim inputs), so every
shape assert passes and a stale catalog cannot announce itself. The only instrument that exposes
one is a RANDOM control at the same budget: **1.0 = encode everything to zero**, and a catalog
that loses to random directions is doing worse than nothing.

Corpus: `rwkv-infer --dump-corpus` / `--dump-shift-corpus` on the certified gen-5 parity model
(`reference_realcyc/rwkv_ref_558.safetensors`), users 107/136/156, stride 1 -> **15,854 WKV states
(79,270 joint (u,v) vectors) and 23,787 shift vectors**. Held out by USER (fit on 107+136, score
on 156), which is the honest analogue of fitting on a handful of users and deploying to thousands.

## The numbers

| catalog | held-out rel. L2 | random control | verdict |
|---|---|---|---|
| `pq_cb_wkv_c80_b10.txt` (live) | **0.9416** | 0.9582 | 1.7% better than random |
| refit on gen-5 (2 users) | **0.5185** | -- | what a refit buys: **+0.4231 (45%)** |
| oracle (fit on the holdout) | 0.2932 | -- | floor-ish at 1024 centroids |
| `pq_cb_shift_c80_m2b12.txt` (live), TS | **1.0352** | 0.9700 | **worse than encoding to ZERO** |
| `pq_cb_shift_c80_m2b12.txt` (live), CS | **1.0811** | 0.9728 | **worse than encoding to ZERO** |
| shift refit on gen-5 (2 users), TS / CS | **0.3933 / 0.3814** | -- | generalises to a held-out user |

## What shipped

* **`reference/pq_cb_wkv_gen5_b10.txt`** -- header `1 10 32 16 1024`, byte-compatible with the
  live catalog, so **deploy state size is unchanged**.
* **`reference/pq_cb_shift_gen5_m2b12.txt`** -- header `2 12 40 80 4096`, likewise.
* Both load in the Rust engine under `RWKV_LOWRANK_PQ` / `RWKV_SHIFT_PQ` and produce quantized
  states (smoke-tested with `card:int4,note:int4` + norm 1 bit).
* The OLD catalogs are NOT replaced or deleted: they are correct for the published-features
  trunk, which is what every recorded QAT number was measured on.

## The caveat that keeps this honest

Reconstruction error **cannot rank two working catalogs** for logloss -- the 2026-08-15 measurement
is explicit that the learned catalog which cut the QAT tax 45% reconstructs WORSE than the frozen
one it started from (0.513 vs 0.334), because centroids train on the TASK loss and nothing
optimizes them for reconstruction. **That is not what this is.** This is a catalog at or past the
encode-to-zero bound, i.e. the q72u signature, and the q72u swap was priced at a measured
**+0.003235 ahead / +0.004183 imm** of recovered cost for zero extra bytes. Going from
worse-than-zero to a working catalog is a repair, not a ranking.

⚠ The run also sets `RWKV_QAT_PQ_LEARN=1` / `RWKV_QAT_SHIFT_PQ_LEARN=1`, so the catalogs adapt
during training and would have recovered *some* of this. That is an argument for the size of the
loss being uncertain, not for leaving a broken starting point in place.

## Owed before the QAT arm launches

1. **Re-run this on ws10's WS-final checkpoint**, which is the model arm 2 actually branches from.
   realcyc and ws10 share architecture, features and recipe and differ only in budget, so today's
   refit is the right interim -- but the final catalog should come from the final trunk. The whole
   procedure is ~10 min of CPU: dump corpus, `wkv_cb_staleness.py`, `shift_cb_staleness.py`, refit.
2. **Enlarge the corpus.** 4,096 centroids per (role, pos) chunk from ~10k training vectors is thin
   (the 2026-08-12 shift refit reached 0.1902 held-out against this one's 0.3933). More users needs
   more parity traces; `export_rnn_trace.py` hardcodes `REF_USERS = [107, 136, 156]`.
