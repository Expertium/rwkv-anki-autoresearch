# kdalpha025 (KD `alpha_decay` 0.5 → 0.25) — pre-registered gate rule

Written 2026-08-19 09:30, before the run starts (it is third in the chain, ~7 h out). Its mechanism
prediction was registered earlier still, in `mk_kdalpha025.py`'s docstring: **it improves**, because
KD pays through target-variance reduction (an EARLY good) while calibration matters LATE, so lowering
alpha in decay should improve calibration at the moment it counts.

## ⚠ THE BOTH-MODES RULE APPLIES. Do NOT use `--curve-side` here

This is worth stating explicitly because the opposite conclusion is easy to reach from the record's
own framing, and it would be wrong.

**KD alpha rewrites BOTH objectives, from the same `kd_mix` tuple:**

| `srs_model.py` | what it rewrites | which objective |
|---|---|---|
| 1263 | `label_y = α·teacher_curve + (1−α)·hard` | curve → **ahead** |
| **1354** | `_km2_target = α·teacher_p + (1−α)·one_hot(label_rating)`, and `p_loss` is then replaced by soft-target CE against it | rating → **imm** |

Both are gated on the same `if kd_mix is not None`, and `_km_alpha` / `_km2_alpha` are the same value
unpacked from the same tuple. So `RWKV_KD_ALPHA` is a **direct** lever on the imm objective, not an
indirect one acting through the shared trunk.

### Why the curve-side exception looks applicable and is not

CLAUDE.md's curve-side exception is verified for **iter 46's self-distillation**, and correctly so:
that lever rewrote only `label_y`, so "the imm objective is `p_loss` = cross-entropy on
`label_rating`, which the lever never touches" was true *of that lever*.

It is **not** true of an external-teacher KD alpha, which replaces `p_loss` itself. The generalisation
from "KD rewrites `label_y`" to "KD is curve-side" is the trap.

Note also that the caveat the record does carry — that `label_y` reaches `p_binary_loss`, which is
skipped because `pbin_scale = 0` — names the **weaker** of the two imm paths and omits the stronger
one. Verified here: `RWKV_PBIN_SCALE` is never set in this runner, so `p_binary_loss` really is
excluded from the loss (`srs_model.py:1401-1404` adds it only when `pbin_scale != 0`). That check is
sound and simply not the binding one. ⚠ The record cites this term as `srs_model.py:1128`; it now
lives at **1365**.

### It also gives iter 55 a cleaner mechanism

iter 55 (alpha 0.9) returned imm **−0.000116**, a real regression outside the ±7.5e-5 floor. Under a
curve-side reading that could only have arrived through the trunk, which would be a surprisingly
large indirect effect. Under the actual code it is direct: alpha 0.9 replaced 90% of the imm target
with teacher probabilities, so the rating head inherited the teacher's miscalibration head-on. The
regression is exactly where the mechanism puts it.

## The rule

```
paired_pvalue.py --cand-ahead result/RWKV-kdalpha025.jsonl --cand-imm result/RWKV-P-kdalpha025.jsonl \
                 --champ-ahead result/RWKV-iter53_muonlora.jsonl \
                 --champ-imm result/RWKV-P-iter53_muonlora.jsonl --intersect
```

Accept only if **raw ≥ 0.0001 in BOTH modes vs iter 53** and **p < 0.0001 in both**. No
`--curve-side`.

Report **both** deltas: vs **iter 45** is the controlled effect of the lever (this run is iter 45's
recipe with one variable changed), vs **iter 53** is the gate.

## Prediction, and what each outcome means

* **Registered prediction: improves.** The calibration mechanism has now earned its keep twice
  before costing a run, and it predicted iter 55's direction correctly.
* **A null bounds the lever from the other side.** With iter 45 (alpha 0.5 beats no-KD-in-decay) and
  iter 55 (0.9 loses to 0.5) already bracketing it, a null at 0.25 would place the decay optimum
  near 0.5 and **close the alpha-schedule lever** rather than leaving it half-explored.
* Either way the number is assigned at verdict time; `kdalpha025` is the first run deliberately named
  for its lever with **no number in its path**.
