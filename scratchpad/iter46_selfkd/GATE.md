# Iter 46 — the gate, PRE-REGISTERED before the run reported

Written 2026-08-12, while iter 46's WS phase was still running and hours before any number exists.
Fixing the criterion after seeing the result is the failure the budget calibration was careful to
avoid; this file is the record that it was fixed first.

> **⚠ This lives here and NOT in `run_iter46.cmd` because that runner is EXECUTING.** cmd.exe
> re-reads a batch file from a saved byte offset every time a command returns, so inserting lines
> into a running `.cmd` corrupts the resume point — that is exactly how iter 43's chain died. I
> edited the header here by reflex, caught it, and reverted to byte-identical launched content
> before the WS command returned (the whole window sat inside a ~2.5 h python invocation, so
> cmd.exe never re-read it). **Annotate a running experiment in a SEPARATE file.**

## The rule (Andrew, 2026-08-12)

> "ahead better, imm not (statistically) significantly worse"

Iter 46 is a **curve-side** lever, so it is judged on ahead with a do-no-harm guard on imm, not on
the usual both-modes rule.

**Why the both-modes bar does not apply here.** The self-distillation teacher is `.detach()`ed, so
no gradient reaches the rating head; and the imm objective is `p_loss` = cross-entropy on
**`label_rating`**, which this lever never touches — it rewrites `label_y`, which feeds only
`curve_loss` / `curve_raw_loss` / the PAVA probe target. imm can therefore move **only through the
shared trunk**. Demanding imm also improve by ≥0.0001 would be demanding a side effect the
mechanism cannot produce. (`CLAUDE.md` §9 already stated the general form: "a curve-side change
moves only one of the two gate modes".)

⚠ `label_y` *does* reach one imm-side term — `p_binary_loss`, `srs_model.py:1128` — but
`pbin_scale = 0` in this recipe, so it is skipped. **If `RWKV_PBIN_SCALE` is ever turned on, this
exception stops being valid.**

## Accept iff both hold

1. **ahead**: raw improvement **≥ 0.0001** vs the iter-45 champion, with one-sided paired
   Wilcoxon **p < 0.0001**.
2. **imm**: **not significantly worse** = NOT (imm by-user mean declines AND the one-sided paired
   Wilcoxon for "candidate worse" gives **p < 0.05**).

Everything else is unchanged and expected to be trivially satisfied, since this is a training-only
change: `size` identical on all 2500, params exactly 558,212, card/note/deck state unchanged,
nan_users 0.

**Both halves of the harm test are load-bearing, and iter 44 is the proof.** Its imm mean moved
−0.000001 (nominally worse) while the *rank* test said the candidate was BETTER at p = 1e-4 — most
users improved slightly, a few worsened a lot. A rank-only guard would fail a magnitude-null
iteration; a mean-only guard fires on noise. Requiring both means the guard trips only when the
decline is real in the metric the gate actually uses (the by-user mean).

## Command

```
python optimization/paired_pvalue.py --curve-side --intersect \
  --champ-ahead result/RWKV-iter45_kddecay-s0.jsonl \
  --champ-imm   result/RWKV-P-iter45_kddecay-s0.jsonl \
  --cand-ahead  result/RWKV-iter46_selfkd-s0.jsonl \
  --cand-imm    result/RWKV-P-iter46_selfkd-s0.jsonl
```

Exit 0 = pass. `--harm-alpha` tunes the 0.05. The tool also *notes*, without failing, an imm
decline larger than the 7.5e-5 same-capacity noise floor that the rank test did not call worse —
that is a "look before promoting" signal, not an automatic reject.

## Scope

Use `--curve-side` **only** for levers that touch the curve/ahead objective alone: self-distillation,
PAVA lambda, ahead-target and monotonicity changes, duration handling. Trunk / optimizer / capacity
/ topology changes keep the BOTH-modes rule, because those genuinely can move both metrics.
Precedent for the shape: iter 36 (PAVA lambda 0.1 → 0.2), directed-accepted on a 5.9:1
ahead-for-imm trade.
