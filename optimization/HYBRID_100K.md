# A hybrid FSRS-RWKV at <=100k parameters — design note

**Status: DESIGN + CPU MEASUREMENT ONLY. Nothing here has been trained.** Andrew's ask
(2026-08-24): *"Thanks to FSRS-7, we know that it's possible to make a solid spaced
repetition algorithm with as little as 34 params. GRU in srs-benchmark beats FSRS-7 and only
has 503 params. So the recurrent core for using only interval lengths and grades is simple
in some sense. So it's probably possible to compress RWKV even further by using a much
simpler recurrent core for interval lengths + grades, and allocating >=99% of parameters to
how other input features are processed. More concretely, try thinking how to make a hybrid
FSRS-RWKV with <=100k parameters."*

Tooling: `scratchpad/hybrid100k/` — `param_map.py` (where the 558,212 live),
`budget.py` + `compose.py` (price a candidate by BUILDING it), `stream_budget.py` (what each
stream actually contributes). All CPU, one thread.

---

## 1. The answer up front

**<=100k is reachable today, with no new code — and Andrew's rebalance is expressible as a
one-file arch change.** Three exactly-priced designs:

| design | params | feature MLP | heads | 5 streams | card state (floats) |
|---|---|---|---|---|---|
| champion (d=80, 13 layers) | 558,212 | 56,080 (10%) | 62,822 | 439,310 (79%) | 2,880 |
| **A** d=32, ctx 1/2/2/2, fm4 | **84,007** | 16,032 (19%) | ~11,574 | 56,401 (67%) | 1,152 |
| **B** d=32, ctx 1/2/2/2, fm8 | **100,263** | 32,032 (32%) | ~11,574 | 56,401 (56%) | 1,152 |
| **C** d=32, ctx 1/1/1/1, fm12 | **99,596** | 48,032 (48%) | ~11,574 | 39,478 (40%) | 1,152 |

(`fm` = `features_fc_mult`; ctx = layer depth of note/deck/preset/user. Card stays 2 layers.
Every row was constructed and counted, not estimated — see §5 for why estimating is unsafe.)

B and C are **matched at ~100k and differ only in where the parameters sit**, so they turn
Andrew's hypothesis into a controlled two-arm experiment rather than a belief.

But three things in the premise need correcting first, and they change what the budget should
buy.

---

## 2. Correction 1 — FSRS's 34 and GRU's 503 are PER USER; ours is one frozen net

This is the load-bearing one.

| model | params | how it is fitted | by-user mean LogLoss (10k users) |
|---|---|---|---|
| FSRS-6 | 21 | **per user**, per time-series split | 0.38424 |
| FSRS-7 | 34 | **per user** | 0.32056 |
| FSRS-7 + recency | 34 | **per user** | 0.31782 |
| GRU | 503 | **per user**, warm-started from a reptile meta-init | 0.31457 |
| RWKV (ahead) | 2.76M | **frozen, user-independent** | 0.29743 |
| RWKV-P (imm) | 2.76M | **frozen, user-independent** | 0.26600 |

(Computed from `srs-benchmark/result/*.jsonl`, n=10000 each. The mode comparable to
FSRS/GRU is **ahead**; `-P`/imm is a different, easier task.)

Across 10,000 users, FSRS-7 is **340,000** parameters of user-specific information and GRU is
**5.03M**. Our whole model is 558,212 — *for everyone*. So "GRU does it with 503" does not
say the function is 503 parameters big. It says the function is 503 parameters big **once you
already know the user**.

**The correct decomposition is therefore not "recurrent core vs feature processing". It is:**

- the **update rule** over (interval, grade) — genuinely cheap; GRU proves 7 state dimensions
  and ~500 weights suffice;
- **amortized inference of the per-user / per-deck parameters** — the thing FSRS gets by
  running an optimizer per user, and which a frozen net must do *in-context* from history.

The second is what our context streams are for, and it is where the parameters actually are.

## 3. Correction 2 — the (interval, grade) recurrent core is already only 13%

Measured breakdown of the champion under its exact env (`param_map.py`):

| pool | params | share |
|---|---|---|
| **card stream** (2 layers) — the per-card recurrence | **72,508** | **13.0%** |
| note stream (1 layer) | 42,573 | 7.6% |
| deck stream (4 layers) | 145,413 | 26.0% |
| preset stream (3 layers) | 89,408 | 16.0% |
| user stream (3 layers) | 89,408 | 16.0% |
| input feature MLP (92 -> 320 -> 80) | 56,080 | 10.0% |
| output heads | 62,822 | 11.3% |

By role: WKV recurrence 59.8%, channel mixers 9.4%, LoRA projections 5.4%, norms/lerps 4.1%,
feature MLP 10.0%, heads 11.3%.

**Deleting the card stream outright takes 558k to 486k.** The <=100k target cannot come from
simplifying the interval/grade core — it must come from the **four context streams, 65.7% of
the model**, whose job is not the interval/grade recurrence at all.

## 4. Correction 3 — the "simple head" half of the idea is already shipped

`RWKV_GRU_HEAD=3` replaces the 128-basis-curve mixture with exactly srs-benchmark's GRU form:
a power-law mixture `sum_i w_i (1 + t/s_i)^(-d_i)` with N=3 (GRU uses N=2). Verified in code:
under `gru_on`, `ahead_linear` and `w_linear` both collapse to `Linear(1,1)`
(`srs_model_rnn.py:215-228`).

**Consequence worth recording: `num_curves` and `num_points` are DEAD CONFIG KNOBS on this
trunk.** Setting them 128 -> 64 changes the parameter count by exactly zero. The d=32-era
"SRS heads 128 -> 64" lever no longer exists here; the live head knob is `head_fc_mult`.

---

## 5. What the streams actually contribute — and the one finding that kills the obvious design

`stream_budget.py` hooks every stream's layers during real CPU inference and measures the
delta each stream adds to the residual chain. 3 VAL users, ~5.5k state-advancing reviews each.

| stream | rel \|delta\| | eff. rank (participation ratio) | dims for 95% var | within-entity var |
|---|---|---|---|---|
| card | 0.334 | 5.55 | 19.0 | 0.428 |
| note | 0.098 | 7.04 | 25.3 | 0.680 |
| deck | 0.404 | 6.92 | 34.7 | 0.890 |
| preset | 0.258 | 9.52 | 31.7 | *(degenerate — see below)* |
| user | 0.257 | 8.37 | 28.7 | n/a (one entity per user) |

**(a) Every stream injects a ~6–10 dimensional signal into an 80-dimensional trunk.** No
stream's contribution has a participation ratio above 9.6. d=80 is generous for what any one
stream does.

**(a2) But the streams do NOT all share one narrow subspace — and this is what sizes the
trunk.** `union_rank.py`, on two users independently (5,262 and 5,842 reviews); the two agree
closely, which is the check that matters for a single-user statistic:

| cloud | eff. rank (PR) | dims for 95% var |
|---|---|---|
| final representation `x` (what the heads see) | 5.80 / 5.21 | 31 / 33 |
| sum of all five stream deltas | 5.99 / 5.19 | 32 / 33 |
| **union — all five streams' deltas stacked** | **12.06 / 12.89** | **44 / 48** |
| *(if the five were fully disjoint)* | — | *~137 / ~149* |

Each stream's own top-8 basis captures 70–85% of its own energy but only 10–32% of any other
stream's, and the overlap is largest between *adjacent scopes* (deck<->preset 0.24–0.27,
preset<->user 0.28–0.32) — exactly the shape the scope ladder predicts.

**Sizing consequence, and the conservative statistic is the one to use.** Participation ratio
(6–12) says the signal is concentrated; d95 (31–44) says the tail is long. Both describe a
model that *had* 80 dimensions to spread into, so neither is a hard requirement — a model
trained narrower would re-encode. But the risk here is **under**-sizing, and a typical-case
statistic cannot bound a worst case (the lesson from iter 51's median-vs-max blow-up). So take
**d95(union) = 44–48** as the cautious trunk width, i.e. **d≈48, not d=32**.

**(b) THE STRUCTURAL WASTE: the interleaved residual chain forces all five streams to share
one `d_model`.** One vector is threaded through all 13 layers, so every stream pays `4d^2` per
layer at the *same* `d` — even though each injects under 10 effective dimensions. This is the
single largest inefficiency in the model, and it is precisely the constraint Andrew's
restructuring would break.

**(c) The contributions are largely TIME-VARYING within an entity, not static per-entity
codes.** Card is 43% within-entity, note 68%, deck 89%. **This kills the cleanest version of
the hybrid** — a hypernetwork that emits a constant per-entity code `z` for a tiny card core
would discard most of what the context streams do. Any conditioning path must itself stay
recurrent.

> **DO NOT QUOTE preset's 0.998 or read anything into it.** Users have 1–2 presets, so there
> is essentially no between-entity variance to measure inside one user and the decomposition
> is degenerate by construction. The card / note / deck numbers rest on 461–1302 entities and
> are meaningful; preset's does not. The user stream's cross-user analogue: the per-user mean
> delta varies across users by 43% of its own magnitude, so there *is* real per-user identity
> in it.

> **AND THIS IS A SIZING PRIOR, NOT AN ACCURACY CLAIM.** Contribution magnitude is not value —
> that is exactly what the delta-rule ablation taught (0.15 of eigenvalue movement, +0.208 imm
> when removed). The note stream's small `rel|delta|` of 0.098 is a reason to *ablate it and
> measure*, never a reason to cut it.

---

## 6. Why "≥99% on feature processing" should not be the target

Two measured reasons, and they point the same way:

1. **capacity-at-5k is 0/3.** Three separate capacity additions (num_curves/points 64->128,
   channel_mixer 1.0->1.5, and iter 49's user/preset layer-0 mixers at +4.7% params) all
   returned nothing. Pouring ~99k into the feature MLP is a fourth capacity add, on the side
   of the model that is already a 2-layer MLP with a 320-wide hidden layer.
2. **Shrinking has 4.95x of precedent at ~0.001 cost** (A0 -> A18: 2.76M -> 558k for
   +0.00096 ahead / +0.00053 imm), while the model is *data*-limited, not capacity-limited.

So the promising half of Andrew's idea is the **shrink**, not the **reallocation**. The
reallocation is still worth testing — that is what arms B and C are for — but it should be
tested, not assumed, and it should not be the reason to spend the budget.

Reaching literally ">=99% on feature processing" is also structurally impossible while any
real recurrence survives: at ~100k total, arm C already pushes the non-recurrent half to 60%
(48k feature MLP + 11.6k heads), and going further means deleting streams that finding (c)
says are doing time-varying work.

---

## 6b. The conflict the measurements create — and the design that resolves it

d95(union)=44 argues for a d≈48 trunk. But **d=48 does not fit under 100k with the current
shared-width architecture**, even gutted (all constructed, not estimated):

| d=48 design | layers | params |
|---|---|---|
| ctx 1/1/1/1 | 6 | 134,200 |
| ctx 1/1/1/1, `head_fc_mult=2` | 6 | 123,544 |
| card 1L + ctx 1/1/1/1 | 5 | 122,573 |
| card 1L + ctx 1/1/1/1 + hm2 | 5 | **111,917** — still over |

So with one shared width the choice is: d=48 and miss the budget, or d=32 and truncate the
union's 95% tail. **That dilemma is an artifact of the residual chain, not of the task.**

**The design the measurements actually point to: decouple per-stream width from trunk width.**
Give each stream an internal width `d_s` with small projections in and out of a shared trunk
of width `d_t`. Justified directly by §5: each stream injects only ~6–10 effective dimensions
(so `d_s` can be small), while the union needs ~44 (so `d_t` must stay wide).

Cost per layer changes from `4·d_t²` to `4·d_s² + 2·d_t·d_s`:

| | shared d=48 | `d_s`=24, `d_t`=48 | `d_s`=16, `d_t`=48 |
|---|---|---|---|
| per layer (no cmix) | 9,216 | 4,608 | 2,560 |

> **ESTIMATE, NOT CONSTRUCTED.** Unlike every other number in this note, the following was not
> built and counted — the code does not exist yet. §1's own rule says estimates are unsafe, so
> treat it as a feasibility sketch: at `d_s`=16, `d_t`=48 with 9 layers, the streams land near
> ~34k, leaving ~66k for the feature MLP and heads at `d_t`=48. That fits, keeps the trunk wide
> enough for the measured union, **and** funds the feature-side rebalance Andrew asked for.

Per-card state also follows `d_s`, not `d_t`: at `d_s`=16 (H=1, K=16) it is 576 floats against
the champion's 2,880.

**Cost of this option: real code.** Per-stream projections touch `rwkv_model.py`, the arch
config, the RNN deploy path, and the Rust engine. It is not a one-file arch change like arms
A/B/C, and it would need its own three-way-parity case.

## 7. Recommended experiment order

All three arms are one generated arch file each; no model-code change is required.

1. **Arm A (84,007 params, fm4)** — the pure shrink, structure preserved. Cheapest test of
   "can 6.6x fewer parameters hold the line?", and it doubles as **the direct test of the
   d95(union)=44 concern**: A runs at d=32, so if the union's 95% tail really matters, A is
   where it shows up. Run it first, because its result decides whether §6b's decoupled design
   is needed at all — a healthy A makes that code unnecessary, a bad A is the evidence that
   justifies writing it.
2. **Arm B (100,263, fm8) vs Arm C (99,596, fm12 + shallower context)** — matched total,
   feature MLP at 32% vs 48%. This is Andrew's hypothesis as a controlled comparison. Only
   worth running if A is not a disaster.

**Pre-registration — ANDREW 2026-08-28: use the TRACK-2 RATIO GATE, not the efficiency budget.**
An earlier version of this section pre-registered "both modes within +0.0015", which is ~3x too
loose. The correct rule already exists in `research_5k.md:17` and was used for the whole A-series
param ladder:

    100,000 * (LL_candidate - LL_champion) / (params_champion - params_candidate) <= 0.0001
    in BOTH modes; params must strictly decrease.

What that actually allows, against the 558,212-param champion:

| arm | params | params removed | max acceptable dLL, per mode |
|---|---|---|---|
| A | 84,007 | 474,205 (-85.0%) | **+0.000474** |
| B | 100,263 | 457,949 (-82.0%) | **+0.000458** |
| C | 99,596 | 458,616 (-82.2%) | **+0.000459** |

**The precedent says this is achievable.** The A0 -> A18 ladder itself (2,762,884 -> 558,212 for
+0.00096 ahead / +0.00053 imm) scores **0.0000435 / 0.0000240** on this exact gate -- 2.3x and
4.2x INSIDE the bar. If arm A cuts at a similar rate it lands near +0.0002, comfortably within
its +0.000474 allowance.

⚠ **THE RATIO GATE CANNOT JUDGE V1/V2, and this must not be discovered at verdict time.** The
FSRS-core variants keep the context streams, so they remove only ~68k params -- an allowance of
**+0.000068 per mode, which is BELOW the +/-7.5e-5 noise floor**. The gate would be measuring
noise. That is not a defect in the gate; it is a statement that V1/V2's value is not in
PARAMETERS at all. Their prize is per-card STATE (2,880 floats -> 3, a 960x cut) and the deletion
of the PQ codebook machinery. They need a state-based criterion, and it should be agreed BEFORE
the run, not after.

**What would falsify the whole direction:** if A costs materially more than ~+0.002 in either
mode, the 4.95x precedent does not extend and the remaining path is narrower streams with a
*decoupled* width (§5b), which needs real code.

---

## 8. Open risks, stated honestly

- **The delta rule is massively load-bearing** (+0.208 imm at inference-time ablation, ~390x
  the entire A0->A18 ladder). A GRU-style core has no delta rule. Arms A/B/C keep RWKV-7
  layers and so keep it; a *genuine* GRU core would not, and that is a much bigger bet than
  the parameter count suggests.
- **State bytes are not automatically better.** Card state falls 2,880 -> 1,152 floats, but the
  deployed 9 B/card comes from a quantization recipe that exploits the WKV state's low-rank
  structure at ~256x. A smaller state is not automatically cheaper *in bytes*, and the
  codebooks are shape-fitted — they would need refitting (`pq_cb_wkv_c80_b10.txt` is C=80).
- **CPU speed follows op count, not width, in the Python RNN path** (measured: a 4.5x
  arithmetic cut bought 1.24x). Arms A/B cut 13 layers to 9; that is the part that converts.
  The Rust path does convert width (2.39x measured).
- **These arms cut width and depth together**, whereas the A0->A18 ladder was width-only at
  constant structure. The precedent is weaker than a straight extrapolation implies.

---

## 9. "Reuse FSRS-7's formulas inside RWKV" (Andrew, 2026-08-24)

> *"I wonder if we could literally reuse FSRS-7's formulas inside of RWKV (well, with
> modifications so that S depends on other input features, but still)"*

### 9.1 What FSRS-7 actually is (read from `srs-benchmark/models/fsrs_v7.py`, not recalled)

- **State: 3 floats per card** — `S_long`, `S_short`, `D`.
- **Stability update** (`next_stability`, applied twice with parameter blocks at index 7 for
  long and 15 for short — the same functional form, different weights):
  - `sinc  = exp(w[k]-1.5) * (11-D) * S^(-w[k+1]) * (exp((1-R)*w[k+2]) - 1) * hard * easy + 1`
  - `S_succ = max(pls, S * sinc)`,  `pls = min(S, w[k+3]*((S+1)^w[k+4] - 1)*exp((1-R)*w[k+5]))`
- **Difficulty update:** `D += linear_damping(-w6*(rating-3), D)` (scaled by `R+0.1` on a lapse),
  then a fixed 1%/99% mean reversion to `init_d(4)`, clamped to [1,10].
- **Forgetting curve:** a normalized 2-component mixture, `(w1*r1 + w2*r2)/(w1+w2)`, where
  both components are `r = (1 + t/s')^(-d')` — `r1` from `S_short`, `r2` from `S_long` with
  difficulty acting on the timescale.

### 9.2 The curve half is already ours, and ours is strictly more general

Our head under `RWKV_GRU_HEAD=3` is `sum_i w_i (1 + t/s_i)^(-d_i)`, N=3, with `w` softmaxed
and **`w, s, d` predicted per review from the trunk**. Both of FSRS-7's components are of
exactly the form `(1 + t/s')^(-d')` (`short_component_recall:184`, `forgetting_curve:205`),
mixed with normalized weights.

**So FSRS-7's forgetting curve is a 2-component special case of the head we already run, whose
coefficients are additionally constrained to be fixed functions of a 3-float state.** Adopting
it would be a *restriction*, not an addition. There is nothing to gain on this half.

### 9.3 The genuinely new half — and the sharp version of the idea

The real difference is not the curve. It is **what drives the curve**:

| | drives the curve | per-card state |
|---|---|---|
| FSRS-7 | 3 floats through fixed forms with **34 global** parameters | **3 floats** |
| ours | an 80-dim trunk through learned per-review projections | **2,880 floats** |

So the proposal worth testing is:

> **Replace the card stream's WKV recurrence with FSRS-7's `(S_long, S_short, D)` update, and
> have the trunk EMIT the 34 parameters per review instead of them being global constants.**

That is exactly Andrew's "S depends on other input features", and it lands precisely on §2's
correction: FSRS obtains its per-user parameters by **running an optimizer per user**; here the
context streams would **predict them in-context**. Amortized inference of a mechanistic model's
parameters — the context streams keep their job, the card core gets a structure.

### 9.4 What it buys

1. **The deploy budget, which is per-card state.** 2,880 floats -> 3 (or 3 + k free dims). Note
   this is not automatically fewer *bytes*: the current recipe already compresses 2,880 floats
   to 9 B/card at ~256x. But it removes the entire PQ machinery — no codebooks to fit, no
   catalog staleness, no rank-1 truncation — and FSRS already ships (S, D) per card in Anki, so
   the state is provably deployable at Anki scale.
2. **Parameters.** The card stream is 72,508; an FSRS core plus a 34-output emitter is a few
   thousand. That alone is most of the way from 558k toward the <=100k target, *while keeping
   the context streams intact* — a different and gentler route than §1's arms.
3. **Inductive bias as regularization**, which our own record supports: capacity-at-5k is 0/3
   and the productive optimizer axis turned out to be Muon-as-*regularizer*. A constrained,
   monotone-by-construction state update is the same kind of lever.
4. It is **`adopted` provenance** — an external published algorithm with a reference
   implementation — which is what the strict-alternation rule requires of the next iteration.

### 9.5 Risks, in order

- **THE MAIN ONE, AND IT IS BEING MEASURED: the delta rule.** FSRS has no key-selective
  erase-then-write. The record's +0.208 imm ablation was measured **globally**, across all five
  streams, so it cannot say whether that value lives in the card stream (which this replaces)
  or in the context streams (which it keeps). Attributing a global ablation to one component is
  the exact error the record keeps logging. `scratchpad/hybrid100k/card_delta_ablate.py` runs
  three paired arms — baseline / card-only `a=0` / all-streams `a=0` — so the card share is
  read against a control measured on the same users and harness. **Result pending; this gates
  the idea.**
- **`R` feeds back into the state update.** `next_stability` consumes the predicted
  retrievability, so the head's output re-enters the recurrence. That is a structural coupling
  which train, eval and both deploy paths must compute identically (the §9 three-way rule).
  Related prior art: iter 48's `RWKV_RCOUPLE` fed the curve logit into the rating head and
  returned an exact tie — a *different* coupling (head->head, not head->state), but it is the
  nearest thing we have tried.
- **Range constraints are where blow-ups live.** FSRS clamps `D` to [1,10], `S` to 36500, plus
  34 box clamps and cross-parameter monotonicity. If the trunk emits these, every one needs a
  bounded parameterization — and iter 51's lesson applies: check the **max**, never the median.
- **3 floats may be too narrow.** §5 measures the card stream's contribution at participation
  ratio 5.55 (19 dims for 95%). A 3-float state is below that. The natural hedge is **FSRS's 3
  structured dims plus k free dims** (say k=5), keeping the inductive bias while leaving
  headroom — and k=0 vs k>0 is itself a clean ablation.

### 9.6 Two variants worth separating

- **V1, state only:** FSRS recurrence for `(S_long, S_short, D)`; keep our 3-component head,
  which reads the FSRS state *and* the trunk. Least restrictive; tests the state structure
  alone.
- **V2, state + curve:** FSRS recurrence *and* FSRS's 2-component curve driven by it. Maximum
  inductive bias, smallest, most interpretable — and, per §9.2, strictly less expressive than
  what we run today.

V1 first. V2 only if V1 holds, because V2 confounds "structured state" with "restricted curve"
and §9.2 says the restriction has no upside on its own.

---

## 10. ⚠ THE STATE-SIZE CASE FOR V1/V2 WAS OVERSTATED BY ~200x (Andrew, 2026-08-28)

Andrew: *"Yes, we go from 2,800 floats to 3, but IIRC with quantization the actual number of
bits per card is like 185 or so right now."* He is right, and it changes the recommendation.

**The deployed card state on THIS trunk is 185 bits, computed from the live codebook headers:**

| component | bits |
|---|---|
| WKV, `pq_cb_wkv_c80_b10.txt` (`1 10 32 16 1024`): 5 heads x 2 layers x 10 | 100 |
| WKV norms, `RWKV_QAT_NORM_BITS=1` | 10 |
| shifts, `pq_cb_shift_c80_m2b12.txt` (`2 12 40 80 4096`) = 24 b/vector x 3 vectors | 72 |
| shift norms | 3 |
| **total** | **185 bits = 23.1 B/card** |

(3 shift vectors, not 4: `RWKV_STRIP_CMIX` contains `card_id:1`, so layer 1 has no channel
mixer and therefore no channel shift.)

⚠ **THE "9 B/card" FIGURE REPEATED ACROSS THE DOCS IS THE d=32-ERA q72u NUMBER.** It appears in
`CPU_INFERENCE.md`, `CONTENT_EMBEDDINGS.md`, `LIT_REVIEW.md`, `HISTORY.md` and earlier sections
of THIS file, and every instance inherited it from that era. Nobody recomputed it for the d=80
trunk. Treat any per-card byte figure older than this section as d=32's.

**So the comparison V1 actually offers is 185 bits -> ~32-40 bits, i.e. 4.6-5.8x** (two
stabilities at ~12 bits each on a log scale, difficulty at 8). Not the 960x that
"2,880 floats -> 3 floats" suggests: that ratio is fp32-to-fp32, and **deployment already
compresses the card state 498x**. Sections 9.4 and 9.6 above overstate this; read them with
this correction.

**WHAT SURVIVES AS THE CASE FOR V1, and it is no longer primarily about size:**

1. ~5x on the binding deploy budget -- real (463 KB -> ~100 KB for a 20k-card collection), not
   transformative.
2. **Deleting the PQ machinery outright, which is the strongest argument and is not about bytes.**
   The codebooks are shape-fitted, go stale within a run, must be refitted whenever `d_model` or
   `H` changes, and fail SILENTLY when they are not -- the q72u catalogs measured *worse than
   random* on this trunk for weeks (held-out 1.0107 vs 0.9576 for random directions) with no
   assert firing, because K=16 kept them dimensionally valid. Three scalars with fixed physical
   ranges have no catalog, no fit, no staleness, and no refit-on-arch-change.
3. Interpretability, which matters if this ships inside Anki.

**AGAINST:** the most code of the five arms, the most parity risk, an inference-time upper bound
of ~+0.011 imm from the card-scoped delta ablation, and no gate that can judge it (section 7).

**=> V1/V2 is an ENGINEERING-SIMPLIFICATION play, not a size play, and should be argued on that
basis or not run.** A/B/C are unaffected: they are parameter cuts judged by a gate that works.

## 11. LAUNCH READINESS (2026-08-29) -- all three arms are armed and preflight-green

The GPU has been busy with Andrew's FSRS benchmark since 08-27, so the runners were built while
waiting rather than after. Each arm is one command:

    powershell -File scratchpad/detach.ps1 -Script C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\hybrid100k\run_hybA.cmd

| arm | params | dparams | max tolerable regression per mode | runner |
|---|---|---|---|---|
| A | 84,007 | 474,205 | 0.000474 | `run_hybA.cmd` |
| B | 100,263 | 457,949 | 0.000458 | `run_hybB.cmd` |
| C | 99,596 | 458,616 | 0.000459 | `run_hybC.cmd` |

Order: **A first** (the cheapest and the largest param cut), then B vs C as the matched ~100k
feature-MLP-vs-depth rebalance.

**`mk_runner.py` is a GENERATOR, not another clone**, which is what CLAUDE.md's bug-rate audit
asked for: the root cause of the 08-17/18 failure run was runners cloned across lineages, each
inheriting its ancestor's defects (27 carry the `endlocal`-before-the-marker bug alone). Every
known scar is a baked-in assert **on the generated output**: `cd /d` present, every `%VAR%` set
before first use, the terminal marker before `endlocal`, no `< > & | ^` in a `REM`, the alpha
guard strings built from the alphas the runner actually SETS, and artifact gates rather than exit
codes. That direction matters -- mk53/mk54 asserted only that stale text did not leak IN, and the
half that broke was required setup silently not surviving.

**Phase 0b is new and it is the one specific to this phase: `assert_arch.py`.**
`RWKV_STRIP_CMIX` matches the literal `"<stream>:<layer_id>"` and **silently ignores an entry that
matches no layer** (`rwkv_model.py:578`). Every arm changes the per-stream depths, so a stale list
half-applies and trains a different model than the one priced, with no error anywhere. The guard
asserts the exact param count **and** that every strip entry matched a real layer.

**★ IT CAUGHT A REAL ERROR OF MINE BEFORE ANY GPU TIME, WHICH IS THE ONLY EVIDENCE THAT MATTERS
FOR A GUARD.** `mk_runner.py` initially restated each arm's strip list, and two of the three were
wrong: arm B has the SAME depths as A, so an invented `deck_id:2` matched nothing, and arm C is
2/1/1/1/1, so six of nine entries matched nothing. Both arms would have trained happily and been
recorded against the wrong denominator.
**The bug was the DUPLICATION, not the typing** -- `mk_arch.py` already owns the depths, the priced
params and `strip_for()`. Fixed at the root: `mk_arch.py` grew an `if __name__ == "__main__"` guard
so it can be imported, and `mk_runner.py` now derives every arm from it. Verified afterwards by
running each runner's own phase-0 guard under the exact env that runner sets: 84,007 / 100,263 /
99,596, all OK, and `preflight_runner.py` passes all three.

**Recipe is the champion's, unchanged** -- seed 4321, WS 1 epoch, decay ratio 1.0, Muon with
`INCLUDE_LORA`, KD alpha 0.9 in WS and 0.5 in decay. The arm is exactly `RWKV_ARCH_MODULE` plus
that arm's strip list, so the comparison is architecture-only.
**The KD dump stays valid for a smaller student:** it replays the d=128 teacher's outputs by step
index and checks a per-step `labels_sum`, and batch composition depends on the db, user range,
MAX_TRAIN_GLOBAL_LEN, fetch-process count and seeds -- never on the student's architecture.

### The gate is now a tool, written before the runs report

`optimization/ratio_gate.py`. Reads the result jsonls, zero GPU cost:

    python optimization/ratio_gate.py --params-cand 84007 --params-champ 558212 \
        --cand-ahead result/RWKV-hybA.jsonl --cand-imm result/RWKV-P-hybA.jsonl \
        --champ-ahead result/RWKV-iter53_muonlora.jsonl \
        --champ-imm result/RWKV-P-iter53_muonlora.jsonl --intersect

Written BEFORE the arms report, deliberately: a gate computed after seeing the numbers has a free
parameter in it, and this repo has the scar -- the accept bar drifted 0.0003 -> a rounded 0.00005
-> a raw 0.0001, each step defensible, and only the written-down version made the drift visible.

**Validated three ways, not asserted.**
1. It reproduces the recorded A0 -> A18 ladder from the actual jsonls: **4.29e-5 ahead / 2.35e-5
   imm** against the 4.35e-5 / 2.40e-5 on record. The small gap is the 2,498-user intersection
   (A0 skipped 7 users to the NaN guard), not the arithmetic.
2. Hold the cost fixed and shrink the saving to 500k and it REJECTS both modes.
3. A candidate that GREW is refused outright rather than divided by. That is the one an ad-hoc
   calculation gets silently wrong: a negative denominator flips the inequality, so a bigger and
   worse model would print ACCEPT.

The p-values are printed as CONTEXT and do not gate. The ratio rule is a magnitude rule, and the
p<0.0001 accuracy gate belongs to the "candidate is better" protocol, which a shrink arm is not
claiming. They matter for reading the result: a regression inside the +/-7.5e-5 noise floor means
something quite different from a real one, and the tool says which.

### The chain is ARMED (2026-08-29 11:42) -- it starts itself when the GPU frees

Three detached waiters, all parented to WmiPrvSE so Esc and session teardown cannot kill them:

| pid | waiter | waits for |
|---|---|---|
| 31620 | `wait_gpu_then_hybA.cmd` | `gpu_free.py` exits 0 **twice in a row** (~4 min apart) |
| 41636 | `wait_then_hybB.cmd` | an anchored `DONE_EXIT_` in `hybA.log` |
| 30316 | `wait_then_hybC.cmd` | an anchored `DONE_EXIT_` in `hybB.log` |

**To stop it:** `Stop-Process -Id 31620,41636,30316 -Force`, or just delete the `.cmd` files --
and if an arm is already training, kill its runner too. The lock files `hyb{A,B,C}.launched`
prevent a double launch if the nudge path fires at the same moment.

**Why two GPU readings and not one.** `gpu_free.py 120` already averages over two minutes, but a
benchmark BETWEEN phases is idle for longer than that, and starting on top of Andrew's benchmark
is the thing CLAUDE.md forbids outright -- two processes once deadlocked in WDDM paging for 2.7 h.
Same two-witness rule as the 08-20 false alarm and the midnight flight-recorder alarm.

**Every mechanism was proven by EXECUTION, not by reading**, which is the standing rule here and
it paid twice:
* The anchored trigger: a log containing only PROSE that mentions `DONE_EXIT_` gives
  `findstr /B` RC=1 (does not fire) while the unanchored form gives RC=0 -- the 2026-07-26 trap
  reproduced live, then avoided.
* The lock short-circuit and the `ping -n 61` sleep both behave as intended headless.

**★ AND THE LAUNCH ITSELF FAILED TWICE BEFORE IT WORKED, SILENTLY -- worth recording because
`detach.ps1` is used for every run in this repo.** Arming all three in one tight loop left all
three DEAD within seconds, having written no log at all: `detach.ps1` still printed
`detached_pid=...`, so the only evidence of failure was `Get-CimInstance` returning nothing and
the absent logs. A single launch, verified before the next, works every time and survives across
tool calls. **THE RULE: `detached_pid=` is a receipt for `Win32_Process.Create` returning, not for
the process still existing. Launch one at a time and verify each by OS truth before the next.**
The tell was in the output and easy to wave away -- `parent_pid= ()` instead of
`parent_pid=2764 (WmiPrvSE)` means the process had ALREADY exited by the time detach.ps1 queried
it, microseconds later.
⚠ Two of my diagnostic detours were my own test harness, not the system: MSYS rewrote `cmd /c`
into a path (needs `cmd //c`) and bash `printf` ate the `\U` in `C:\Users`. Neither affects the
waiters, which cmd.exe runs natively.

### Arm A LAUNCHED 2026-08-29 21:11 — after the first launch died in 9 seconds

Andrew confirmed the GPU free at 21:05. **The waiter had not fired**: `gpu_free.py` requires mean
utilisation below 8%, and an idle desktop with browsers sits at ~13%. The threshold was tuned
against a machine running a benchmark, never against a machine merely awake. Launched manually
instead — the lock file is what stops the two paths racing, and it did its job.

**Then phase 0a failed in 9 seconds, three times, and it was my bug in the generator.** The
scripted-eval smoke loads the champion's **d=80** `i45` checkpoint, but the runner had already set
this arm's **d=32** arch, so `load_state_dict` produced ~200 size mismatches. I copied that line
from `run_iter52.cmd`, where the arch and the checkpoint matched — **the exact clone-inheritance
failure this generator was written to prevent, committed inside the generator itself.**

The cascade was instructive: arm A's `DONE_EXIT_45` immediately fired B's waiter, whose identical
failure fired C's. All three were dead within 40 seconds. **That is the chain working as designed**
— a fast failure propagates fast and costs nothing — but it means a phase-0 bug takes out the whole
queue, so phase 0 must be right before arming.

**The fix, and why it is a plain set/restore rather than anything cleverer:** phase 0a now sets the
CHAMPION's arch and strip list, runs the smoke, and restores the arm's own. Two rejected
alternatives, each rejected by a guard:
* A nested `setlocal`/`endlocal` — rejected because `preflight_runner.py`'s endlocal check is
  **nesting-blind** and flagged the inner `endlocal` as the stray-endlocal bug it was built for.
  Teaching it to track depth is more code and more risk than not nesting.
* Leaving the old assert as-is — it said "a champion strip entry must not APPEAR in an arm's
  runner", which is now false by construction.

**★ THE GENERALIZABLE PART: PRESENCE WAS ALWAYS THE WRONG PREDICATE.** Which arch is in force in a
`.cmd` is **positional**, so "does this string appear in the file" cannot answer it. Both checks are
now positional and assert **both ends**: the last `set RWKV_ARCH_MODULE` before the smoke must be
the CHAMPION's, and the last one before `train_rwkv` must be the ARM's. A one-ended check would have
passed the broken runner.

Verified by execution before relaunching: the smoke returns `SCRIPTED EVAL OK user=5001` under the
champion arch with the arm's other env set. Then live —

    hybA SMOKE_OK   21:11:59
    hybA ARCH_OK 84007 params  21:12:01

so phase 0b independently re-confirmed the model is the priced one. Training at 77% GPU, 2.3 GB.
B and C waiters re-armed on fresh logs.

## 12. RESULTS — the hypothesis does not pay, and arm C found a bug instead

| arm | params | ahead cost | imm cost | ratio (bar 0.0001) | verdict |
|---|---|---|---|---|---|
| A (84,007) | 6.6x smaller | +0.002711 | +0.003386 | 0.000572 / 0.000714 | **reject** (iter 60) |
| B (100,263) | 5.6x smaller | +0.002837 | +0.003570 | 0.000620 / 0.000779 | **reject** (iter 61) |
| C (99,596) | 5.6x smaller | — | — | — | **no verdict: CUDA fault** |

### Two findings, and the second is the one that answers Andrew's question

**1. A knee in the parameter-efficiency curve (iter 60).** A0 -> A18 cut 2,762,884 -> 558,212
(4.95x) for +0.00096 / +0.00053. iter 53 -> arm A cuts 558,212 -> 84,007 (6.6x) for
+0.00271 / +0.00339 — a comparable size ratio at **~5x the cost per parameter**. The width ladder
had already taken the cheap shrinkage.

**2. Widening the feature pathway does not substitute for recurrent capacity (iter 61).** Arm B is
arm A with `features_fc_mult` doubled and **identical depths** — 16,256 extra parameters placed
exactly where the hypothesis says they belong. It is **worse in both modes** (+0.002837 vs
+0.002711 ahead, +0.003570 vs +0.003386 imm). Small, but the wrong sign, and it is a single-variable
comparison.

**Together those say the <=100k target is reachable neither by shrinking this architecture
proportionally nor by rebalancing its budget toward the feature MLP.** That is a real answer to the
premise rather than a failure to reach a number: FSRS-7 and the 503-param GRU do get by on a tiny
recurrent core, but they are also not doing what this trunk does with the other 90-odd input
features — and the arms show that trunk's capacity is not idle fat.

⚠ **What is NOT tested: a genuinely different recurrent core.** Both arms shrink or rebalance the
*same* RWKV-7 core. Andrew's proposal also contemplated replacing it with something FSRS-shaped
(`rwkv/fsrs_core.py` + `fsrs_stream.py` are written and port-verified but never wired in). These
results do not speak to that.

### Arm C: a CUDA illegal memory access in the INTERLEAVE path — filed, not fixed

Two runs, two faults, at step **791** and step **146**. Diagnosis, in order:

* **Not divergence.** C's loss curve matches A and B almost exactly (1.938 / 2.062 / 1.890 / 2.023
  at steps 25/50/100/145 vs A's 1.938 / 2.062 / 1.896 / 2.018).
* **Not deterministic in the data.** The two runs are bit-identical through step 145 and died at
  different steps — so it is an out-of-bounds access that usually lands in mapped memory and only
  occasionally faults. ⚠ **That means arm C's numbers would be untrustworthy even if it finished.**
* **It IS the interleave path.** `diag_c_noilv.cmd` — arm C exactly, with `RWKV_INTERLEAVE`
  removed and capped at 800 steps, past both failure points — ran **clean, 0 faults**.
* **"One stream in a round" is necessary but NOT sufficient.** The CHAMPION's own schedule has a
  single-participant round (depths 2/1/4/3/3 -> round 3 runs deck alone) and never faults. What is
  unique to arm C is that the lone participant is `card_id`, the **first** stream — and the gather
  code builds each stream's layout on top of the previous one's (`cur = torch.cat(parts)`,
  "the next stream indexes into THIS stream's layout").

**Next step if it is ever picked up:** run depths (2,1,2,1,1), which gives round 1 two participants,
and (1,1,1,1,1), which gives one round and no interleaving at all. That brackets whether the trigger
is the count or the identity of the lone stream.

**Not fixed now, deliberately.** The trunk never uses these depths, arms A and B already answer the
research question, and Andrew's next phase is the dispatch speedup. Filed here and in the bug-hunt
matrix rather than chased.

## 13. V1 IS NEXT (Andrew, 2026-08-30) — with two corrections found while scoping it

> *"Let's test the FSRS-7-shaped core next, I believe there were 2 experiments for that."*

Two there are: **V1 (state only)** — FSRS's `(S_long, S_short, D)` recurrence replaces the card
stream's WKV, with the trunk EMITTING the 34 parameters per review, keeping our 3-component GRU
head — and **V2 (state + curve)**, which additionally adopts FSRS's 2-component curve. V1 first;
V2 only if V1 holds, because V2 confounds "structured state" with "restricted curve" and §9.2
already says the restriction has no upside on its own.

### ✓ The gate §9.5 set for V1 has PASSED, and it was measured before any GPU time

`card_delta_ablate.py`, 3 paired VAL users, inference-time `a=0`:

| ablation | imm | ahead |
|---|---|---|
| card stream only | +0.01137 | +0.00966 |
| all streams | +0.19442 | +0.07011 |
| **card share of the global imm cost** | **5.8%** | |

So ~94% of the delta rule's value lives OUTSIDE the card stream, which V1 keeps. The risk §9.5
called "the main one" is retired. ⚠ But note the absolute number: replacing the card recurrence
still has **+0.011 imm** to recover, ~21x what the whole A0->A18 ladder cost. It is an
inference-time upper bound and V1 substitutes structure rather than deleting it — but that is the
size of the hole. (n=3 users; small.)

### ⚠ CORRECTION 1 — "most of the way from 558k toward <=100k" IS WRONG

§9.4 claimed removing the card stream gets us "most of the way" to the <=100k target. Measured by
constructing the model, not by arithmetic on a remembered figure:

| | params |
|---|---|
| champion | 558,212 |
| champion with **card depth 0** | **485,704** |
| + FSRS core (emit 2,754 + writeback 400, n_free=0) | **488,858** |
| saved | **69,354 = 12.4%** |

558k - 72.5k is 486k, which is nowhere near 100k. **The context streams hold the bulk of the
parameters, exactly as §5 measured** — the card stream was never the place the budget lives. Same
error class as §10's 200x state-size overstatement: a real mechanism, an unchecked magnitude.

**=> V1's case is NOT parameters.** It is (a) per-card deploy state 2,880 floats -> 3, which also
deletes the entire PQ/codebook machinery, (b) inductive bias as regularization, which this record
supports (capacity-at-5k is 0/3; Muon paid as a regularizer), and (c) `adopted` provenance, which
the strict-alternation rule requires of the next iteration.

### ⚠ CORRECTION 2 — the RATIO gate is the WRONG gate here, and it is pre-registered as such NOW

At dparams = 69,354 the ratio budget would be **0.0000694 per mode** — *tighter* than the ordinary
0.0001 accept bar. Judging a 960x state reduction by a parameter-ratio rule is a category error:
V1 barely moves parameters and moves state enormously.

**PRE-REGISTERED GATE for V1 — CLAUDE.md's SIZE/SPEED exception:** accept iff **both modes stay
within +0.0015 of the champion** AND per-card state strictly shrinks. That is the rule already used
for H=2/K=16, which was accepted on exactly this argument (Pareto-dominant at accuracy-parity).
Registering it before the run, because choosing the gate after seeing the number is how a bar drifts.

### Implementation state — what exists and what is owed

**Exists and verified:** `rwkv/fsrs_core.py` (FSRS-7 math, port-verified at machine precision
against `srs-benchmark/models/fsrs_v7.py`), `rwkv/fsrs_stream.py` (`FsrsCardCore` + `run_sequence`,
18/18 smoke), the flag `RWKV_FSRS_CARD=<n_free>` (0 = pure FSRS, k>0 = k free dims),
`scratchpad/hybrid100k/arch_fsrs_v1.py` (champion arch, card depth 0 — constructs, and the
interleave schedule correctly sits the card stream out of every round).

**Owed:** the wiring in `srs_model.py` (training) and `srs_model_rnn.py` (deploy), plus a
three-way-parity case — CLAUDE.md requires one per new arch env flag, and
`parity_train_vs_rnn.py` is single-stack so it structurally cannot see this. Two inputs the core
needs are already available in the forward: **`log_dt_N`** (natural-log elapsed SECONDS in
canonical row order, built for `RWKV_RGATE`; FSRS wants days, so `exp(log_dt)/86400`) and the
rating one-hot at `_COL_R1`. `batch_skips` already marks the query/probe rows that must not
advance the state, which is exactly what `run_sequence`'s `skip_BT` expects.

## 14. V1 IS WIRED AND CORRECT — but it is ~12x slower per step, which changes the plan

**The build is done and verified** (2026-08-30, CPU): `RWKV_FSRS_CARD=<n_free>` wires
`FsrsCardCore` into BOTH `srs_model.py` (training) and `srs_model_rnn.py` (deploy).

| check | result |
|---|---|
| flag OFF, champion arch | **exactly 558,212 params** in both classes |
| flag OFF, scripted eval | LogLoss **identical** to the pre-edit run (0.001693 / 0.000246) |
| flag ON, V1 arch | **488,858** in both classes, same rating column resolved |
| training scan vs deploy stepping | agree at **0.000e+00** |
| skip semantics | a skipped row outputs, state unchanged; an unskipped row moves it |
| n_free=5 variant | builds, agrees, costs 1,250 params |

`scratchpad/parity3/smoke_fsrs_card.py`, 16 checks. It needs its own file because
`parity_train_vs_rnn.py` is single-stack and never constructs these classes.

### Three implementation findings worth keeping

* **An empty `ModuleList` is how you make an optional submodule truly inert.** The codebase's
  existing pattern (root Parameters + 1x1 dummies for the untaken branch, as the GRU head does)
  ADDS parameters when the flag is off. A zero-entry ModuleList has zero parameters and
  TorchScript unrolls its iteration to nothing.
* **A scripted function cannot close over a module-level float.** Measured
  (`scratchpad/tsglobal.py`): a bare global FAILS, the same value as a DEFAULT ARGUMENT scripts
  fine. **`Final[...]` does NOT fix it** — it applies to module attributes, not module-level
  globals. So `fsrs_core`'s clamp bounds are now defaulted parameters: written once, captured at
  def time, callers unaffected. The port re-verifies at 7.1e-15 afterwards.
* ⚠ **Randomize PARAMETERS, never `state_dict()`, when perturbing weights in a smoke.**
  `state_dict()` includes the `clip_lo`/`clip_hi` BUFFERS — the FSRS parameter ranges — and random
  bounds make `bounded_w` produce NaN. The smoke's non-vacuity check caught it. The n_free=5 arm
  escaped by luck (a different-sized state_dict consumed the RNG differently from the same seed),
  which is exactly how a bug like this survives into a "passing" test.

### ⚠ THE PROBLEM: 0.123 steps/s, 11.7x slower than a hybrid arm

Measured at step 260 with warmup provably over (0.11 -> 0.177 -> 0.123, oscillating, not
climbing). That is **~50 h for a full run** against ~6 h for an arm.

**The cause is structural and was foreseeable from our own profile.** V1 replaces a FUSED CUDA WKV
kernel — one launch per layer — with a Python loop of ~40 elementwise ops per review per card
sequence. `TRAINING_SPEED.md` measured this step as **85% CPU-dispatch-bound**, so adding
thousands of dispatches is the worst possible shape. It is not a bug in the wiring; the wiring is
verified correct.

⚠ Note the irony worth recording: the same measurement that motivates the queued dispatch-speedup
phase also predicts this slowdown, and I did not price it before building. **A new recurrence
implemented in Python op-by-op should be costed in DISPATCHES before it is costed in parameters.**

