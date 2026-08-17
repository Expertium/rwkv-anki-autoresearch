# Research proposal queue (ranked)

> **Why this file exists.** The 2026-08-10 ranking was produced by Andrew's 3-agent protocol and
> then lived only in the chat transcript. A compaction ate it, and the next session could
> reconstruct only the top of the list from `research_5k_notes.md`. **Idea generation is expensive
> and its output is not reproducible — write the ranked list HERE the moment it is produced, before
> implementing anything from it.** Same rule as the iteration log: if it is not on disk, it did not
> happen.

## The protocol (Andrew, 2026-08-10)

1. Spin up **3 subagents**. Each writes **5 proposals** spanning **at least 2 families** — not five
   variants of one idea.
2. Give them **different priors** so they do not converge (the 2026-08-10 set used: literature /
   domain-knowledge / reject-log steelman).
3. Rank all 15 (fewer after dedup) by expected LogLoss reduction. Implement the top one.

Andrew also asked how idea generation itself could be improved; that discussion is in the
2026-08-10 transcript and its actionable residue is the "different priors" rule above.

## Standing constraints every proposal must satisfy

* Accept gate: **raw ≥0.0001 in BOTH modes** vs the current champion + paired Wilcoxon p<0.0001
  both modes (CLAUDE.md "ACCEPTANCE GATE"). Same-capacity noise floor is ±7.5e-5, so an idea whose
  honest expected effect is ~1e-5 is not worth a 6-hour run.
* Hierarchy invariant `card→deck→note→preset→user` (the CODE's order) and the same 92-dim inputs.
* **Prefer changes that add no deploy debt.** `rust/rwkv-infer` is already 2 gaps behind
  (interleave schedule, stream order — `TRACK2_PORT_PLAN.md`). A training-only change ships free;
  a forward-pass change adds a gap and a mandatory fresh parity trace.

## The 2026-08-10 ranking

⚠ **Partially lost.** Items 1–6 survived (via the compaction focus + `research_5k_notes.md`);
7–15 did not. Do not treat the absence of 7–15 as evidence they were bad.

| # | proposal | family | status |
|---|---|---|---|
| 1 | **KD through the decay phase** — since iter 34 adopted decay_ratio=1.0, decay is half of all training and runs on pure hard labels. Zero code: the runner simply does not clear `RWKV_KD_MIX`. | distillation | **iter 45**, ran 2026-08-11 |
| 2 | **Retrievability-coupled rating head** — feed logit R(t) from the curve head into the Again logit. | architecture | **iter 48**, REJECTED as an exact tie 2026-08-15 — and with iter 46 it CLOSES the ahead-vs-imm-gap family (see below) |
| 3 | **Privileged self-distillation (imm → ahead)** — soften the ahead target toward the model's own better-informed imm estimate of the same event. | distillation | **iter 46**, implemented 2026-08-11 |
| 4 | Duration dropout — per-row Bernoulli dropout of the duration feature (the corrected retry of iter 33's intent). | input-robustness | queued |
| 5 | NorMuon / PolarExpress — refinements to the accepted Muon optimizer; the Newton-Schulz orthogonality error was measured real (0.19–0.31 RMS). | optimizer | **PolarExpress = iter 51, FAILED (NaN at step 411), closed on mechanism 2026-08-16. NorMuon untried but DEMOTED — see "Muon is a regularizer" below.** |
| 6 | Restore user/preset L0 channel-mixers (`RWKV_STRIP_CMIX`) — zero code, just a shorter strip list. | capacity | **iter 49**, REJECTED 2026-08-16 — +0.000067 ahead at p=0.11 (a coin flip) / +0.000087 imm; both under the bar |

### Why #3 was promoted over #2 (2026-08-11)

Both exploit the same measured headroom — the **ahead-vs-imm information gap**
(`research_5k_notes.md`): identical per-user `size` on all 2500 VAL users, imm better than ahead on
**2497** of them, mean gap **0.032411**. Three reasons the ordering flipped on inspection:

1. **Deploy debt.** #3 is a loss-side change only: train differs, eval and CPU/Rust inference are
   untouched, param count unchanged (558,212 exactly). #2 edits the forward pass and would add a
   **9th** Rust port gap plus a mandatory fresh parity trace.
2. **Prior evidence.** Soft targets on this head from a better-informed teacher have been accepted
   three times (iters 32/35/39) with a dose curve monotone up to α=0.9 — this model wants teacher
   signal. #3 supplies one for free; #2 is a novel structural coupling with no in-repo precedent.
3. **#3 partly subsumes #2's mechanism.** #2's appeal was creating a gradient path from the
   better-conditioned imm objective into the curve head; #3 creates that path directly, through the
   target, without the architectural risk that the rating head degrades by leaning on R(t).

⚠ **The gap is an UPPER BOUND, not a target.** The query row sees the intervening reviews and the
exact lag; the ahead head structurally cannot, and predicting cold from history *is* the task.
Distillation can transfer the variance-reduction part (a calibrated target beats a 0/1 draw — which
is exactly why the external-teacher α peaks at 0.9), never the information part.

### ★★ RE-RANK 2026-08-15: #2 IS PROMOTED TO THE TOP OF THE UNTRIED LIST — iter 46 killed the
### argument that demoted it

Of the three reasons above, **reason 3 was the load-bearing one and it is now refuted by the very
experiment it justified.** #3 ran as **iter 46** and was a clean null (ahead -0.000023 / imm
+0.000016, both inside the noise floor). Its diagnosis: the teacher shares the trunk AND the forward
pass, so its soft target only re-expresses what the student already computes. **So #3 did NOT subsume
#2's mechanism — it demonstrated that the substitute does not work**, and iter 46's own stated
conclusion is the argument FOR #2: *"Closing that gap requires changing what the ahead path COMPUTES
or is FED, not what it is FIT to."* #2 is exactly a change to what it computes.
Reason 2 (prior evidence favours soft targets) is also weaker now: the external-teacher sub-family is
4/4, but the SELF-teacher variant is 0/1, and #2 is not a distillation idea at all.
**Reason 1 stands and is the only real objection: #2 edits the forward pass**, so it adds a Rust port
gap and needs a fresh parity trace. That is a cost to schedule, not a reason to keep skipping it —
and the two gaps that prompted the warning (interleave schedule, stream order) were CLOSED
2026-08-11, so the port is no longer behind.
⚠ Keep the upper-bound caveat in view: the 0.032 ahead-vs-imm gap is NOT a target. The query row sees
the intervening reviews and the exact lag; predicting cold from history IS the task. #2's mechanism is
to give the rating head a better-conditioned input, not to import information the ahead path cannot
have.

**REVISED ORDER OF UNTRIED WORK (2026-08-15).** Two queues with very different unit costs, which
should drive the choice: an ALGORITHMIC iteration is ~2.6 h train + ~2.9 h plain eval ≈ **5.5 h**; a
QAT iteration needs the quant-aware eval, measured at **10 h 18 m**, so ≈ **13 h**. At the measured
algorithmic rate the remaining `still_needed` (+0.00165 / +0.00100) is ~15-18 iterations, i.e. the
algorithmic loop is where the wall-clock goes and cheap iterations matter.

| rank | item | queue | cost | why here |
|---|---|---|---|---|
| ~~1~~ | ~~#2 retrievability-coupled rating head~~ | algorithmic | DONE | **iter 48, REJECTED (exact tie).** The coupling was LEARNED and sign-correct but negligible — the trunk already carries retrievability. Family CLOSED with iter 46. |
| 2 | #6 restore user/preset L0 cmix | algorithmic | 5.5 h, zero code | free to try, capacity family untested at this trunk |
| 3 | QAT#2 KD from the PLAIN iter-45 teacher | QAT | 13 h + 2-3 h dump | targets the deploy objective directly; distillation is 4/5 |
| 4 | #4 duration dropout | algorithmic | 5.5 h | training-only, no deploy debt |
| ~~5~~ | ~~#5 NorMuon / PolarExpress~~ | algorithmic | — | **PolarExpress ran as iter 51 and FAILED structurally; NorMuon DEMOTED on mechanism (it refines descent, and descent is the half of Muon that has stopped paying at our budget). Do not rank it without a new argument.** |
| 6 | QAT#3 `RWKV_CB_LR_MULT` sweep | QAT | 13 h/point | now motivated (catalogs ARE the active lever) but expensive per point |

⚠ **The QAT items are NOT dead** — a 2026-08-15 summary called the QAT vein "worked out", which was
wrong: items 3 and 6 here were never tried and neither is reconstruction-motivated. What IS closed is
rank-1 regularization (measured), the four norm/catalog levers (mechanism), and anything justified by
reconstruction error alone (catalog init, index bits) — see `research_5k_notes.md`.

### Two traps found while implementing #3 — both would have wasted a full run

Recorded because they generalize, not as iteration narrative.

* **The probe path is not the ahead objective.** The obvious home for a soft ahead target is
  `_pava_probe_loss`, which already carries a query-row index. But the champion does **not** set
  `RWKV_AHEAD_PROBE_ONLY` and runs `PROBE_DENSITY=0.08`, so the probe path covers 8% of reviews at
  λ=0.2 ≈ **3% of the ahead objective's weight** — far under the 7.5e-5 noise floor. *Check which
  env flags the CHAMPION actually sets before assuming a code path is load-bearing; iter 33's
  settings did not persist into the champion recipe.*
* **`probe_query` is the wrong review for a teacher.** It joins on `review_th[q] == review_th[r]`
  — review r's OWN decision point, correct for PAVA's pooling weights. But a real row's ahead label
  is the **next** review of that card (`label_review_th = groupby("card_id")["review_th"].shift(-1)`,
  `data_processing.py:292-303`). The teacher must join on `review_th[q] == label_review_th[r]`.
  Measured on real LMDB rows: the two joins differ on **100.0%** of probe rows. The correct join
  gives **100.00% coverage** of ahead-scored rows (388,156/388,156 over 5 users, 0 violations) —
  `ahead_rows == query_rows == teachers` exactly, which is the structural reason the two metrics
  report identical per-user `size`.

## QAT-IMPROVEMENT SUB-QUEUE (Andrew 2026-08-13: "keep the current quantization recipe... let's try improving QAT first")

Constraint: the quantizer's STRUCTURE is FROZEN (rank-1 int4 card/note + PQ b10 WKV + m2b12 shift +
1-bit norms => 185 b/card, 105 b/note). Levers are the training procedure only. Baseline to beat =
qtaxc_m2b12 cell 2: **0.301882 / 0.271594** on the VAL half (tax +0.004185 / +0.006219 vs iter 45).

1. **LEARNABLE CODEBOOKS (running first).** Andrew's own doctrine, and the d=32 quant endgame's
   winning lever ("huge learnable catalogs beat the product form"). Infrastructure is COMPLETE
   contrary to CLAUDE.md's "queued" note: PQ_LEARN envs train both catalogs (kernel-side grads,
   wd=0 groups, resume-safe), and exports fire at every ckpt save. The only missing wiring was the
   eval pointing at the exported files — runner-level. Zero deploy-size change (values change, not
   structure). Single-variable vs qtaxc: only `RWKV_QAT_PQ_LEARN=1 RWKV_QAT_SHIFT_PQ_LEARN=1`.
   Extra rationale on this trunk: the k-means catalogs are fitted on the PLAIN model's states, and
   the state distribution shifts during QAT — co-training tracks it.
2. **KD from the PLAIN iter-45 teacher.** The classic fp32-teacher QAT recipe. Distinct from the
   current d=128 teacher: it targets "what the plain model would predict", which IS the deploy
   objective (minimize the tax), not just low logloss. Needs a new dump (forward-only pass over the
   decay window, ~2-3 h) — the dump/mix infra exists. Could stack with the d=128 teacher (mix) or
   replace it; run as replace first (cleaner attribution).
3. **RWKV_CB_LR_MULT sweep** — only if #1 helps; the champion cb_lr=1x is a d=32-era HP.
4. **Longer/rescheduled QAT fine-tune — DEPRIORITIZED by evidence:** the paired train-loss trace
   shows the QAT-vs-plain gap is a constant offset from step ~500 (adaptation saturates fast);
   more steps of the same schedule will not close it.
5. Quantization-strength annealing (stochastic quant ramp) — real code; only if 1-2 disappoint.

Screening tool for all of these: the paired train-loss gap vs qtaxc's decay trace (same WS start,
seed, db => steps pair exactly; visible by step ~1000, i.e. ~50 min in, zero GPU beyond the run
itself). ⚠ Valid here because candidates share the quantizer and regularization — the documented
train-loss-prune bias applies to REGULARIZATION levers, not matched-config QAT variants.
Gate for accepting a QAT improvement: cell-2-style eval on the FULL VAL half beats qtaxc_m2b12's
0.301882 / 0.271594 with the usual paired Wilcoxon; there is no cell 3 (meaningless for structural
quant — see research_5k_notes 2026-08-13).

## ★★ THE AHEAD-VS-IMM-GAP FAMILY IS CLOSED (0/2, on mechanism) — 2026-08-15

Both routes into the 0.032411 gap are now measured and both are exact nulls:

| iter | route | ahead / imm delta |
|---|---|---|
| 46 | soft targets (privileged self-distillation imm→ahead) | −0.000023 / +0.000016 |
| 48 | an architectural path (R(t) into the rating logits) | +0.000009 / +0.000013 |

Iter 48 is the decisive one because its coefficients were **zero-init**, so the null is separable:
the model **learned a sign-correct coupling** (Again −0.0138) and it bought nothing, i.e. it USED
R(t) and gained nothing. **The trunk representation already carries the retrievability information
the rating head needs.**

**This file predicted it.** The standing caveat above — *"the gap is an UPPER BOUND, not a target"*,
because the query row sees the intervening reviews and the exact lag while the ahead row structurally
cannot — is now demonstrated twice rather than argued once. **Do not rank a third routing variant.**
What remains legitimately open is changing what the ahead path is **FED** (new input features, the
endgame's step 2), which attacks information CONTENT rather than its routing.


## ★ AFTER ITER 49: capacity-at-5k is 0/3 — a standing constraint on new proposals (2026-08-16)

Three structurally different capacity adds now agree:

| iter | where the params went | ahead / imm |
|---|---|---|
| — | num_curves/points 64→128, channel_mixer 1.0→1.5 (100-user era) | rejected |
| — | WS 18 epochs / 8-epoch decay (100-user era) | rejected |
| 49 | user/preset **layer-0** channel mixers, +26,070 params (+4.7%) | +0.000067 (p=0.11) / +0.000087 |

Iter 49 is the sharpest of the three because it put the params back **exactly where the cmix
ablations had removed the most**, and got noise on ahead. **Do not rank a fourth width/depth add
without a mechanism argument that distinguishes it from these three.** The consistent reading, now
confirmed at 5k on a 4.95x smaller trunk, is the 100-user era's: this model is **data-limited, not
capacity-limited**, and training/topology levers are where the wins have come from.

## Off-queue: the DECK TREE (Andrew's direct ask, iter 50) -- REJECTED as an exact tie, 2026-08-16

Not from the ranked list — Andrew asked for it directly: `card->note->deck->preset->global` becomes
`card->note->(deck, depth_level)->preset->global`. Running at **L=2** (parent level only, 49.21% of
reviews, 17 layer-steps, 558,292 params). **L=3 is the better test in principle** — the deck-depth
histogram peaks at 4, not 1 — and is the natural follow-up **if L=2 shows signal**, but it needs the
memory addressed first: L=3 sits at 95% VRAM, which is this machine's documented WDDM paging cliff,
with a GPU co-tenant.

**Two implementation lessons for anyone proposing another stream:**
1. **Cost it on B, not on rows.** The WKV state is per SEQUENCE, so a grouping that produces many
   short sequences is expensive even when padded volume and kernel-launch counts look fine (they
   both did; neither counts B). Giving inactive rows singleton sequences cost ~780 MB of extra
   state before backward saves.
2. **Do not time it until compile warmup is provably over** — several HUNDRED steps on this stack.
   Three early windows said 2.5-4x slower; steady state was 1.35x, matching the shape prediction.

**VERDICT (2026-08-16): exact tie, +0.000007 at p=0.52 / -0.000024 at p=0.86.** The level embedding
WAS learned (L2=1.77, ~2x a features2card row), so the model used the parent-deck level and gained
nothing. **Mechanism: the 5-stream ladder already brackets that scope** (deck below, preset/user
above), so an intermediate level is interpolation, not new evidence.
**⚠ This DEMOTES the L=3 follow-up.** Deeper ancestors interpolate even closer to preset/user, so if
the parent level -- most distinct scope, widest reach (49.21%) -- is a coin flip, levels 3-4 are a
worse bet, not a better one. Do NOT rank L=3 as "we only tested the shallow case".
**And it closes the topology family's last open direction:** existence of cross-scope paths pays
(iter 41), choreography does not (42-44), more scopes do not (50).

## ★★ MUON IS A REGULARIZER AT OUR BUDGET — this demotes the rest of the optimizer family (2026-08-16)

Full measurement + tables: `research_5k_notes.md`. Prompted by Andrew's recollection that Muon was
"way better than Adam initially, but only mildly better at the end", which the archive confirms and
sharpens. On the matched pair iter 29 (Muon) vs iter 26 (AdamW) — the only difference is the three
`RWKV_MUON_*` env vars — the **train**-loss advantage decays across 6,554 paired steps from
+0.01446 / +0.09809 (first decile) to **−0.00058 / +0.00097** (last), i.e. on `ahead` it INVERTS,
while the **held-out** advantage holds at **+0.001909 / +0.001913**.

Training to a higher train loss and a lower eval loss is the signature of a regularizer. **Muon's
value here is generalization, not speed of descent.**

**Ranking consequence.** PolarExpress and NorMuon are both refinements of the *descent* — more
accurate orthogonalization, better per-neuron scaling. They target the half that has already stopped
paying. PolarExpress additionally failed structurally (iter 51: an accurate polynomial has p(1)→1,
but production's `a+b+c`=0.7010 makes p(1)=0.70 a *contraction*, which is what keeps thin rank-1
momentum matrices stable at σ_max≈1). **A future optimizer proposal must name which half it attacks**,
and one aimed at descent quality has to say why it would move held-out loss when the existing descent
advantage does not.

⚠ Demotion on mechanism, not closure (conduct rule 5). And the measurement is one matched pair at the
older 6,554-step / MAX=32768 budget; the current recipe runs 10,935 steps with an 8× lower Muon LR.

## ★★★ ANDREW 2026-08-17: AT LEAST 10 MORE ALGORITHMIC ITERATIONS BEFORE THE FEATURES PHASE

> *"It seems a bit too early to give up on algorithmic improvements, give it at least 10 more iters.
> There is no way the current architecture and training are so optimal that no improvement is
> possible. Well, it's possible that there won't be any improvement over 10 iters, but still."*

This **overrides** the reading that closed families + a 0-for-6 run meant the loop was done. The
features work stays where it is (implemented, inert, waiting) and the GPU goes back to algorithmic
iterations. Note this is also the standing conduct rule's spirit — ">= 50 research iterations before
even considering declaring nothing left to improve" — applied to a phase that had quietly started
behaving as if the number were a ceiling rather than a floor.

**The generation problem is real and worth stating**, because it is what made the loop look dry: the
families with a hit rate are mostly closed *as families*, so the next 10 candidates have to come from
mechanisms rather than from "one more variant of X". The list below is built that way — each entry
names the measurement that motivates it and what would distinguish it from the reject that looks
similar.

| rank | candidate | family | mechanism, and what makes it NOT a repeat | cost |
|---|---|---|---|---|
| 1 | **Muon on the LoRA matrices** (`RWKV_MUON_INCLUDE_LORA`) | optimizer / regularization | The Muon group rule excludes any param whose name contains `lora` or `scale` (`train_rwkv.py:150-153`), so ~10.3% of params (57,412) run on AdamW — and on this trunk that group is dominated by the rank-4/rank-2 LoRA projections the A18 width ladder introduced. If Muon pays through **spectral regularization** (measured 2026-08-16), the most anisotropic matrices in the model are exactly the ones currently not getting it. Distinct from iter 51: no schedule change, the production triple and its p(1)=0.70 contraction are untouched. ⚠ Pre-register the counter-hypothesis — flattening a deliberately low-rank factorization's update may destroy what the factorization is for. | 5.5 h, ~2 lines |
| 2 | **Ensemble teacher: d=128 + a frozen past champion** | distillation | Distillation is 4/5 and is the only family with a real hit rate. Iter 46's null is NOT a precedent against this: its teacher shared the trunk AND the forward pass, so the soft target re-expressed what the student already computed. A frozen iter-41/45 checkpoint is a genuinely different function evaluated in a separate pass. Mechanism: KD pays here through target-variance reduction (which is why alpha peaks at 0.9), and averaging two independent teachers reduces it further. | 5.5 h + ~2 h dump |
| 3 | **Spacing-effect monotonicity on the curve** | curve-shape constraints | The family is 2/3 (PAVA, lambda=0.2). The GRU head already gives monotone-in-t and convex-in-t *by construction* (`(1+t/S)^-d`, d>0, and a nonneg mixture of convex functions is convex — checked in code, so do NOT propose either of those). What is NOT imposed is monotonicity in **review count**: stability should not decrease after a successful review. That is a real SRS structural fact and an un-used constraint. | 5.5 h |
| 4 | **De-confound WS:decay at a FIXED budget** (1+1 vs 1.6+0.4) | schedule | `research_5k_notes.md` already establishes that iter 34's decay_ratio 0.25 -> 1.0 gain is confounded with a 1.25 -> 2.0 epoch budget change, and that the log-linear budget curve explains +0.00084 of the +0.00145. So the ratio itself rests on one confounded point, and the endgame's 10+2 split is *spending* +0.0006 it cannot demonstrate. This measures it. | ~10 h |
| 5 | **Hint/feature distillation from the d=128 trunk** | distillation | Output-KD is 4/5; the classic next step is matching an intermediate representation, which targets the trunk directly rather than the two heads. Needs a learned projection (d=128 -> d=80) since the widths differ — the only entry here with real implementation risk. | 5.5 h + dump |

⚠ Ranks 1-3 are cheap and single-variable; run them first and re-rank on what they say. Rank 1 is
also the fastest to reject: if extending Muon's coverage does nothing, the "Muon = regularizer"
reading loses its most direct prediction, which is worth knowing early.

### Queue state 2026-08-17 — the first two are BUILT and CHAINED behind the running QAT job

| # | iter | lever | cost | status |
|---|---|---|---|---|
| 1 | **52** | KD `alpha_decay` 0.5 -> 0.9 | **~3.5 h** | armed behind QAT#2 |
| 2 | **53** | `RWKV_MUON_INCLUDE_LORA=1` | ~6.2 h | armed behind iter 52 |

**★ A COST DISCOVERY WORTH MORE THAN EITHER ITERATION: a decay-only lever costs ~3.5 h, not ~5.5 h.**
Iter 52 warm-starts from the champion's own `i45_ws_10935` — the same checkpoint the QAT arms use —
which is EXACT rather than an approximation, because the lever cannot touch WS. Any future candidate
that only changes the decay phase (KD schedule, decay LR shape, decay-phase regularization) gets the
same 36% discount. Iter 53 does NOT: an optimizer change acts from step 1.

**Iter 53's lever, in one line:** `get_optimizer` excludes any param whose name contains `lora`, so
**27,520 params in 94 tensors (4.9%)** have always run on AdamW — and after the A18 width ladder made
LoRA rank load-bearing, those are the most anisotropic matrices in the model. If Muon pays through
spectral regularization, that is exactly where it is missing. Kept single-variable by giving them
their own group at wd=0.0 (the value they already had); dropping them into `decay_params` would have
moved the optimizer AND the weight decay at once. Smoke: `scratchpad/iter53_muonlora/smoke_muon_lora.py`
verifies by param IDENTITY, not by count — 94 tensors move, nothing leaves Muon, wd unchanged, and
with the flag off the partition is exactly the historical one.
⚠ **The counter-hypothesis is pre-registered**: flattening the update of a deliberately low-rank
factorization may destroy what the factorization is for. A regression would bound the regularizer
reading more sharply than a win would confirm it.

⚠ **Chaining accepts a known basis subtlety.** Iter 53 is built on the ITER-45 recipe, so if iter 52
wins and promotes, iter 53's controlled comparison stays iter-53-vs-iter-45 while the champion it
must beat has moved. Accepted deliberately: the levers are orthogonal, iter 53's value is the
mechanism test, and chaining keeps the GPU busy rather than idling until a human reads a verdict.

## ★★★ THE 10-ITERATION PLAN (Andrew asked for it explicitly, 2026-08-17)

**Planning constraint that shapes the whole list:** the families with a hit rate are mostly closed
*as families* (ahead-vs-imm routing 0/2 mechanism-closed, topology 1/4 closed, capacity 0/3,
low-rank-friendly regularization 0/1 mechanism-closed, HP tuning closed, state-size ladder 0/5).
So "one more variant of X" is not available for most of X. Every entry below names the measurement
that motivates it and what distinguishes it from the reject it resembles — and where that is thin,
it says so.

**Second planning constraint, and it is new: DECAY-ONLY LEVERS COST 36% LESS.** They warm-start from
the champion's own `i45_ws_10935` exactly (the lever cannot touch WS), so ~3.5 h instead of ~5.5 h.
Worth actively looking for decay-only formulations of any candidate.

> ⚠ **ITERATION NUMBERS CORRECTED 2026-08-17.** This table originally numbered the ensemble teacher as iter 54; the armed iter 54 is the learnable channel-mixer exponent (added later, from Andrew's expressiveness catch), so everything below it shifted by one. The QUEUE STATE table further down is the authority on what is actually built.

| # | iter | lever | family | cost | why it is here |
|---|---|---|---|---|---|
| 1 | **52** | KD `alpha_decay` 0.5 -> 0.9 | distillation | **3.5 h** | ARMED. The 0.5 was never chosen — iter 45 won because its runner didn't clear the var. Its own notes flag this as open. |
| 2 | **53** | `RWKV_MUON_INCLUDE_LORA=1` | optimizer/reg | 6.2 h | ARMED. 27,520 params (4.9%) excluded from Muon by a name rule that predates the width ladder; the most anisotropic matrices in the model. Direct prediction of the regularizer finding. |
| 3 | **54** | **`RWKV_CMIX_POW=1`, learnable channel-mixer exponent** | EXPRESSIVENESS | 6.2 h | ARMED. Added after this table was first written -- see "EXPRESSIVENESS vs CAPACITY" below. The only candidate of Andrew's named class that survives the redundancy test. Carries the first Rust deploy debt in a while. |
| 4 | 55 | **Ensemble teacher: d=128 + a frozen past champion** | distillation | 5.5 h + 2 h dump | External-teacher KD is **4/4**. KD pays here through target-VARIANCE reduction (which is why α peaks at 0.9); averaging two independent teachers reduces it further. NOT iter 46: that teacher shared the trunk and the forward pass. |
| 5 | 56 | **Decay LR SHAPE** (cosine -> linear / 1−sqrt) | schedule | **3.5 h** | The tuner swept `decay_ratio`, never the decay *shape*, and WSD literature puts the action in that phase. Decay-only, so it is the cheapest untried lever in the queue. |
| 6 | 57 | **Spacing-effect monotonicity** | curve constraints | 5.5 h | Family is **2/3**. ⚠ Do NOT propose monotone-in-t or convex-in-t: the GRU head gives both BY CONSTRUCTION (`(1+t/S)^-d`, d>0, nonneg mixture — checked in code). Monotonicity in REVIEW COUNT is a real SRS fact and is not imposed. |
| 7 | 58 | **Fixed-budget WS:decay de-confound** (1+1 vs 1.6+0.4) | schedule | ~10 h | Iter 34's `decay_ratio 0.25 -> 1.0` gain is confounded with a 1.25 -> 2.0 epoch budget change; the log-linear budget curve explains +0.00084 of the +0.00145. The endgame's 10+2 split is SPENDING ~+0.0006 that rests on one confounded point. |
| 8 | 59 | **Weight decay on the LoRA group** | regularization | 5.5 h | Those params sit at wd=0.0 today, never chosen — they inherited it from `other_params`. One number, and it pairs with iter 53's outcome either way (if Muon helps them, wd is the other half of the same question; if it hurts, wd is the gentler version). |
| 9 | 60 | **Horizon reweighting of the curve loss** | objective | 5.5 h | Long intervals are rare and hard, so the curve objective is dominated by short t. ⚠ Distinct from iter 37 (by-USER weighting, mechanism-refuted in every size quartile): this addresses a coverage imbalance in **t**, which is a property of the data, not of user size. |
| 10 | 61 | **Hint/feature distillation from the d=128 trunk** | distillation | 5.5 h + dump | Output-KD is 4/4; matching an intermediate representation targets the TRUNK rather than the two heads. Needs a learned d=128 -> d=80 projection — the only entry with real implementation risk. |
| 11 | 62 | **Born-again: fresh student, iter-45 champion as SOLE teacher** | distillation | 5.5 h + dump | The BAN phenomenon (same-capacity student beats its teacher). Cleanly distinct from iter 46: separate forward pass, different weights, frozen. Ranked below #3 because #3 keeps the known-good teacher and only ADDS ours, so it risks less. #3's result should inform whether this is worth running at all. |

### ⚠ Honest note on where this list thins out
Ranks 1-6 each rest on a specific measurement in this repo. Ranks 7-10 are weaker: 7 and 8 are
"a knob nobody chose" arguments, and 9-10 are literature-standard moves without local evidence.
**The plan is therefore to re-run Andrew's 3-agent generation protocol once ranks 1-5 have reported**,
rather than to commit to 7-10 now — the first five results will close or reopen whole branches, and
idea generation is cheap relative to a 5.5 h run.

### What would change the plan
* **If iter 53 wins**, the regularizer reading gains its first confirmation and the queue should tilt
  toward more of it (scale matrices onto Muon, #7, structured regularizers). **If it regresses**, that
  bounds the reading harder than a win would confirm it, and #7 becomes the gentler retry.
* **If iter 52 wins**, the KD schedule is live and `alpha_decay=0.25` (the other bracket point) is a
  cheap 3.5 h follow-up; #3 also gets more attractive.
* **If both are nulls**, distillation and optimizer are the last two families with a hit rate, and
  the honest read is that the trunk is near its ceiling at this budget — which is an argument for
  moving to features sooner, not for grinding out ranks 7-10.

## ★★★ EXPRESSIVENESS vs CAPACITY — Andrew's catch, 2026-08-17, and it was a real hole

> *"No changes to architecture that improve expressiveness? Stuff like activation functions with
> learnable params."*

**He is right and the omission was a logical error on my part.** The queue had no such entry because
"capacity-at-5k is 0/3" was doing the work of an argument it cannot support. Those three rejects all
added **more of the same functional form** — user/preset layer-0 channel mixers (iter 49, +4.7%
params), `num_curves`/`num_points` 64->128, `channel_mixer` 1.0->1.5. **None of them tested whether a
RICHER functional form at ~fixed parameter count helps.** Expressiveness and capacity are different
axes and the record only covers one.

### ★ THE SCREEN THAT MAKES THIS CHEAP: the REDUNDANCY TEST
Before measuring anything, ask whether an adjacent **free linear layer can absorb the new parameter**.
If it can, the proposal adds exactly zero expressiveness — it is a reparameterization, and the
optimizer already has the freedom it claims to add. In this architecture that kills a whole class at
zero cost:

* **Learnable slope/gain on `tanh` or `sigmoid`** — every one of them is sandwiched between free
  linear layers (`B(tanh(A(x)))`, `sigmoid(linear(x))`), so `tanh(s*A(x))` is just `A` rescaled and
  `s*tanh(...)` is just `B` rescaled. **Dead by algebra.**
* **Cross-head mixing after the WKV** — `W_o` is already full-width over the flattened `H*K`
  dimension, so heads already mix inside the layer. **Dead by algebra.**
* **A learnable EXPONENT survives**: `relu(c*k)^p = c^p * relu(k)^p`, so rescaling the input only
  rescales the output — the CURVATURE is set by `p` alone and no linear rescale reproduces it.

### Three hard bounds measured against the champion, all NOT binding (CPU, minutes, no GPU)
Tools: `scratchpad/expressiveness/decay_floor_probe.py` and the two inline probes in the
2026-08-17 transcript. Each reads the reachable envelope straight from the checkpoint —
`B(tanh(A(x)))` with `tanh` in `[-1,1]` gives an exact range of `bias +/- sum_j |B[.,j]|`, so no
forward pass is needed.

| bound | what it caps | measured on iter 45 | verdict |
|---|---|---|---|
| `_d = **-0.5** - softplus(-d_lora)` (`rwkv_model.py:915`) | fastest decay: `w >= 0.5452`, half-life 1.14 steps | median FASTEST reachable `w` per tensor is **0.954-0.994** (half-life 15-115 steps); **0.3%** of channels can get within 0.05 of the floor | **DEAD.** The model lives at the SLOW end (`w@rest` 0.984-0.998) and never approaches the cap. Making the constant learnable cannot buy anything. |
| `tanh` inside every LoRA bottleneck | saturation collapses an already rank-2/4 bottleneck | typical input `||A_row||` = **1.08-1.50** (tanh retains 42% of linear slope at 1.0, ~20% at 1.5); only 9.5% of `v_lora` rows exceed 2 | **DEAD twice over** — not saturating, and a learnable slope is redundant by the test above. |
| `a = sigmoid(a_lora)`, RWKV-7's in-context learning rate | `a` in (0,1): no over-correction, no full skip | reachable band is median **[0.41, 0.60]**; **0.0%** of channels reach `a<0.05`, **0.1%** reach `a>0.95` | **DEAD.** Nowhere near the bound — the model chose a narrow half-strength band, so the sigmoid is not what is limiting it. |

**That is three plausible-looking architecture proposals killed for ~20 minutes of CPU**, which is
the same discipline that killed the NS-step-count lever. It also says something worth carrying: the
trunk operates in a *narrow, smooth* regime — slow decay, mid-range `a`, unsaturated `tanh` — rather
than pressed against any of its parameterization's limits.

### WHAT SURVIVES: iter 54 candidate — learnable exponent in the channel mixer
`rwkv_model.py:633` is `o = W_v(relu(k)**2)` — RWKV-7's squared-ReLU FFN, with the exponent a
**hardcoded 2**. Make it `relu(k)**p` with `p` a learnable scalar (init 2.0): **13 params**, one per
block, and it survives the redundancy test. This is exactly the class Andrew named, applied at the
one place in the network where a fixed exponent is doing real shape work.
* Sweep granularity as part of the same iteration: global / per-stream / per-block. Per-block is 13
  params and the natural default.
* ⚠ **Implementation note that will bite:** `d/dp relu(k)^p = relu(k)^p * log(relu(k))`, which
  diverges as `k -> 0+`. Use `(relu(k) + eps)^p` or clamp the base; a naive version will produce
  NaNs in the first hundred steps and look like the iter-51 failure.
* ⚠ **Deploy debt:** this is a forward-pass change, so it needs the Rust port plus a fresh parity
  trace. First such candidate in a while — the last several were training-only.

**Ranked into the plan at #4**, displacing the decay LR shape to #5: it is a different FAMILY from
everything else queued (all of which are training-recipe or optimizer levers), and the whole point of
Andrew's catch is that the family was missing.

### QUEUE STATE 2026-08-17 (late) -- three iterations ARMED, and the plan re-ranked itself

| rank | iter | lever | cost | status |
|---|---|---|---|---|
| 1 | **52** | KD `alpha_decay` 0.5 -> 0.9 | 3.5 h | ARMED behind QAT#2 |
| 2 | **53** | `RWKV_MUON_INCLUDE_LORA=1` | 6.2 h | ARMED behind 52 |
| 3 | **54** | `RWKV_CMIX_POW=1` -- learnable channel-mixer exponent | 6.2 h | ARMED behind 53 |
| 4 | 55 | Ensemble teacher: d=128 + a frozen past champion | 5.5 h + 2 h dump | next to build |
| 5 | 56 | Decay LR SHAPE (cosine -> linear / 1-sqrt) | 3.5 h | decay-only, cheapest untried |
| 6 | 57 | Spacing-effect monotonicity in REVIEW COUNT | 5.5 h | curve constraints 2/3 |
| 7 | 58 | Fixed-budget WS:decay de-confound | ~10 h | settles a +0.0006 the endgame is spending |
| ~~-~~ | ~~-~~ | ~~Delta-rule authority (`a = c*sigmoid`)~~ | ~~5.5 h~~ | **KILLED by measurement before launch -- see LIT_REVIEW.md** |

**Iter 54 was promoted into the armed set** because Andrew's expressiveness catch opened a family
the plan had no entry for, and it is the only candidate of that family that survived the redundancy
test. It carries the first Rust deploy debt in a while (forward-pass change -> channel mixer + fresh
parity trace), which is a cost to schedule if it wins, not a reason to skip it.

**STOPPING AT THREE DEEP ON PURPOSE.** Iters 53 and 54 are already built on the iter-45 recipe, so a
win for iter 52 leaves them measuring against a superseded champion. Chaining a fourth would compound
that for no gain -- the GPU is booked ~34 h and the first verdict lands well before the queue drains.
Build 55 only once 52 has reported.

### The verdict protocol for these three, decided in advance
Each is a single variable vs iter 45 on the PLAIN rectified basis (0.297697 / 0.265375, VAL half
5001-7500), both-modes rule, `paired_pvalue` for the p-gate. Two of them have a SEPARABLE diagnostic
that should be reported alongside the number, the way iters 48 and 50 were:
* **iter 53** -- did the LoRA group's behaviour change at all? Compare the LoRA weight statistics
  against iter 45's checkpoint. A null with visibly different weights means Muon reached them and
  bought nothing; a null with identical statistics means something is wrong with the grouping.
* **iter 54** -- print the 13 learned `cmix_pow` values. Still 2.0 = the lever was inert; moved = the
  model used it and the verdict is only about whether that helped. This is the same shape as iter
  50's level embedding training to L2=1.77 and gaining nothing, which is what made that null
  interpretable rather than merely disappointing.

