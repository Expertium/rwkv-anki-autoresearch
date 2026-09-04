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
| 4 | -- | **Ensemble teacher** ⚠ **DEMOTED to rank 7 on 2026-08-17** -- see "ENSEMBLE-TEACHER SCREEN" below; the proposed 2nd teacher is the 1st teacher's own student | distillation | 5.5 h + 2 h dump | External-teacher KD is **4/4**. KD pays here through target-VARIANCE reduction (which is why α peaks at 0.9); averaging two independent teachers reduces it further. NOT iter 46: that teacher shared the trunk and the forward pass. |
| 5 | -- | **Decay LR SHAPE** (cosine -> linear / 1−sqrt) | schedule | **3.5 h** | The tuner swept `decay_ratio`, never the decay *shape*, and WSD literature puts the action in that phase. Decay-only, so it is the cheapest untried lever in the queue. |
| ~~6~~ | -- | ~~**Spacing-effect monotonicity**~~ **KILLED 2026-08-17 by the second screen** -- the violating rows are CALIBRATED (+0.0019 +/- 0.0029) and carry no excess loss, so the drops are correct inference; the between-group gap points the OTHER way. See "SECOND SPACING SCREEN". | curve constraints | 5.5 h saved | Family is **2/3**. ⚠ Do NOT propose monotone-in-t or convex-in-t: the GRU head gives both BY CONSTRUCTION (`(1+t/S)^-d`, d>0, nonneg mixture — checked in code). Monotonicity in REVIEW COUNT is a real SRS fact and is not imposed. |
| 7 | -- | **Fixed-budget WS:decay de-confound** (1+1 vs 1.6+0.4) | schedule | ~10 h | Iter 34's `decay_ratio 0.25 -> 1.0` gain is confounded with a 1.25 -> 2.0 epoch budget change; the log-linear budget curve explains +0.00084 of the +0.00145. The endgame's 10+2 split is SPENDING ~+0.0006 that rests on one confounded point. |
| 8 | -- | **Weight decay on the LoRA group** | regularization | 5.5 h | Those params sit at wd=0.0 today, never chosen — they inherited it from `other_params`. One number, and it pairs with iter 53's outcome either way (if Muon helps them, wd is the other half of the same question; if it hurts, wd is the gentler version). |
| ~~9~~ | -- | ~~**Horizon reweighting of the curve loss**~~ **KILLED 2026-08-17 by the horizon screen** -- long intervals are NOT rare (32.8% of rows at t>=21d) and the calibration gap has NO horizon trend. See "HORIZON SCREEN", which replaced it with the recalibration candidate. | objective | 5.5 h | Long intervals are rare and hard, so the curve objective is dominated by short t. ⚠ Distinct from iter 37 (by-USER weighting, mechanism-refuted in every size quartile): this addresses a coverage imbalance in **t**, which is a property of the data, not of user size. |
| 10 | -- | **Hint/feature distillation from the d=128 trunk** | distillation | 5.5 h + dump | Output-KD is 4/4; matching an intermediate representation targets the TRUNK rather than the two heads. Needs a learned d=128 -> d=80 projection — the only entry with real implementation risk. |
| 11 | -- | **Born-again: fresh student, iter-45 champion as SOLE teacher** | distillation | 5.5 h + dump | The BAN phenomenon (same-capacity student beats its teacher). Cleanly distinct from iter 46: separate forward pass, different weights, frozen. Ranked below #3 because #3 keeps the known-good teacher and only ADDS ours, so it risks less. #3's result should inform whether this is worth running at all. |

### ★★ ITER 53 SPECTRAL PRE-REGISTRATION (2026-08-17, written at step ~1150 of 10,935)
Full text + method: `scratchpad/iter53_muonlora/PREREG.md`. Tool `spectra.py`, CPU seconds, run on
the **step-matched** pair `i53_ws_1000` vs `i45_ws_1000` (same recipe/seed, augmentation off — the
control iter 47 failed to use). Two results, both before any eval number existed:
* **The lever is strongly ENGAGED** — `lora_*` stable rank **+48.26% median** (max +151.6%) against
  **−2.49%** on an INTERNAL control (`*scale*` — excluded from Muon in BOTH runs, so it cannot
  have moved), a 20:1 ratio. Unlike iters 48/50 (learned-but-negligible), the intervention is
  large, so a null would mean the mechanism doesn't pay, not that the flag did nothing.
  ⚠ `||W||_F` moved **+22.85% median / 220% max** at matched `wd=0` (verified: the LoRA group is
  explicitly `weight_decay: 0.0`, so no wd confound) — **but the inert control moved +10.28% too**,
  so most of that is indirect coupling, not the lever. Only the stable-rank change attributes.
* **★ CORRECTED 2026-08-18 — THE PREMISE HOLDS, and my first correction of it was wrong.** I
  reported 0.695 vs 0.846 ("18%, ~9x weaker than claimed") from the **CANDIDATE** checkpoint,
  which by step 1000 had already had the flag raise its LoRA stable rank 48% — **the premise
  measured on the treated model.** On the CHAMPION it is **0.5082 vs 0.8524** at step 1000 and
  0.5197 vs 0.8146 at WS-final: the LoRA matrices sit at ~60-64% of the Muon-managed group's
  relative spread, stably across training. `spectra.py` now reads the control for Q1.
  **What survives is the measurement rule:** raw stable rank (2.01 vs 17.94) is not comparable
  across shapes and overstates this ~9x, `÷ min(shape)` (0.52 vs 0.23) *inverts* the sign, and
  only a **shape-matched random reference** is meaningful. Plus: **a premise must be measured on
  the UNTREATED model** — same error family as iter 47's step-50-vs-final comparison.
* **★ WS-FINAL (step 10,935): engagement grows and the NORM GROWTH DOES NOT SATURATE.**
  `lora_*` stable rank +48.26% → +50.88% → **+66.60%** across steps 1k/2k/10.9k, inert control
  +2.72%. But ‖W‖_F goes +22.85% → +36.43% → **+70.56%** (max 372%) while the inert group's
  indirect drift falls to +2.04% — so it is the lever's own and it is still climbing. Muon takes
  a fixed-norm step where Adam's adapts to gradient scale, and this group has `wd=0`.
  **→ This PROMOTES rank 8** (LoRA weight decay) from "a knob nobody chose" to **the fix for
  this run's failure mode if it fails** — as Muon *plus* decay, not decay instead of Muon.
  It also sharpens the counter-hypothesis: the lever **overshoots**, taking LoRA from the
  champion's 0.52 past the 0.81 where the rest of the model sits, to 0.828.
* **Prediction: null or small harm**, because a rank-4 bottleneck exists *to* concentrate and the
  flag pushes it further toward flat. If null, rank 8 (LoRA weight decay) is the gentler retry of
  the SAME question, not an independent lever.

### ★★ THE GRAFT POLICY — harvesting the sub-bar positives the loop currently throws away (2026-08-17)

**The structural observation.** The accept bar is raw ≥0.0001 in both modes; the same-capacity noise
floor is ±7.5e-5. The bar is therefore only **1.33x the floor**, which leaves a band of effects that
are *real, reproducible and rank-significant* yet get discarded whole. That is not a flaw in the
gate — the gate was tightened to 0.0001 for good reason (iters 41/43/44 are mutually
indistinguishable at ≤7.5e-5) — but it does mean the loop deletes information it paid ~5.5 h to
acquire.

**The concrete instance.** iter 49 (restore the user/preset layer-0 channel mixers) measured
**imm +0.000087 at p=5.3e-16** — as statistically certain as any *accepted* result in the log — and
was rejected purely on magnitude. Its ahead side (+0.000067, p=0.11) is a coin flip, so it is one
real half-effect, not two.

**The policy, pre-registered so it is not invented after a convenient result:**

> If **≥2** of the queued iterations (52, 53, 54, 55, 57) return **sub-bar but rank-significant**
> positives *in the same mode*, run ONE graft combining them (plus iter 49) before concluding the
> algorithmic loop is exhausted.

**Why a graft of MEASURED singles is a much better bet than iter 31's was.** Iter 31 grafted three
changes simultaneously and can never say which paid. Here every ingredient already has its own
single-lever number, so the graft is attributable by subtraction — and the singles are what make the
additivity assumption checkable rather than assumed.

⚠ **The pre-registered counter-hypothesis, and the repo has already demonstrated it.** Effects need
not add. Iters 41/42/43 are exactly this: the interleave+reorder BUNDLE was worth +0.000216..+0.000611,
but order-alone was a small NEGATIVE and interleave-alone equalled the bundle — **the entire gain was
one component and the other contributed nothing.** So a graft that lands *below* the sum of its parts
is the expected case, not a surprise, and a graft that lands below its best single component means
the levers interfere.

⚠ **Do not graft levers that touch the same mechanism.** iter 49 (adds channel-mixer capacity) and
iter 54 (`RWKV_CMIX_POW`, changes the channel mixer's functional form) are both channel-mixer
changes; if both land positive, they are the *least* likely pair to be independent. Prefer pairs
from different families.

⚠ **Not available today:** of the six rejects since iter 46, only iter 49 qualifies (38 is mooted —
iter 39's α=0.9 dominates the same lever; 48, 50 and 56 are inside the noise floor; 47 regressed).
**One ingredient is not a graft.** This entry is a policy waiting on the chain, not a runnable
proposal.

### ★★ RANK 8 RE-SPECIFIED BY A DOSE SCREEN (2026-08-18) — the obvious value is a NULL by construction

iter 53's accept promoted rank 8 (weight decay on the LoRA group) from "a knob nobody chose" to the
fix for its one open worry: the deployed LoRA `‖W‖_F` is **+62.4%** over the champion's and does not
saturate, because Muon's step is fixed-norm × LR and that group runs at `wd = 0`. The obvious
implementation is to give it the **0.01** every other Muon group carries. **Pure arithmetic says that
would measure nothing.**

**The mechanics.** Muon adds a fixed-norm step scaled by LR, so the per-step norm *increment* is
∝ LR. Decoupled decay removes `LR · wd_lr_scale · wd · ‖W‖`, also ∝ LR. **The LR cancels in the
equilibrium**, and what remains is a brake time constant measured in STEPS:

| `wd` | time constant (steps) | vs the 21,870-step run (WS + decay) |
|---|---|---|
| **0.01** | 100,000 | **4.57×** — brake never engages |
| 0.02 | 50,000 | 2.29× |
| **0.05** | 20,000 | **0.91×** — acts on the run's own timescale |
| 0.1 | 10,000 | 0.46× |

Integrating the actual schedule (warmup 400 → flat → `1 − sin(πx/2)`) confirms it: `wd = 0.01`
shrinks the norm by **13.7%** across WS + decay, against a **+62.4%** growth — it offsets barely a
fifth of the thing it is meant to brake, and a null would then be uninterpretable (dose too small, or
mechanism wrong?).

**→ SPECIFICATION: run it at `wd = 0.05` on the LoRA group only.** That is the dose whose time
constant matches the run length, and it is already inside the tuner's swept range
`[0.01, 0.05, 0.1]`, so it is not an exotic value. Keep every other group at its current decay so the
LoRA group's `wd` is the single variable — the same discipline that made iter 53 attributable.

⚠ **Pre-registered reading, both ways.** iter 53 won *with* the norm growth, so the growth is not
obviously harmful — this tests whether it is harmful *anyway*, not whether it is required. If
`wd = 0.05` REGRESSES, the growth is load-bearing and the 10x endgame needs no brake after all, which
is worth knowing for a 4-day run. If it IMPROVES, the endgame must carry it.

### ~~DELETED CARDS~~ — SHELVED BY ANDREW 2026-08-18: *"forget about removing deleted cards.
### Focus on algorithmic improvements."* Both variants are off the queue. The screen below is kept
### because its measurements are reusable, not because the item is live.

**Two facts worth carrying out of it, independent of the proposal:**
* **srs-benchmark evaluates deleted cards** -- no filter on card existence anywhere in
  `data_loader.py` / `evaluate.py` / `script.py`, and deletion is absent from the README's filter
  list. So they can never be removed from EVAL. This is what the `size` gate enforces.
* **Every deleted card in a user is pooled into ONE synthetic deck and ONE synthetic preset**
  (`deck_id`/`preset_id` <- a bare `ID_PLACEHOLDER`; `note_id` is unique per card and escapes
  this). That fake deck is the LARGEST in the user for 3 of 8 sampled users, median rank 2.
  **Anyone reasoning about the deck or preset stream should know this exists**, whether or not it
  is ever changed.

#### The original screen (kept for its numbers)


> *"I believe right now RWKV is trained on deleted cards as well, their deck and preset IDs are -1.
> Check if removing deleted cards from training degrades log loss. If it doesn't, that's a free win."*

**Premise confirmed, and the −1 is real — I was wrong to "correct" it.** BOTH sentinels exist, in
different files. `srs-benchmark/data_loader.py:65` does `dataset.fillna(-1, inplace=True)` after the
card/deck left-merges — that is the −1 Andrew remembered. Our vendored `rwkv/data_processing.py`
instead fills with `ID_PLACEHOLDER = 314159265358979323` (lines 285-296). I checked our file, found
the placeholder, and told him his figure was wrong; it was right about the benchmark. **Check the
file the claim is about, not the nearest file that resembles it.**
The important detail is *which* ids our pipeline fills, since that is what trains the model:

    note_id    <- ID_PLACEHOLDER + card_id   -> UNIQUE per card; notes are NOT pooled
    deck_id    <- ID_PLACEHOLDER             -> a BARE CONSTANT
    preset_id  <- ID_PLACEHOLDER             -> a BARE CONSTANT

**So every deleted card in a user collapses into ONE fake deck and ONE fake preset.**

### It is not a rounding error (`scratchpad/deleted_cards/prevalence.py`, CPU, minutes)

| range | reviews from deleted cards | cards deleted |
|---|---|---|
| train (1–5000) | **15.18%** | 20.14% |
| eval (5001–7500) | **9.42%** | 13.63% |

Per-user spread is enormous — 0.04% (user 555) to **66.85%** (user 101).

### ★ The mechanism, measured (`fake_deck_size.py`) — this is the real argument
The synthetic deck is the **LARGEST deck in the user for 3 of 8 sampled users, median rank 2**. User
101 has 105 real decks and puts 66.85% of its reviews in the fake one; user 1200, 38.36%. The deck
stream is a per-deck recurrence, so it is spending much of its capacity summarising a group whose
members share nothing except having been deleted. **That is a fabricated grouping, not merely extra
data** — a concrete reason to expect signal rather than a null.

### ⚠ But 9.42% of EVAL rows are deleted-card rows, which shapes the design
Dropping them from training while still being graded on them conflates *"the data was useful"* with
*"we stopped training on what we are tested on"*. The asymmetry is worth stating, because it still
makes the run worth doing:
* **An IMPROVEMENT is unambiguous** — if dropping 15% of training data still wins while 9.4% of the
  eval distribution is exactly what was dropped, the fabricated-deck harm must dominate.
* **A DEGRADATION is uninterpretable** on its own.

### ★★ COMPATIBILITY CHECKED (Andrew's stop, 2026-08-18): srs-benchmark DOES evaluate deleted cards

> *"If it uses deleted cards for evaluation, then we can't remove them, since we need our evals to
> be compatible with the srs-benchmark methodology."*

**Verified in the upstream code, three ways:**
1. `data_loader.py::load_user_data` builds the dataset from **revlogs** (`create_features(df_revlogs)`),
   then left-merges cards and decks purely to ADD columns, and fills the misses with −1. No row is
   removed.
2. Neither `data_loader.py`, `evaluate.py` nor `script.py` contains any filter on card existence.
   The README enumerates the filters that *are* applied — same-day reviews, manual due-date changes,
   filtered-deck entries, an outlier filter — and **deletion is not among them**.
3. Our own pipeline agrees by construction: `data_processing.py:257` asserts
   `len(df) == df_len` straight after the merges, so it cannot be dropping rows either.

**=> EVAL MUST KEEP THEM. That is settled and non-negotiable** — it is what the `size` gate has been
enforcing all along, and it is why the published RWKV numbers are comparable at all.

**What this does and does not forbid:**
* **Variant A (drop from TRAINING)** is still runnable — it changes nothing about what is evaluated.
  But it now carries a named cost: the model is graded on 9.42% of rows whose *kind* it never
  trained on. Confounded on a loss, unambiguous on a win (see the asymmetry above).
* **Variant B (un-pool, drop nothing)** has **no compatibility question at all**. Identical rows in
  training and in eval; only the deck/preset GROUPING changes. Andrew's objection does not touch it.

**→ RUN B FIRST.** The check just removed the only argument for preferring A.

### → Two variants, no LMDB rebuild for either
The stored deck ids are raw (`data_processing` never factorizes them), so `ID_PLACEHOLDER` is
detectable at batch-prep time. Both variants are runtime changes in `prepare_batch.build_module_data`:

| variant | what it does | property |
|---|---|---|
| **A** (as proposed) | drop deleted-card rows from training | loses 15% of data; confounded on a loss |
| **B** (from the screen) | keep every row, but give each deleted card its OWN deck/preset id — exactly what `note_id` already does | **no data loss, no train/test mismatch**, and it tests the fabricated grouping directly |

**B is the sharper test and the better bet**: it removes only the fabrication, so it cannot lose on
data volume, and a null cleanly means the pooling was harmless. Implement one env flag with two
modes (`drop` / `split`) so both are one run each.

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


## ★★ ENSEMBLE-TEACHER SCREEN (2026-08-17) — the two teachers are NOT independent; DEMOTED

Run BEFORE building it, from two dumps already on disk, **zero GPU**:
`scratchpad/ensemble_screen/teacher_agreement.py` (output in `result.txt`). Both dumps cover the
IDENTICAL batch stream — `labels_sum` matches on all 14 sampled steps — so 789,158 predicted ahead
rows correspond one-for-one and no forward pass was needed.

### The finding that decides it: teacher B is teacher A's STUDENT
The proposal's mechanism is *"averaging two INDEPENDENT teachers reduces target variance further"*.
**They are not independent, and it is verifiable in the runners rather than argued.** The proposed
second teacher is the frozen iter-45 champion, and `scratchpad/iter45_kddecay/run_iter45.cmd:43,82`
sets `DUMP=C:
wkv_kd_dump	128_seedpair_65k` — the d=128 teacher's own dump. Iter 45 sits at the
end of a four-iteration lineage (32 / 35 / 39 / 45) **every one of which was trained to imitate
teacher A**. So the "second opinion" was fitted to the first opinion.

This is the same shape as iter 46's null, one step removed. Iter 46 failed because its teacher shared
the trunk and the forward pass, so the soft target re-expressed what the student already computed;
here the second teacher is a separate forward pass with different weights, but it was *optimized to
agree with the first*. Measured `r(A,B) = 0.9460`, consistent with that — though with no
non-distilled control dump the correlation cannot be ATTRIBUTED to distillation on its own. The
lineage is the argument; `r` corroborates it.

### And the intervention is small next to one we have already priced
The mix is linear in probability space (`srs_model.py:1189`,
`label_y = alpha*teacher + (1-alpha)*hard`), so both changes are in the same units:

| intervention | target shift |
|---|---|
| iter 39, alpha 0.5 → 0.9 (ACCEPTED, +0.000158 / +0.000153) | 0.0568 |
| ensemble, teacher A → 0.5*(A+B) at alpha=0.9 | **0.0117 (21%)** |

Linear projection: **+0.000033 / +0.000032** — under the 0.0001 bar and inside the ±7.5e-5 noise
floor. ⚠ Both directions of caveat, stated because the projection is crude: effect need not scale
linearly with shift magnitude (variance reduction is a different mechanism from a systematic move
toward the teacher, and could be more efficient per unit), and `E|teacher-hard|` is a
calibration-based estimate (`2*E[p(1-p)]` = 0.1419) on a corpus that is largely the teacher's own
training data, where it is genuinely more accurate — which would shrink the iter-39 shift and raise
the ratio. Neither caveat plausibly spans the 3× gap to the bar.

### The one number that argues FOR the lever
Disagreement is **concentrated where the loss lives**: mean `|A-B|` is 0.0429 on the 45.3% of rows
where the teacher is uncertain (p < 0.95) versus 0.0119 on confident rows, and **74.9% of the total
disagreement mass sits on the uncertain rows**. So the mean understates the effective intervention by
~3.6× on the rows that carry the gradient. That is why this is a DEMOTION and not a kill.

Also worth knowing: averaging necessarily blurs the target (mean `|p-0.5|` 0.4204 → 0.4131,
**-1.74%** vs the more confident teacher), and this model's KD dose-response peaks at alpha=0.9 —
it wants MORE teacher signal, not a softer target. That is a headwind specific to a mixing lever.

### What would make an ensemble worth running
The second teacher must sit **outside the KD lineage**. Candidates, with their problems:
* **iter 31 or A18** — both predate iter 32, the first KD iteration on this trunk, so neither was
  fitted to teacher A. **Verified in their own runners** — `iter31_algo/run_iter31_algo.cmd`
  and `track2_a18/run_track2_a18.cmd` both set no `RWKV_KD_MIX` — not taken from the lineage
  table.
  Genuinely non-distilled, but weaker (iter 31: 0.298909 / 0.267637) and still same-data,
  same-trunk.
* **A different-seed retrain** — the cleanest source of decorrelated error, but the teacher costs a
  full training run to produce.
* **⚠⚠ NOT `pretrain/RWKV_trained_on_5000_10000.pth`.** It is the obvious "second big teacher" and
  it is **disqualified by leakage**: it was trained on users 5000-10000, which contains our entire
  VAL half (5001-7500) and TEST half (7501-10000). Distilling from it would inject knowledge of the
  eval users into the model, and no gate we run would catch it — the numbers would simply improve.
  The teacher in use (`RWKV_trained_on_101_4999.pth`) is the correct one and `iter10_kd/run_kd_dump.cmd:4`
  already flags exactly this ("never saw eval users 5001-10000"). Keep it that way.

**RANKING:** #55 drops below the decay-LR-shape and spacing-effect entries — both are cheaper and
neither has a measured headwind. If it is run later, run it with iter 31 as teacher B, not iter 45.



## ★★★ ITER 55 (APPROVED BY ANDREW 2026-08-17): RETRIEVABILITY-GATED STATE UPDATE

Andrew asked for genuinely big architectural changes, was pointed at FSRS, and approved this one
(*"#1 sounds good"*). **Design recorded BEFORE implementation** so a compaction cannot lose it.

### The mechanism, read from FSRS's own code (`srs-benchmark/models/fsrs_v7.py:263-312`)
```
R     = forgetting_curve(dt, S, S_short, D)   # time enters the READOUT
S_new = next_stability(S, D, R, rating)       # and R then GATES the state update
D_new = next_difficulty(D, rating, R)
```
**FSRS's state does NOT decay in real time** -- it is unchanged between reviews. What pays is that
the *size of the update* is scaled by the model's own predicted retrievability: recall something you
were likely to have forgotten and stability jumps; recall something you'd have remembered anyway and
it barely moves. That is the spacing effect as STRUCTURE.
⚠ An earlier note in this session proposed making our state decay as `w^dt` "like FSRS". That was
WRONG -- inferred from the forgetting curve without reading `step()`. Corrected here.

### Why our model is the right target
RWKV-7 already HAS the corresponding mechanism: the delta rule, whose `a` is literally an in-context
learning rate (`rwkv_model.py:948`, `a = sigmoid(a_lora(...))`), with delta-direction eigenvalue
`w - a*||kappa||^2`. **And we measured it sitting idle**: `a*||kappa||^2 ~ 0.13` against a reachable
~0.95, moving the eigenvalue ~0.15 against a decay of ~0.98. So the one RWKV-7 mechanism that
structurally matches FSRS's core update is the one this trunk barely uses.

### ★ THE DESIGN CONSTRAINT THAT SHAPES THE WHOLE LEVER (get this wrong and it is a capacity add)
A naive version computes `rhat` as another learned function of the same `x` the `a`-LoRA already
sees, and adds it to the a-logit. **That fails the REDUNDANCY TEST** -- it is extra capacity on an
existing gate, and capacity-at-5k is 0/3. What makes it genuinely new is using **FSRS's FUNCTIONAL
FORM**:

    rhat = (1 + dt / s_hat) ** (-d_hat)

a retrievability computed from a learned stability and the **actual elapsed time**. A rank-4 LoRA
cannot express that interaction between a learned scalar and an input feature, so it survives the
test. Then, in logit space with a ZERO-INIT gain so the model is bit-identical at init:

    a_logit += gain * (1 - rhat)        # FSRS sign: lower expected recall => larger update

### Implementation plan
* **Plumbing is the real work.** The time-mixer sees only the projected `d_model` hidden state;
  `dt` must be threaded from `srs_model` down through `RWKV7` -> block -> time_mixer. The column is
  `CARD_FEATURE_COLUMNS.index("scaled_elapsed_seconds") == 2`.
* **SCOPE v1 = the CARD stream only** (depth 2). That column means "gap since THIS CARD's previous
  review", which is exactly FSRS's `dt` for the card stream; gathered into deck/preset it silently
  becomes a different quantity. Scope string like the QAT scopes, e.g. `RWKV_RGATE=card`.
* **Params:** 2 linears (d_model->1) + 1 scalar per gated layer ~= 161 x 2 = **~322 (+0.06%)**.
* **Deploy:** state size UNCHANGED (no new state), so the frozen 9-byte card budget holds. But it IS
  a forward-pass change => `rwkv_rnn_model.py` mirror + a `parity_train_vs_rnn.py` case + the Rust
  port + a fresh parity trace. Same debt class as iter 54.
* **Guards:** default OFF and inert; zero-init gain means even ON it starts bit-identical;
  `smoke_scripted_eval.sh` before launch (mandatory after touching these files).

### Pre-registered counter-hypotheses (write the verdict against these, not after the fact)
1. **`gain` trains to ~0** => surprise-gating adds nothing here; the generic recurrence already does
   what FSRS needs explicit structure for. A real finding, same shape as iter 48's learned-but-
   negligible coupling.
2. **`gain` moves but logloss does not** => the trunk already had the information (iter 48/50 shape).
3. **It helps** => the delta rule was idle because nothing was pointing it at the right signal, and
   the natural follow-ups are gating `note` too, and tying `rhat` to the real curve head.
⚠ Report the learned `gain` per gated layer as the separable diagnostic, exactly as iter 54 reports
its 13 `cmix_pow` values.

### ✅ BUILT 2026-08-17 (`ef1853f`) — code complete, NOT yet armed
`RWKV_RGATE=card`, default off. **+324 params exactly as pre-registered**; per-entity state sizes
unchanged, so the frozen 9-byte card budget holds. Threading: `log_dt` is recovered from
`scaled_elapsed_seconds` in canonical row order and gathered per split with x's own indices, then
passed `RWKV7 -> block -> time_mixer`; mirrored in `rwkv_rnn_model.py` + `srs_model_rnn.py`.
Implemented in log space as `exp(-d * softplus(log_dt - log_s))` — the same function for every real
input, so there is no clamp and no NaN branch, and `rhat` in (0,1] bounds the added term to
`[0,|gain|]` (a WORST-CASE bound, the check iter 51 lacked).
**Optimizer placement is deliberate:** all three tensors land in `other_params` at wd=0 (a (1,C)
weight has squeeze-rank 1, so `train_rwkv`'s ">=2-D matrix" rule skips it and Muon never sees it).
Weight decay on a zero-init gain would pin it at zero and make counter-hypothesis 1 unfalsifiable.

**Verification (all green, CPU, zero GPU):**
* `parity_train_vs_rnn.py` — 2 new cases, **10/10 pass**. Gate case 1.43e-06, with dt-sensitivity
  2.96e-02 on BOTH paths and parity 1.67e-06 at a second dt. That second half is load-bearing:
  agreement between two paths that both IGNORE dt is a matched no-op, not parity. Scope case
  confirms `RWKV_RGATE=note,deck` leaves a card stack ungated.
* **NEW `scratchpad/parity3/smoke_rgate.py`** (real chunk, real `prepare()`) — the single-stream
  harness cannot see the PLUMBING, which is the risky half: `log_dt` is the first raw input feature
  ever threaded into the recurrence, and an off-by-one in the gather would silently gate every
  review on some OTHER review's elapsed time. **ON@gain=0 vs OFF = 0.000e+00** (identical checksums
  to 10 dp), so the zero-init inertness claim is measured rather than asserted; gain=0.8 moves
  1.88e-03; all 8 rgate tensors get finite non-zero grads; recovered `log_dt` is physically
  log-seconds (median 6.68 = 13 min, max 15.86 = 89 days).

**★ FOUND BY THAT SMOKE, and it would have become folklore otherwise: the LMDB stores features in
BFLOAT16.** The first-review sentinel's standardized value −1.9117082534 is held as −1.9140625, and
un-standardizing multiplies the error by std=5.21, so a first review recovers as
**`log_dt` = −0.01227, not 0.0** (17.9% of rows in a real chunk). Substantively identical — `rhat`≈1
there, so the gate contributes ~0 exactly as intended — but any `== 0` sentinel test can never pass,
and the same ±0.02 log-space quantization (~2% in dt) rides on every row. Documented in the model
and in the smoke.

**STILL OWED BEFORE ARMING:** `smoke_scripted_eval.sh` (GPU-gated, waiting on the QAT#2 eval;
mandatory after touching `srs_model.py` — a PLAIN eval is the only path that scripts the model, so
iter 48's bug class is invisible to training AND to QAT evals), plus the Rust port + fresh parity
trace, the same deploy debt iter 54 carries.

## ★★ SPACING-EFFECT SCREEN (2026-08-17) — the constraint BINDS HARD, but the proposal as written is WRONG

Run before building it, CPU only, no GPU: `scratchpad/spacing_screen/monotonicity_probe.py`
(champion driven through the deploy RNN path over whole user histories; each card's stored curve is
re-read at fixed horizons 1d/7d/30d/180d and consecutive reviews of the same card are compared).
Output in `result.txt`. Comparing at a FIXED horizon is what makes it a statement about the model —
comparing each curve at its own interval would conflate "stability changed" with "the interval
changed".

**INSTRUMENT VERIFIED FIRST, and this is why the numbers are usable:** the probe reproduces the
`reference_iter41` trace's certified `py_pred_ahead` on all 4,215 ahead rows at **exactly
0.000e+00** (`verify_probe.py`) — the same predictions the Rust port was certified against. As a
free by-product that also proves `imm_predict` is genuinely state-read-only, since the probe skips
it and the exporter does not.

### The result: violations are common, and they split by BUTTON
n = 8,862 consecutive same-card pairs, users 107 + 136. Violation = predicted retention at the fixed
horizon DECREASED. At 30d:

| button | pairs | R decreased |
|---|---|---|
| Again | 663 | 59.3% |
| **Hard** | 2,326 | **65.9%** |
| Good | 5,659 | 39.7% |
| Easy | 214 | 38.3% |

**The proposal says "penalise a stability decrease after a SUCCESSFUL review", and `rating >= 2`
makes Hard a success.** So a blanket rule would fight the model on **66% of Hard transitions** —
where a decrease is not an error but correct inference, Hard being direct evidence the card is
harder than assumed. **Any implementation must be Good/Easy-conditional.** That alone is worth the
screen: the blanket version was the obvious one to write.

### The sanity check that forced the reinterpretation
Predicted retention at a fixed horizon **falls over a card's life** — only 20-27% of cards end
higher than they started (mean R(30d) 0.9356 → 0.8688). That is the opposite of the naive SRS
expectation, so before trusting any violation rate it had to be explained.

**It is difficulty selection, and that is measurable from the DATA ALONE with no model involved:**
lapse rate per card rises monotonically with how many reviews the card received — **1.9% (1 review)
→ 9.0% (5-7) → 21.2% (8-12) → 46.4% (21+)**, Spearman rho **0.4867** over 10,797 cards spanning two
train and two held-out users. Cards that survive to many reviews are the hard ones, so a model that
infers difficulty *should* lower its retention estimate as evidence accumulates. The declining trend
is correct behaviour, not a probe artifact — which is exactly why the per-step rates could then be
read at face value.

### ⚠ And the "structural fact" is FSRS's modelling assumption, not a proven property of memory
In FSRS, `R(t) = (1 + FACTOR*t/S)^-DECAY` with DECAY **fixed**, so `R(t_fixed)` is a monotone
function of `S` alone and cannot fall after a successful review. Our curve is a **mixture** of
`(1 + t/s_i)^(-d_i)` with **learnable per-curve `d_i` and mixture weights**, so it can lower
`R(t_fixed)` while stability grows — it has strictly more freedom, and it *uses* that freedom on
~40% of Good transitions. So the constraint is not free structure being left on the table; it is a
restriction our (more accurate) model currently declines to obey.

**VERDICT: keep it queued, re-specified.** Good/Easy-conditional, and understood as a
*regularizer* — the same shape as PAVA, which is the family's accepted win: it may buy generalization
while costing train loss. Do NOT pitch it as "imposing a fact the model is getting wrong". The screen
did not kill it, but it replaced the implementation and the rationale.

## ★★★ SECOND SPACING SCREEN (2026-08-17) — THE LEVER IS DEAD, and the data points the other way

`scratchpad/spacing_screen/violation_calibration.py`, output in `calibration_result.txt`. ~90 min of
CPU, no GPU, 90,278 reviews over 5 TRAIN-range users (107/136/156/178/203), champion driven through
the deploy RNN path — the same instrument that reproduces the certified `reference_iter41` ahead
predictions at exactly 0.000e+00.

The first screen established the constraint BINDS (39.7% of Good / 38.3% of Easy transitions lower
predicted retention at a fixed horizon). **Necessary, not sufficient: a violated constraint is only
worth imposing if the violations are WRONG.**

### Why the obvious test would have been wrong, and what replaces it
Comparing logloss on rows following a violation vs following a non-violation is CONFOUNDED by a
mechanism the first screen itself measured: violations concentrate on hard cards (per-card lapse rate
1.9% → 46.4% with review count, ρ=0.4867), and hard cards carry more loss whatever the model does.
**Calibration is the confound-free version** — a hard card has a low `p` AND a low `y`, so its gap
stays near zero; only a systematically mis-set state moves `mean(y) − mean(p)`.

### The result: the violations are CORRECT INFERENCE

| group | n | mean p | mean y | calibration gap | logloss |
|---|---|---|---|---|---|
| **after Good/Easy WITH violation** | 17,026 | 0.9588 | 0.9607 | **+0.0019 ± 0.0029** | 0.1323 |
| after Good/Easy, no violation | 49,199 | 0.9680 | 0.9642 | **−0.0038 ± 0.0016** | 0.1316 |

**The primary reading needs no between-group comparison and carries no confound: the violating rows
are CALIBRATED.** Their gap is statistically indistinguishable from zero. The lever's entire premise
is that the drop is an error, which would appear as a clearly positive gap right there. It does not.
**The target does not exist.**

Corroborating, from the same table: the violating rows carry **no excess loss** — 0.1323 vs 0.1316 —
despite being genuinely harder rows (mean p is 0.009 lower). Adjusted for difficulty they are
predicted *better* than the rows the constraint would leave alone.

### ⚠ And the between-group difference points AGAINST the lever, not for it
The violation-minus-control gap is **+0.0057 ± 0.0033**, and it holds in **all six**
predicted-probability bins (the strict, difficulty-matched form) — so it is not a mix artifact. But
read where it comes from: the violating group is calibrated and the **NON-violating group is
overconfident** (−0.0038, n=49,199). The model's error after a successful review is *failing to lower
R enough*, in exactly the rows the regularizer would leave untouched — while the rows it would
penalize are the correct ones. **A penalty on drops would push the wrong way.**

⚠ **Do NOT promote "non-violating rows are overconfident" to an established finding without a lagged
control.** That comparison partitions on a MODEL-INTERNAL change (the sign of ΔR), which invites
regression to the mean: "R dropped" preferentially selects transiently-low estimates that then
regress upward, and vice versa. The symmetric straddle of zero (+0.0019 / −0.0038) is exactly the
signature. Binning on `p` controls the level, not the change. The primary verdict above does not
depend on this comparison at all, which is why it stands regardless.

### VERDICT: REMOVE from the queue (plan rank 6). Family stays 2/3.
This is the **fifth cheap screen to change the ranking**, and the second to change it for this one
lever — the first re-specified the implementation (Good/Easy-conditional), this one removes the
justification for any implementation. ~90 min of CPU against a 5.5 h GPU run that would have measured
a regularizer pulling against the data.
**What is NOT closed:** the curve-shape family (2/3, PAVA and λ=0.2 are its wins) and monotonicity in
`t`, which the GRU head already gives by construction. Only the review-count monotonicity constraint
is dead.

## ★★★ HORIZON SCREEN (2026-08-17) — rank 9 DIES, and the run turned up something better

`scratchpad/spacing_screen/calibration_by.py` (output `calib_by_result.txt`, records cached in
`calib_records.npz`) reuses the spacing screen's instrument and slices 83,478 predictions several
ways, so the next question costs a slice rather than another 90 minutes of CPU.

### Rank 9's premise fails on BOTH halves

| t bucket | n | mean p | mean y | gap | logloss |
|---|---|---|---|---|---|
| <1d | 21,587 | 0.9727 | 0.9733 | +0.0006 | 0.0955 |
| 1-3d | 9,045 | 0.9659 | 0.9595 | **−0.0064** | 0.1325 |
| 3-7d | 10,877 | 0.9681 | 0.9689 | +0.0009 | 0.1081 |
| 7-21d | 14,616 | 0.9642 | 0.9595 | **−0.0047** | 0.1367 |
| 21-60d | 11,731 | 0.9560 | 0.9515 | **−0.0045** | 0.1611 |
| 60-180d | 8,910 | 0.9594 | 0.9529 | **−0.0066** | 0.1691 |
| >180d | 6,712 | 0.9590 | 0.9549 | −0.0041 | 0.1455 |

* **"Long intervals are RARE" — FALSE. 32.8% of scored rows sit at t ≥ 21 days.** The horizon
  distribution is remarkably spread; the curve objective is not dominated by short t.
* **"…and HARD, so the model underfits the tail" — the loss column rises with t, but that is not the
  test.** Long gaps are genuinely harder; that is the task, not an error. The test is the GAP, and it
  shows **no horizon trend**: 60-180d (−0.0066) is indistinguishable from 1-3d (−0.0064), while <1d
  and 3-7d are calibrated. **Reweighting toward long t would be correcting an error that is not
  there.** Sixth screen to change the ranking.

### ★★ THE REAL FINDING: the champion is systematically OVERCONFIDENT, and KD is why

Read the sign column: the gap is **negative nearly everywhere**, overall **−0.00292** — and this is
measured on **TRAIN-range users**, which is what makes it a finding. Binary cross-entropy is a proper
scoring rule: at its optimum `mean(p) == mean(y)` within any input-determined group. A persistent gap
on data the model trained on means it is not at the calibration optimum **for the hard labels**.

**The mechanism is in the code, not inferred** (`srs_model.py:1261-1263`):

    label_y = alpha * teacher_curve + (1 - alpha) * label_y      # alpha = 0.9 WS, 0.5 decay

**The curve head is not trained to predict the outcome. It is trained to predict a blend that is
mostly the d=128 teacher's probability**, so the teacher's calibration is inherited and nothing in the
objective pulls it back to the data's frequency. KD is the log's best family (4/4) and it pays through
target-VARIANCE reduction — but variance reduction and calibration are separable, and only one of them
was ever measured.

**Size of the prize** (`recalibration_prize.py`, held out, base logloss 0.1295 on this population):

| correction | params | held-out gain |
|---|---|---|
| logit shift `z + b`, b = −0.093 | 1 | **+0.000115** |
| Platt `a·z + b`, a = 1.033 | 2 | **+0.000131** |

Just over the 0.0001 accept bar and comfortably clear of the 7.5e-5 noise floor.

⚠ **Caveats, none of which I can discharge from this run.** (a) TRAIN-range users and ALL
predecessor-having rows, not the benchmark's equalized subset — the absolute 0.1295 is NOT comparable
to the 0.2977 gate number; only the relative gain transfers. (b) The held-out split is 2-fold by ROW,
not by user — weaker than the gate's setting. Fixed for future runs (`calibration_by.py` now records
the user id; delete the npz and re-run with `REUSE=0` to get a per-user split). (c) My first
back-of-envelope estimate was **+0.00037, ~3x too high**, because it used a bucket-level gap where the
overall gap is −0.00292 — the measured number is the one to quote.

### The candidate, and a pre-registered test of it that is ALREADY RUNNING
Two routes, with different costs:
1. **Training-side: `alpha_decay` → 0.25, or a short pure-hard-label tail at the very end of decay.**
   Lets the model recalibrate to the data while keeping the KD-learned representation. Cheap (3.5 h,
   decay-only, same dump) and already listed as open in-family — this gives it a MECHANISM it did not
   have. ⚠ It trades against the variance reduction that made KD win, so the +0.00013 is a ceiling
   on the calibration half only.
2. **Post-hoc recalibration** harvests it without that trade, but it is a forward-pass change on the
   curve path and adds deploy debt (the GRU head's output is a mixture, so a logit shift is not
   foldable into an existing linear bias).

**★★ PRE-REGISTRATION CONFIRMED (2026-08-18).** The test was: *"iter 52 raises `alpha_decay` to
0.9, i.e. MORE teacher in decay. If the KD-calibration cost is real and binding, iter 52 should
come back neutral-to-negative."* **It came back NEGATIVE** — ahead −0.000043 (a tie), imm
−0.000116 (a real regression), logged as iter 55. The excluded outcome was the one the WS dose
curve predicted (0.5 → 0.75 → 0.9 monotone up), so the prediction had content.
**→ α=0.9 wins in WS and loses in decay.** Variance reduction is an EARLY good; calibration is a
LATE one. iter 45 showed *some* teacher in decay helps, so the decay optimum is INTERIOR, near
0.5. **The informative direction is now `alpha_decay` 0.25** — the calibration mechanism predicts
it IMPROVES, and a null bounds the effect. Decay-only, ~6.1 h, same dump, zero code. **Promoted
to the front of the queue**, alongside the re-specified rank 8 (`wd=0.05` on the LoRA group).

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

### QUEUE STATE 2026-08-17 (late) -- three ARMED, and the unbuilt list is re-ranked

> **⚠ UNBUILT CANDIDATES DELIBERATELY CARRY NO ITERATION NUMBER.** Pre-assigning them is what
> desynchronised this file's two tables (the ensemble teacher was written as "iter 54", then the
> real iter 54 became the cmix exponent, and every later row was off by one). A number is assigned
> when the runner is built, not when the idea is ranked.

| rank | iter | lever | cost | status |
|---|---|---|---|---|
| 1 | **52** | KD `alpha_decay` 0.5 -> 0.9 | 3.5 h | ARMED behind QAT#2 |
| 2 | **53** | `RWKV_MUON_INCLUDE_LORA=1` | 6.2 h | ARMED behind 52 |
| 3 | **54** | `RWKV_CMIX_POW=1` -- learnable channel-mixer exponent | 6.2 h | ARMED behind 53 |
| 4 | -- | Decay LR SHAPE (cosine -> linear / 1-sqrt) | **3.5 h** | next to build; decay-only, cheapest untried, no measured headwind |
| 5 | -- | Spacing-effect monotonicity in REVIEW COUNT | 5.5 h | curve constraints 2/3; screen it first (see below) |
| 6 | -- | Fixed-budget WS:decay de-confound | ~10 h | settles a +0.0006 the endgame is spending |
| 7 | -- | Ensemble teacher | 5.5 h + 2 h dump | **DEMOTED 2026-08-17 by a zero-GPU screen** -- the proposed 2nd teacher is the 1st teacher's own student. Only worth running with iter 31 as teacher B. |
| ~~-~~ | ~~-~~ | ~~Delta-rule authority (`a = c*sigmoid`)~~ | ~~5.5 h~~ | **KILLED by measurement before launch -- see LIT_REVIEW.md** |

**★ THE CHEAP-SCREEN HABIT: FOUR SCREENS, SIX CANDIDATES RE-RANKED — RUN ONE BEFORE EVERY
BUILD.** The expressiveness bounds (3 killed), the delta-rule authority probe (1 killed), the
ensemble screen (1 demoted) and the spacing-effect screen (1 re-specified). Each cost
minutes-to-an-hour of CPU against the 5.5-13 h GPU runs it redirected. **Before building the decay-LR-shape
and spacing-effect entries, ask what measurement would kill them.** For spacing-effect that question
has an obvious answer and it is not yet run: *how often does the champion actually violate
monotonicity of stability in review count?* If it essentially never does, the constraint is
non-binding and the lever is dead the same way the decay floor was. That needs per-review `log-S`
from a forward pass, so it is CPU-minutes on a few users rather than free -- but it is still ~100x
cheaper than the run it would cancel.

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


## ★★ DECAY-SHAPE SCREEN (2026-08-17) — rank 5 was mis-specified twice; the lever survives with a BETTER motivation

Pure arithmetic, no GPU, minutes. Rank 5 read *"Decay LR SHAPE (cosine -> linear / 1−sqrt)"*. Both
halves of that are wrong, and the screen also supplies the argument the entry was missing.

**1. WE ARE NOT ON A COSINE.** `train_rwkv.py`'s decay lambda is `1 + cos(pi/2*(1+x))`, whose name
(`cosine_down`) is what made it look like one. It is **identically `1 − sin(pi x/2)`** (agreement
4.4e-16) — far more aggressive than a standard cosine.

**2. `1−sqrt(x)` IS NOT AN ALTERNATIVE DIRECTION.** It sits essentially on top of the current
schedule. Integrated LR multiplier over the decay ("LR mass") and the midpoint value:

| shape | LR mass | vs current | f(0.5) |
|---|---|---|---|
| **CURRENT** `1 − sin(pi x/2)` | 0.3634 | 1.000x | 0.293 |
| `1 − sqrt(x)` | 0.3333 | 0.917x | **0.293** |
| linear `1 − x` | 0.5000 | **1.376x** | 0.500 |
| standard cosine `.5(1+cos pi x)` | 0.5000 | **1.376x** | 0.500 |

So the only informative direction is **more LR mass**.

**★ AND THAT REPLACES THE MOTIVATION WITH A MUCH BETTER ONE.** "Shape is unexplored" was never an
argument. But **iter 34 adopted `decay_ratio` 0.25 → 1.0** — 4x the decay steps — and it was the
phase's largest gain, i.e. direct local evidence that **more decay-LR-mass helps**. `decay_ratio`
buys mass by spending WALL-CLOCK; the shape buys **1.376x of it at an identical step count, for
free**. That is worth a run.

⚠ **PRE-REGISTERED COUNTER-HYPOTHESIS:** linear and standard cosine have the *same* mass (0.5) and
different profiles, so **a win here is a MASS result, not a SHAPE result**, until those two are
compared against each other. If iter 57 wins, the follow-up that separates them is standard cosine
at matched mass — do not claim "shape matters" from this run alone.

**BUILT AS ITER 57** (`RWKV_DECAY_SHAPE=linear`, decay-only, ~3.5 h, queued behind iter 55). Params
unchanged at 558,212 (a schedule has no weights); no deploy debt. Perfectly controlled by
construction: it warm-starts from the champion's own `i45_ws_10935`, so WS is literally the
champion's and the lever cannot touch it; KD alpha stays at the champion's 0.5 because iter 52 is
the run that moves it. The default `RWKV_DECAY_SHAPE` path was verified **bit-identical to the
historical schedule over all 10,935 decay steps** before queueing — mandatory, because a chain's
later phases import whatever is on disk *then* and iter 53 was mid-flight.

⚠ **That verification's own step count was wrong at first (corrected 2026-08-18).** It was checked
over **2,734** steps — the tuner-era `decay_ratio = 0.25` figure — when the real decay is **10,935**,
the same as WS, because iter 34 adopted `decay_ratio = 1.0` (`EPOCHS = 1.0` in every `*_decay.toml`;
the champion's own checkpoint is `i45_d_10935`). Re-verified at the true length: **bit-identical
across all 10,936 values**, so no in-flight run was affected. The same stale figure had the LIVE
chain costed ~40% low — a full iteration is **9.2 h** and a decay-only **6.1 h**, not 3.5 h.
**And the clamp turns out to fix a latent bug:** the original `1 + cos(pi/2*(1+x))` lets the LR
*rise again* past `total_steps` (0.735 at t=20,000), where the clamped form pins it at 0.
Unreachable through LambdaLR's counter, but wrong if it ever were.

## ★ INVENTED, 2026-09-02: CALENDAR-AWARE FORGETTING CURVE (from the featB ahead/imm split)

> **✗ KILLED 2026-09-04 17:40 BY ITS OWN PRE-REGISTERED COUNTER-HYPOTHESIS, before any code.** The LOO
> sweep's `t_since_any` arm (featB checkpoint, n=300) costs **+0.001325 ahead / +0.005076 imm** -- i.e.
> the ENTIRE clock group (+0.001366 / +0.005072 from the grouped ablation). The calendar columns are
> each worth ~+0.00008 ahead and ~0 imm (tod +0.000082/+0.000005, dow +0.000083/+0.000002, doy
> +0.000074/-0.000001, is_weekend +0.000083/-0.000002; tod_dev +0.000115/+0.000023). So the clock
> gain is SESSION RECENCY (seconds since the user's last review of any card), which is not a function
> of `t` and cannot be supplied to a prediction for a future date. The lever's ceiling collapses to
> ~0.0001 ahead, under its own 0.0005 abort line. Not building it; the deploy question it raised is
> moot. ⚠ Note what this does NOT say: imm's reliance on `t_since_any_review` IS deployable -- Anki
> knows the last review time of any card when it asks for R(now) -- so the featB imm gain is real
> deploy value; only the AHEAD/scheduling path is structurally blind to it.


**Provenance: invented** (ours, from the featB measurement; the queue's next `adopted` slot is
unaffected -- this is filed for the invented slot after it).

**The observation.** featB moved imm +0.002371 and ahead +0.000303 (7.8:1; ~28:1 review-weighted).
The structural reason is in `prepare_batch`: a real row's ahead label is the card's NEXT review, so
the curve at row k is scored on review k+1 with only rows <= k as input, while the query row that
scores the same event for imm carries review k+1's own feature vector -- including its time of day,
weekday, day of year and seconds-since-any-review. The 10 clock columns are therefore visible to imm
and invisible to ahead by construction. (Detail: `research_5k_verbose.md`, featB, "WHY AHEAD MOVED
SO LITTLE".)

**The lever.** The clock of the evaluation time is NOT unknown at prediction time: `tod`, `dow`,
`doy` of the predicted review are deterministic functions of `review_time(k) + t`, and a live
scheduler knows `now()` when it asks for R(now). So the curve head can be conditioned on the calendar
phase of the point it is evaluated at, with no new input columns: for each of the `num_points`
evaluation offsets (and for `label_elapsed_seconds` in the loss), compute `sin/cos` of the target
tod/dow/doy from the row's `review_time` plus the offset, and feed them into the GRU head's
per-point input (it already takes the offset `t`). Train, eval and deploy compute the same quantity
(§9 three-way parity): Rust gets `review_time` per row already in the -id layout.

**What it can be worth: bounded by the ablation.** `abl_clock` (armed) measures how much of featB's
imm gain rests on the 10 clock columns; that number is the CEILING for ahead under this lever, since
ahead would then see the same information for the scored review. If `abl_clock` costs imm < 0.0005,
the lever is not worth a run.

**Constraints it must respect.** (a) PAVA / monotonicity in t: a calendar-conditioned curve is no
longer monotone in t by construction (a review at 03:00 can be predicted harder than one at 15:00 a
few hours later). PAVA still rectifies at eval, and `RWKV_PAVA_LAMBDA` still trains toward
monotonicity, so the deploy contract (zeroed duration + PAVA + no residual) is unchanged -- but the
lever and the rectifier pull in opposite directions on exactly the diurnal wiggle, so measure the
rect-vs-unrect gap on the candidate. (b) `-id` lineage only: needs `review_time`, which the
published dbs do not carry. (c) Gate: this is a curve-side lever -> the curve-side exception
applies (ahead >= +0.0001 at p<1e-4, imm not significantly worse).

**Pre-registered counter-hypothesis.** The imm gain from the clock columns may come mostly from
`t_since_any_review` (seconds since ANY card was reviewed -- session structure), which is NOT a
function of `t` and cannot be supplied to ahead. If the ablation of `t_since_any_review` alone
explains most of `abl_clock`, the lever's ceiling collapses; run that single-column ablation
(~3 h, one eval) before building.

**Implementation sketch (2026-09-04, written while the LOO sweep runs; NO code until it reports).**
The GRU head is per-ROW, not per-point: `_gru_heads(x_w)` emits (w, S, d) for N=3 curves once per row
and `gru_forgetting_curve` evaluates the closed form at the label's `t` (`srs_model.py:872-874`,
`:1014`). So "condition the head on the target phase" is best done as a t-dependent MODULATION of the
curve's logit, not a change to the head's input:
`logit R'(t) = logit R(t) + sum_k c_k(x_w) * phi_k(t)`, with `phi` = the target-time clock pairs
obtained by ROTATING the row's own `tod_sin/cos` (P = 86400 s) and `dow_sin/cos` (P = 7 d) by
2*pi*t/P -- `sin(a+b) = sin a cos b + cos a sin b` -- so no new input column, exact at train, eval
and RNN deploy alike (the Rust engine has the same feature vector). `c_k = Linear(w_head_dim -> 4)`,
zero-init => byte-identical at start, ~1.3k params. Flag `RWKV_CAL_CURVE=1`, default off = inert;
`doy` left out (the dataset spans few years per user and the ablation put the annual half in the
dead-weight group). Parity: add a case to `parity_train_vs_rnn.py` AND a 4-point rotation identity
check (rotating by exactly P must reproduce the pair). Gate: curve-side exception.
**⚠ The deploy question that this makes concrete, for Andrew when the LOO reports:** PAVA at eval/deploy
flattens the intra-day wiggle, and the Rust interval solver inverts the RECTIFIED curve -- so at
deploy the lever keeps only its monotone envelope unless the solver is made calendar-aware (solve
R'(now + t) for t, which is a CONTRACT change). Measure rect-vs-unrect on the candidate first; if the
gain lives in the wiggle, it is deploy-invisible under the current contract and the lever is worth
nothing as shipped.

## ★★ RANKED QUEUE 2026-09-04 18:20 -- the 3-agent refill (13 distinct levers from 15 proposals)

Full texts: `scratchpad/proposals_2026-09-04/{literature,domain,steelman}.md` (written to disk by the
agents themselves). Reference for every band = realcyc 0.298083 / 0.263592 (n=2,499); AHEAD is binding.
Two agents converged INDEPENDENTLY on rank 1 (literature via CORAL, domain via FSRS's graded stability
updates + a data screen it already ran: Hard-vs-Good lapse gap 1.4x at matched t, positive in 14/18
users) and on rank 3 (born-again KD). The strict alternation says the next slot is INVENTED.

| rank | lever | provenance | gate | expected ahead | GPU | CPU screen (kill rule) |
|---|---|---|---|---|---|---|
| 1 | **Ordinal next-rating supervision on the curve logit** (`label_rating` on real rows = the k+1 button, consumed by NOTHING today; CORAL cutpoints on `z = logit R(t)`, 2-4 train-only params) | **adopted** (Cao/Mirjalili/Raschka 2020 CORAL, arXiv 1901.07884; Frank & Hall 2001) | curve-side | +0.0001..+0.0004 | 1 run | RNN pass: among successes with t>=1 d, Hard share must FALL and Easy share RISE with decile of logit p (|rho|>=0.8 both); AUC(Easy vs Hard) > 0.75 => already separated, dead |
| 2 | **Duration dropout on input dim 8** (per-row Bernoulli zeroing of `scaled_duration` on real rows, train only; iter 33's prescribed clean retry) | **invented** | BOTH modes | +0.00015..+0.0005 rectified | 1 run | RNN pass: ahead cost of zeroing the current row's duration on THIS checkpoint < +0.0004 => dead |
| 3 | **Born-again KD from the frozen realcyc checkpoint** (existing dump path, alphas 0.9/0.5 unchanged, fresh student) | **adopted** (Furlanello et al. 2018, arXiv 1805.04770; Mobahi et al. 2020) | BOTH modes | +0.00015..+0.00045 | 2 h dump + 1 run | teacher train-minus-val gap > 0.010 => memorising, dead; calibration gap on train users |
| 4 | imm-scale 0.5 (buy ahead with imm; pbin's trade run in the untested direction) | invented | **DIRECTED -- Andrew** | +0.00015..+0.00045 (imm -0.0002..-0.0005) | 1 run | gradient cosine ahead vs imm on the trunk: > +0.5 => inert |
| 5 | Multi-horizon button ordering on the probes (hinge on adjacent-button order at t x {1/8, 8}) | invented | curve-side | 0..+0.0003 | 1 run | crossing rate at other horizons < 3% => dead |
| 6 | SAM, decay-only first (LookSAM fallback) | adopted (Foret et al. 2021) | BOTH | 0..+0.0003 | 2x decay | sharpness gap at rho 0.05 < 0.002 => flat already |
| 7 | Auxiliary next-interval regression head (train-only, 321 params) | adopted (Caruana 1997; Time-LSTM) | BOTH | +0.00005..+0.0002 | 1 run | ridge probe R^2 > 0.85 => trunk already carries it |
| 8 | Odds-power monotone residual on the curve logit | invented | curve-side; **contract sign-off** | +0.0001..+0.0003 | 1 run + port | per-user oracle fit < +0.0002 => dead |
| 9 | Probe density 0.08 -> 0.20 (zero code) | invented | curve-side | 0..+0.0002 | 1 run, +30% WS | none; overlaps rank 2 |
| 10 | Curve-logit recalibration (shift+temperature), KD-off re-screen | invented | curve-side | 0..+0.00013 | 1 h CPU + eval | held-out by-user prize < +0.0001 => dead |
| 11 | Muon coverage of the 26 `*scale*` matrices | adopted (Moonlight) | BOTH | 0..+0.0001 | filler only | update anisotropy < 0.5 => dead |
| 12 | Probe the first review (train + rectified eval) | invented | **contract -- Andrew** | -0.0001..+0.0002 vs re-scored base | re-score + 1 run | share of scored rows < 5% => note only |
| 13 | Chunk-continuous training (state carry across a user's chunks) | invented | BOTH | 0..+0.0003 | multi-day | reset-cost curve flat past 16k rows => dead |

**ORDER UNDER STRICT ALTERNATION (next = invented): 2 -> 1 -> (4 if Andrew approves, else 5) -> 6 or 7 -> ...**
**★ ANDREW 2026-09-04 20:35: "Let's skip KD."** => rank 3 (born-again KD) is REMOVED, the teacher retrain
(old queue order 7) is CANCELLED, and the features lineage stays KD-OFF through phase 4; the GPU days go to
the endgame's own 10x run. The next ADOPTED slot after ordcut is therefore rank 6 (SAM, decay-only) or rank 7
(auxiliary next-interval head), screened first.

**★ SCREENS RUN 2026-09-04 19:07 (`scratchpad/proposals_2026-09-04/screen_pass.py`, realcyc checkpoint,
10 train-range users, 218,841 predicted rows):**
* **Rank 2 GO.** Zeroing the current review's duration costs **+0.001388 ahead** by-user mean (8 of 10
  users positive, max +0.003581) vs the +0.0004 kill line -- the constraint is binding on this
  checkpoint, matching iter 31's +0.001451 on the published set. Building it as the INVENTED slot.
* **Rank 1 survives in a REDUCED form.** Among successes at t >= 1 d, the Hard share falls perfectly
  monotonically with the model's own logit R (Spearman -1.000, 0.166 -> 0.016 across deciles), but the
  Easy share is U-SHAPED (0.143 -> 0.053 -> 0.099; rho -0.73 vs the +0.8 wanted): a shared latent
  explains Hard-vs-Good but not Easy. AUC(Good vs Hard) 0.737 -- separated, not solved (kill line 0.75).
  => implement ONE cutpoint (Again < Hard < {Good, Easy}), not two; the Easy cut would distort calibration.
* **Rank 10 alive:** by-user calibration gap on train users -0.00836 (overconfident; kill line 0.001).

* **Rank 7 (auxiliary next-interval head) KILLED 20:50 by its own second rule** (`aux_probe.py`, realcyc,
  6 train users, leave-one-user-out ridge from the curve head's (w, S, d) to log(1+next interval)):
  held-out R^2 0.648 (below the 0.85 "already carried" line -- the head does NOT fully encode the
  scheduler's interval), BUT corr(|residual|, per-row ahead BCE) = **+0.018**, i.e. the interval
  information the trunk lacks is UNRELATED to where ahead errs. An aux task teaches what it is not
  missing. => next adopted slot after ordcut = rank 6 (SAM, decay-only), sharpness screen first.
* **Rank 6 (SAM) GO 21:10** (`sam_probe.py`, realcyc, 12 real training chunks, CPU, fp32, dropout off):
  the SAM ascent L(w + rho g/||g||) - L(w) at rho=0.05 is **median +0.0230, min +0.0097 (chunk 109, 1.4%
  of its L0), max +0.0951**; at rho=0.01 median +0.0036. Every chunk is far above the 0.002 kill line
  and above the L0-relative 0.5% line: the minimum is SHARP at SAM's scale. The lever is ALIVE and is
  the adopted slot after ordcut (decay-only first, LookSAM fallback). Caveat carried: the per-chunk
  gradient includes batch noise, which is also what SAM's own ascent uses.
* **Rank 5 (multi-horizon button ordering) GO 21:47** (`button_probe.py`, realcyc, 4 train users, every
  20th labelled row = 1,478 probes, RAW counterfactual curves): adjacent-button order violations on
  **29.9% of rows at the label's own t** (the same-t rectifier pools these), **32.5% at 1 d, 32.1% at 7 d,
  35.6% at 30 d, 48.8% at 180 d**; the button ORDER differs from its order at label-t on 10.8 / 11.4 /
  16.5 / 34.1% of rows; median |R_Good - R_Hard| at 30 d = 0.070. Far above the 3% kill line: the
  constraint is NOT satisfied off the label horizon. => the INVENTED slot after sam.
**Rank 5 BUILT 22:00** as `RWKV_PAVA_HORIZON_LAMBDA` (+ `RWKV_PAVA_HORIZON_FACTORS`, default 0.125,8): hinge on adjacent-button order of the 4 probe curves at t x factor, probe rows only, curve head only; `scratchpad/hord/smoke_hord.py` 8/8 (positive on real probes, exactly lambda*hinge added, same-t PAVA untouched, zero on coinciding curves, gradient reaches the GRU head, scripts). Runner/PREREG/waiter follow sam's verdict, which decides whether the base recipe carries SAM in its decay.
**CHAIN EXTENDED 21:26: -> sam** (decay-only, `RWKV_SAM_RHO=0.05`; `rwkv/sam.py` unit-tested on CPU with the real model/chunk/loss, 8/8 incl. an independent gradient at w+e matching to 0.00e+00 and a snapshot-based bit-exact restore -- the add/sub restore was NOT bit-exact and the test caught it; `sam/auto_control.py` picks the base; PREREG written).
**CHAIN ARMED 19:20 (all waiters WMI-detached, each gating on the previous log's anchored marker):**
LOO phase 2 (`feat_loo/loo_p2.log`, running) -> **durdrop** (`scratchpad/durdrop/`, PREREG written,
both-modes gate vs realcyc) -> **ordcut** (`scratchpad/ordcut/`, PREREG written, curve-side gate;
`auto_control.py` applies the mechanical gate to durdrop's result and regenerates the runner on
durdrop's recipe if it ACCEPTS, else keeps realcyc -- so the 2-minute hand-off window is no longer a
human race). Both levers implemented behind default-off flags (`RWKV_DUR_DROP`, `RWKV_ORD_LAMBDA`),
smokes PASS, realcyc's checkpoint still loads strictly.
Ranks 1 and 2 share ONE CPU instrument (the deploy-RNN pass on realcyc over ~10 train + ~10 VAL users
recording, per scored row, the curve logit with and without the current duration, y, t, this row's
rating) -- run it once before building either.

## ★ QUEUE STATE 2026-09-02 22:20 -- the features chain, then the ADOPTED slot

| order | run | lever | provenance | control | state |
|---|---|---|---|---|---|
| 1 | gen4base (phase 2) | none -- the features lineage's baseline (gen 4, KD-off) | -- | -- | **DONE 09-03 13:33** -- 0.298089 / 0.263548, n=2,499 (user 6701 excluded: WDDM ceiling); size baseline snapshotted |
| 2 | feat_ablate | inference-time ablation of featB's 23 columns in 4 groups | measurement | featB | **DONE 07:51** -- clock carries the asymmetry (imm +0.0051 / ahead +0.0014), struct symmetric (+0.0009 each), pseudo cycles dead weight (+0.00008 / -0.00001). `research_5k_verbose.md` |
| 3 | rebuild5 (CPU) | gen-5 dbs, `RWKV_REAL_CYCLES=1` | -- | -- | armed on gen4base DECAY_OK + 25 GB RAM |
| 4 | realcyc | real-time cycles replace the 7 pseudo cycles + row 11 | Andrew | gen4base | **DONE 09-04 05:38 -- EXACT TIE** (+0.000007 / -0.000045, both inside the floor; pre-registered P3). Rejected as accuracy; **ADOPTED as the lineage layout by directive: gen 5 is the lineage db, realcyc = the reference** |
| 5 | **lorawd** | **`RWKV_MUON_LORA_WD=0.05`** -- decoupled wd on the LoRA Muon group | **adopted** (Moonlight, arXiv 2502.16982; dose from the 2026-08-18 screen = rank 8) | **realcyc** (regenerated on its gen-5 recipe 05:40) (`mk_lorawd.py realcyc`) | **DONE 09-04 15:53 = iter 62, REJECTED** (ahead -0.000040 / imm -0.000024, both inside the floor; imm rank-significant). P3 held (LoRA norm ratio 0.811, engaged) -- the growth is restoring and harmless; **endgame: no LoRA brake by default**. No 0.2 retry |
| 6 | (invented slot) | calendar-aware curve head. **Ablation reported 2026-09-03: clock reliance imm +0.00507 vs ahead +0.00137, gap 0.0037 -- abort line (0.0005) cleared; the gap is the loose ceiling.** Build after the LOO sweep separates `t_since_any_review` (not a function of t) from the calendar columns | invented | the then-champion | ceiling known; awaiting LOO |
| 5b | **abl_batchpos** | zero ONE column, `scaled_creation_batch_pos_1h`, at featB's input | Andrew 2026-09-03 ("seems useless") | featB | **DONE** -- NOT unused at inference: +0.000217 ahead (p=2e-7) / +0.000089 imm; the LOO sweep ranks it against the rest |
| 5c | **feat_loo** (~8 h) | leave-one-out: 19 arms, one per feature (sin/cos pairs together), 300 users each | Andrew 2026-09-03 ("ideally ablate everything") | featB | **RUNNING since 09-04 15:54** (fired on lorawd's marker); `scratchpad/feat_loo/` |
| 7 | teacher retrain | a d=128 teacher native to the final layout | infrastructure | -- | **CANCELLED 09-04 (Andrew: "Let's skip KD")** -- costed at ~4 GPU days for a useful teacher (`scratchpad/teacher_gen5/TIMING.md`) |

