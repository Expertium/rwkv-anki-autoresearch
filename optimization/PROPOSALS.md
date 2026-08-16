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
