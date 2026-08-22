# Workload efficiency by replay — FSRS-7 vs RWKV-Curve

**Andrew's idea, 2026-08-21.** Estimate the relative efficiency of two forgetting-curve
algorithms *without a simulator*. We cannot simulate RWKV realistically — it consumes review
duration, time of day, per-day review counts, deck/preset context — and faking a joint
distribution over all of that would be a bigger and less trustworthy piece of work than the
thing it is meant to measure. So instead of simulating a schedule, **replay the real one** and
ask a counterfactual at every step.

> Replay a user's real review history in chronological order. After **every** review, ask each
> algorithm: *what interval would you assign to this card right now, at desired retention DR?*
> A card on an interval of `L` days is seen `1/L` times per day, so the collection's workload
> on calendar day `D` is `W(D) = sum over active cards of 1/L`. The efficiency estimate is
> `W_FSRS(D) / W_RWKV(D)`, averaged over days and users, repeated for each DR level.

DR levels: **99, 95, 90, 85, 80, 75, 70 %**.

Everything lives in `scratchpad/workload/`.

---

## 1. What is actually run

| step | script | cost |
|---|---|---|
| canonical replay table (one row set for both arms) | `build_table.py` | seconds |
| FSRS-7 arm — replay + interval inversion | `fsrs_arm.py` (srs-benchmark venv) | seconds |
| RWKV-Curve arm — replay + interval inversion | `rwkv_arm.py` | **20.2 reviews/s, 1 thread** |
| daily workload + ratio | `combine.py` | seconds |
| the report (3 tables) | `analyze.py` | seconds |
| orchestration, 2 workers | `run_pipeline.py`, `run_phase1.cmd` | — |

**Zero GPU.** Both arms are CPU-only and single-threaded; the job costs exactly as many CPU
threads as `run_pipeline.py` has workers (2).

### Users

**5001–7500 only** — the VAL half. The RWKV champion trained on 1–5000, so its intervals there
would be fitted rather than predicted, while FSRS-7's per-user parameters are fitted for every
user by construction. Restricting to held-out users is what stops the comparison being rigged.

Stratified by collection size, because size spans three orders of magnitude (5th percentile
6.0k reviews, 99th 520k) and neither a uniform nor a review-weighted sample represents it.

- **Phase 1** — 24 users, 416k reviews, bands 5–10k / 10–20k / 20–40k. Ran in 3.11 h.
- **Phase 3** — 40 users, 484k reviews, bands 5–12k / 12–25k. Ran in 3.64 h.
- **Phase 2** (12 users, 1.28M reviews, 40k–320k) was **defined and deliberately not run.**
  Phase 1 measured the ratio's size-dependence as flat (Spearman rho −0.05…−0.31), so twelve
  giant users would have cost ~9 h to add twelve data points, while the same wall clock bought
  forty small ones. Statistical power per CPU-hour is the scarce resource here.

Total: **65 users, 906k reviews, 6.75 h on 2 threads.** `select_users.py` refuses to re-pick a
user an earlier phase already replayed, so the phases compose without silently shrinking.

### Parameters: FSRS-7 is *not* re-optimized

srs-benchmark already stores the per-user optimized 34-parameter vector for all 10,000 users
(`result/FSRS-7-short-secs.jsonl`, written 2026-06-26). Re-running the optimizer would
reproduce it at enormous cost. The vector used is copied into each output's `.meta.json`, so a
re-benchmark overwriting the source file cannot silently change what was measured.

⚠ The **non-equalized** file is the right one. The `-equalize_test_with_non_secs` variant's
parameters genuinely differ (median max-abs difference **0.094** over 200 users, max 1.28) —
equalization changes which rows *train*, not only which rows score — and the canonical replay
table is the full raw stream, which is what the plain `--short --secs` pipeline fits.

The `sched_penalties` variant is run as a second FSRS arm. Its two penalties exist to stop
FSRS-7 asking for sub-second intervals at high DR, which is *precisely* the regime this study
measures, and its accuracy is within noise of the plain variant (README: RMSE 0.3438 vs
0.3437). It is arguably the fairer scheduler-side arm.

---

## 2. Three decisions that the number depends on

### 2.1 One row set, asserted

FSRS and RWKV have different preprocessing pipelines that keep different rows. If each arm
replayed its own, the ratio would compare two different card populations and mean nothing. One
canonical table is built (from `get_rwkv_data`, the unfiltered raw stream) and both arms replay
it; `rwkv_arm.py` re-derives the frame and **asserts** it matches column by column, and
`combine.py` asserts the two arms see identical active-card counts on every day.

### 2.2 The 1-day floor is mandatory, not cosmetic

`sum(1/interval)` is dominated by whatever produces the shortest intervals. Unfloored, user
5100's workload comes out at **4.4 million reviews/day** — the 1-second clamp, not a quantity.
A scheduler inside Anki cannot act on a sub-day interval for a review card anyway. So the
headline floors intervals at 1 day; the unfloored version is reported beside it, never instead
of it.

The floor is doing real work at the top of the DR range: at DR=99 % FSRS-7 wants a sub-second
interval on **77 %** of user 5100's rows (RWKV: 24 %). This is the known FSRS-7 pathology the
`sched_penalties` variant exists to fix. **Read the 99 % and 95 % rows with suspicion.**

### 2.3 Only the REVIEW queue counts — and this one is not optional

The first run of the metric measured the *learning* queue, not the algorithms. `sum(1/L)`
weights short intervals enormously, and **64–98 % of the unfiltered workload came from
intervals the 1-day floor had to rewrite — at every DR level**, both arms. Those windows are
same-day learning and relearning steps, which Anki drives from fixed learning steps, not from
FSRS or RWKV. Including them makes the ratio a measurement of the floor: the DR=99 % row came
out at 0.9997, which is not a finding about either algorithm.

So the default counts a scheduling decision only if the card sat in Anki's **review queue**
during the window it is being charged for — row *j* counts iff the card's next review has
`state == 2`. (`state` is the state *before* a review — 0 new, 1 learning, 2 review,
3 relearning, 4 filtered — so the next review's state is the state during window *j*.) The
mask is derived from the shared review table, never from either arm's intervals, so both arms
are always restricted identically.

Effect on user 5100, floored share of workload:

| DR | all rows: FSRS / RWKV | review queue: FSRS / RWKV |
|---|---|---|
| 99 % | 96.7 / 97.9 % | 95.3 / 97.1 % |
| 90 % | 75.2 / 76.9 % | **38.7 / 47.9 %** |
| 80 % | 67.1 / 79.6 % | **26.8 / 23.8 %** |
| 70 % | 64.1 / 83.3 % | **30.3 / 15.1 %** |

⚠ **DR = 99 % stays floor-dominated even in the review queue (95–97 %), and 95 % is still
67–83 %.** Those two rows are not usable as efficiency numbers; they are reported for
completeness and should be read as "both algorithms want sub-day intervals here".
`--queue all` reproduces the unfiltered version as a sensitivity check.

### 2.4 A card is active *between* two observed reviews

Default (`alive`): review *j* of card *c* contributes `1/L_j` to every day from its own day up
to the day of that card's next review. A card's last observed review ends its life. The
alternative (`persist`, reported as a sensitivity check) keeps abandoned cards contributing
forever, which inflates both arms and lets years-dead cards dominate.

`alive` has a clean property: if an algorithm's interval equalled the real gap, each review
would contribute exactly 1, so a card's lifetime contribution is its real review count.

---

## 3. Both replays are validated against the benchmark

Neither arm reimplements its model. The FSRS arm imports `models.fsrs_v7.FSRS7` from
srs-benchmark and calls its own `step()` / `forgetting_curve()`; the RWKV arm uses
`SrsRWKVRnn.button_heads` / `button_curves`, the parity-verified deploy path.

### FSRS-7: exact, per row (`validate_fsrs.py`)

A LogLoss comparison against the recorded per-user number is *not* a real check — the row sets
differ, so a genuine implementation error can hide behind "different rows". (It nearly did:
delta was −0.0006 on user 5100 but **+0.029** on user 5530.)

So the row set was removed from the comparison instead: run srs-benchmark's own
`create_features` + `Collection.batch_predict` on a user, run this replay, and compare
**per-row retention**.

| user | rows compared | max abs diff | mean abs diff |
|---|---|---|---|
| 5100 | 4,905 | **1.9e-07** | 3.7e-08 |
| 5530 | 24,355 | **2.3e-07** | 4.1e-08 |

The arm's recurrence and curve *are* the benchmark's. The earlier LogLoss deltas were purely a
row-set effect.

⚠ **The first version of that validation reported a mismatch (mean 0.039) and was itself
wrong.** srs-benchmark builds its history prefixes *before* `_common_postprocessing` drops the
`delta_t == 0` rows, so a scored row's `tensor` still contains reviews that are absent from the
frame — user 5100's `review_th` 16 is a card's *second* review, is labelled `i=1`, and carries
`tensor [[0., 1.]]`, the dropped first one. Replaying only the surviving rows gives a different
state. Replaying the full raw stream is what reproduces the benchmark exactly.

### RWKV-Curve

Same rows as FSRS-7 above, user 5100: replay LogLoss **0.4002** vs FSRS-7's **0.4078** — RWKV
better by 0.0076, the direction and rough size the leaderboard implies. On the benchmark's own
equalized subset (4,085 rows, identical `size` in both result files) the recorded values are
**0.3909** vs **0.4084**.

**Interval inversion accuracy** (`probe_inversion.py`). The RWKV arm reads `t(DR)` off a
441-point log-*t* grid by interpolation rather than bisecting seven times on a curve that costs
four forward passes. Round-tripping — evaluate the exact rectified curve at the returned `t*` —
gives `|R(t*) − DR|` of median **2e-06**, max **1.8e-05**, i.e. ~2,700× below the 0.05 spacing
between DR levels. Negligible.

### The deploy contract is followed, and it is load-bearing

Per CLAUDE.md's deploy contract: duration of the most recent review zeroed, PAVA applied, no
piecewise ahead correction. That costs **five** forward passes per review (four counterfactual
buttons, because PAVA pools *across* buttons, plus one to advance the state with the real
duration). Both shortcuts were measured rather than assumed (`probe_contract.py`, user 5100,
400 reviews, log-ratio vs the full contract):

| shortcut | speed | median | p90 abs | rows differing >5 % |
|---|---|---|---|---|
| no PAVA (2 fwd) | 2.5× | 0.0000 | 0.0000 | **2–3 %** |
| no PAVA + real duration (1 fwd) | 5× | **+0.07 … +0.08** | ~0.9 | **70–78 %** |

PAVA is the identity on ~97 % of rows but rewrites the rest by up to 4×; and feeding the real
duration lengthens intervals by ~8 % at the median, which is a genuine information leak (at
scheduling time Anki must show all four button intervals *before* the user presses, so the
duration of the press is not yet observable). Both shortcuts were rejected; the full contract is
what ran.

Training-probe rows and the deploy button rows were checked to be the same construction —
`insert_probes` copies the target row and changes only the grade one-hot and the imputed
duration, exactly as `button_heads` does — so this is not a train/deploy divergence.

---

## 4. The caveat that decides how the result reads

**A more accurate model does not automatically mean less work.** An *overconfident* model asks
for longer intervals, so it wins on workload while quietly under-delivering the retention it
promised. Comparing at equal *nominal* DR is therefore only an efficiency comparison if both
models deliver what they claim.

So the report has three tables, and only the third is an efficiency claim:

1. **Nominal** — workload ratio at the same stated DR. What was asked for.
2. **Calibration** — each arm's own scheduling curve evaluated at the interval that actually
   happened, against what actually happened. Decides whether table 1 can be read as efficiency.
3. **Matched** — workload at equal *realized* retention: each arm's nominal DR axis is mapped
   through its own empirical calibration curve, so "80 % realized" is looked up on each side at
   whatever nominal DR that arm needs in order to actually achieve 80 %.

Plus two diagnostics that turn out to matter more than expected:

**The absolute check.** Neither LogLoss nor a ratio can tell you whether the *inverted*
intervals are sane in absolute terms — inverting a curve to a fixed DR is an extrapolation, and
LogLoss only scores the horizons that happened. But the user's own review load is a fact. So
each arm's workload is compared against `W_actual`, computed the same way from the gaps the user
really used. On user 5100 (actual load 3.22 review-queue reviews/day, realized retention 0.790):
FSRS reproduces the real load at DR ≈ 75 %, RWKV at DR ≈ 72 %. Both land in a sane range —
neither is off by an order of magnitude — so the comparison is measuring something real.

**Calibration by horizon.** If a curve decays too fast with *t*, its bias goes increasingly
negative as the horizon grows: it under-predicts recall exactly where it must extrapolate, which
is what the inversion depends on. This is the diagnostic that would distinguish "RWKV correctly
believes memory is weaker" from "RWKV's curve shape is wrong away from the observed horizons" —
a distinction LogLoss cannot make, because both models are only ever scored at one horizon per
row and nothing in either training objective constrains the curve's *shape* across *t*.

### Other caveats, stated rather than buried

- **Neither model is trained to produce intervals.** Both are trained to predict recall at the
  horizons that actually occurred, which follow the user's real schedule. Inverting the curve to
  a fixed DR is an *extrapolation* for both, and extrapolation quality is not what LogLoss
  measures. This is the deepest limitation of the method and it applies symmetrically.
- **FSRS-7 gets its per-user parameters fitted on the user's whole history**, including the
  future relative to any replay day. That is generous to FSRS and is the conservative direction
  for a comparison against a frozen net.
- **Window lengths come from the real schedule**, so day-weighting inherits whatever algorithm
  the user actually used. Identical for both arms, so it cancels in the ratio.
- **RWKV's ID encodings are random draws** (seeded at 1234). The model must be robust to them;
  the seed sensitivity is a cheap robustness check, not yet run.
- Users are real, including degenerate ones: user 5001 has a 0.1 % Again rate, and FSRS-7 fitted
  to it returns the 100-year ceiling at 4 of 7 DR levels. Nothing is excluded on that basis;
  per-user ratios are reported so outliers stay visible, and the headline is the median.

---

## 5. Results — n = 65 users, 906k reviews

### The headline

**At the same nominal desired retention, RWKV-Curve asks for MORE reviews/day than FSRS-7 —
about 1.2–1.3× at the median, more on the geometric mean — despite predicting recall better on
the identical rows (LogLoss 0.2900 vs 0.2949).**

Per-user ratio = that user's total FSRS reviews ÷ total RWKV reviews over their whole history.

| DR | median | geomean | p25…p75 | frac > 1 | sign-test p |
|---|---|---|---|---|---|
| 99 % | 1.004 | 0.894 | 0.76…1.10 | 0.55 | 0.382 |
| 95 % | 0.884 | 0.754 | 0.60…1.07 | 0.35 | **0.025** |
| 90 % | 0.786 | 0.733 | 0.63…1.12 | 0.35 | **0.025** |
| 85 % | 0.833 | 0.660 | 0.49…1.17 | 0.38 | 0.082 |
| 80 % | 0.803 | 0.581 | 0.44…1.13 | 0.38 | 0.082 |
| 75 % | 0.854 | 0.537 | 0.33…1.08 | 0.32 | **0.006** |
| 70 % | 0.828 | 0.488 | 0.28…1.04 | 0.32 | **0.006** |

Every DR below 99 % points the same way; four levels clear p < 0.05. The per-user spread stays
wide (p25…p75 roughly 0.3…1.1), so this is a **direction with a rough magnitude**, not a precise
number — which is all a replay-based counterfactual can honestly give.

DR = 99 % is the floor's own artefact (83–90 % of its workload is intervals clipped to one day)
and should not be read as a result.

Robust to every variant tried:

| variant | median @ 90 % | median @ 80 % | median @ 70 % |
|---|---|---|---|
| headline (review queue, 1-day floor) | 0.786 | 0.803 | 0.828 |
| `sched_penalties` FSRS parameters (n=64) | 0.871 | 0.832 | 0.876 |
| `--queue all` (learning steps included) | 0.842 | 0.945 | 0.848 |
| no floor | 0.527 | 0.644 | 0.555 |
| `--mode persist` | 0.786 | 0.803 | 0.828 |

(`persist` is *identical* to `alive` here, and necessarily so: the queue mask already drops each
card's last review, which is the only row the two activity definitions treat differently.)

Size-dependence is **flat** (Spearman rho −0.18…−0.34), which is what made phase 3's forty small
users a better buy than phase 2's twelve giant ones.

### Why — and the n = 25 explanation was WRONG

The 25-user cut showed RWKV under-predicting recall by 0.55 pp overall and I attributed the
workload gap to that. **At n = 65 the overall bias is essentially symmetric and that explanation
collapses:**

| arm | LogLoss | mean predicted | mean observed | bias |
|---|---|---|---|---|
| FSRS-7 | 0.2949 | 0.8602 | 0.8587 | **+0.0015** |
| RWKV-Curve | **0.2900** | 0.8570 | 0.8587 | **−0.0017** |

Both are well calibrated *on average*. The matched-retention correction consequently moves
almost nothing — to realize 90 %, FSRS needs a nominal 0.9005 and RWKV 0.9045 — and the ratio
stays put (median 0.732 at 90 % realized, p = 0.006). **So the workload gap is not a calibration
gap.**

It is a **shape** gap. Bias (mean predicted − mean observed) by actual horizon:

| actual interval | n | observed | FSRS bias | RWKV bias |
|---|---|---|---|---|
| < 1 d | 246,332 | 0.8297 | +0.0037 | +0.0032 |
| 1–3 d | 121,166 | 0.8614 | −0.0020 | −0.0056 |
| 3–7 d | 117,413 | 0.8897 | −0.0050 | −0.0065 |
| 7–21 d | 115,893 | 0.8992 | −0.0098 | **−0.0158** |
| 21–60 d | 65,766 | 0.8685 | **+0.0172** | +0.0049 |
| 60–180 d | 32,314 | 0.8180 | +0.0246 | +0.0238 |
| > 180 d | 16,900 | 0.8079 | +0.0134 | +0.0110 |

Across the whole 1–60 day band — where scheduling actually happens — **FSRS-7's curve sits above
RWKV-Curve's relative to truth**: more negative for RWKV at 1–21 d, far more positive for FSRS at
21–60 d. A higher curve at fixed DR is a longer interval, and a longer interval is less work.
The two errors cancel in the global mean, which is why the aggregate calibration looks clean for
both.

⚠ **And part of FSRS's advantage is over-optimism, not efficiency.** At 21–60 days FSRS
over-predicts recall by **+1.7 pp** against RWKV's +0.5 pp: its longer intervals there
under-deliver the retention they promise. The matched-retention table cannot correct for this,
because it maps a *global* calibration curve keyed on predicted probability, while the
miscalibration is horizon-dependent. A horizon-conditional correction is the obvious next
refinement and is not implemented.

**The structural point stands, and it is the research finding.** Neither model is trained to
produce intervals. Both are scored at exactly **one horizon per row** — the interval that
actually happened — and **nothing in either objective constrains the curve's shape across t**.
PAVA constrains ordering across *buttons*, not decay across *time*. FSRS-7 gets shape for free
from a rigid 8-parameter form; RWKV-Curve's learned 3-component mixture can fit every observed
horizon well and still be wrong *between* them, which is exactly where interval inversion lives.

**LogLoss cannot see this. A model can win the benchmark and lose the scheduler.**

### The absolute check

Neither LogLoss nor a ratio says whether the inverted intervals are sane in absolute terms. The
user's own review load is a fact, so: median over users of W_model ÷ W_actual (mean actual
review-queue load 13.0 reviews/day).

| DR | FSRS / actual | RWKV / actual |
|---|---|---|
| 90 % | 2.26 | 2.60 |
| 85 % | 1.28 | 1.25 |
| 80 % | **0.91** | **0.85** |
| 75 % | 0.42 | 0.55 |

Both land in a sane range: users' real behaviour sits near DR ≈ 82–83 % on both arms, which is
where real Anki users actually are. Neither model's inversion is off by an order of magnitude,
so the comparison is measuring something real.

⚠ The per-user spread behind those medians is nonetheless enormous — individual users run from
0.02× to 37× their actual load. Most of it is genuine (users study at wildly different retention
targets, and a card whose real gap is 150 days while the model wants 4 days legitimately
contributes 37×), but it is the dominant source of the wide ratio quartiles, and it is why the
median rather than the mean is the headline.

### Single-user preview (user 5100, 5,842 reviews) — NOT a result, a pipeline check

| DR | nominal ratio | realized FSRS | realized RWKV | matched ratio |
|---|---|---|---|---|
| 95 % | 0.868 | 0.943 | 0.963 | 1.054 |
| 90 % | 0.849 | 0.897 | 0.886 | 0.823 |
| 85 % | 0.865 | 0.862 | 0.886 | 0.855 |
| 80 % | 0.824 | 0.802 | 0.806 | 0.813 |
| 75 % | 0.790 | 0.748 | 0.716 | 0.726 |
| 70 % | 0.752 | 0.702 | 0.736 | 0.695 |

Calibration on the same 4,905 rows: FSRS bias **−0.0015**, RWKV bias **−0.0083** (both
*under*-confident; RWKV more so, which by itself makes RWKV ask for shorter intervals than it
needs to). One user's calibration curve is too noisy to invert reliably — note it is not
monotone above 90 % — which is exactly why the real table pools 24 users.

---

## 6. What this changes, and what it does not

**It does not change any gate.** Nothing here is a research-phase accept/reject: the acceptance
gate is LogLoss on the VAL half, RWKV-Curve still wins it, and no champion moves.

**It does add a target the LogLoss gate is blind to.** The curve's shape across `t` is
unconstrained by the training objective, and it is the only thing interval inversion depends on.
Concretely, in rough priority order:

1. **Multi-horizon supervision on the curve head.** The probe machinery already exists —
   `insert_probes` builds counterfactual rows and keeps the target's `label_elapsed_seconds`.
   Probing the same row at *several* horizons, with the pooled-BCE target it already uses, would
   put gradient on the curve's decay rate rather than only on its value at one point. This is the
   natural first attempt and it reuses code that is already parity-verified.
2. **A shape prior.** FSRS-7's advantage here comes from rigidity, not from information; an
   explicit monotone-decay-rate constraint on the GRU mixture would buy some of that without
   giving up the learned form.
3. **Report interval-space error alongside LogLoss** for any future curve-side iteration, so a
   change that improves prediction while degrading scheduling is visible when it happens rather
   than a year later.

⚠ **Do not read this as "FSRS-7 schedules better".** Part of FSRS's lower workload is genuine
and part is over-optimism at 21–60 days (+1.7 pp) that the global matched-retention correction
cannot remove. Separating those needs a horizon-conditional retention match, which is the honest
next measurement and is not done.

## 7. Reproducing

```
.venv/Scripts/python.exe scratchpad/workload/select_users.py 1
powershell -NoProfile -File scratchpad/detach.ps1 -Script <abs path>/scratchpad/workload/run_phase1.cmd
.venv/Scripts/python.exe scratchpad/workload/analyze.py --queue review --json-out report.json
```

Validation harnesses, each cheap and each worth re-running after any change to either arm:

| script | what it proves | cost |
|---|---|---|
| `validate_fsrs.py <uids>` | the FSRS arm IS srs-benchmark's model, per row | seconds |
| `probe_inversion.py` | grid inversion lands at the requested retention | ~1 min |
| `probe_contract.py` | what the two deploy-contract shortcuts would cost | ~1 min |
| `probe_speed.py` | per-review cost breakdown of the RWKV path | ~30 s |

Per-user outputs and the parameter vectors actually used are in `scratchpad/workload/out/`
(`*.meta.json`), so the run stays reproducible even if srs-benchmark's result files are
regenerated.
