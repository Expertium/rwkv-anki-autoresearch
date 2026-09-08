# Pre-registration: what arm 1 (`w10plain`) means, written BEFORE the number exists

Arm 1 is not a gated candidate. It is the endgame's plain 10+2 number, the control every decay
branch is measured against, and the test of three separate claims the record has been carrying on
credit. Writing the readings down first is the point: two of the three comparisons available are
NOT clean, and it is easy to pick the flattering one after the fact.

Recorded 2026-09-09, with ws10 at step 59,638 of 109,350 and no number in existence.

## The reference numbers

| model | params | budget | basis | ahead | imm |
|---|---|---|---|---|---|
| `realcyc` | 563,652 | 1 + 1 ep | **gen-5** (`label_filter_db_id_e2s`) | 0.298083 | 0.263592 |
| old d=128 `base5k` | 2,762,884 | ~12 ep (upstream) | **published** (`label_filter_db`) | 0.294623 | 0.263586 |
| stop criterion (Andrew 2026-09-02) | -- | -- | set before gen 5 existed | <= 0.2950 | <= 0.2640 |

Both reference rows are on the same 2,499 VAL users. `w10plain` will be on the gen-5 basis, so
**only the realcyc row is a clean comparison.**

## Q1 -- is the budget premise real? (CLEAN: same model, same basis, same recipe)

`w10plain` vs `realcyc` differs in exactly one thing: 1 + 1 epochs becomes 10 + 2. The 2026-08-11
calibration measured a 3x-budget step at +0.002 and projected **+0.0042 at 10x**.

* **>= +0.0035 ahead** -- the premise holds and phase 6 earned its ~4 days.
* **+0.0015 .. +0.0035** -- real but well under projection; the log-linear extrapolation from a
  3x step overstated a 10x one, which is the ordinary way such extrapolations fail.
* **< +0.0015** -- the premise is refuted at this budget. Then the remaining ahead error is
  neither architecture (three screens), nor out-of-family structure (FSRS-7 adds nothing at a
  fitted blend weight of 0.02), nor budget -- and phase 4's stop criterion is unreachable by any
  lever now known. That is a genuine fork and it is Andrew's.

⚠ **A flat WS validation is NOT a prediction of this.** ws10's stable-phase ahead validation moved
0.3237 (step 10k) -> 0.3230 (44k), i.e. -0.0007 over three epochs. Under WSD that is expected --
at constant peak LR the loss sits at an LR noise floor and the gain is converted by the DECAY.
Do not read the flat stable phase as an early answer to Q1 in either direction.

## Q2 -- does capacity bind at the real budget? (INFORMATIVE, NOT CLEAN)

Every capacity reject in the record -- "capacity-at-5k 0/3", "the d=32 model is DATA-limited, not
capacity-limited" -- was measured at ~1.25 epochs, a budget at which the model demonstrably could
not use more capacity. ws10's own gap screen says the model is still FIT-limited at 4 epochs. So
capacity is **unresolved at the real budget**, and arm 1 is the first evidence either way:
563k @ 12 ep against 2.76M @ ~12 ep.

**This comparison is NOT clean and cannot be made clean.** The d=128 model takes 92-dim published
features and structurally cannot forward the gen-5 109-dim layout (measured: the `teacher_114`
screen, where re-laying-out its input projection cost +0.020 ahead). So it can never be scored on
the gen-5 basis, and the two numbers differ in basis as well as in model and budget.

**Size of the basis effect, estimated, not measured.** The gen-5 basis removes ~0.19% of rows that
are 1.46x EASIER than average (7.07% vs 10.31% failure), which raises the mean by itself. Here the
scored totals are 126,657,015 (gen-5) against 126,874,015 (published), -0.171%, and 2,158 of 2,499
users differ. Treating it as a pure removal gives `f * (L - L_easy) / (1 - f)` ~= **+0.00008**, so
the basis is worth ~0.0001, not ~0.003. ⚠ It is NOT a pure removal -- TimeSeriesSplit boundaries
move so rows also ENTER (measured -1437 / +87 on user 5402) -- so treat this as an order of
magnitude, not a correction to subtract.

* **arm 1 ahead <= ~0.2950** -- 563k matches 2.76M at a matched budget; capacity does not bind and
  "stop making the model smaller" cost nothing.
* **arm 1 ahead >= ~0.2965** -- the ~0.003 gap to the d=128 model survives 10x the budget, and
  capacity becomes the live suspect for the first time with evidence behind it. That would be a
  decision for Andrew, since he fixed the size for this run.

## Q3 -- the stop criterion, and a gap in it that is Andrew's to close

The criterion (ahead <= 0.2950, imm <= 0.2640) was set 2026-09-02, **before gen 5 existed**, and
this repo's own rule says to read the target on the current lineage's numbers rather than carrying
an absolute across a basis change. **The target was never re-derived for the gen-5 basis.** By the
estimate above the shift is small (~0.0001), so the criterion is very nearly transferable -- but
"very nearly" is a judgement, not a measurement, and the honest move is to say so rather than to
quietly apply the old number.

Report arm 1 against the criterion, and flag explicitly that the criterion's basis predates the
lineage it is being applied to.

## What I will report

Both modes' by-user means over the 2,499 users, `size` checked against
`optimization/size_baseline_id_e2s.json` (a mismatch is a pipeline bug, not a result), nan_users,
the paired Wilcoxon against realcyc, and each of Q1/Q2/Q3 answered against the lines above --
including the ones that go against the plan.
