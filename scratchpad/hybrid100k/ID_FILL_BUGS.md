# The NaN-metadata id fill: a precision bug, a train/deploy divergence, and a blind guard

Found 2026-08-26 while auditing the pipeline at Andrew's request. All three are the same family
as Bug A/B (2026-08-21) and all three are still live in code. Probes:
`scratchpad/hybrid100k/placeholder_precision.py`.

## 1. `ID_PLACEHOLDER + card_id` is computed in float64 and loses its uniqueness

`data_processing.py:286` fills a NaN `note_id` with `ID_PLACEHOLDER + card_id`, and its own
comment says this is "precisely so each such card gets a UNIQUE placeholder". But
`df["note_id"]` **contains NaN at that moment**, so pandas holds it as float64 and the addition
happens there. `ID_PLACEHOLDER = 314159265358979323 ≈ 3.14e17`, far past float64's exact-integer
limit of 2^53 ≈ 9.0e15. **float64 spacing at that magnitude is 64**, so any two card_ids closer
than 64 collapse onto the same placeholder — *before* `create_sample`'s int64 cast ever runs.

**=> the 2026-08-21 int32 -> int64 fix widened the DESTINATION while the VALUE had already been
destroyed upstream.** It is a real improvement (all-collapse-to-one became 64-wide buckets) but
it does not achieve the intent.

Measured over 6 users / 49,186 cards per regime:

| id regime | distinct placeholders intended | via float64 | identity lost |
|---|---|---|---|
| PUBLISHED (factorized small ints) | 49,186 | 812 | **98.3%** |
| `-id` (raw epoch-ms) | 49,186 | 30,869 | **37.2%** |

The published set is worse because factorized ids are dense: every card in a 64-wide block
shares a note. The `-id` set fails wherever cards were created within 64 ms of each other, i.e.
exactly during a bulk add or an import.

Proven by EXECUTING the real path, not by reasoning about it — four cards with epoch-ms ids
1 / 19 / 119 ms apart produce three distinct placeholders instead of four, and `note_id` is
float64 both before and after the assignment.

**Scope:** only cards with a NaN `note_id`, which is **19.28% of all reviews** in the published
set (per-user median 4.1%, p90 70.0%).

## 2. TRAIN vs DEPLOY use DIFFERENT FILL RULES — the §9 divergence, again

| path | rule | result for NaN-note cards |
|---|---|---|
| TRAINING (`data_processing.py:286`) | `ID_PLACEHOLDER + card_id` | one note entity **per card** |
| DEPLOY (`run_as_rnn.py:255-263`, `add_id`) | `ID_PLACEHOLDER` (a **constant**) | **all** such cards share ONE note |

`run_as_rnn`'s `add_id` applies the same constant to `note_id`, `deck_id` and `preset_id` alike.
Deploy then keys its state dict on that value (`self.note_states[note_id]`, :152), so every
NaN-note card in the collection shares a single note stream.

This is Bug A's shape with the direction reversed. Before the int64 fix, training also collapsed
them (to one), so the two paths were accidentally closer. **The int64 fix made them diverge.**

**Which side is right: TRAINING.** A NaN note_id means the metadata is missing, and the
no-information default is to give the card its own note — not to pool thousands of unrelated
cards into one stream, which is precisely the damage Bug A did. The code comment already states
this intent. So the fix is to make DEPLOY compute `ID_PLACEHOLDER + card_id` for `note_id`,
and to do the arithmetic in int64 on both sides.

⚠ `deck_id` / `preset_id` take a bare constant on BOTH sides, so they agree and have no
precision issue (a constant rounds to one float64 value consistently). Only `note_id` differs.

## 3. Why the guard did not catch it: it SIMULATES deploy instead of RUNNING it

`scratchpad/parity3/smoke_id_identity.py` was written on 2026-08-21 for exactly this class, and
it does compare the real partition rather than entity counts. But its model of deploy is
`df[name]` — the frame **after** the training-side fill (`smoke_id_identity.py:84`,
"DEPLOY's view: run_as_rnn iterates the frame and keys on the raw value").

So both sides of its comparison inherit the SAME fill, and the comparison is structurally
incapable of seeing a fill-RULE difference. It caught Bug A because that was a *storage*
truncation downstream of the shared fill; it is blind to a divergence introduced by the fill
itself.

**The lesson, and it is the reusable one:** a parity guard that MODELS the other path can only
test what the model already assumes is shared. To catch a divergence in the rule, the guard must
EXECUTE both paths. Same family as the rgate smoke whose control inherited the treatment from
`os.environ` — a control that is derived from the thing under test is not a control.

## What to do

1. **Do the fill in int64 on both sides**, e.g. build the column as an int64 numpy array and
   assign it whole, rather than writing into a float64 column. A smaller constant would also
   work arithmetically but changes entity identity, so it is not free either.
2. **Make `run_as_rnn.add_id` use `ID_PLACEHOLDER + card_id` for `note_id`**, matching training.
3. **Fix the guard to execute `run_as_rnn`'s fill** rather than reading the training-filled frame,
   and add a case with cards created within 64 ms of each other.

⚠ **TIMING: this gates the published-db rebuild decision.** The champion trained on
`train_db_5k_h1`, which predates the int64 fix and therefore carries Bug A. If those dbs are
rebuilt with ONLY the int64 fix, they will still lose 98.3% of the intended note identity on
19.28% of reviews. **Both fixes must land before any rebuild**, or the rebuild banks a smaller
version of the same bug and re-bases the champion for nothing.
