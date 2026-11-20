import io

PROP = "optimization/PROPOSALS.md"
BOARD = "optimization/GPU_RUN_BOARD.md"

result = """
### 2026-11-15 -- TA3 pre-test RESULT: the mechanism IS present in this data (positive)

Andrew 2026-11-15: *"It seems this approach with adaptive WD isn't fruitful. Let's move on to T3."*
Built `rwkv/ta3.py` (env-gated, inert when unset) + a trainer hook, and ran the measure-only
probe (`RWKV_TA3_PROBE`) on the REAL training batch stream of an adaptive-style attempt, from
the incumbent `adaptive_173416.pth`, scope `cmix` (30,016 params), A/Q split at the batch's time
midpoint, dropout masks pinned so every control is paired, no optimizer step taken
(`scratchpad/ta3_probe_20261115/`, 30 steps).

| eta | mean dL_Q downhill | mean dL_Q random | mean dL_Q uphill | downhill<random | Wilcoxon p |
|---:|---:|---:|---:|---:|---:|
| 0.03 | -0.000309 | -0.000008 | +0.000295 | 27/30 | 1.0e-07 |
| 0.10 | -0.000899 | -0.000008 | +0.001076 | 27/30 | 2.4e-07 |
| 0.30 | -0.002125 | +0.000001 | +0.003768 | 27/30 | 2.6e-06 |
| 1.00 | -0.000630 | +0.000003 | +0.018284 | 20/30 | 1.9e-01 |

- `cos(g_A, g_Q)` over all parameters: mean **+0.39**, positive on 28/30 steps -> the two halves
  are genuinely correlated, which is TA3's premise.
- The equal-norm RANDOM control is ~0 (mean -8e-6..+3e-6, ~50% negative), so the measurement is
  clean and paired; downhill beats both random and uphill at every eta <= 0.3.
- The response is monotone in eta up to 0.3 and turns over at 1.0 (curvature dominates), so the
  useful window is eta ~ 0.1-0.3. At eta=0.3 the per-element displacement is 0.00173, i.e. about
  1.7x the optimizer's own lr (0.001) -- a sane scale, not a knife-edge.
- Evidence boundary: this is a FIRST-ORDER, single-batch measurement with a positive control.
  It says the direction exists; it does not show that adding the TA3 step to training beats the
  plain recipe at equal epochs. Only a training run can.

**Read the probe's controls with care:** an earlier revision of the probe passed the sign in two
places and its "downhill" row was really the ascent (which inverts the whole reading). `_shift`
now takes a descent scale, and that convention is documented in the module.

Implementation note for a full arm: the A/Q split is a LOSSMASK change (`has_label`, column 4),
so the fetched batch stream stays byte-identical between the TA3 arm and its control. Each TA3
step does two half-batch forward/backward passes plus a shift/restore, and the gradient the
optimizer sees is (n_A*g_A + n_Q*g_Q)/(n_A+n_Q), which is the gradient of the same full-batch row
mean. A TA3 failure aborts the run (exit 44) instead of degrading into a plain step.
"""

text = io.open(PROP, encoding="utf-8").read()
if "TA3 pre-test RESULT" in text:
    raise SystemExit("PROPOSALS.md already has the pre-test result")
io.open(PROP, "w", encoding="utf-8", newline="\n").write(text.rstrip("\n") + "\n" + result)
print("PROPOSALS_APPENDED")

lines = io.open(BOARD, encoding="utf-8").read().split("\n")
row = (
    u"| 10 | TA3 (Training-Time Testing) arm | **Pre-test passed; 0.04-epoch smoke running** | "
    u"`rwkv/ta3.py` + trainer hook, env-gated (`RWKV_TA3=1`, inert and byte-identical when unset). "
    u"Each optimizer step splits the batch into A and Q by time position (a `has_label` loss-mask change, "
    u"so the data stream is unchanged), takes a normalized DOWNHILL trial step on A (RWKV_TA3_ETA=0.3, "
    u"scope `cmix`), takes Q's gradient at the perturbed weights, restores the weights exactly, and steps "
    u"on (n_A*g_A + n_Q*g_Q)/(n_A+n_Q). Measure-only probe on 30 real batches: downhill on A lowers Q's loss "
    u"by -0.00031/-0.00090/-0.00213 at eta 0.03/0.1/0.3, beating an equal-norm random control 27/30 "
    u"(Wilcoxon p 1e-07..2.6e-06) and the uphill direction at every eta; random is ~0; cos(g_A,g_Q) +0.39. | "
    u"Probe: `scratchpad/ta3_probe_20261115/` (30 steps, no optimizer step, dropout masks pinned). "
    u"Smoke: `scratchpad/ta3_smoke_20261115/` (`EPOCHS=0.04`, `RWKV_TA3=1`, real path, isolated tag). "
    u"The probe is a first-order mechanism check, not accuracy evidence; a training arm at matched epochs "
    u"against a matched plain control is what decides. Full-arm budget needs Andrew's decision."
)
lines.append(row)
io.open(BOARD, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("BOARD_ROW10_ADDED")
