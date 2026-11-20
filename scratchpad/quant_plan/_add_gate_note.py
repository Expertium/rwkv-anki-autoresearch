"""Insert Andrew's 2026-11-20 22:23 gate definition into RESULTS.md (replaces the anchor-only framing)."""
from pathlib import Path

P = Path(__file__).resolve().parent / "RESULTS.md"
t = P.read_text(encoding="utf-8")

anchor = "are paired per user against the baseline row on the same 750 users."
assert t.count(anchor) == 1, t.count(anchor)

add = """

**Gate (Andrew, 2026-11-20 22:23): a result is judged against "whatever run that is exactly identical
EXCEPT for quantization"** - the fp32 control trained with the same recipe, data and step budget
(`qnj_it36ctl` for every 124-step point, `qnj_it36ctl496` for the 496-step point). The anchor-champion
comparison stays in the tables for context only: it mixes in a different training run and different
data, so its 0.0085 "win" must NOT be quoted as the quantization cost. Andrew's priority order for
further improvement: **(1) smaller state, (2) lower log loss against the identical unquantized run,
(3) speed** (10% slower is acceptable).

**Best point at the time of writing:** `l1i8` (rank-1 int8 on both streams), 1,024 card + 3,072 note
bits, full tax **+0.000503 ahead / +0.001009 imm** against the 124-step fp32 control. The bit ladder
(`it36qatsmall`: `l1i4`, `l1i4s2`, `l1i2` at 124 steps, plus `l1i8` at 496 steps) walks DOWN from there."""

P.write_text(t.replace(anchor, anchor + add), encoding="utf-8")
print("gate paragraph added; lines:", len(P.read_text(encoding="utf-8").splitlines()))