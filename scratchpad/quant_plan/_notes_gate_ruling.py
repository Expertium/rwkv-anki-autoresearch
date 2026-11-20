"""Prepend a live-status bullet about Andrew's 2026-11-20 gate ruling and the bit-ladder chain."""
from pathlib import Path

P = Path(__file__).resolve().parent / "NOTES.md"
t = P.read_text(encoding="utf-8")

anchor = "## Live status\n"
assert t.count(anchor) == 1

blk = """
* 2026-11-20 22:25 - **Andrew REDEFINED THE GATE and re-ranked the axes.** A result is judged against
  "whatever run that is exactly identical EXCEPT for quantization" - the fp32 control with the same
  recipe, data and step budget (`qnj_it36ctl` for 124-step points, `qnj_it36ctl496` for the 496-step
  point). The anchor-champion comparison (+0.0085/+0.0071) is context only and must NOT be quoted as
  the quantization cost; the believable number for `l1i8` is **+0.000503 ahead / +0.001009 imm**.
  Priority for further work: **(1) smaller state (highest), (2) lower log loss against that identical
  unquantized run, (3) speed** (10% slower is acceptable). Consequence: the arm now walks DOWN the bit
  ladder instead of up.
* 2026-11-20 22:25 - **`it36qatsmall_chain.py` launched** (detached, log `it36qatsmall_chain.log`):
  job 0 scores the 496-step fp32 control `qat_it36ctl496/ctl_496.pth` (trained 10:21 today, never
  evaluated because the machine went down at ~10:28); then, all at 124 steps and all against
  `qnj_it36ctl`: `l1i4` (512/1,536 bits), `l1i4s2` (384/1,152 bits), `l1i2` (256/768 bits); then
  `l1i8` at 496 steps against the 496-step twin. Each point = 0.04-epoch smoke + QAT + one quantized
  eval (~20 min), so the result is a bits-vs-tax curve. Added `l1i3`/`l1i2` to
  `quant_sweep.CANDIDATES` (they were only in the Rust grid before).
* 2026-11-20 22:24 - **The 10:28 outage:** the previous chains died mid-run. `ctl_496.pth` completed
  cleanly (10:21), but `qat496` (candidate `c2_nl1i8`, 496 steps) was killed around global step 250;
  its `qat_it36qat496/` folder holds only `qat_50.pth` and must NOT be scored or resumed. Nothing else
  was lost - all four PTQ rows and rows 8/10 were already complete.
"""
t = t.replace(anchor, anchor + blk)
P.write_text(t, encoding="utf-8")
print("prepended; lines:", len(t.splitlines()))