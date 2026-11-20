"""One-shot: append Row 11 amendment 6 to optimization/GPU_RUN_BOARD.md.

`edit` cannot match this file (it carries a BOM), so the board is appended to with python, the
same way amendments 1-5 were written.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "optimization" / "GPU_RUN_BOARD.md"

blk = """
Row 11 amendment 6 (2026-11-20 22:50, Moscow). **The gate is redefined: the anchor is the run that is
identical EXCEPT for quantization, and the 496-step control shows the decay length dominates.**
Andrew's ruling (22:22): every quantized point is judged against the *same* run with quantization OFF,
not against the iter36 champion. For the 124-step decay that twin is `qnj_it36ctl` = 0.333802 ahead /
0.306211 imm (750-user seeded roster), and the previously quoted "-0.0085/-0.0071 vs the champion" is
WITHDRAWN: the champion `rwkv_iter36_124.pth` differs in data and step budget, so that number was
never a quantization result. `scratchpad/quant_plan/RESULTS.md` was rebuilt around row 0 = the control
(rows 1-4 = the gate-comparable PTQ/QAT points, then the fp32-twin table, the tax decomposition, and
Appendix A holding the PTQ-on-champion numbers explicitly marked non-comparable); `NOTES.md` carries
the same ruling plus the resume hazards.

New reference point, row 0b: the SAME 124-step recipe run for 496 steps with no quantization
(`qnj_it36ctl496`) scores **0.321659 ahead / 0.294614 imm** - 0.012143/0.011598 BETTER than row 0
(ub95 +0.012911/+0.012336, p ~1e-108) and 0.021154/0.019659 better than the iter36 champion. So at
this scale the decay length moves the loss about 25x more than the entire quantization tax does
(0.0121/0.0116 vs 0.0005/0.0010), and any 496-step candidate must be compared against this twin.

REBUILD, NOT RESUME: the 10:28 machine outage killed `it36qat496_chain` while the 496-step `c2_nl1i8`
QAT was around global step 250, leaving `scratchpad/quant_plan/qat_it36qat496/qat_50.pth` with an
EMPTY train.log. That checkpoint must NOT be scored and that folder must NOT be resumed. Its clean
control (`qat_it36ctl496/ctl_496.pth`, finished 10:21:28) survived and is the row 0b above.

RUNNING, one GPU job at a time: `it36qatsmall_chain` (launched 2026-11-20 22:25, log
`scratchpad/quant_plan/it36qatsmall_chain.log`). Job 0 scored the 496-step control (done 22:45:52,
rc=0, 20.4 min). Then the bit ladder, each point = 0.04-epoch smoke + a 124-step QAT decay from
`rwkv_iter36_558.pth` + one quantized eval against `qnj_it36ctl`, for the reductions that matter most
to Andrew's first axis: `l1i4` (512/1,536 bits), `l1i4s2` (384/1,152), `l1i2` (256/768); and finally
`l1i8` at 496 steps (1,024/3,072 bits) against row 0b. Every point skips itself if its merged result
already exists, so a restart is safe.

Deliverables are on GitHub: commit `69121550` ("Document the card/note state-quantization arm, gated
on the identical unquantized twin") - 421 files, the arm's runners, RESULTS.md/NOTES.md, the board and
research-log updates, and new .gitignore rules for the regenerable per-user prediction dumps. The
2,158 locally deleted stale scratchpad paths were deliberately NOT committed. The deployed budget
stays 403 card bits / 214 note bits; no deploy-contract change is proposed.
"""

with P.open("a", encoding="utf-8") as fh:
    fh.write(blk)

text = P.read_text(encoding="utf-8")
print(f"board lines now: {len(text.splitlines())}")
print(f"amendment markers: {text.count('Row 11 amendment 6')}")
