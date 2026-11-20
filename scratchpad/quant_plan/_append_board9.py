"""Append board amendment 9 (the relaxed QAT step budget + chain 4). The board is CRLF + BOM, so
`edit` cannot match it; append bytes in binary mode instead."""
from pathlib import Path

P = Path("optimization/GPU_RUN_BOARD.md")
blk = """
Row 11 amendment 9 (2026-11-21 01:10, Moscow). **Andrew relaxed the QAT step budget, and chain 4
(`it36qatsmall4`) is queued to exploit it.** The 124-step QAT length was the champion's own decay
length, so every earlier QAT point copied it; Andrew now allows any length: "instead of using 124
steps, you can use more steps. It will make results less apples-to-apples comparable, but if it
ultimately ends up turning a failing recipe into a winning one, it's fine". Chain 4 (launched detached
01:05:50, lock `it36qatsmall4_chain.lock`, waits behind chain 3) runs: (1) the 992-step fp32 twin
`it36ctl992` -> tag `qnj_it36ctl992`, scored against `qnj_it36_124` for context only; (2) `l1i8` at 992
steps against that twin (RESULTS row 11); (3) `l1i4` at 992 steps only if the 496-step `l1i4` ahead tax
beat the 124-step one (row 12). Every point keeps a same-length fp32 twin: longer training helps fp32
too (row 0b is 0.0121/0.0116 better than row 0), so without its own twin a longer point cannot separate
more training from quantization. 992 = 8x the champion's 124 and 2x the last measured point, so it is
the last sensible length on this curve. The tax rises from 124 to 496 steps (+0.000503/+0.001009 ->
+0.001132/+0.001674); if it keeps rising at 992, this route is closed for good and only bit allocation
is left. Chain 3 (the `l1i8c`/`l1i8n` attribution pair plus the `cn8` mechanism diagnostic) still runs
before it. GPU at amendment time: chain 2's `l1i4`@496 QAT, 6.3 GB / 71%; free RAM 41.8 GB.
"""
with P.open("ab") as fh:
    fh.write(blk.encode("utf-8"))
text = P.read_text(encoding="utf-8-sig")
print("amendment 9 markers:", text.count("Row 11 amendment 9"))
print("board lines:", text.count("\n") + 1)
