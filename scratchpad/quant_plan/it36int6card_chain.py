"""Chain 18: the MIXED-precision int6 points - the CARD-only rung, then its NOTE-only twin.

WHY. Chain 17's ladder coarsens card and note together (`l2i6s4`), so a failure cannot say WHERE the
int6 tax comes from. An 11-user deploy screen (`rust_wide_screen_it36int6split.json`, per-user paired
deltas vs `l2i8s4`) splits it:

  * `l2i6c` card rank-2 int6 + note rank-1 int8 = 1,024/2,304 = 3,328 bits (256 BELOW row 17) costs
    +0.000115 imm [+0.000018, +0.000218] and +0.000052 ahead [-0.000007, +0.000109] - nearly free at
    PTQ, and the ahead head is not measurably worse at all.
  * `l2i6n` card rank-2 int8 + note rank-1 int6 = 1,280/1,920 = 3,200 bits (384 BELOW row 17) costs
    +0.000307 imm [+0.000124, +0.000558] and +0.000268 ahead [+0.000079, +0.000601].

So the CARD is the stream whose int6 is worth a 750-user QAT point, and `l2i6n` is the follow-up that
says whether the NOTE can be cut at int6 as well (the shrink chain tests the note at int3, which is
coarser; if that fails and int6 passes, `l2i6n` is the note's frontier).

Order: least-aggressive first, so a pass is banked before the riskier point is spent. Waits behind
`it36int6` (chain 17), which asks the same factor-precision question with BOTH streams coarsened - the
coarse answer should come first, and if chain 17 fails this chain is what isolates the card.

Outside `autorun/QUEUE.md` on purpose: it is the interactive session's chain, like `it36e1`,
`it36e1shrink` and `it36int6`, so the hourly supervisor must not start, kill, restart or score it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "quant_plan"
sys.path.insert(0, str(RUN))
import it36ladder_lib as lib  # noqa: E402  the shared wedge watchdog + retry

BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"
            "rwkv-infer|probe_get_result|check_corruption")
POINTS = [
    ("l2i6c", "it36l2i6c", 124, "qnj_it36ctl"),
    ("l2i6n", "it36l2i6n", 124, "qnj_it36ctl"),
]
lib.WAIT_MAX_MIN = 2400
sys.exit(lib.chain_main("it36int6card", "it36int6_chain.log",
                        "=== it36int6 chain end ===", POINTS, BUSY_PAT))
