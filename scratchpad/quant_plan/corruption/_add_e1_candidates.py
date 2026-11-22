"""Add the E1 (tail-energy penalty) candidates to `quant_sweep.CANDIDATES`.

Raw-bytes insertion: `quant_sweep.py` mixes CRLF and LF, and the anchor lines are CRLF, so a
line-based text edit is fragile. The anchor must appear exactly once, and the new block is written
with the same CRLF endings. The script then imports the module and prints the bit accounting of the
three new candidates, which is the proof that the insertion is syntactically and semantically valid.

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/_add_e1_candidates.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QS = HERE.parent / "quant_sweep.py"

ANCHOR = (
    b'    "l2i8s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",\r\n'
    b'                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},\r\n'
    b'               "card rank-2 int8 factors + int4 shifts (1,280/2,304 bits)", None),\r\n'
)

BLOCK = [
    '    # ---- E1: the tail-energy penalty (RWKV_QAT_TAIL_PENALTY), zero state bits, zero deploy cost --',
    "    # The gate's tax is RANK TRUNCATION (`cn8`, full-matrix int8, is +0.000033/+0.000022), and the",
    "    # measured tail of row 17's own states (corruption/tail_energy_l2i8s4.json: 5,388 raw states on",
    "    # the eval path, users 7213/7044/6960) is: the card at rank 2 keeps 0.99381 of the state energy",
    "    # (tail 0.62%), the note at rank 1 keeps 0.96943 (tail 3.06%). The deploy path stores a rank-r",
    "    # matrix, so a state whose energy is EXACTLY rank-r loses nothing to the truncation; the tail is",
    "    # precisely the information the deploy path throws away, and the QAT LOSS CANNOT SEE IT, because",
    "    # the straight-through estimator passes gradient through the truncation as if it were identity.",
    "    # The penalty adds `lambda * mean over streams of the tail fraction` to the training loss. It is",
    "    # a RATIO, so it is scale-invariant: shrinking the whole state does not reduce it, only a",
    "    # steeper spectrum does. Lambda is bracketed by the measured tail (mean ~0.0245 over the four",
    "    # quantized layers): 1.0 is ~2.4% of the ~1.02 training loss, 10.0 is ~25%.",
    "    # RWKV_QAT_FUSED=0 is REQUIRED (the fused rank-1 kernel truncates inside the kernel), and it is",
    "    # also the documented bit-parity twin of the fused kernels, so `l2i8s4pf` separates \"fused off\"",
    '    # from "penalty on" - without it a small E1 delta could be misattributed. Added 2026-11-22.',
    '    "l2i8s4pf": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",',
    '                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",',
    '                  "RWKV_QAT_FUSED": "0"},',
    '                 "row 17\'s recipe on the Python per-step QAT loop, NO penalty - the E1 control (1,280/2,304)", None),',
    '    "l2i8s4e1a": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",',
    '                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",',
    '                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "1.0"},',
    '                  "row 17 + tail-energy penalty lambda=1.0 (~2.4% of the loss) (1,280/2,304)", None),',
    '    "l2i8s4e1b": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",',
    '                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",',
    '                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0"},',
    '                  "row 17 + tail-energy penalty lambda=10.0 (~25% of the loss) (1,280/2,304)", None),',
]
NEW = "\r\n".join(BLOCK).encode("utf-8") + b"\r\n"


def main() -> int:
    raw = QS.read_bytes()
    hits = raw.count(ANCHOR)
    if hits != 1:
        print(f"ABORT: anchor found {hits} times, expected exactly 1")
        return 1
    if b"l2i8s4e1a" in raw:
        print("ABORT: the E1 candidates are already present")
        return 1
    QS.write_bytes(raw.replace(ANCHOR, ANCHOR + NEW))
    print(f"patched {QS} (+{len(NEW)} bytes)")

    sys.path.insert(0, str(HERE.parent))
    import quant_sweep as qs  # noqa: E402

    for name in ("l2i8s4", "l2i8s4pf", "l2i8s4e1a", "l2i8s4e1b"):
        env, comment, override = qs.CANDIDATES[name]
        bits = qs.bits_for(env, override)
        print(f"{name:12s} bits={bits}  penalty={env.get('RWKV_QAT_TAIL_PENALTY', '-')}"
              f"  fused={env.get('RWKV_QAT_FUSED', '1')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
