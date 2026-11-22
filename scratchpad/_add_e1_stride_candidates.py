"""Register the stride-guarded E1 candidates (l2i8s4e1c/d/e) in quant_sweep.CANDIDATES.

Inserted directly after the `l2i8s4e1b` entry, with CRLF line endings (the file is CRLF-dominant:
0 bare LF). The block also records WHY the first two E1 points could not train.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "scratchpad" / "quant_plan" / "quant_sweep.py"

ANCHOR = ('''    "l2i8s4e1b": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0"},
                  "row 17 + tail-energy penalty lambda=10.0 (~25% of the loss) (1,280/2,304)", None),
''')

BLOCK = '''    # ---- E1, second attempt: the SAME penalty with a step subsample (RWKV_QAT_TAIL_STRIDE) -------
    # `e1a`/`e1b` above are kept as the record: both raised CUDA OOM at 10.34 GiB of the 11.00 GiB
    # cap BEFORE their first training step (2026-11-23 00:18/00:19), because the penalty's
    # differentiable `svdvals` runs once per TIMESTEP per truncated layer and every one of those
    # graphs stays alive until backward, so its memory grows with the training window. The control
    # `l2i8s4pf` (same layout, penalty off) trains 6 steps in 21.4 s in the same 10.34 GiB, so the
    # penalty's own graph is the whole difference.
    # The module now accumulates on ONE step in `RWKV_QAT_TAIL_STRIDE` per stream (default 8). The
    # mean over the sampled steps is an unbiased estimate of the mean tail over the window, at
    # 1/stride of the graph memory - a smoke first proved the counting (scratchpad/_test_e1_stride.py)
    # and the real trainer path (corruption/e1_cpu_smoke.py with E1SMOKE_STRIDE).
    # Lambda still follows the measured-tail bracket (mean tail ~0.0245 over the four quantized
    # layers, so 1.0 ~ 2.4% of the ~1.02 loss, 3.0 ~ 7%, 10.0 ~ 25%).
    "l2i8s4e1c": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "1.0",
                   "RWKV_QAT_TAIL_STRIDE": "8"},
                  "row 17 + tail-energy penalty lambda=1.0, stride 8 (~2.4% of the loss) (1,280/2,304)", None),
    "l2i8s4e1d": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "3.0",
                   "RWKV_QAT_TAIL_STRIDE": "8"},
                  "row 17 + tail-energy penalty lambda=3.0, stride 8 (~7% of the loss) (1,280/2,304)", None),
    "l2i8s4e1e": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
                   "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
                   "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0",
                   "RWKV_QAT_TAIL_STRIDE": "8"},
                  "row 17 + tail-energy penalty lambda=10.0, stride 8 (~25% of the loss) (1,280/2,304)", None),
'''


def main() -> int:
    raw = P.read_bytes().decode("utf-8")
    lf = raw.replace("\r\n", "\n")
    print("bare LF:", lf.count("\n") - lf.count("\r\n"), "| CRLF lines:", raw.count("\r\n"))
    if lf.count(ANCHOR) != 1:
        print(f"FAIL: anchor found {lf.count(ANCHOR)} times")
        return 1
    out_lf = lf.replace(ANCHOR, ANCHOR + BLOCK)
    if out_lf.count("RWKV_QAT_TAIL_STRIDE") != 5:
        print(f"FAIL: expected 5 stride refs, got {out_lf.count('RWKV_QAT_TAIL_STRIDE')}")
        return 1
    out = out_lf.replace("\n", "\r\n")
    P.write_bytes(out.encode("utf-8"))
    chk = P.read_bytes().decode("utf-8")
    if chk != out:
        print("FAIL: round-trip mismatch")
        return 1
    print("inserted l2i8s4e1c / l2i8s4e1d / l2i8s4e1e")
    return 0


if __name__ == "__main__":
    sys.exit(main())
