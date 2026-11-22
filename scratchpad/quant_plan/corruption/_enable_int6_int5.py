"""Enable the int6/int5 factor rungs end to end (raw-bytes patch, all edits additive).

THREE edits, all in one place so the change is reviewable:

1. `rwkv/architecture.py` `_QMAX` -- the ONLY thing that blocks a QAT run at int6/int5. Without it
   `card:2:int6` raises `KeyError: 'int6'` in `_apply_qat_scopes`. Purely ADDITIVE: no existing
   candidate names int6/int5, so no existing recipe changes behaviour. Both quantization paths are
   already generic in qmax -- the fused kernels use `fminf(fmaxf(roundf(x/qmax), -qmax), qmax) * scale`
   (rwkv7_cuda.cu:473/474/740/808) and the Python twin uses `amax/qmax` (rwkv_ops.py:326/337) -- so no
   kernel work is needed. The Rust deploy engine has ALWAYS had these levels (`parse_level`,
   model.rs:2104-2105), which is why `rust_grid.py` could price them with no code change at all.

2. `scratchpad/quant_plan/quant_sweep.py` `_QMAX` + `_QBITS` -- the sweep's own copies, used by
   `bits_for`. `_QBITS` must carry 6/5 or the reported bit counts are wrong.

3. `scratchpad/quant_plan/quant_sweep.CANDIDATES` -- the three ladder points. Layout is row 17's exact
   shape (card rank 2, note rank 1, int4 shifts); the ONLY change is factor precision, so each point
   isolates it:

     l2i6s4     card:2:int6  note:1:int6  int4 shifts   1,024/1,920  = 2,944 bits  (640 below row 17)
     l2i5s4     card:2:int5  note:1:int5  int4 shifts     896/1,728  = 2,624 bits  (960 below row 17,
                                                                                   192 below row 21)
     l2i4s4n4   card:2:int4  note:1:int4  int4 shifts     768/1,536  = 2,304 bits  (bottom rung, 36%
                                                                                   below row 17; the
                                                                                   BOTH-int4 combination
                                                                                   is untested -- row 21
                                                                                   was note-int4 only and
                                                                                   row 23 card-int4 only)
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCH = ROOT / "rwkv" / "architecture.py"
QS = ROOT / "scratchpad" / "quant_plan" / "quant_sweep.py"

OLD_QMAX = '_QMAX = {"int8": 127.0, "int4": 7.0, "int3": 3.0, "int2": 1.0, "fp32": float("inf")}'
NEW_QMAX = (
    '# int6/int5 are NEW here (2026-11-22). They have always existed in the Rust deploy engine\n'
    '# (`parse_level`, model.rs:2104) and in rust_grid.QBITS, but not in the training path, so no QAT\n'
    '# run could train for them. Additive only: no existing recipe names them. Motive is axis 1 -- the\n'
    '# factor-precision rungs BELOW int8 at row 17\'s exact layout, e.g. card:2:int6 + note:1:int6 =\n'
    '# 1,024/1,920 bits, which is 640 bits under row 17\'s 1,280/2,304.\n'
    '_QMAX = {"int8": 127.0, "int6": 31.0, "int5": 15.0, "int4": 7.0, "int3": 3.0, "int2": 1.0,\n'
    '         "fp32": float("inf")}'
)

OLD_QBITS = '_QBITS = {"int8": 8, "int4": 4, "int3": 3, "int2": 2, "fp32": 32}'
NEW_QBITS = '_QBITS = {"int8": 8, "int6": 6, "int5": 5, "int4": 4, "int3": 3, "int2": 2, "fp32": 32}'

ANCHOR_BODY = [
    '    "l3i8s4n4e1b": ({"RWKV_QAT_LOWRANK_SCOPE": "card:3:int8,note:1:int4",',
    '                     "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",',
    '                     "RWKV_QAT_FUSED": "0", "RWKV_QAT_TAIL_PENALTY": "10.0"},',
    '                    "row 22\'s layout + tail penalty lambda=10.0 (1,792/1,536 = 3,328)", None),',
]

NEW_CANDS = [
    "    # THE int6/int5 FACTOR RUNGS (2026-11-22) -- axis 1. Row 17's EXACT layout (card rank 2, note",
    "    # rank 1, int4 shifts), with only the FACTOR PRECISION changed, so each point isolates it.",
    "    # Every earlier rung below row 17 used int4 factors; int6/int5 were never runnable at all",
    "    # because _QMAX lacked the keys (patched above). Bits:",
    "    #   l2i6s4   card 2x2x32x6 + 64x4 = 1,024  note 3x(2x1x32x6 + 64x4) = 1,920  = 2,944",
    "    #   l2i5s4   card 2x2x32x5 + 64x4 =   896  note 3x(2x1x32x5 + 64x4) = 1,728  = 2,624",
    "    #   l2i4s4n4 card 2x2x32x4 + 64x4 =   768  note 3x(2x1x32x4 + 64x4) = 1,536  = 2,304",
    "    # A 2-user Rust PTQ probe on the row-17 checkpoint said int6 costs +0.000003 ahead / +0.000002",
    "    # imm vs int8 and int5 +0.000017/-0.000073 (rust_grid_int6_probe.json) -- i.e. no visible",
    "    # penalty for 640/960 fewer bits. That probe CANNOT settle the gate: calibrated against rows 21",
    "    # and 23 it reproduces the note-int4 cost at only 54% and gets the card-int4 sign WRONG. So",
    "    # these are full 750-user QAT points, not PTQ.",
    '    "l2i6s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int6,note:1:int6",',
    '                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},',
    '               "row 17 layout at int6 factors (1,024/1,920 = 2,944 bits)", None),',
    '    "l2i5s4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int5,note:1:int5",',
    '                "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},',
    '               "row 17 layout at int5 factors (896/1,728 = 2,624 bits)", None),',
    '    "l2i4s4n4": ({"RWKV_QAT_LOWRANK_SCOPE": "card:2:int4,note:1:int4",',
    '                  "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4"},',
    '                 "row 17 layout at int4 factors, the bottom rung (768/1,536 = 2,304 bits)", None),',
]


def patch(path, pairs, nl):
    raw = path.read_bytes()
    for old, new in pairs:
        ob = old.replace("\n", nl).encode("utf-8")
        n = raw.count(ob)
        assert n == 1, f"{path.name}: anchor {old[:40]!r} count {n}, expected 1"
        raw = raw.replace(ob, new.replace("\n", nl).encode("utf-8"))
    path.write_bytes(raw)
    print(f"patched {path.name}")


def main() -> int:
    a = ARCH.read_bytes()
    if b'"int6": 31.0' in a:
        print("architecture.py already has int6; skipping")
    else:
        patch(ARCH, [(OLD_QMAX, NEW_QMAX)], "\n")

    q = QS.read_bytes()
    if b'"int6": 31.0' in q:
        print("quant_sweep.py already has int6; skipping")
    else:
        patch(QS, [(OLD_QMAX, NEW_QMAX), (OLD_QBITS, NEW_QBITS)], "\r\n")
        # the candidate block is CRLF
        anchor = "\r\n".join(ANCHOR_BODY).encode("utf-8")
        raw = QS.read_bytes()
        n = raw.count(anchor)
        assert n == 1, f"candidate anchor count {n}, expected 1"
        QS.write_bytes(raw.replace(anchor, anchor + b"\r\n" + "\r\n".join(NEW_CANDS).encode("utf-8")))

    # verify
    import importlib.util
    spec = importlib.util.spec_from_file_location("quant_sweep", QS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("l2i8s4", "l2i6s4", "l2i5s4", "l2i4s4n4"):
        env, comment, ov = mod.CANDIDATES[name]
        b = mod.bits_for(env, ov)
        print(f"  {name}: card={b['card']}b note={b['note']}b total={b['card'] + b['note']}"
              f"  | {comment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
