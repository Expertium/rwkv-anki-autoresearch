"""Reorder the queue so the corruption check runs FIRST (2026-11-22 11:30, Andrew: "still no corruption
checks done").

The corruption check was queued behind it36shrink because the shrink chain was already running. It is
the cheap one (50 users, minutes per point) and it is Andrew's headline deliverable, so:
  * it36corrupt   waits on `it36longdecay` (which ended at 02:29:54) -> it can start at once;
  * it36shrink    waits on `it36corrupt`            -> it runs right after the verdicts land;
  * it36equalbits waits on `it36shrink` (unchanged), it36combo on equalbits, it36corrupt2 on combo.
Also adds the four shrink tags to corrupt2's gate-guarded point list and to check_corruption's CONFIGS,
so a passing shrink row is swept too.
"""
import pathlib

RUN = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan")
COR = RUN / "corruption"


def sub(path, old, new, count=1):
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        print(f"{p.name}: already patched")
        return
    if old not in text:
        raise SystemExit(f"{p.name}: anchor not found:\n{old[:200]}")
    p.write_text(text.replace(old, new, count), encoding="utf-8")
    print(f"{p.name}: patched")


# 1. it36corrupt now waits for it36longdecay (already ended) instead of it36shrink.
sub(RUN / "it36corrupt_chain.py",
    'PREV_LOG = RUN / "it36shrink_chain.log"\nPREV_END = "=== it36shrink chain end ==="',
    'PREV_LOG = RUN / "it36longdecay_chain.log"\nPREV_END = "=== it36longdecay chain end ==="')

# 2. it36shrink now waits for it36corrupt instead of it36longdecay.
sub(RUN / "it36shrink_chain.py",
    'sys.exit(lib.chain_main("it36shrink", "it36longdecay_chain.log",\n'
    '                            "=== it36longdecay chain end ===", POINTS, BUSY_PAT))',
    'sys.exit(lib.chain_main("it36shrink", "it36corrupt_chain.log",\n'
    '                            "=== it36corrupt chain end ===", POINTS, BUSY_PAT))')

# 3. corrupt2 also sweeps the four shrink rows.
sub(RUN / "it36corrupt2_chain.py",
    '    ("l2i4s4n8s3", "it36l2i4s4n8s3", "chain 13 point 3"),\n]',
    '    ("l2i4s4n8s3", "it36l2i4s4n8s3", "chain 13 point 3"),\n'
    '    ("l3i4s4n4", "it36l3i4s4n4", "shrink point 1 (card rank-3 int4 + note int4)"),\n'
    '    ("l2i8s4n3", "it36l2i8s4n3", "shrink point 2 (note rank-1 int3 factors)"),\n'
    '    ("l2i8s2n4", "it36l2i8s2n4", "shrink point 3 (card int2 shifts + note int4)"),\n'
    '    ("l3i8s4n4s2", "it36l3i8s4n4s2", "shrink point 4 (note int4 + int2 shifts)"),\n]')

# 4. check_corruption.py learns the four shrink tags.
sub(COR / "check_corruption.py",
    '    "l2i4s4n8s3": (RUN / "qat_it36l2i4s4n8s3" / "qat_124.pth", "l2i4s4n8s3", "chain 13 point 3"),\n}',
    '    "l2i4s4n8s3": (RUN / "qat_it36l2i4s4n8s3" / "qat_124.pth", "l2i4s4n8s3", "chain 13 point 3"),\n'
    '    "l3i4s4n4": (RUN / "qat_it36l3i4s4n4" / "qat_124.pth", "l3i4s4n4", "shrink point 1"),\n'
    '    "l2i8s4n3": (RUN / "qat_it36l2i8s4n3" / "qat_124.pth", "l2i8s4n3", "shrink point 2"),\n'
    '    "l2i8s2n4": (RUN / "qat_it36l2i8s2n4" / "qat_124.pth", "l2i8s2n4", "shrink point 3"),\n'
    '    "l3i8s4n4s2": (RUN / "qat_it36l3i8s4n4s2" / "qat_124.pth", "l3i8s4n4s2", "shrink point 4"),\n}')
print("done")