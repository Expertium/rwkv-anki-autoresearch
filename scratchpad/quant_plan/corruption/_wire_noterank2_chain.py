"""Wire chain 14 (rank-2 note) into the queue.

1. `it36shrink_chain.py` now waits for `it36noterank2` instead of `it36corrupt`, and its BUSY_PAT gains
   the corruption patterns (it is queued behind a chain that can run the corruption driver).
2. `it36corrupt2_chain.py` gains the four chain-14 tags, so the gate-guarded corruption sweep covers
   every new row (it skips a row whose own sweep fails the log-loss gate).
3. `corruption/check_corruption.py` CONFIGS gains the same four tags.

Run once; every replacement asserts exactly one match.
"""
import re
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]


def patch(path: Path, pairs) -> None:
    text = path.open(encoding="utf-8", newline="").read()
    for pat, rep, what in pairs:
        # a function replacement keeps backslashes literal (no \1 / \n escape surprises)
        text, n = re.subn(pat, lambda _m, r=rep: r, text, count=1)
        if n != 1:
            raise SystemExit(f"{path.name}: {what}: expected 1 match, got {n}")
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print(f"patched {path.name}")


# 1. shrink
patch(RUN / "it36shrink_chain.py", [
    (r'BUSY_PAT = \("qat_decay\.py.*?rwkv-infer"\)',
     'BUSY_PAT = ("qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|rust_speed.py|rust_grid.py|"\n'
     '            "rwkv-infer|probe_get_result|check_corruption")',
     "BUSY_PAT"),
    (r'sys\.exit\(lib\.chain_main\("it36shrink", "it36corrupt_chain\.log",\s*\n\s*"=== it36corrupt chain end ===", POINTS, BUSY_PAT\)\)',
     'sys.exit(lib.chain_main("it36shrink", "it36noterank2_chain.log",\n'
     '                        "=== it36noterank2 chain end ===", POINTS, BUSY_PAT))',
     "predecessor -> it36noterank2"),
])

# 2. corrupt2 points
new_points = "\n".join([
    '    ("l2i8n2i8", "it36l2i8n2i8", "chain 14 point 1 (note rank-2 int8 + int4 shifts)"),',
    '    ("l3i8n2i8", "it36l3i8n2i8", "chain 14 point 2 (rank-3 card + note rank-2 int8)"),',
    '    ("l2i8n2i8s8", "it36l2i8n2i8s8", "chain 14 point 3 (note rank-2 int8 + int8 shifts)"),',
    '    ("l2i8n3i8", "it36l2i8n3i8", "chain 14 point 4 (note rank-3 int8)"),',
])
patch(RUN / "it36corrupt2_chain.py", [
    (r'(\("l3i8s4n4s2", "it36l3i8s4n4s2", "shrink point 4 \(note int4 \+ int2 shifts\)"\),)',
     r"\1\n" + new_points,
     "chain 14 points"),
])

# 3. the corruption driver's CONFIGS
new_cfgs = "\n".join([
    '    "l2i8n2i8": (RUN / "qat_it36l2i8n2i8" / "qat_124.pth", "l2i8n2i8", "chain 14 point 1 (note rank-2 int8)"),',
    '    "l3i8n2i8": (RUN / "qat_it36l3i8n2i8" / "qat_124.pth", "l3i8n2i8", "chain 14 point 2"),',
    '    "l2i8n2i8s8": (RUN / "qat_it36l2i8n2i8s8" / "qat_124.pth", "l2i8n2i8s8", "chain 14 point 3"),',
    '    "l2i8n3i8": (RUN / "qat_it36l2i8n3i8" / "qat_124.pth", "l2i8n3i8", "chain 14 point 4"),',
])
patch(RUN / "corruption" / "check_corruption.py", [
    (r'(    "l3i8s4n4s2": \(RUN / "qat_it36l3i8s4n4s2" / "qat_124\.pth", "l3i8s4n4s2", "shrink point 4"\),)',
     r"\1\n" + new_cfgs,
     "chain 14 configs"),
])
