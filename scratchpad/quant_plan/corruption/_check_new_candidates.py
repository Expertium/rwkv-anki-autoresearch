import sys, pathlib
ROOT = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch")
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import quant_sweep as qs

for name in ("l3i4s4", "l2i8s2", "n1i8s3", "n2i4s3"):
    env, comment, over = qs.CANDIDATES[name]
    b = qs.bits_for(env, over)
    ok = b["card"] <= qs.BAR_BITS["card"] and b["note"] <= qs.BAR_BITS["note"]
    print(f"{name:8s} card={b['card']:6d} (bar {qs.BAR_BITS['card']}) note={b['note']:6d} "
          f"(bar {qs.BAR_BITS['note']}) ok={ok}")
    assert ok, f"{name} is over the bar"
print("quant_sweep.py imports and all four candidates are inside the bit bars")
