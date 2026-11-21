import sys, pathlib
ROOT = pathlib.Path(r"C:\Users\Andrew\rwkv-anki-autoresearch")
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import quant_sweep as qs

expect = {"l3i4s4": 3328, "l2i8s2": 3456, "n1i8s3": 3392, "n2i4s3": 3392,
          "l3i4s4n8s3": 3136, "l3i4s4n3": 2368, "l2i4s4n8s3": 2880}
for name, total in expect.items():
    env, comment, over = qs.CANDIDATES[name]
    b = qs.bits_for(env, over)
    ok = b["card"] <= qs.BAR_BITS["card"] and b["note"] <= qs.BAR_BITS["note"]
    assert ok, f"{name} is over the bar: {b}"
    assert b["card"] + b["note"] == total, f"{name}: {b['card']}+{b['note']} != {total}"
    print(f"{name:12s} card={b['card']:5d} note={b['note']:5d} total={b['card']+b['note']:5d} ok={ok}")
print("all candidates present, inside the bars, and at the expected totals")
