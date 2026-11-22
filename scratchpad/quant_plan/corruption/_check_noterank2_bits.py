"""Check the bit arithmetic of the rank-2-note candidates against the bars."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quant_sweep as q  # noqa: E402

print("bars:", q.BAR_BITS, "naive int4:", q.INT4_BITS)
for k in ("l2i8n2i8", "l3i8n2i8", "l2i8n2i8s8", "l2i8n3i8", "l2i8s4", "l3i8s4", "n2i4s4"):
    if k not in q.CANDIDATES:
        print(f"{k:<12} MISSING")
        continue
    env = q.CANDIDATES[k][0]
    b = q.bits_for(env)
    print(f"{k:<12} card {b['card']:5d}  note {b['note']:5d}  total {b['card'] + b['note']:5d}"
          f"  card_ok={b['card'] <= 2176} note_ok={b['note'] <= 6528}")
