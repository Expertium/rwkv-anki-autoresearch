from pathlib import Path

P = Path(r"C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\quant_plan\rust_grid.py")
raw = P.read_bytes()

DUP = (b'                     "RWKV_QUANT_SHIFTS": "1",\r\n'
       b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},\r\n'
       b'                     "RWKV_QUANT_SHIFTS": "1",\r\n'
       b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},\r\n')
FIX = (b'                     "RWKV_QUANT_SHIFTS": "1",\r\n'
       b'                     "RWKV_STATE_SHIFT_LEVEL": "card:int4,note:int2"},\r\n')
n = raw.count(DUP)
print("dup found", n)
if n != 1:
    raise SystemExit("ABORT")
P.write_bytes(raw.replace(DUP, FIX, 1))

import sys
sys.path.insert(0, str(P.parent))
import rust_grid as rg

for name in ("l2i6s4n1i8s2", "l2i6s4n1i8s3", "l2i5s4n1i8s3", "l2i4s4n1i8s3", "l2i5s4n1i8s2"):
    e = rg.CONFIGS[name]
    bad = [k for k in e if not k.startswith("RWKV_")]
    b = rg.bits_for(e)
    print(f"  {name:16s} keys={len(e)} bogus={bad} bits={b} total={sum(b.values())}")
print("CONFIGS size", len(rg.CONFIGS))
