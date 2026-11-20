"""Fill the constraint-6 speed cells of RESULTS.md from the idle-machine `rust_speed` run.

Run: `rust_speed.py --weights rust_artifacts/it36qatl1i8_124.safetensors --configs
c2_nl1i8,cn2,l1i8,l1i4 --reps 2 --tag it36qatl1i8` (2026-11-20 10:06, GPU otherwise idle).
plain 2,863.0 rev/s / 6,061.7 states/s; c2_nl1i8 2,586.0 / 0.903x / 0.894 median; cn2 2,628.4 /
0.918x / 0.903; l1i8 2,586.0 / 0.903x / 0.898; l1i4 2,586.0 / 0.903x / 0.894. All PASS.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "RESULTS.md"
t = P.read_text(encoding="utf-8")

REPL = [
    # main table: rows that still had a placeholder
    ("| 4 | PTQ: card full-matrix int2 + note rank-1 int8 | 2,176 (272) | 3,072 (384) | see speed note | see speed note |",
     "| 4 | PTQ: card full-matrix int2 + note rank-1 int8 | 2,176 (272) | 3,072 (384) | 2,586 | 0.903x |"),
    ("| 6 | PTQ card int2 + note rank-1 int8 applied to row 5 | 2,176 (272) | 3,072 (384) | see speed note | see speed note |",
     "| 6 | PTQ card int2 + note rank-1 int8 applied to row 5 | 2,176 (272) | 3,072 (384) | 2,586 | 0.903x |"),
    ("THE DEPLOY ROW** | 2,176 (272) | 3,072 (384) | see speed note | see speed note |",
     "THE DEPLOY ROW** | 2,176 (272) | 3,072 (384) | 2,586 | 0.903x |"),
    ("| 9 | QAT both streams full-matrix int2 (the naive at-the-bar scheme), evaluated WITH quantization | 2,176 (272) | 6,528 (816) | see speed note | see speed note |",
     "| 9 | QAT both streams full-matrix int2 (the naive at-the-bar scheme), evaluated WITH quantization | 2,176 (272) | 6,528 (816) | 2,628 | 0.918x |"),
    # speed section table
    ("| c2_nl1i8 | card int2 + note rank-1 int8 (row 8's config) | *running* | *running* | *running* | *running* | *running* | *running* |",
     "| c2_nl1i8 | card int2 + note rank-1 int8 (row 8's config) | 2,586 | 0.903x | 4,873 | 0.804x | 0.894 | PASS |"),
    ("4. The `c2_nl1i8` row (row 8's config) is being re-measured on the idle machine together with\n   `plain,cn2,l1i8,l1i4`, since no earlier run covered it.",
     "4. The `c2_nl1i8`/`l1i8` rows come from the IDLE-machine run `rust_speed the it36qatl1i8 tag`\n"
     "   (`rust_speed_it36qatl1i8.json`, 2026-11-20 10:06, weights\n"
     "   `rust_artifacts/it36qatl1i8_124.safetensors`, 2 reps, best kept): plain 2,863.0 rev/s /\n"
     "   6,061.7 states/s; c2_nl1i8 2,586.0 / 0.903x / median 0.894; cn2 2,628.4 / 0.918x / 0.903;\n"
     "   l1i8 2,586.0 / 0.903x / 0.898; l1i4 2,586.0 / 0.903x / 0.894 - every config PASSES 0.75x, so\n"
     "   constraint 6 holds for the recommended recipes too."),
]
for old, new in REPL:
    n = t.count(old)
    assert n == 1, (n, old[:70])
    t = t.replace(old, new)
assert "see speed note" not in t
assert "*running*" not in t
P.write_text(t, encoding="utf-8")
print("RESULTS.md speed cells filled; lines:", len(t.splitlines()))
