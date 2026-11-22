"""Export a fitted fixed basis (fixedbasis_<tag>.npz) to the TEXT format the Rust deploy engine reads.

The Rust engine has no numpy, so the basis travels as whitespace-separated text -- the same convention
`RWKV_SHIFT_ROT` already uses (line 1 = a header, then the numbers). Format:

    <n_entries>
    <name> <layer> <stride> <K*stride floats P, row-major> <K*stride floats Q, row-major>

`stride` is the number of basis columns actually stored (the fitted M); the engine reads only the
first m <= stride columns, so ONE file serves every m in RWKV_STATE_FIXEDBASIS_SCOPE.

Per-(stream, layer) bases are preferred over the pooled `__ALL` key: measured capture on the note is
0.87-0.97 per layer vs 0.86 pooled at m=4 (corruption/fixedbasis_*.json).

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/export_fixedbasis_rust.py \
      --npz scratchpad/quant_plan/corruption/fixedbasis_l2i8s4all.npz \
      --out scratchpad/quant_plan/corruption/fixedbasis_l2i8s4all.txt [--streams card,note] [--expand-all]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LAYERS = {"card": 1, "note": 3, "deck": 4, "preset": 3, "user": 3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", default="", help="comma list to restrict to (default: every key)")
    ap.add_argument("--expand-all", action="store_true",
                    help="also write the pooled __ALL basis for streams/layers with no per-layer key")
    args = ap.parse_args()

    npz = Path(args.npz)
    keep = {s.strip() for s in args.streams.split(",") if s.strip()}
    doc = np.load(npz)
    entries = []  # (name, layer, stride, P, Q)
    pooled = {}
    for k in sorted(doc.files):
        m = re.fullmatch(r"P__([A-Za-z]+)__(L(\d+)|ALL)", k)
        if not m:
            continue
        name, layer = m.group(1), m.group(2)
        qk = k.replace("P__", "Q__", 1)
        assert qk in doc.files, f"{npz}: {k} has no {qk}"
        if layer == "ALL":
            pooled[name] = (k, qk)
            continue
        entries.append((name, int(m.group(3)), doc[k], doc[qk]))

    if not entries and not pooled:
        raise SystemExit(f"{npz}: no P__<stream>__L<layer> / __ALL keys found (files: {list(doc.files)})")
    have = {(n, l) for n, l, _, _ in entries}
    if args.expand_all:
        for name, (pk, qk) in sorted(pooled.items()):
            for l in range(LAYERS.get(name, 0)):
                if (name, l) not in have:
                    entries.append((name, l, doc[pk], doc[qk]))
    if keep:
        entries = [e for e in entries if e[0] in keep]
    if not entries:
        raise SystemExit(f"{npz}: nothing left after --streams {sorted(keep)}")
    entries.sort(key=lambda e: (e[0], e[1]))

    lines = [str(len(entries))]
    report = []
    for name, layer, p, q in entries:
        assert p.shape == q.shape, f"{npz}: {name}/L{layer} P{p.shape} != Q{q.shape}"
        stride = int(p.shape[1])
        lines.append(f"{name} {layer} {stride}")
        lines.append(" ".join(f"{v:.8g}" for v in np.asarray(p, dtype=np.float32).reshape(-1)))
        lines.append(" ".join(f"{v:.8g}" for v in np.asarray(q, dtype=np.float32).reshape(-1)))
        report.append({"stream": name, "layer": layer, "stride": stride,
                       "p_norm": float(np.linalg.norm(p)), "q_norm": float(np.linalg.norm(q))})
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(entries)} entries)")
    for r in report:
        print(f"  {r['stream']:>5} L{r['layer']}  stride={r['stride']:>2}  band={r['stream']}:{r['layer']}  "
              f"|P|={r['p_norm']:.4f} |Q|={r['q_norm']:.4f} (a K x stride orthonormal-ish basis: "
              f"|P|^2 ~ stride = {r['stride']})")
    meta = {"npz": str(npz), "out": str(out), "entries": report}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
