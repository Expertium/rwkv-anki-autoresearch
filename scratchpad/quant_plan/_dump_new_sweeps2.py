import json
import pathlib

STAMPS = {
    "it36l1i4496": "row: l1i4 @496",
    "it36small8s4": "row 9: l1i8s4 @124",
    "it36small4s8": "row 10: l1i4s8 @124",
    "it36l1i4s2496": "l1i4s2 @496",
    "it36attrc": "attr: card rank-1 int8 only",
    "it36attrn": "attr: note rank-1 int8 only",
    "it36cn8": "diag: full-matrix int8",
    "it36ctl992": "ctl 992 steps (fp32)",
    "it36l1i8992": "l1i8 @992",
}
for s, label in STAMPS.items():
    d = json.loads(pathlib.Path("scratchpad/quant_plan/quant_sweep_" + s + ".json").read_text())
    for r in d["rows"]:
        a, i = r["gate"]["ahead"], r["gate"]["imm"]
        print(f"### {label}  [{s}] name={r['name']} bits={r['bits']} ref={d['ref']} "
              f"secs={r['secs']:.0f} bits_ok={r['bits_ok']}")
        print(f"    ahead {a['mean_cur']:.6f} (ref {a['mean_ref']:.6f}) d {a['delta']:+.6f} "
              f"ub95 {a['ub95']:+.6f} p {a['p_worse']:.3g}")
        print(f"    imm   {i['mean_cur']:.6f} (ref {i['mean_ref']:.6f}) d {i['delta']:+.6f} "
              f"ub95 {i['ub95']:+.6f} p {i['p_worse']:.3g}")
