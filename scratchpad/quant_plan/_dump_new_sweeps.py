import json
import pathlib

STAMPS = ["it36l1i4496", "it36small8s4", "it36small4s8", "it36l1i4s2496",
          "it36attrc", "it36attrn", "it36cn8", "it36ctl992", "it36l1i8992"]
for s in STAMPS:
    p = pathlib.Path("scratchpad/quant_plan/quant_sweep_" + s + ".json")
    d = json.loads(p.read_text())
    print("===", s, "ref=", d.get("ref"), "model=", pathlib.Path(d.get("model", "?")).name)
    for r in d["rows"]:
        g = r.get("gate", {})
        a = g.get("ahead", {}) or {}
        i = g.get("imm", {}) or {}
        print("  %-8s bits=%s ahead=%.6f imm=%.6f | d_ahead %+.6f ub95 %+.6f p %.3g | "
              "d_imm %+.6f ub95 %+.6f p %.3g" % (
                  r.get("name"), r.get("bits"), r.get("ahead", 0), r.get("imm", 0),
                  a.get("delta", 0), a.get("ub95", 0), a.get("p_worse", 0),
                  i.get("delta", 0), i.get("ub95", 0), i.get("p_worse", 0)))
