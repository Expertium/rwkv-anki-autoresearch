import json

d = json.load(open("scratchpad/quant_plan/corruption/state_subspace_l2i8s4all.json", encoding="utf-8"))
by = d.get("by_key", d)
for k, v in by.items():
    if not isinstance(v, dict):
        continue
    print("==", k, "n=", v.get("n"), "energy=", v.get("energy"))
    for f in ("svd", "tdpca", "vpca"):
        cur = v.get(f)
        if isinstance(cur, dict):
            print("   ", f, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in cur.items() if not isinstance(vv, dict)})
    loo = v.get("loo")
    if isinstance(loo, dict):
        for f, cur in loo.items():
            if isinstance(cur, dict) and "mean" in cur:
                print("    LOO", f, "mean", round(cur["mean"], 4), "min", round(cur.get("min", float("nan")), 4))
