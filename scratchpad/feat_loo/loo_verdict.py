"""Leave-one-out verdict: per feature, the paired per-user cost of zeroing it at featB's input,
both modes, on the users the arm scored. Sorted by imm cost. ~0 in both modes == the model does not
rely on it, which bounds its retrained value at ~0 too -- a DROP candidate. Reliance, not value.
"""
import json, os
import numpy as np
from scipy.stats import wilcoxon
groups = json.load(open("scratchpad/feat_loo/arms.json"))
def load(tag, mode):
    f = "result/RWKV-%s.jsonl" % tag if mode == "ahead" else "result/RWKV-P-%s.jsonl" % tag
    if not os.path.exists(f): return None
    return {json.loads(l)["user"]: json.loads(l)["metrics"]["LogLoss"] for l in open(f) if l.strip()}
ctrl = {m: load("featB", m) for m in ("ahead", "imm")}
rows = []
for tag, cs in groups:
    arm = {m: load("loo_" + tag, m) for m in ("ahead", "imm")}
    if any(v is None for v in arm.values()):
        rows.append((tag, cs, None)); continue
    out = {}
    for m in ("ahead", "imm"):
        u = sorted(set(arm[m]) & set(ctrl[m])); d = np.array([arm[m][k] - ctrl[m][k] for k in u])
        out[m] = (d.mean(), wilcoxon(d, alternative="greater").pvalue if len(u) > 10 and np.any(d != 0) else float("nan"), len(u))
    rows.append((tag, cs, out))
print("=" * 96); print("LEAVE-ONE-OUT on featB (cost of zeroing = arm minus control; + means the feature helps)"); print("=" * 96)
print("%-28s %12s %10s %12s %10s %5s" % ("feature", "ahead cost", "p", "imm cost", "p", "n"))
done = [r for r in rows if r[2] is not None]
for tag, cs, o in sorted(done, key=lambda r: -r[2]["imm"][0]):
    print("%-28s %+12.6f %10.1e %+12.6f %10.1e %5d" % (tag, o["ahead"][0], o["ahead"][1], o["imm"][0], o["imm"][1], o["imm"][2]))
for tag, cs, o in rows:
    if o is None: print("%-28s MISSING" % tag)
print()
print("DROP CANDIDATES (|cost| < 0.0001 in BOTH modes, i.e. under the accept bar):")
for tag, cs, o in done:
    if abs(o["ahead"][0]) < 1e-4 and abs(o["imm"][0]) < 1e-4: print("  " + tag + "  (" + ",".join(cs) + ")")
