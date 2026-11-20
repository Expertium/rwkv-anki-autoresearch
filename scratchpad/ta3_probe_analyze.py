import io
import json
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = "scratchpad/ta3_probe_20261115/ta3_probe.jsonl"
recs = [json.loads(line) for line in io.open(path, encoding="utf-8") if line.strip()]
print("probed steps:", len(recs), "| scope:", recs[0]["scope"],
      "| n_scope_params:", recs[0]["n_scope_params"],
      "| per-element displacement at eta=0.3:", round(recs[0]["per_element_eta_default"], 6))
print("mean loss_a:", round(statistics.mean(r["loss_a"] for r in recs), 4),
      "| mean loss_q:", round(statistics.mean(r["loss_q"] for r in recs), 4))
print("grad_norm_a mean:", round(statistics.mean(r["grad_norm_a"] for r in recs), 4),
      "| grad_norm_q mean:", round(statistics.mean(r["grad_norm_q"] for r in recs), 4))
cos = [r["cos_gA_gQ"] for r in recs if r["cos_gA_gQ"] is not None]
print("cos(g_A,g_Q) all params: mean", round(statistics.mean(cos), 3),
      "| min", round(min(cos), 3), "| max", round(max(cos), 3),
      "| positive on", sum(1 for c in cos if c > 0), "/", len(cos))
if "cos_gA_gQ_scope" in recs[0] and recs[0]["cos_gA_gQ_scope"] is not None:
    cs = [r["cos_gA_gQ_scope"] for r in recs if r["cos_gA_gQ_scope"] is not None]
    print("cos scoped:", round(statistics.mean(cs), 3), "| positive on",
          sum(1 for c in cs if c > 0), "/", len(cs))
    sl = [r["first_order_slope"] for r in recs]
    print("first-order slope dL_Q/dscale at 0: mean", round(statistics.mean(sl), 5),
          "| negative on", sum(1 for s in sl if s < 0), "/", len(sl))

by = {}
for rec in recs:
    for row in rec["trial_rows"]:
        by.setdefault((row["eta"], row["control"]), []).append(row["delta_q"])

print()
print(f"{'eta':>6} {'control':>9} {'mean dL_Q':>12} {'median':>11} {'steps<0':>9}")
for eta in sorted({e for e, _ in by}):
    for control in ("downhill", "random", "uphill"):
        vals = by[(eta, control)]
        print(f"{eta:>6} {control:>9} {statistics.mean(vals):>12.6f} "
              f"{statistics.median(vals):>11.6f} {sum(1 for v in vals if v < 0):>5}/{len(vals)}")

print()
print("PAIRED (TA3 needs downhill < random and downhill < uphill)")
try:
    from scipy.stats import wilcoxon

    have_scipy = True
except Exception:
    have_scipy = False
for eta in sorted({e for e, _ in by}):
    d, r, u = by[(eta, "downhill")], by[(eta, "random")], by[(eta, "uphill")]
    n = len(d)
    dr = [a - b for a, b in zip(d, r)]
    du = [a - b for a, b in zip(d, u)]
    line = (f"  eta={eta:<5} downhill<random {sum(1 for x in dr if x < 0):>3}/{n}  "
            f"downhill<uphill {sum(1 for x in du if x < 0):>3}/{n}  "
            f"mean(d-r) {statistics.mean(dr):+.6f}  mean(d-u) {statistics.mean(du):+.6f}")
    if have_scipy:
        try:
            line += f"  p(d<r) {wilcoxon(dr, alternative='less').pvalue:.2e}"
        except Exception:
            line += "  p(d<r) n/a"
    print(line)
