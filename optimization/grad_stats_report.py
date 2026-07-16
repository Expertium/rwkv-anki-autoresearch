"""Aggregate an RWKV_GRAD_STATS json into an ablation-target report (Andrew 2026-07-16).

Usage: python optimization/grad_stats_report.py <grad_stats.json> [--top 25]

Two views:
1) LAYER RANKING -- params grouped by structural unit (stream/layer/head/input-FC),
   numel-weighted mean of mean|grad| and mean|grad*w| (saliency). Low saliency = the unit
   barely moves the loss = ablation candidate. |grad*w| is reported alongside plain |grad|
   because at convergence grads -> 0 for important params too; g*w is the first-order
   estimate of the loss change from zeroing the weight (SNIP-style).
2) NO-OP SUSPECTS -- per-tensor near-no-op flags with a TYPE-AWARE reference:
   - additive/general weights (W_*, lora, heads): no-op reference 0 (frac |w|<0.01)
   - norm gains (layer_norm/group_norm .weight): no-op reference 1 (frac |w-1|<0.01)
   - lerp factors (rkvdag_lerp): no-op at 0 OR 1 depending on which input wins -- both
     fractions shown, interpret manually
   - biases: reference 0
   Caveats printed with the report: near-0 pre-norm channels are rescaled downstream
   (not automatically dead); LoRA A~0 with large B must be judged on the PRODUCT; params
   zero-init BY DESIGN (W_o, gate loras) being small late may mean slow growth, not
   uselessness.
"""

import argparse
import json
import re
from collections import defaultdict

STREAMS = ["card", "deck", "note", "preset", "user"]  # architecture.py modules order


def unit_of(name):
    m = re.match(r"rwkv_modules\.(\d)\.blocks\.(\d+)\.(time_mixer|channel_mixer)", name)
    if m:
        return f"{STREAMS[int(m.group(1))]}.L{m.group(2)}.{m.group(3)}"
    m = re.match(r"rwkv_modules\.(\d)\.blocks\.(\d+)", name)
    if m:
        return f"{STREAMS[int(m.group(1))]}.L{m.group(2)}.other"
    m = re.match(r"rwkv_modules\.(\d)", name)
    if m:
        return f"{STREAMS[int(m.group(1))]}.stream_other"
    if name.startswith("features2card"):
        return "input_fc"
    for h in ("head_w", "head_ahead", "head_p", "head_d", "w_linear", "ahead_linear",
              "p_linear", "prehead"):
        if name.startswith(h):
            return "srs_heads"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    with open(args.path) as f:
        stats = json.load(f)

    groups = defaultdict(lambda: {"numel": 0, "g": 0.0, "gw": 0.0, "tensors": 0})
    for name, s in stats.items():
        g = groups[unit_of(name)]
        g["numel"] += s["numel"]
        g["g"] += s["mean_abs_grad"] * s["numel"]
        g["gw"] += s["mean_abs_grad_x_w"] * s["numel"]
        g["tensors"] += 1

    rows = sorted(groups.items(), key=lambda kv: kv[1]["gw"] / kv[1]["numel"])
    print(f"== LAYER RANKING (ascending saliency = best ablation targets first; "
          f"{len(stats)} tensors, {sum(g['numel'] for g in groups.values()):,} params) ==")
    print(f"{'unit':34s} {'numel':>9s} {'mean|g|':>12s} {'mean|g*w|':>12s}")
    for name, g in rows:
        print(f"{name:34s} {g['numel']:9,d} {g['g'] / g['numel']:12.4e} "
              f"{g['gw'] / g['numel']:12.4e}")

    print("\n== NO-OP SUSPECTS (type-aware reference) ==")
    suspects = []
    for name, s in stats.items():
        is_norm = ("norm" in name and name.endswith(".weight"))
        is_lerp = "rkvdag_lerp" in name
        if is_norm:
            score, ref = s["final_frac_within_0.01_of_1"], "==1"
        elif is_lerp:
            score = max(s["final_frac_absw_lt_0.01"], s["final_frac_within_0.01_of_1"])
            ref = "==0|1"
        else:
            score, ref = s["final_frac_absw_lt_0.01"], "==0"
        if score > 0.5:
            suspects.append((score, name, ref, s["numel"], s["mean_abs_grad_x_w"]))
    for score, name, ref, numel, gw in sorted(suspects, reverse=True)[: args.top]:
        print(f"  {score * 100:5.1f}% near-no-op ({ref})  {name}  "
              f"(numel {numel:,}, saliency {gw:.2e})")
    if not suspects:
        print("  none above the 50% threshold")
    print("\ncaveats: pre-norm near-0 channels get rescaled downstream; judge LoRA pairs "
          "on the A@B product; zero-init-by-design params (W_o, gate loras) small late "
          "may be slow growth, not uselessness.")


if __name__ == "__main__":
    main()
