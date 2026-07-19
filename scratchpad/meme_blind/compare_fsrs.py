"""Blind-RWKV vs FSRS-7 paired comparison (meme run record, 2026-07-19)."""
import json
import statistics

from scipy.stats import wilcoxon

FSRS_PATH = r"C:\Users\Andrew\srs-benchmark\result\FSRS-7-sched_penalties-short-secs-recency.jsonl"

fsrs = {r["user"]: r["metrics"]["LogLoss"]
        for r in map(json.loads, open(FSRS_PATH, encoding="utf-8"))
        if 5001 <= r["user"] <= 10000}
blind = {r["user"]: r["metrics"]["LogLoss"]
         for r in map(json.loads, open("result/RWKV-meme_blind.jsonl", encoding="utf-8"))}
blind_imm = {r["user"]: r["metrics"]["LogLoss"]
             for r in map(json.loads, open("result/RWKV-P-meme_blind.jsonl", encoding="utf-8"))}

keys = sorted(set(fsrs) & set(blind))
print("common users:", len(keys))

for name, model in [("blind-RWKV ahead", blind), ("blind-RWKV imm", blind_imm)]:
    d = [fsrs[k] - model[k] for k in keys]  # positive = blind better on that user
    mean_f = statistics.mean(fsrs[k] for k in keys)
    mean_m = statistics.mean(model[k] for k in keys)
    wins = sum(1 for x in d if x > 0)
    p_fsrs_better = wilcoxon(d, alternative="less").pvalue
    print(f"{name}: mean {mean_m:.6f} vs FSRS-7 {mean_f:.6f} -> "
          f"delta {mean_m - mean_f:+.6f} (positive = FSRS wins); "
          f"blind wins {wins}/{len(keys)} users ({100 * wins / len(keys):.1f}%); "
          f"wilcoxon p(FSRS better) {p_fsrs_better:.3e}")
