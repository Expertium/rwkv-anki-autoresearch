"""Record champ5k_r1 results: research_log.jsonl entry + tuner-baseline means (5001-5200)."""
import json

def user_means(path, lo, hi):
    lls = [json.loads(l)["metrics"]["LogLoss"] for l in open(path)
           if lo <= json.loads(l)["user"] <= hi]
    return sum(lls) / len(lls), len(lls)

a_all, n_a = user_means("result/RWKV-champ5k_r1.jsonl", 5001, 10000)
i_all, n_i = user_means("result/RWKV-P-champ5k_r1.jsonl", 5001, 10000)
a_tune, n_at = user_means("result/RWKV-champ5k_r1.jsonl", 5001, 5200)
i_tune, n_it = user_means("result/RWKV-P-champ5k_r1.jsonl", 5001, 5200)
print(f"full: ahead {a_all:.6f} (n={n_a})  imm {i_all:.6f} (n={n_i})")
print(f"tuner baseline 5001-5200: ahead {a_tune:.6f} (n={n_at})  imm {i_tune:.6f} (n={n_it})")

rec = {
    "exp": "champ5k_r1",
    "change": "FIRST 5k champion run: train 1-5000 (2ep WS = 13108 steps + 0.5ep decay = 3277), "
              "quant-aware q72u with per-run LEARNABLE codebooks (cb exports at both seams), "
              "champion HPs (peak_lr 1e-3, warmup 200, wd 0.01, clip 0.25), MAX=110000, eval 5001-10000",
    "params": 193724,
    "ahead": round(a_all, 6),
    "imm": round(i_all, 6),
    "d_ahead": round(0.296385 - a_all, 6),
    "d_imm": round(0.264905 - i_all, 6),
    "status": "5k CHAMPION (phase starting point)",
    "note": "PROMOTED to champion_5k.json (ckpt champ5kd_3277.pth + its learned cbs; WS step trace = the "
            "Wilcoxon prune ref). vs d=128 fp target on the same 5000 users: -0.010187 ahead / -0.013418 imm "
            "(paired one-sided Wilcoxon p=1.0 both -- behind the target, as expected for the 15x-smaller "
            "quant-aware model; the phase's job is closing this). Wall-clock: WS 5h00m (~1.36 s/step avg), "
            "decay 72 min, eval 2x-sharded 66 min (+14 min resume) -> ~7.0h clean pipeline. Two latent bugs "
            "hit+fixed: LEARN=1 optim resume param-group mismatch (f71f43b), per-user lmdb env leak in "
            "get_benchmark_info killed eval shard 0 at user 2007 (7d095e3; n=5000 finish gate caught it). "
            "Tuner baseline (5001-5200 subset): ahead %.6f imm %.6f. Next: hp_tuner_5k loop." % (a_tune, i_tune),
}
with open("optimization/research_log.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("research_log.jsonl appended")
