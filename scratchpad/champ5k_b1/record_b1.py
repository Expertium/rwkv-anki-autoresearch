"""Record champ5k_b1 (budget A/B winner): research_log entry + tuner baseline means (5001-5200)."""
import json

def user_means(path, lo, hi):
    lls = [json.loads(l)["metrics"]["LogLoss"] for l in open(path)
           if lo <= json.loads(l)["user"] <= hi]
    return sum(lls) / len(lls), len(lls)

a_all, n_a = user_means("result/RWKV-champ5k_b1.jsonl", 5001, 10000)
i_all, n_i = user_means("result/RWKV-P-champ5k_b1.jsonl", 5001, 10000)
a_tune, n_at = user_means("result/RWKV-champ5k_b1.jsonl", 5001, 5200)
i_tune, n_it = user_means("result/RWKV-P-champ5k_b1.jsonl", 5001, 5200)
assert n_a == 5000 and n_i == 5000 and n_at == 200 and n_it == 200
print(f"full: ahead {a_all:.6f}  imm {i_all:.6f}")
print(f"tuner baseline 5001-5200: ahead {a_tune:.6f}  imm {i_tune:.6f}")

rec = {
    "exp": "champ5k_b1",
    "change": "BUDGET A/B: champion recipe at HALF budget -- WS 1 ep (6554 steps) + 0.25 ep decay "
              "(1638), everything else identical to champ5k_r1 (champion HPs, q72u learnable cbs)",
    "params": 193724,
    "ahead": round(a_all, 6),
    "imm": round(i_all, 6),
    "d_ahead": round(0.306572 - a_all, 6),
    "d_imm": round(0.278323 - i_all, 6),
    "status": "NEW CHAMPION (size/speed: budget halved)",
    "note": "vs champ5k_r1 paired on the same 5000 users: ahead -0.000058 (p=0.31, indistinguishable), "
            "imm +0.000430 BETTER (p=6.1e-62). The 2nd WS epoch (same 5000 users reshuffled) adds "
            "NOTHING -- consistent with the data-variety-beats-repetition lesson. ADOPTED (Andrew's "
            "rule): WS 1 ep + ratio-0.25 decay is now the budget for ALL 5k runs (tuner trials AND "
            "research runs); champion runs drop ~7h -> ~3.5h. SIZE/SPEED-class accept (efficiency "
            "budget, p-gate exempt; imm improvement is a bonus). Promoted: ckpt champ5kb1d_1638.pth + "
            "its learned cbs; 6554-step WS trace = the new prune ref. Wall-clock: WS 2h27m, decay 37m, "
            "eval 89m (2-sharded).",
}
with open("optimization/research_log.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("research_log.jsonl appended")
