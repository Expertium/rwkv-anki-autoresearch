"""Compare the fresh d=128 VAL-half eval against (a) the 2026-07-03 run and (b) our models.

(a) is a DRIFT CHECK, and it is the point of re-running a number we already had: if current code
no longer reproduces the July baseline, the phase's "we are ~0.0037 behind upstream" anchor has
moved and every comparison resting on it needs revisiting. Reports the per-user max |diff| as
well as the mean, because a mean can hide a few users moving a lot.

(b) is the comparison Andrew asked for. ⚠ It is UNRECTIFIED-vs-UNRECTIFIED by construction: the
d=128 model is scored as intended (no rectifier, piecewise correction ON), so it must be paired
with our UNRECTIFIED numbers, never the rectified deploy ones.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAL = (5001, 7500)


def load(tag, mode):
    p = os.path.join(ROOT, "result", f"RWKV-{tag}.jsonl" if mode == "ahead" else f"RWKV-P-{tag}.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        u = int(r["user"])
        if VAL[0] <= u <= VAL[1]:
            out[u] = r["metrics"]["LogLoss"]
    return out


def mean(d):
    return sum(d.values()) / len(d) if d else float("nan")


def main():
    new = {m: load("base128_val", m) for m in ("ahead", "imm")}
    if not new["ahead"]:
        raise SystemExit("no base128_val results found")
    print(f"d=128 FRESH (5001-7500, n={len(new['ahead'])}): "
          f"ahead {mean(new['ahead']):.6f}  imm {mean(new['imm']):.6f}")

    old = {m: load("base5k", m) for m in ("ahead", "imm")}
    if old["ahead"]:
        print(f"d=128 2026-07-03 same range: ahead {mean(old['ahead']):.6f}  "
              f"imm {mean(old['imm']):.6f}")
        print("--- DRIFT CHECK (fresh - July; ~0 means current code still reproduces it) ---")
        for m in ("ahead", "imm"):
            common = sorted(set(new[m]) & set(old[m]))
            if not common:
                continue
            dm = mean({u: new[m][u] for u in common}) - mean({u: old[m][u] for u in common})
            worst = max(common, key=lambda u: abs(new[m][u] - old[m][u]))
            print(f"  {m:5s} mean diff {dm:+.6f}   max per-user |diff| "
                  f"{abs(new[m][worst]-old[m][worst]):.6f} (user {worst})   n={len(common)}")
            if abs(dm) > 1e-4:
                print(f"  ⚠ {m}: mean moved by more than 1e-4 -- the July anchor does NOT "
                      f"reproduce under current code. Do not use it until explained.")
    else:
        print("(no 2026-07-03 base5k jsonl found -- drift check skipped)")

    print("--- vs our models on the SAME range, UNRECTIFIED both sides ---")
    for tag, label in (("iter31_algo", "iter 31"), ("iter32_kd", "iter 32 champion")):
        ours = {m: load(tag, m) for m in ("ahead", "imm")}
        if not ours["ahead"]:
            continue
        for m in ("ahead", "imm"):
            common = sorted(set(new[m]) & set(ours[m]))
            if not common:
                continue
            g = mean({u: ours[m][u] for u in common}) - mean({u: new[m][u] for u in common})
            print(f"  {label:16s} {m:5s} {mean({u: ours[m][u] for u in common}):.6f}  "
                  f"gap to d=128 {g:+.6f}  (positive = d=128 better)")


if __name__ == "__main__":
    main()
