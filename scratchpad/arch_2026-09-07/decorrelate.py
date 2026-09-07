"""Do two STRUCTURALLY DIFFERENT models make the SAME errors on ahead? The decisive test for
"is the remaining ahead loss noise, or a blind spot shared by this architecture family?"

WHY THIS QUESTION. The 2026-07-03 entropy-floor estimate (research_5k_notes.md task #18) put the
AHEAD floor at 0.2994 with the two d=128 models at 0.2992/0.2993 -- i.e. no gap -- because the
estimator COLLAPSED: cross-model residual covariance 0.0950 vs each model's own Brier 0.0955, so two
disjoint-trained models disagreed on only ~1% of Brier. That note states the limit plainly: the
estimator cannot separate TRUE NOISE from a blind spot SHARED by the family, and two models of the
same family erring identically is exactly what a shared blind spot looks like. The way to tell them
apart is to add a structurally different model and see whether its errors DECORRELATE.

THE PAIR USED HERE (the most different pair available without new training):
  A = realcyc      d=80, H=5, 563,652 params, gen-5 REAL-TIMESTAMP features (109 dims), 2026 recipe
  B = the pretrained d=128 model (RWKV_trained_on_101_4999.pth), 2,762,884 params, PUBLISHED
      features (92 dims), upstream's ~12-epoch recipe
5x the parameters, different feature layout, different training era. Same RWKV-7 family, which is the
honest limit of an in-house contrast -- a genuinely out-of-family contrast (FSRS-6) needs per-review
predictions from srs-benchmark and is Andrew's repo to run.

USERS: VAL half (5001+), where BOTH models are out-of-sample -- the same design as the 2026-07-03
estimate, which scored users unseen by both. ⚠ This is a DIAGNOSTIC: it selects no candidate, sets no
threshold and enters no gate, so it does not tune on VAL.

ALIGNMENT: each model reads its own dataset (-id vs published), and the two share `review_th`, the
per-user review index. Row counts are identical user-for-user (verified 2026-07-26 over 6 users /
363,598 reviews) and day_offset differs on 0.001%, so review_th aligns; the script INTERSECTS on it
and reports how many rows survived, so a silent misalignment shows up as a low intersection.

REPORTED: per-model Brier and BCE; the cross-model residual covariance E[(y-pA)(y-pB)] against each
model's own Brier (the 2026-07-03 statistic); the correlation of residuals; and the oracle 50/50
ensemble, whose gain over the better model is what decorrelation is worth in the metric's own units.

Usage: decorrelate.py dump  A|B <users...>     -- one arm, writes a per-arm npz
       decorrelate.py cmp                      -- compare the two arms
"""
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "cmp"
OUT = "scratchpad/arch_2026-09-07"

if MODE == "dump":
    arm = sys.argv[2]
    users = [int(u) for u in sys.argv[3:]] or [5001, 5002, 5003, 5004]
    os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")
    sys.path.insert(0, os.getcwd())
    common = {"RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_ZERO_FEATURES": ""}
    if arm == "A":
        env = dict(common, **{
            "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
            "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
            "RWKV_STRIP_L0_VLORA": "1", "RWKV_STATE_CLAMP_TAU": "300",
            "RWKV_STATE_CLAMP_WINDOW": "32768",
            "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
            "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1"})
        ckpt, data = "scratchpad/realcyc/rc_d_10935.pth", r"C:\Users\Andrew\anki-revlogs-10k-id"
    else:
        env = dict(common, **{"RWKV_ARCH_MODULE": "scratchpad/architecture_old_d128.py"})
        ckpt, data = "pretrain/RWKV_trained_on_101_4999.pth", r"C:\Users\Andrew\anki-revlogs-10k"
    for k, v in env.items():
        os.environ[k] = v
    os.environ["RWKV_CHAMP_CKPT"] = ckpt
    from pathlib import Path

    import numpy as np
    import torch

    torch.set_num_threads(4)
    import rwkv.run_as_rnn as rnn_mod
    from rwkv.data_processing import get_rwkv_data

    RTH, P, Y, T, U = [], [], [], [], []
    for uid in users:
        torch.manual_seed(uid)
        df = get_rwkv_data(Path(data), uid).sort_values("review_th", kind="stable").reset_index(drop=True)
        proc = rnn_mod.RNNProcess(path=ckpt, device=torch.device("cpu"), dtype=torch.float32)
        with torch.inference_mode():
            for row in df.to_dict("records"):
                curve, _ = proc.run(row, skip=False)
                if int(row["has_label"]) == 1:
                    t = float(row["label_elapsed_seconds"])
                    RTH.append(int(row["review_th"])); P.append(float(proc.predict_func(curve, t)))
                    Y.append(float(row["label_y"])); T.append(t); U.append(uid)
        print(f"arm {arm} user {uid}: {len(df):,} rows -> {len(P):,} labelled", flush=True)
    f = f"{OUT}/decor_{arm}.npz"
    np.savez_compressed(f, rth=np.array(RTH), p=np.array(P), y=np.array(Y), t=np.array(T), u=np.array(U))
    print(f"wrote {f}: {len(P):,} rows")
    raise SystemExit(0)

import numpy as np

A = np.load(f"{OUT}/decor_A.npz")
B = np.load(f"{OUT}/decor_B.npz")
EPS = 1e-6


def bce(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


keyA = {(int(u), int(r)): i for i, (u, r) in enumerate(zip(A["u"], A["rth"]))}
ia, ib = [], []
for j, (u, r) in enumerate(zip(B["u"], B["rth"])):
    i = keyA.get((int(u), int(r)))
    if i is not None:
        ia.append(i); ib.append(j)
ia, ib = np.array(ia), np.array(ib)
pA, pB, y = A["p"][ia], B["p"][ib], A["y"][ia]
ymatch = float((A["y"][ia] == B["y"][ib]).mean())
print(f"arm A rows {len(A['p']):,}   arm B rows {len(B['p']):,}   intersected {len(ia):,}")
print(f"label agreement on the intersection: {ymatch:.4f}  (must be ~1.0, else review_th does not align)")
if ymatch < 0.99:
    print("NO VERDICT -- the two arms are not scoring the same reviews.")
    raise SystemExit(3)
rA, rB = y - pA, y - pB
brA, brB = float((rA ** 2).mean()), float((rB ** 2).mean())
cov = float((rA * rB).mean())
print(f"\nA (realcyc, 563k, gen-5 features): BCE {bce(pA, y):.5f}  Brier {brA:.5f}")
print(f"B (d=128 pretrained, 2.76M, published): BCE {bce(pB, y):.5f}  Brier {brB:.5f}")
print(f"\ncross-model residual covariance E[(y-pA)(y-pB)] = {cov:.5f}")
print(f"  vs A's Brier {brA:.5f} (ratio {cov/brA:.4f}) and B's Brier {brB:.5f} (ratio {cov/brB:.4f})")
print(f"  residual correlation = {float(np.corrcoef(rA, rB)[0,1]):.4f}")
print(f"  prediction correlation = {float(np.corrcoef(pA, pB)[0,1]):.4f}")
ens = bce(0.5 * (pA + pB), y)
print(f"\n50/50 ensemble BCE {ens:.5f}  vs best single {min(bce(pA,y), bce(pB,y)):.5f}  "
      f"=> ensemble gain {min(bce(pA,y), bce(pB,y)) - ens:+.5f}")
print("\nREADING: covariance/Brier ~1.0 => the two models err on the SAME reviews; the 2026-07-03")
print("'family saturated' result then extends across a 5x parameter change AND a feature-layout")
print("change, which is strong evidence the residual is noise or a deep shared blind spot.")
print("A ratio meaningfully below 1, or a real ensemble gain, means there IS structure one model")
print("captures and the other does not -- and an architecture that captures both has headroom.")
