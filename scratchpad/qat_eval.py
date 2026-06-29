"""Evaluate a QAT'd model deployed with state quant vs (a) its own fp32, (b) the iter36 champion
fp32, and gate vs iter0. Reads three rust pred sets for the 17 gate users:
  reference/rust_pred_fp32_{u}.json    = iter36 champion fp32 (existing baseline)
  reference/rust_pred_qatfp32_{u}.json = QAT model, NO quant
  reference/rust_pred_qatq_{u}.json    = QAT model + deploy state quant (card int2 + note int4)
Reports by-user-mean imm/ahead LogLoss for each + the key deltas + the iter0 gate.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

REF = Path("reference")
ITER0_IMM, ITER0_AHEAD, BUDGET = 0.319475, 0.374046, 0.0015
USERS = [107, 110, 116, 120, 121, 128, 136, 146, 150, 151, 156, 159, 162, 165, 175, 176, 187]


def load_preds(path):
    d = json.load(open(path))
    rth = d["review_th"]
    imm = {rt: p for rt, p in zip(rth, d["pred_imm"])}
    ahead = {rt: p for rt, p in zip(rth, d["pred_ahead"]) if p is not None}
    return imm, ahead


def ll(label_bin, predmap, keys):
    return log_loss([label_bin[rt] for rt in keys], [predmap[rt] for rt in keys], labels=[0, 1])


def main():
    sets = {"champ_fp32": "rust_pred_fp32", "qat_fp32": "rust_pred_qatfp32", "qat_quant": "rust_pred_qatq"}
    agg = {k: {"imm": [], "ahead": []} for k in sets}
    used = []
    for u in USERS:
        meta_p = REF / f"trace_user_{u}.json"
        if not meta_p.exists() or not all((REF / f"{p}_{u}.json").exists() for p in sets.values()):
            continue
        meta = json.load(open(meta_p))
        eq = meta["equalize_review_ths"]
        label_rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
        label_bin = {rt: int(np.clip(label_rating[rt], 0, 1)) for rt in eq}
        preds = {k: load_preds(REF / f"{p}_{u}.json") for k, p in sets.items()}
        # common keys across all sets
        imm_keys = [rt for rt in eq if all(rt in preds[k][0] for k in sets)]
        ah_keys = [rt for rt in eq if all(rt in preds[k][1] for k in sets)]
        for k in sets:
            agg[k]["imm"].append(ll(label_bin, preds[k][0], imm_keys))
            agg[k]["ahead"].append(ll(label_bin, preds[k][1], ah_keys))
        used.append(u)

    m = {k: {kk: np.mean(v) for kk, v in agg[k].items()} for k in sets}
    print(f"users: {len(used)}  {used}")
    print(f"{'set':<12} {'imm':>10} {'ahead':>10}")
    for k in ["champ_fp32", "qat_fp32", "qat_quant"]:
        print(f"{k:<12} {m[k]['imm']:>10.6f} {m[k]['ahead']:>10.6f}")
    print()
    print("KEY DELTAS:")
    print(f"  QAT fine-tune effect (qat_fp32 - champ_fp32):   imm {m['qat_fp32']['imm']-m['champ_fp32']['imm']:+.6f}  ahead {m['qat_fp32']['ahead']-m['champ_fp32']['ahead']:+.6f}")
    print(f"  pure quant cost on QAT (qat_quant - qat_fp32):  imm {m['qat_quant']['imm']-m['qat_fp32']['imm']:+.6f}  ahead {m['qat_quant']['ahead']-m['qat_fp32']['ahead']:+.6f}")
    print(f"  deployed vs champ fp32 (qat_quant - champ_fp32):imm {m['qat_quant']['imm']-m['champ_fp32']['imm']:+.6f}  ahead {m['qat_quant']['ahead']-m['champ_fp32']['ahead']:+.6f}")
    print()
    iq, aq = m["qat_quant"]["imm"], m["qat_quant"]["ahead"]
    print(f"GATE vs iter0 (+{BUDGET}):  imm {iq:.6f} <= {ITER0_IMM+BUDGET:.6f} {'PASS' if iq<=ITER0_IMM+BUDGET else 'FAIL'}"
          f"  |  ahead {aq:.6f} <= {ITER0_AHEAD+BUDGET:.6f} {'PASS' if aq<=ITER0_AHEAD+BUDGET else 'FAIL'}")
    # compare to PTQ card int4+note int4 (the non-QAT version of a similar-aggression config)
    print("REFERENCE: PTQ (iter36) card int4+note int4 deployed = imm 0.299632 (+0.003569 over champ fp32); "
          "card int2+note int4 PTQ would be worse. QAT target: get qat_quant imm near champ_fp32 (0.296064).")


if __name__ == "__main__":
    main()
