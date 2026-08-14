#!/usr/bin/env python
"""Does the TEACHER's imm carry information about our AHEAD's error that OUR OWN imm does not?

The question (Andrew, 2026-08-13). Distilling an imm head into the ahead head is
privileged-information distillation: imm sees the current row, ahead does not, so imm's soft
target is a less noisy estimate of the same event. iter 46 tested it with OUR OWN imm as teacher
and got a tie; its stated cause was that the teacher shared the trunk and the forward pass, so the
target only re-expressed what the student already computed. The d=128 teacher's imm is a genuinely
different function, so iter 46 does not settle that variant.

But it only earns a 9 h decay run if it is not simply a copy of ours. This decides that offline.

METHOD. Two dumps over the IDENTICAL batch stream (`labels_sum` proves identity per step):
teacher = C:\\rwkv_kd_dump\\t128_seedpair_65k, ours = a fresh dump of the champion. Per labelled
review, with y = 1[rating >= 2] (rating 1 = Again = fail):

    o_ahead  our curve probability          -- the STUDENT
    o_imm    our imm P(recall) = 1 - P(Again)   -- iter 46's teacher
    t_imm    the d=128 teacher's imm P(recall)  -- the proposed teacher

Then regress our ahead's residual r = y - o_ahead on the two candidate signals:

    step 1: r ~ (o_imm - o_ahead)                 -- what iter 46 already had
    step 2: r ~ (o_imm - o_ahead) + (t_imm - o_ahead)

**The INCREMENTAL R-squared of step 2 over step 1 is the answer.** ~0 => the teacher's imm is
redundant given ours, iter 46 covers it, and the branch closes without spending GPU. Materially
positive => the teacher's imm points at errors ours cannot see, and the run is justified.

Usage: python imm_independence.py [teacher_dir] [ours_dir] [--max-steps N]
"""
import glob
import os
import sys

import numpy as np
import torch

T_DEFAULT = r"C:\rwkv_kd_dump\t128_seedpair_65k"
O_DEFAULT = r"C:\rwkv_kd_dump\ours_i45_immcorr"


def p_recall_from_imm(p_imm_all):
    """P(recall) = 1 - P(Again). p_imm_all is already a probability simplex over 4 ratings."""
    a = p_imm_all.float().numpy()
    s = a.sum(-1)
    if not np.allclose(s[np.isfinite(s)], 1.0, atol=2e-2):
        print(f"  WARNING: p_imm_all rows do not sum to 1 (mean {np.nanmean(s):.4f}) -- "
              f"treating as logits and softmaxing")
        a = torch.softmax(torch.from_numpy(a), dim=-1).numpy()
    return 1.0 - a[..., 0]


def logloss(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def r2_of(X, r):
    """OLS R^2 of r on [1, X]."""
    A = np.column_stack([np.ones(len(r))] + [x for x in X])
    beta, *_ = np.linalg.lstsq(A, r, rcond=None)
    resid = r - A @ beta
    ss_tot = ((r - r.mean()) ** 2).sum()
    return 1.0 - (resid ** 2).sum() / ss_tot, beta


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tdir = args[0] if args else T_DEFAULT
    odir = args[1] if len(args) > 1 else O_DEFAULT
    max_steps = 10 ** 9
    if "--max-steps" in sys.argv:
        max_steps = int(sys.argv[sys.argv.index("--max-steps") + 1])

    osteps = {int(os.path.basename(p)[5:-3]): p for p in glob.glob(os.path.join(odir, "step_*.pt"))}
    tsteps = {int(os.path.basename(p)[5:-3]): p for p in glob.glob(os.path.join(tdir, "step_*.pt"))}
    common = sorted(set(osteps) & set(tsteps))[:max_steps]
    if not common:
        raise SystemExit(f"no common steps between {tdir} and {odir}")
    print(f"{len(common)} common steps (teacher {len(tsteps)}, ours {len(osteps)})")

    O_A, O_I, T_I, Y = [], [], [], []
    mismatch = 0
    for st in common:
        o = torch.load(osteps[st], weights_only=False)
        t = torch.load(tsteps[st], weights_only=False)
        if abs(o["labels_sum"] - t["labels_sum"]) > 1e-3:
            mismatch += 1
            continue
        if "has_label" not in o:
            raise SystemExit("our dump lacks labels -- regenerate with RWKV_KD_DUMP_LABELS=1")
        # ⚠ p_curve is valid ONLY on ahead rows and p_imm ONLY on query rows -- DISJOINT sets.
        # The eval joins them by label_review_th (srs_model.py ~1352) and so must we; the first
        # version of this script masked both to has_label and silently compared different reviews
        # (its tell was "our ahead" scoring 1.98 logloss against an eval value of 0.298).
        if "label_review_th" not in o:
            raise SystemExit("dump lacks label_review_th -- regenerate (RWKV_KD_DUMP_LABELS=1)")
        hl = o["has_label"].numpy().astype(bool)
        isq = o["is_query"].numpy().astype(bool)
        a_m, q_m = hl & ~isq, hl & isq
        if a_m.sum() == 0 or q_m.sum() == 0:
            continue
        rth = o["label_review_th"].numpy()
        oa_p = o["p_curve"].float().numpy()
        oi_p = p_recall_from_imm(o["p_imm_all"])
        ti_p = p_recall_from_imm(t["p_imm_all"])
        rat = o["label_rating"].numpy()
        # join ahead-row -> query-row by review index (the teacher shares the row layout exactly,
        # since both dumps walk the identical batch stream, so its query rows are the same rows)
        qidx = {int(r): i for i, r in zip(np.flatnonzero(q_m), rth[q_m])}
        ai = np.flatnonzero(a_m)
        pair = [(i, qidx[int(rth[i])]) for i in ai if int(rth[i]) in qidx]
        if not pair:
            continue
        ia = np.array([p[0] for p in pair]); iq = np.array([p[1] for p in pair])
        oa = oa_p.reshape(-1)[ia]; oi = oi_p.reshape(-1)[iq]; ti = ti_p.reshape(-1)[iq]
        y = (rat.reshape(-1)[ia] >= 2).astype(np.float64)
        ok = np.isfinite(oa) & np.isfinite(oi) & np.isfinite(ti)
        O_A.append(oa[ok]); O_I.append(oi[ok]); T_I.append(ti[ok]); Y.append(y[ok])
    if mismatch:
        print(f"  WARNING: {mismatch} steps dropped on labels_sum mismatch (stream drift)")
    o_ahead = np.concatenate(O_A).astype(np.float64)
    o_imm = np.concatenate(O_I).astype(np.float64)
    t_imm = np.concatenate(T_I).astype(np.float64)
    y = np.concatenate(Y)
    print(f"{len(y):,} labelled reviews; base rate {y.mean():.4f}\n")

    print("logloss on these reviews (sanity -- ahead is a HARDER task, not a worse model):")
    for name, p in (("our ahead", o_ahead), ("our imm", o_imm), ("teacher imm", t_imm)):
        print(f"  {name:12s} {logloss(p, y):.5f}")

    print(f"\ncorr(teacher imm, our imm)      = {np.corrcoef(t_imm, o_imm)[0,1]:.4f}")
    print(f"corr(teacher imm, our ahead)    = {np.corrcoef(t_imm, o_ahead)[0,1]:.4f}")
    print(f"corr(our imm,     our ahead)    = {np.corrcoef(o_imm, o_ahead)[0,1]:.4f}")

    r = y - o_ahead
    d_ours = o_imm - o_ahead
    d_teach = t_imm - o_ahead
    r2_1, b1 = r2_of([d_ours], r)
    r2_2, b2 = r2_of([d_ours, d_teach], r)
    r2_t, _ = r2_of([d_teach], r)
    print(f"\nexplaining our AHEAD's residual r = y - p_ahead:")
    print(f"  R^2 with our imm alone         = {r2_1:.5f}   (this is what iter 46 had)")
    print(f"  R^2 with teacher imm alone     = {r2_t:.5f}")
    print(f"  R^2 with BOTH                  = {r2_2:.5f}")
    print(f"  ** INCREMENTAL R^2 of the teacher over ours = {r2_2 - r2_1:.5f} **")
    print(f"  (fitted weights with both: our imm {b2[1]:+.4f}, teacher imm {b2[2]:+.4f})")

    inc = r2_2 - r2_1
    print()
    if inc < 0.002:
        print("VERDICT: the teacher's imm is REDUNDANT given ours -- iter 46 already covers this")
        print("         branch. Close it; do not spend a decay run.")
    else:
        print("VERDICT: the teacher's imm explains error our own imm cannot -- the independence")
        print("         mechanism is real here, and the decay run is justified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
