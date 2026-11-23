"""Bucket the DEPLOY quantization tax by a card's answer count.

Why (feedback, 2026-11-23): a dataset-wide average hides the damage. The other team's int4-everywhere
state cost +0.015% on the whole set but +0.18%..+0.74% on cards with 128+ answers, because a state
that is re-encoded at every review compounds its error over the card's life. This tool splits the
Rust deploy engine's per-review tax into their answer-count buckets (1-2, 3-4, 5-8, 9-15, 16-31,
32-127, 128+) and adds the per-review |delta p| readout (p50 / P99 / max, share of reviews and share
of cards moving by more than 0.05 and 0.10).

Input: `rust_pred_<u>.json` per config (the engine's per-review predictions) + the review log, whose
row order defines `review_th`. The bucket of review row i is (number of earlier rows of the same
card) + 1, i.e. the ordinal of that answer within the card.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/bucket_tax.py \
      --weights scratchpad/quant_plan/rust_artifacts/it36l2i8_124.safetensors \
      --trace-dir scratchpad/quant_plan/trace_corrupt50 \
      --configs plain,l2i8s4,l2i8fb4n --tag bucket17
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_grid as rg  # noqa: E402
import rust_score  # noqa: E402

DATA = Path(r"C:\Users\Andrew\anki-revlogs-10k")
EDGES = [(1, 2), (3, 4), (5, 8), (9, 15), (16, 31), (32, 127), (128, 1 << 30)]


def bucket_name(ordinal):
    for lo, hi in EDGES:
        if lo <= ordinal <= hi:
            return f"{lo}-{hi}" if hi < (1 << 30) else "128+"
    return "0"


def answer_ordinals(user):
    """1-based ordinal of each review within its card, in revlog order (== review_th order)."""
    df = pd.read_parquet(DATA / "revlogs" / f"user_id={user}", columns=["card_id"])
    return (df.groupby("card_id").cumcount() + 1).to_numpy()


def load_preds(preds_dir, users):
    """preds_dir holds rust_pred_<u>.json for ONE config."""
    out = {}
    for u in users:
        p = Path(preds_dir) / f"rust_pred_{u}.json"
        if p.exists():
            out[u] = json.loads(p.read_text(encoding="utf-8"))
    return out


def review_table(trace_dir, preds_imm, preds_ahead, config, users):
    """One row per (user, review_th) that exists in the config's predictions."""
    rows = []
    for u in users:
        r = preds_imm.get(u)
        if r is None:
            continue
        ordn = answer_ordinals(u)
        meta = json.loads((trace_dir / f"trace_user_{u}.json").read_text(encoding="utf-8"))
        rating = {int(k): int(v) for k, v in meta["label_rating"].items()}
        ahead = dict(zip(preds_ahead[u]["review_th"], preds_ahead[u]["pred_ahead"])) if u in preds_ahead else {}
        rows.append(pd.DataFrame({
            "user": u,
            "review_th": r["review_th"],
            "p_imm": r["pred_imm"],
            "p_ahead": [ahead.get(rt) for rt in r["review_th"]],
            "y": [int(np.clip(rating.get(rt, 0), 0, 1)) for rt in r["review_th"]],
            "ordinal": [int(ordn[rt - 1]) if rt - 1 < len(ordn) else -1 for rt in r["review_th"]],
        }))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    df["config"] = config
    return df


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bucket_rows(dfq, dfp):
    """Paired per-review frame: plain vs quantized, with the loss delta and the |delta p| columns."""
    keys = ["user", "review_th"]
    cols = ["p_imm", "p_ahead", "y", "ordinal"]
    m = dfp[keys + cols].merge(dfq[keys + cols], on=keys, suffixes=("_p", "_q"))
    m["bucket"] = [bucket_name(o) for o in m["ordinal_p"]]
    m["dimm"] = logloss(m["p_imm_q"], m["y_p"]) - logloss(m["p_imm_p"], m["y_p"])
    ok_a = m["p_ahead_p"].notna() & m["p_ahead_q"].notna()
    m["dahead"] = np.where(ok_a, logloss(m["p_ahead_q"], m["y_p"]) - logloss(m["p_ahead_p"], m["y_p"]), np.nan)
    m["dimm_p"] = (m["p_imm_q"] - m["p_imm_p"]).abs()
    m["dahead_p"] = np.where(ok_a, (m["p_ahead_q"] - m["p_ahead_p"]).abs(), np.nan)
    m["limm_p"] = logloss(m["p_imm_p"], m["y_p"])
    m["limm_q"] = logloss(m["p_imm_q"], m["y_p"])
    m["lahead_p"] = np.where(ok_a, logloss(m["p_ahead_p"], m["y_p"]), np.nan)
    m["lahead_q"] = np.where(ok_a, logloss(m["p_ahead_q"], m["y_p"]), np.nan)
    return m


def move_stats(d):
    a = np.asarray(d, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "p50": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()),
            "share_gt_0.05": float((a > 0.05).mean()), "share_gt_0.10": float((a > 0.10).mean())}


def summarise(m, label):
    """Loss / relative-% / |delta p| summary for one subset of the paired frame."""
    out = {"n_reviews": int(len(m)), "n_users": int(m["user"].nunique())}
    out["plain_imm"] = float(m["limm_p"].mean())
    out["quant_imm"] = float(m["limm_q"].mean())
    out["d_imm"] = float(m["dimm"].mean())
    out["rel_imm_pct"] = 100.0 * out["d_imm"] / out["plain_imm"] if out["plain_imm"] else float("nan")
    a = m[m["p_ahead_p"].notna() & m["p_ahead_q"].notna()]
    out["n_ahead"] = int(len(a))
    if len(a):
        out["plain_ahead"] = float(a["lahead_p"].mean())
        out["quant_ahead"] = float(a["lahead_q"].mean())
        out["d_ahead"] = float(a["dahead"].mean())
        out["rel_ahead_pct"] = 100.0 * out["d_ahead"] / out["plain_ahead"] if out["plain_ahead"] else float("nan")
    out["move_imm"] = move_stats(m["dimm_p"])
    out["move_ahead"] = move_stats(m["dahead_p"])
    out["label"] = label
    return out


def card_share(m, col, thr):
    """Share of (user, card) pairs with at least one review moving more than `thr` on `col`."""
    d = m[["user", "card_id", col]].dropna()
    if d.empty:
        return float("nan")
    hit = d.groupby(["user", "card_id"])[col].max() > thr
    return float(hit.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "scratchpad/quant_plan/rust_artifacts/it36l2i8_124.safetensors"))
    ap.add_argument("--wmap", default="", help="per-config weights: name=path,name=path (else --weights)")
    ap.add_argument("--preds-map", default="",
                    help="reuse an existing prediction dir for a config: name=dir (that config is not re-run)")
    ap.add_argument("--trace-dir", default="scratchpad/quant_plan/trace_corrupt50")
    ap.add_argument("--configs", default="plain,l2i8s4,l2i8fb4n")
    ap.add_argument("--users", default="")
    ap.add_argument("--preds-root", default=str(ROOT / "scratchpad/quant_plan/rust_grid_preds"))
    ap.add_argument("--tag", default="bucket")
    ap.add_argument("--no-run", action="store_true", help="reuse existing predictions")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    trace_dir = ROOT / args.trace_dir
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    if "plain" not in names:
        names = ["plain"] + names
    if args.users:
        users = [int(u) for u in args.users.split(",")]
    else:
        users = sorted(int(p.stem.split("_")[-1]) for p in trace_dir.glob("trace_user_*.json"))
    preds_root = Path(args.preds_root) / args.tag
    wmap = {}
    for spec in (s.strip() for s in args.wmap.split(",") if s.strip()):
        k, _, v = spec.partition("=")
        wmap[k] = Path(v)
    pmap = {}
    for spec in (s.strip() for s in args.preds_map.split(",") if s.strip()):
        k, _, v = spec.partition("=")
        pmap[k] = Path(v) if Path(v).is_absolute() else ROOT / v
    if not args.no_run:
        for name in names:
            if name in pmap:
                print(f"{name:14s} reused from {pmap[name]}", flush=True)
                continue
            w = wmap.get(name, Path(args.weights))
            pdir, err = rg.run(w, trace_dir, users, name, preds_root)
            print(f"{name:14s} {'OK' if not err else 'FAILED: ' + err}", flush=True)

    frames = {}
    for name in names:
        pr = load_preds(pmap.get(name, preds_root / name), users)
        if not pr:
            print(f"{name}: no predictions")
            continue
        frames[name] = review_table(trace_dir, pr, pr, name, users)
        print(f"{name:14s} rows={len(frames[name])} users={len(pr)}", flush=True)
    if "plain" not in frames:
        print("no plain reference - abort")
        return

    report = {"tag": args.tag, "trace_dir": args.trace_dir, "weights": str(args.weights),
              "users": users, "n_users": len(users), "buckets": [b for b in
              (f"{lo}-{hi}" if hi < (1 << 30) else "128+" for lo, hi in EDGES)], "configs": {}}

    # card identity for the per-card share: (user, card_id). Rebuild it here from the review log.
    cards = {}
    for u in users:
        df = pd.read_parquet(DATA / "revlogs" / f"user_id={u}", columns=["card_id"])
        cards[u] = df["card_id"].to_numpy()

    for name in names:
        if name == "plain":
            continue
        m = bucket_rows(frames[name], frames["plain"])
        m["card_id"] = [cards[u][rt - 1] for u, rt in zip(m["user"], m["review_th"])]
        bits = rg.bits_for(rg.CONFIGS[name])
        rec = {"bits": bits, "all": summarise(m, "all"),
               "buckets": {}, "per_user": {}, "cards_imm_gt_0.05": card_share(m, "dimm_p", 0.05),
               "cards_imm_gt_0.10": card_share(m, "dimm_p", 0.10),
               "cards_ahead_gt_0.05": card_share(m, "dahead_p", 0.05),
               "cards_ahead_gt_0.10": card_share(m, "dahead_p", 0.10)}
        for b in report["buckets"]:
            sub = m[m["bucket"] == b]
            if len(sub):
                rec["buckets"][b] = summarise(sub, b)
        for u, sub in m.groupby("user"):
            rec["per_user"][str(int(u))] = {"n": int(len(sub)), "d_imm": float(sub["dimm"].mean()),
                                            "d_ahead": float(sub["dahead"].mean())}
        report["configs"][name] = rec

        print(f"\n=== {name}  card={bits['card']}b note={bits['note']}b")
        hdr = (f"{'bucket':8s} {'n':>7s} {'d_imm':>10s} {'rel%':>7s} {'d_ahead':>10s} {'rel%':>7s} "
               f"{'|dp|imm p50':>11s} {'p99':>7s} {'max':>7s} {'>0.05':>6s} {'>0.10':>6s}")
        print(hdr)
        for b in ["all"] + report["buckets"]:
            r = rec["all"] if b == "all" else rec["buckets"].get(b)
            if not r:
                continue
            mi = r["move_imm"]
            print(f"{b:8s} {r['n_reviews']:7d} {r['d_imm']:+10.6f} {r['rel_imm_pct']:+7.3f} "
                  f"{r.get('d_ahead', float('nan')):+10.6f} {r.get('rel_ahead_pct', float('nan')):+7.3f} "
                  f"{mi.get('p50', float('nan')):11.5f} {mi.get('p99', 0):7.5f} {mi.get('max', 0):7.5f} "
                  f"{mi.get('share_gt_0.05', 0):6.3f} {mi.get('share_gt_0.10', 0):6.3f}")
        print(f"cards moving >0.05 imm: {rec['cards_imm_gt_0.05']:.4f}   >0.10 imm: {rec['cards_imm_gt_0.10']:.4f}"
              f"   >0.05 ahead: {rec['cards_ahead_gt_0.05']:.4f}   >0.10 ahead: {rec['cards_ahead_gt_0.10']:.4f}")

        np.savez_compressed(ROOT / "scratchpad" / "quant_plan" / f"bucket_rows_{args.tag}_{name}.npz",
                            user=m["user"].to_numpy(), review_th=m["review_th"].to_numpy(),
                            ordinal=m["ordinal_p"].to_numpy(), y=m["y_p"].to_numpy(),
                            p_imm_p=m["p_imm_p"].to_numpy(), p_imm_q=m["p_imm_q"].to_numpy(),
                            p_ahead_p=m["p_ahead_p"].to_numpy(), p_ahead_q=m["p_ahead_q"].to_numpy(),
                            card_id=m["card_id"].to_numpy())

    out = Path(args.out) if args.out else ROOT / "scratchpad" / "quant_plan" / f"bucket_tax_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
