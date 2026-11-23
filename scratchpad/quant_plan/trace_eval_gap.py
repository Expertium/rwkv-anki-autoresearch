"""Size the trace-path (engine) vs eval-path gap per user, on the SAME weights and with NO quantization.

Feedback item (d), 2026-11-23: the other team flagged that our ahead head's trace-vs-eval gap is about
10x the deploy tax and rests on 2 users. This tool measures that gap per user, on every user that has
both a deploy trace and an eval row, and separates two different questions:

  * engine vs eval -- the deploy engine on a trace against the evaluation path's own per-user LogLoss.
                       A real gap here is an INPUT difference (the trace exporter's frame + clock vs the
                       evaluation dataset), so it invalidates absolute comparison ACROSS paths while
                       leaving within-path deltas (i.e. every deploy tax) valid;
  * engine vs trace -- the Rust port against the trace's embedded Python RNN predictions. Only
                       meaningful when the predictions were produced with the SAME checkpoint the trace
                       was exported with (`--same-checkpoint`), because `py_pred_*` is that checkpoint's.

Head mapping: the project names these files the other way round from the intuitive reading --
`result/adw_<tag>.jsonl` is the AHEAD head and `result/adwP_<tag>.jsonl` is the IMMEDIATE head
(`quant_sweep.py` docstring, line 10).

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/trace_eval_gap.py \
      --trace-dir scratchpad/quant_plan/trace_corrupt50 \
      --preds scratchpad/quant_plan/rust_grid_preds/bucket17/plain \
      --tag corrupt_ctl_fp32 --sweep-tag qnj_it36ctl
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import rust_score  # noqa: E402


def read_eval(tag):
    """(user, head) -> LogLoss. adw_ is the AHEAD head, adwP_ the IMMEDIATE head."""
    out = {}
    for stem, head in (("adwP", "imm"), ("adw", "ahead")):
        p = ROOT / "result" / f"{stem}_{tag}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out.setdefault(int(rec["user"]), {})[head] = rec["metrics"]["LogLoss"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default="scratchpad/quant_plan/trace_corrupt50")
    ap.add_argument("--preds", default="scratchpad/quant_plan/rust_grid_preds/bucket17/plain")
    ap.add_argument("--tag", default="corrupt_ctl_fp32",
                    help="eval tag on this roster (falls back to --sweep-tag for users it lacks)")
    ap.add_argument("--sweep-tag", default="qnj_it36ctl")
    ap.add_argument("--same-checkpoint", action="store_true",
                    help="the predictions came from the checkpoint the trace was exported with")
    ap.add_argument("--users", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    trace_dir = ROOT / args.trace_dir
    preds = ROOT / args.preds
    users = ([int(u) for u in args.users.split(",")] if args.users
             else sorted(int(p.stem.split("_")[-1]) for p in preds.glob("rust_pred_*.json")))
    ev = read_eval(args.tag)
    sweep = read_eval(args.sweep_tag)

    def cell(v, width=9):
        return "n/a".rjust(width) if v is None else f"{v:{width}.5f}"

    rows = []
    print(f"{'user':>6s} {'n_rev':>7s} {'eng_imm':>8s} {'ev_imm':>8s} {'d_imm':>8s} | "
          f"{'eng_ah':>8s} {'ev_ah':>8s} {'d_ah':>8s} | {'src':>5s}")
    for u in users:
        got = rust_score.score_user(trace_dir, preds, u)
        meta_path = trace_dir / f"trace_user_{u}.json"
        if not got or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        e = ev.get(u) or sweep.get(u) or {}
        src = "eval" if u in ev else ("sweep" if u in sweep else "-")
        r = {"user": u, "n_reviews": meta["n_reviews"], "eng_imm": got["imm"], "eng_ahead": got["ahead"],
             "ev_imm": e.get("imm"), "ev_ahead": e.get("ahead"), "src": src,
             "py_imm": meta["py_imm_logloss"], "py_ahead": meta["py_ahead_logloss"]}
        r["d_imm"] = None if r["ev_imm"] is None else r["eng_imm"] - r["ev_imm"]
        r["d_ahead"] = None if r["ev_ahead"] is None else r["eng_ahead"] - r["ev_ahead"]
        if args.same_checkpoint:
            r["eng_py_imm"] = r["eng_imm"] - r["py_imm"]
            r["eng_py_ahead"] = r["eng_ahead"] - r["py_ahead"]
        rows.append(r)
        print(f"{u:6d} {r['n_reviews']:7d} {r['eng_imm']:8.5f} {cell(r['ev_imm'], 8)} "
              f"{cell(r['d_imm'], 8)} | {r['eng_ahead']:8.5f} {cell(r['ev_ahead'], 8)} "
              f"{cell(r['d_ahead'], 8)} | {src:>5s}")

    print()
    keys = ["d_imm", "d_ahead"]
    if args.same_checkpoint:
        keys += ["eng_py_imm", "eng_py_ahead"]
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            print(f"{key:12s} n={len(vals):3d} mean={statistics.mean(vals):+.6f} "
                  f"median={statistics.median(vals):+.6f} mean|.|={statistics.mean(abs(v) for v in vals):.6f} "
                  f"max|.|={max(abs(v) for v in vals):.6f}")
    out = (Path(args.out) if args.out else
           ROOT / "scratchpad" / "quant_plan" / f"trace_eval_gap_{Path(args.preds).name}.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
