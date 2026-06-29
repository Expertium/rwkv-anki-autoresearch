"""Evaluate an iteration's result jsonls, check the gates, and append a log record.

Computes by-user-mean LogLoss (ahead+imm) from get_result output, reads iteration 0 from
log.jsonl as the baseline, computes params + per-card state from the CURRENT architecture
(rwkv/architecture.py, i.e. the arch that was trained), checks the gates, and appends a
record to optimization/log.{jsonl,md} via logbook.

Gates (protocol): LogLoss(both modes) not worse than iter0 by >+0.0015; per-card state <=
iter0; total review count identical to iter0.

Usage:
  python optimization/gate.py --number 1 --model rwkv_iter1_558.pth \
      --ahead result/RWKV-iter1.jsonl --imm result/RWKV-P-iter1.jsonl \
      --summary "iter1: d_model 128->64 (N_HEADS 4->2), 3.4x fewer params" \
      [--throughput 1234.5 --wilcoxon-p 3e-4] [--no-write]
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import logbook
from model_stats import build_report

TOL = 0.0015


def by_user_mean(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    lls = [r["metrics"]["LogLoss"] for r in rows]
    size = sum(r["size"] for r in rows)
    return sum(lls) / len(lls), len(rows), size


def iter0():
    for rec in logbook.load():
        if rec["number"] == 0:
            return rec
    raise SystemExit("no iteration 0 in log.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--number", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ahead", required=True)
    ap.add_argument("--imm", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--throughput", type=float, default=None)
    ap.add_argument("--wilcoxon-p", type=float, default=None)
    ap.add_argument("--comment", default="")
    ap.add_argument("--status", default=None,
                    help="adoption outcome (CHAMP-acc/CHAMP-comp/ex-champ/alt/rejected/...); "
                         "default derived from gate pass/fail")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    base = iter0()
    ahead, na, sa = by_user_mean(args.ahead)
    imm, ni, si = by_user_mean(args.imm)
    rep = build_report()  # current architecture.py

    d_ahead = ahead - base["logloss"]["ahead"]
    d_imm = imm - base["logloss"]["imm"]
    ll_ok = (d_ahead <= TOL) and (d_imm <= TOL)
    size_ok = (sa == base["review_count_total"]) and (si == base["review_count_total"])
    state_ok = rep["card_state_floats"] <= base["state_size_floats"]

    print(f"users: ahead {na}, imm {ni}")
    print(f"ahead LogLoss {ahead:.6f}  (iter0 {base['logloss']['ahead']:.6f}, "
          f"delta {d_ahead:+.6f})")
    print(f"imm   LogLoss {imm:.6f}  (iter0 {base['logloss']['imm']:.6f}, "
          f"delta {d_imm:+.6f})")
    print(f"params {rep['total_params']:,}  state {rep['card_state_floats']} floats "
          f"({rep['card_state_kib_f32']} KiB)")
    print(f"GATES: logloss<=+{TOL} {'PASS' if ll_ok else 'FAIL'} | "
          f"size identical {'PASS' if size_ok else 'FAIL'} (got {sa} vs {base['review_count_total']}) | "
          f"state<=iter0 {'PASS' if state_ok else 'FAIL'}")

    accepted = ll_ok and size_ok and state_ok
    status = args.status if args.status else ("rejected" if not accepted else "accepted")
    # Throughput is only measured for kept iterations; rejected ones (incl. gate-passing but
    # not-adopted, via --status rejected) show "n/a" instead of "pending".
    throughput = args.throughput
    if throughput is None and status == "rejected":
        throughput = "n/a"

    rec = {
        "number": args.number,
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "arch": {"d_model": rep["d_model"], "n_heads": rep["n_heads"],
                 "layers": rep["layers"]},
        "logloss": {"ahead": round(ahead, 6), "imm": round(imm, 6)},
        "logloss_delta_vs_iter0": {"ahead": round(d_ahead, 6), "imm": round(d_imm, 6)},
        "params": rep["total_params"],
        "state_size_floats": rep["card_state_floats"],
        "state_kib": rep["card_state_kib_f32"],
        "throughput": throughput,
        "wilcoxon_p": args.wilcoxon_p,
        "review_count_total": sa,
        "review_count_check": "PASS" if size_ok else "FAIL",
        "logloss_tolerance_check": "PASS" if ll_ok else "FAIL",
        "state_size_check": "PASS" if state_ok else "FAIL",
        "summary": args.summary,
        "comment": args.comment,
    }
    if args.no_write:
        print("\n(--no-write) candidate record:")
        print(json.dumps(rec, indent=2, ensure_ascii=False))
    else:
        with open(logbook.JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logbook.rebuild_md()
        print(f"\nappended iteration {args.number} to the log.")


if __name__ == "__main__":
    main()
