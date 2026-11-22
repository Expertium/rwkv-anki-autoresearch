"""Run the Rust finiteness check over every COMPLETE corrupt50 trace - the real coverage gap.

WHY. Andrew's blowup gate is about the 50 users with the longest average review histories, with their
histories NOT truncated. The gate was already run that way in PYTHON (`corruption/check_corruption.py`
via `it36corrupt_chain.py`: row 17 passed on 50/50 users, 158,986 state calls, 0 non-finite). But the
DEPLOY engine's own check (`RWKV_FINITE_CHECK`, `--strict`) had only ever seen the 3-user parity trace,
so the engine - which is what actually ships and which has its own non-finiteness guard - had never
been exercised at those history lengths. `corruption/export_corrupt50_traces.py` is producing those
traces (`trace_corrupt50/`, 5 shards x 10 users) for exactly this.

WHAT IT DOES. Globs the complete traces (both `.json` and `.safetensors` present - a half-written
trace would make the engine fail for the wrong reason), runs `rust_finite.py --strict` once per config
over all of them, and prints the verdict. `--strict` makes the ENGINE exit 3 on any non-finite value,
so a 0 exit is the engine's own statement, not a Python post-check.

NOT a substitute for the Python gate: it is the same gate on the other implementation. Both matter,
because the two quantize the same state through different code.

CAUTION, learned the hard way: `rwkv-infer.exe` matches the chains' BUSY_PAT, and a waiting chain
returns only on `ended AND busy == 0`. So this must be launched INSIDE a running point's window
(smoke 1-3 min + decay 7-10 min + eval 20-56 min), never across a handoff, or it strands the queue.

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/corrupt50_finite_run.py [--tag NAME]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                      # scratchpad/quant_plan
ROOT = HERE.parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
TRACE_DIR = RUN / "trace_corrupt50"
WEIGHTS = ROOT / "scratchpad" / "row17_l2i8s4_bundle" / "weights" / "deploy_iter17_l2i8s4.safetensors"


def complete_users():
    out = []
    for p in sorted(TRACE_DIR.glob("trace_user_*.json")):
        if not p.with_suffix(".safetensors").exists():
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            continue
        out.append(int(p.stem.split("_")[-1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corrupt50")
    ap.add_argument("--configs", default="plain,l2i8s4")
    ap.add_argument("--min-users", type=int, default=40,
                    help="refuse to run on fewer complete traces than this")
    args = ap.parse_args()

    users = complete_users()
    print(f"complete traces: {len(users)} -> {users}")
    if len(users) < args.min_users:
        print(f"ABORT: only {len(users)} complete traces, need >= {args.min_users}; the export is "
              f"still running (or stalled). Re-run later.")
        return 2
    if not WEIGHTS.exists():
        print(f"ABORT: weights missing at {WEIGHTS}")
        return 2

    cmd = [str(PY), "-X", "utf8", str(RUN / "rust_finite.py"),
           "--weights", str(WEIGHTS),
           "--configs", args.configs,
           "--trace-dir", str(TRACE_DIR.relative_to(ROOT)).replace("\\", "/"),
           "--users", ",".join(str(u) for u in users),
           "--tag", args.tag, "--strict"]
    print("running:", " ".join(cmd[1:]))
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(p.stdout[-4000:])
    if p.stderr.strip():
        print("STDERR:", p.stderr[-1500:])

    out = RUN / f"rust_finite_{args.tag}.json"
    verdict = {"tag": args.tag, "n_users": len(users), "users": users, "rc": p.returncode,
               "json": str(out) if out.exists() else None}
    if out.exists():
        doc = json.loads(out.read_text(encoding="utf-8"))
        verdict["per_config"] = {k: v for k, v in doc.items() if k != "users"}
    (RUN / f"rust_finite_{args.tag}_verdict.json").write_text(
        json.dumps(verdict, indent=1), encoding="utf-8")
    print(f"\nrc={p.returncode}  (0 = the ENGINE found nothing non-finite in any user/config)")
    return 0 if p.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
