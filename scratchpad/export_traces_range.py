"""Export RNN traces (features + routing + labels + fp32 Python preds) for a RANGE of users,
so the Rust engine can be run with/without card-only state quant and gated on by-user-mean LogLoss.

Reuses export_rnn_trace.export_user (the exact same feature capture the parity export uses).
The champion checkpoint + safetensors come from RWKV_CHAMP_CKPT / RWKV_CHAMP_SFT (architecture.py
must match that checkpoint). Writes reference/trace_user_{u}.{safetensors,json} per user.

Usage:
  RWKV_CHAMP_CKPT=... RWKV_CHAMP_SFT=... python scratchpad/export_traces_range.py START END
(inclusive START, exclusive END). Default 101 201 (= users 101..200).
"""
import sys
import json
from pathlib import Path

# The wrapper lives in scratchpad/; put the repo root (its parent) on the path so
# `export_rnn_trace` (at the repo root) is importable. CWD must still be the repo root
# (export_rnn_trace uses CWD-relative paths like "reference" and "../anki-revlogs-10k").
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import export_rnn_trace as ert


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 201
    ert.OUT_DIR.mkdir(exist_ok=True)
    # Export the champion weights -> safetensors once (Rust reads these).
    ert.export_weights()
    metas = []
    for u in range(start, end):
        # Resume: skip users already fully exported (both files present + json parses).
        tj = ert.OUT_DIR / f"trace_user_{u}.json"
        ts = ert.OUT_DIR / f"trace_user_{u}.safetensors"
        if tj.exists() and ts.exists():
            try:
                m = json.load(open(tj))
                if "py_imm_logloss" in m and "py_ahead_logloss" in m:
                    metas.append({"user": u, "size": m["size"],
                                  "imm": m["py_imm_logloss"], "ahead": m["py_ahead_logloss"]})
                    print(f"[{len(metas)}] user {u}: SKIP (already exported)", flush=True)
                    continue
            except Exception:
                pass
        m = ert.export_user(u)
        metas.append({"user": u, "size": m["size"],
                      "imm": m["py_imm_logloss"], "ahead": m["py_ahead_logloss"]})
        # running by-user mean so progress is visible / resumable
        imm = sum(x["imm"] for x in metas) / len(metas)
        ah = sum(x["ahead"] for x in metas) / len(metas)
        print(f"[{len(metas)}] user {u}: imm {m['py_imm_logloss']:.6f} ahead "
              f"{m['py_ahead_logloss']:.6f} | running mean imm {imm:.6f} ahead {ah:.6f}",
              flush=True)
    summary = {"start": start, "end": end, "n": len(metas),
               "mean_imm": sum(x["imm"] for x in metas) / len(metas),
               "mean_ahead": sum(x["ahead"] for x in metas) / len(metas),
               "per_user": metas}
    Path("reference/range_fp32_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nFP32 by-user mean over {len(metas)} users: "
          f"imm {summary['mean_imm']:.6f} ahead {summary['mean_ahead']:.6f}")
    print("wrote reference/range_fp32_summary.json")


if __name__ == "__main__":
    main()
