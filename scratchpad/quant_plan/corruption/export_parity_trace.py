"""Export a CURRENT-lineage parity trace: the Python RNN's own predictions for the row-17 recipe.

Why this exists
---------------
`reference/trace_user_*.{safetensors,json}` are the June-2026 traces of the d=128 `rwkv_ref_558`
model. `export_rnn_trace.py`'s own OUT_DIR comment says the current Python can no longer reproduce
their frozen `py_pred_*`, so they are a fixed LABEL-only evaluation set: good for scoring the Rust
engine against the real labels (that is what `rust_grid.py` / `rust_score.py` do), useless for a
Python<->Rust parity check, because the parity target inside them belongs to a different model.

This driver exports a fresh trace with the row-17 checkpoint AND the row-17 quantization env, so
the trace's own `py_pred_imm` / `py_pred_ahead` are the Python QAT path's predictions for the SAME
weights and the SAME quantization the Rust deploy engine runs. A Rust run over this trace is then a
genuine three-way parity witness (training/eval path <-> Python RNN path <-> Rust deploy path) for
the shipped weights, not just a label score.

`RWKV_QAT_FUSED=0` is deliberate: the fused rank-1 CUDA kernel truncates inside the kernel, so its
pre-truncation state is invisible, while the Python per-step loop is the documented bit-parity twin
of the fused kernels and is what the Rust reference solver was verified against.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/corruption/export_parity_trace.py \
      --users 107 --ckpt scratchpad/quant_plan/qat_it36l2i8s4/qat_124.pth \
      --out scratchpad/quant_plan/trace_iter17_parity
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # corruption -> quant_plan -> scratchpad -> repo root
if not (ROOT / "rwkv").is_dir():
    raise SystemExit(f"repo root not found from {HERE} (looked for {ROOT / 'rwkv'})")
sys.path.insert(0, str(HERE))

# Sets the 92-input lineage flags (RWKV_ID_FEATURES=0, RWKV_REAL_CYCLES=0, RWKV_FINAL_LAYOUT=0,
# RWKV_E2S_PUBLISHED=0, RWKV_NO_JIT=1) and the eval's label DB, exactly as the corruption export
# does -- the same env every certified number in this arm was produced under.
import export_corrupt50_traces as drv  # noqa: E402

# The row-17 recipe's quantization, on the Python training-side names. MUST be set before
# export_rnn_trace (and therefore rwkv.model) is imported: the QAT scopes are parsed at import.
os.environ["RWKV_QAT_FUSED"] = "0"
os.environ.setdefault("RWKV_QAT_LOWRANK_SCOPE", "card:2:int8,note:1:int8")
os.environ.setdefault("RWKV_QAT_SHIFT_SCOPE", "card:int4,note:int4")

DEFAULT_OUT = ROOT / "scratchpad" / "quant_plan" / "trace_iter17_parity"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="107")
    ap.add_argument("--ckpt", default=str(drv.DEFAULT_CKPT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    users = [int(u) for u in args.users.split(",") if u.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    import export_rnn_trace as e  # noqa: E402  (imports rwkv.model -> reads the QAT env above)
    import torch  # noqa: E402

    e.MODEL_PATH = str(ckpt)
    e.OUT_DIR = out
    e.LABEL_DB = os.environ["RWKV_EXPORT_LABEL_DB"]
    e.LABEL_DB_SIZE = drv.LABEL_DB_SIZE
    torch.set_num_threads(max(1, args.threads))

    print(f"ckpt {ckpt}\nout {out}\nusers {users}", flush=True)
    print(f"qat lowrank={os.environ['RWKV_QAT_LOWRANK_SCOPE']} "
          f"shift={os.environ['RWKV_QAT_SHIFT_SCOPE']} fused={os.environ['RWKV_QAT_FUSED']}",
          flush=True)

    rows = []
    for u in users:
        t0 = time.time()
        meta = e.export_user(u)
        dt = time.time() - t0
        rec = {"user": u, "secs": round(dt, 1), "n_reviews": int(meta["n_reviews"]),
               "n_eq": int(meta["size"]), "py_imm_logloss": float(meta["py_imm_logloss"]),
               "py_ahead_logloss": float(meta["py_ahead_logloss"])}
        rows.append(rec)
        print(f"user {u}: {dt:.1f}s n={rec['n_reviews']} eq={rec['n_eq']} "
              f"py imm {rec['py_imm_logloss']:.6f} ahead {rec['py_ahead_logloss']:.6f}", flush=True)

    (out / "parity_export.json").write_text(
        json.dumps({"ckpt": str(ckpt), "users": users, "rows": rows,
                    "qat": {"lowrank_scope": os.environ["RWKV_QAT_LOWRANK_SCOPE"],
                            "shift_scope": os.environ["RWKV_QAT_SHIFT_SCOPE"],
                            "fused": os.environ["RWKV_QAT_FUSED"]}}, indent=1), encoding="utf-8")
    print(f"wrote {out / 'parity_export.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
