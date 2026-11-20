"""Evaluate ONE checkpoint on a chosen user roster with the canonical sharded evaluator.

Reuses the adaptive-WD chain's own helper (`adaptive_train.eval_users`) so the evaluation path is
identical to the one every screen in this lineage used: `optimization/eval_sharded.py` with
`rwkv/get_result_config_base5k.toml`, one shard, solo-threshold 0.

Writes result/adw_<tag>.jsonl (ahead) and result/adwP_<tag>.jsonl (immediate), plus a small summary
json under scratchpad/quant_plan/.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/eval_baseline.py \
      --model <checkpoint.pth> --tag <unique-tag> [--users-file <roster.json>] [--limit N]
"""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "adaptive_wd"))
import adaptive_train as ad  # noqa: E402

VAL_DB = "F:/rwkv_lmdb/test_db_5k_adaptive92"
LABEL_DB = "F:/rwkv_lmdb/label_filter_db_id_e2s"
RUN_DIR = ROOT / "scratchpad" / "quant_plan"
DEFAULT_ROSTER = RUN_DIR / "roster750_seed20261119.json"


def eval_env():
    """The eval environment of the 92-input lineage: the four layout flags OFF, nothing else.

    The feature-layout flags are what the model was trained under; training-only flags (weight
    decay, resume scheduling, TA3, SAM, Muon) are irrelevant to evaluation and are deliberately
    left unset so a stray value cannot change the numbers.

    IMPORTANT: start from os.environ.copy(). A bare dict strips SystemRoot/PATH from the child
    process and the shard worker then dies with `OSError: [WinError 10106] The requested service
    provider could not be loaded or initialized` while importing asyncio's _overlapped.
    """
    env = os.environ.copy()
    env.update({
        "RWKV_ID_FEATURES": "0",
        "RWKV_REAL_CYCLES": "0",
        "RWKV_FINAL_LAYOUT": "0",
        "RWKV_E2S_PUBLISHED": "0",
        "RWKV_VRAM_CAP_GB": "11",
    })
    return env


def load_users(path, limit=None):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    users = list(doc["users"] if isinstance(doc, dict) else doc)
    if limit:
        users = users[:limit]
    return users, (doc.get("seed") if isinstance(doc, dict) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--users-file", default=str(DEFAULT_ROSTER))
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N users (sanity check)")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"checkpoint not found: {model}")
    users, seed = load_users(args.users_file, args.limit or None)
    tag = args.tag or f"qbl_{uuid.uuid4().hex[:12]}"
    print(f"model:  {model}")
    print(f"roster: {args.users_file} (seed {seed})  users={len(users)}  tag={tag}")

    res = ad.eval_users(RUN_DIR, ROOT / "rwkv" / "get_result_config_base5k.toml", model, tag,
                        users, VAL_DB, LABEL_DB, eval_env(), shards=1, require_full=True)
    res["tag"] = tag
    res["model"] = str(model)
    res["users_file"] = str(args.users_file)
    out = RUN_DIR / f"eval_summary_{tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"RESULT ahead={res['ahead']:.6f} imm={res['imm']:.6f} n={res['n']}")
    print(f"summary written to {out}")


if __name__ == "__main__":
    main()
