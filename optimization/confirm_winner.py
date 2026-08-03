"""Resolve the HP tuner's winning trial and stage its FULL VAL-half confirmation eval.

WHY THIS EXISTS: the tuner ranks on the 1000-user subset 5001-6000, which is a RANKING PROXY,
not a gate. The champ5k_t1 lesson is explicit that a subset winner can invert at full scale (a
+0.0008/+0.0010 win on 200 users became -0.0005/-0.0007 at n=5000), and while the 1000-user
subset is far better than that 200-user one -- it ranked maxval-vs-iter-31 the same way the full
half did -- a sub-0.001 verdict still has to be confirmed on 5001-7500 before it becomes the
recipe.

This costs NO retraining: the winning trial's decay checkpoint already exists on disk. It is one
eval (~2.5 h) plus a free paired comparison against iter 32's RECTIFIED jsonls, which are the
gate basis from iter 33 on.

Run via scratchpad/tuner65k/run_confirm.cmd, which sets the trunk env (the eval needs it in the
process environment before python starts) and handles the giant-user retry.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "optimization"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "hp_tuner_5k", os.path.join(ROOT, "optimization", "hp_tuner_5k.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

OUT_TOML = f"{ROOT}/scratchpad/tuner65k/confirm_eval.toml"
WINNER_TXT = f"{ROOT}/scratchpad/tuner65k/winner.txt"
TAG = "t65confirm"
VAL_START, VAL_END = 5001, 7500     # the FULL VAL half -- candidates never touch 7501-10000


def main():
    recs = T.load_journal()
    if not recs:
        raise SystemExit("journal empty")
    best = min(recs, key=T.obj)
    if best.get("pruned"):
        raise SystemExit(f"best row {best['name']} is a PRUNED estimate, not a real eval -- refusing")
    # ⚠ The BASELINE row is a legitimate journal row and can legitimately win -- it is the default
    # config, and a grid where nothing beats the default is a real (informative) outcome, not an
    # error. But it has no trial directory: its numbers came from scratchpad/maxval restricted to
    # 5001-6000, not from a run under scratchpad/tuner65k. Without this branch the checkpoint
    # search below just reports "no decay checkpoint -- did the trial finish?", which points the
    # reader at a broken trial instead of at the actual, benign situation.
    if best.get("param") == "baseline":
        raise SystemExit(
            "the BASELINE (default HPs) is still the best row -- no coordinate beat it, so there "
            "is nothing to confirm and nothing to adopt. Its checkpoint lives in scratchpad/maxval, "
            "not under scratchpad/tuner65k. Confirm maxval directly if a full VAL-half number is "
            "wanted.")

    name = best["name"]
    folder = f"{T.TRIAL_DIR}/{name}"
    # the DECAY checkpoint (prefix "<name>d"), highest step, excluding the optimizer files
    cands = []
    for p in glob.glob(f"{folder}/{name}d_*.pth"):
        b = os.path.basename(p)
        if "optim" in b:
            continue
        m = re.match(rf"{re.escape(name)}d_(\d+)\.pth$", b)
        if m:
            cands.append((int(m.group(1)), p.replace("\\", "/")))
    if not cands:
        raise SystemExit(f"no decay checkpoint under {folder} -- did the trial finish?")
    step, ckpt = max(cands)

    # ⚠ KEY NAMES ARE get_result's, NOT the training toml's. Written wrong the first time
    # (TEST_DATASET_LMDB_PATH / TEST_USERS_START, which are TRAINING-side names) and the dry run
    # missed it because it only checked the file PARSED. Parsing is not validation -- the schema
    # check below is. Format mirrors scratchpad/write_eval_toml.py, which is the reference.
    with open(OUT_TOML, "w") as f:
        f.write(f'''# FULL VAL-half confirmation of the HP tuner winner: {name} (decay step {step}).
# Subset (5001-6000) result: ahead {best["ahead"]:.6f} / imm {best["imm"]:.6f}.
# Config: {json.dumps(best["config"])}
FILE_AHEAD = "RWKV-{TAG}"
FILE_IMM = "RWKV-P-{TAG}"
MODEL_PATH = "{ckpt}"
DEVICE = "cuda"
DTYPE = "bfloat16"
DATASET_LMDB_PATH = "F:/rwkv_lmdb/test_db_5k"
DATASET_LMDB_SIZE = 250_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 40_000_000_000
RAW = false
RAW_DB_PATH = "raw/result_db"
RAW_DB_SIZE = 1_000_000_000
USER_START = {VAL_START}
USER_END = {VAL_END}
NUM_FETCH_PROCESSES = 2
''')

    # Schema check: eval_sharded reads these by exact name and dies with a bare KeyError if any
    # is missing, so assert them here where the message can say what to do about it.
    import tomli
    with open(OUT_TOML, "rb") as fh:
        parsed = tomli.load(fh)
    required = ["FILE_AHEAD", "FILE_IMM", "MODEL_PATH", "DEVICE", "DTYPE", "DATASET_LMDB_PATH",
                "DATASET_LMDB_SIZE", "LABEL_FILTER_LMDB_PATH", "LABEL_FILTER_LMDB_SIZE",
                "USER_START", "USER_END", "NUM_FETCH_PROCESSES"]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise SystemExit(f"toml is missing get_result keys {missing} -- compare against "
                         f"scratchpad/write_eval_toml.py")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint does not exist: {ckpt}")
    with open(WINNER_TXT, "w") as f:
        f.write(name + "\n")

    print(f"WINNER {name}  (objective {T.obj(best):.6f} on the 1000-user subset)")
    print(f"  config   {json.dumps(best['config'])}")
    print(f"  peak_lr  {best.get('peak_lr')}   muon_lr {best.get('muon_lr')}")
    print(f"  ckpt     {ckpt}  (decay step {step})")
    print(f"  subset   ahead {best['ahead']:.6f}  imm {best['imm']:.6f}")
    print(f"  -> {OUT_TOML}  (eval {VAL_START}-{VAL_END}, RECTIFIED)")
    print(f"  gate: paired_pvalue vs result/RWKV{{,-P}}-iter32_kd_rect.jsonl "
          f"(iter 32 rectified = 0.300268 / 0.267262)")


if __name__ == "__main__":
    main()
