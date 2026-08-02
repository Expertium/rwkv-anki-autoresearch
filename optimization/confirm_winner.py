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

    with open(OUT_TOML, "w") as f:
        f.write(f'''# FULL VAL-half confirmation of the HP tuner winner: {name} (decay step {step}).
# Subset (5001-6000) result: ahead {best["ahead"]:.6f} / imm {best["imm"]:.6f}.
# Config: {json.dumps(best["config"])}
FILE_AHEAD = "RWKV-{TAG}"
FILE_IMM = "RWKV-P-{TAG}"
MODEL_PATH = "{ckpt}"
TEST_USERS_START = {VAL_START}
TEST_USERS_END = {VAL_END}
TEST_DATASET_LMDB_PATH = "F:/rwkv_lmdb/test_db_5k"
TEST_DATASET_LMDB_SIZE = 250_000_000_000
LABEL_FILTER_LMDB_PATH = "label_filter_db"
LABEL_FILTER_LMDB_SIZE = 40_000_000_000
DEVICE = "cuda"
DTYPE = "bfloat16"
''')
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
