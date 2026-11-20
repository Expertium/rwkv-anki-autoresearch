"""Run the advisor-required TA3 correctness gates on the real path.

Gates (rwkv/ta3.py check mode, `RWKV_TA3_CHECK`):
  1. eta = 0 must reproduce the ordinary full-batch gradient (relative L2 + cosine).
  2. A nonzero eta must change that gradient.
  3. The trial perturbation must restore the weights EXACTLY (max abs drift reported) and leave
     the optimizer state untouched.

Dropout is switched OFF for this check (`RWKV_DROPOUT_SCALE=0`): with dropout on, the A and Q
phases draw different masks and that noise floor would hide an implementation error. The check
never steps the optimizer, so it cannot change any run state.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "adaptive_wd" / "adaptive_fullrun_92_20261105"
SRC = RUN / "attempt_11_epoch_05_wd_0.64"
DST = ROOT / "scratchpad" / "ta3_gradcheck_20261115"
DST.mkdir(parents=True, exist_ok=True)

CHECK_STEPS = 4


def edit_toml(src, dst, values):
    lines = src.read_text(encoding="utf-8").splitlines()
    seen, out = set(), []
    for line in lines:
        raw = line.split("#", 1)[0]
        if "=" in raw:
            key = raw.split("=", 1)[0].strip()
            if key in values:
                if key not in seen:
                    value = values[key]
                    out.append(f"{key} = {json.dumps(value) if isinstance(value, str) else value}")
                    seen.add(key)
                continue
        out.append(line)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    missing = [k for k in values if k not in seen]
    if missing:
        raise SystemExit(f"keys not present in base toml: {missing}")


internal_val_users = [5001, 5003]
(DST / "validation_users.json").write_text(json.dumps(internal_val_users), encoding="utf-8")
edit_toml(SRC / "train.toml", DST / "train.toml", {
    "EPOCHS": 4.0003,
    "STEP_OFFSET": 173417,
    "VALIDATE_EVERY": 100000,
    "VALIDATE_USERS_START": min(internal_val_users),
    "VALIDATE_USERS_END": max(internal_val_users),
    "VALIDATE_USERS_FILE": str(DST / "validation_users.json"),
    "SAVE_MODEL_FOLDER": str(DST),
    "SAVE_MODEL_PREFIX": "ta3check",
    "LOAD_MODEL_FOLDER": str(RUN / "attempt_04_epoch_04_wd_0.01"),
    "LOAD_MODEL_NAME": "adaptive_173416",
    "LOAD_OPTIM_PATH": str(RUN / "attempt_04_epoch_04_wd_0.01" / "adaptive_optim_173416.pth"),
})

env = dict(os.environ)
env.update({
    "RWKV_WEIGHT_DECAY": "0.01",
    "RWKV_WEIGHT_DECAY_CMIX": "0.01",
    "RWKV_WEIGHT_DECAY_HEAD": "0.01",
    "RWKV_VRAM_CAP_GB": "11",
    "RWKV_RESUME_SKIP_GROUPS": "1",
    "RWKV_DECAY_RESUME_SCHED": "1",
    "RWKV_DECAY_SHAPE": "constant",
    "RWKV_ID_FEATURES": "0",
    "RWKV_REAL_CYCLES": "0",
    "RWKV_FINAL_LAYOUT": "0",
    "RWKV_E2S_PUBLISHED": "0",
    "RWKV_DROPOUT_SCALE": "0",
    "RWKV_TA3_FP32_CHECK": "1",
    "RWKV_TA3_CHECK": str(CHECK_STEPS),
    "RWKV_TA3_ETA": os.environ.get("RWKV_TA3_ETA", "0.3"),
    "RWKV_TA3_SCOPE": os.environ.get("RWKV_TA3_SCOPE", "cmix"),
    "RWKV_TA3_FRAC": os.environ.get("RWKV_TA3_FRAC", "0.5"),
})
env.pop("RWKV_MUON", None)

log = DST / "gradcheck.log"
with log.open("w", encoding="utf-8", errors="replace") as handle:
    proc = subprocess.run([sys.executable, "-u", "-m", "rwkv.train_rwkv",
                           "--config", str(DST / "train.toml")],
                          cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
text = log.read_text(encoding="utf-8", errors="replace")
print("returncode:", proc.returncode)
print("traceback:", "Traceback (most recent call last)" in text)
print("--- check lines ---")
for line in text.splitlines():
    if line.startswith("[ta3-check]") or line.startswith("[ta3] scope"):
        print(line[:600])
