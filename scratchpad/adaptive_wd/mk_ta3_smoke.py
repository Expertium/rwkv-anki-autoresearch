"""TA3 0.04-epoch smoke: the real training path with RWKV_TA3=1, in an isolated folder.

Checks plumbing only (training starts, the TA3 phases execute, checkpoints/artifacts are
written, no unexpected skipped batches). It is not accuracy evidence.

Mirrors an adaptive attempt from the SAME start checkpoint and settings, with TA3 as the
only difference, so the smoke also prices TA3's wall-clock cost per step.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "adaptive_wd" / "adaptive_fullrun_92_20261105"
SRC = RUN / "attempt_11_epoch_05_wd_0.64"
DST = ROOT / "scratchpad" / "ta3_smoke_20261115"
DST.mkdir(parents=True, exist_ok=True)

EPOCHS = 0.04            # project rule: a real-path 0.04-epoch smoke for a non-tuning change


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
    "EPOCHS": EPOCHS,
    "STEP_OFFSET": 1,
    "VALIDATE_EVERY": 500,
    "VALIDATE_USERS_START": min(internal_val_users),
    "VALIDATE_USERS_END": max(internal_val_users),
    "VALIDATE_USERS_FILE": str(DST / "validation_users.json"),
    "SAVE_MODEL_FOLDER": str(DST),
    "SAVE_MODEL_PREFIX": "ta3smoke",
})

env = dict(os.environ)
env.update({
    "RWKV_WEIGHT_DECAY": "0.01",
    "RWKV_WEIGHT_DECAY_CMIX": "0.01",
    "RWKV_WEIGHT_DECAY_HEAD": "0.01",
    "RWKV_VRAM_CAP_GB": "11",
    "RWKV_ID_FEATURES": "0",
    "RWKV_REAL_CYCLES": "0",
    "RWKV_FINAL_LAYOUT": "0",
    "RWKV_E2S_PUBLISHED": "0",
    "RWKV_TA3": "1",
    "RWKV_TA3_ETA": os.environ.get("RWKV_TA3_ETA", "0.3"),
    "RWKV_TA3_SCOPE": os.environ.get("RWKV_TA3_SCOPE", "cmix"),
    "RWKV_TA3_FRAC": os.environ.get("RWKV_TA3_FRAC", "0.5"),
})
env.pop("RWKV_MUON", None)
env.pop("RWKV_TA3_PROBE", None)

log = DST / "train.log"
with log.open("w", encoding="utf-8", errors="replace") as handle:
    proc = subprocess.Popen([sys.executable, "-u", "-m", "rwkv.train_rwkv",
                             "--config", str(DST / "train.toml")],
                            cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
print("launched pid", proc.pid, "->", log)
