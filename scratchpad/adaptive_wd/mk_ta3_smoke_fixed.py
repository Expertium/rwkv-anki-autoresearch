"""TA3 0.04-epoch smoke with the FIXED code and the REAL start checkpoint.

Mirrors the plain chain's epoch-1 run (same start = QAT46 step-248 model+optimizer, same data,
same LR shape, same settings) with RWKV_TA3=1 as the only difference, in an isolated folder.

Plumbing + cost check only: training starts, the TA3 path executes on every step, checkpoints and
optimizer pairs are written, no batches are skipped, no tracebacks. Not accuracy evidence.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "adaptive_wd" / "adaptive_fullrun_92_20261105"
SRC = RUN / "attempt_01_epoch_01_wd_0.01"        # the real start config: QAT46 step-248
DST = ROOT / "scratchpad" / "ta3_smoke_fixed_20261115"
DST.mkdir(parents=True, exist_ok=True)


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
    "EPOCHS": 0.04,
    "STEP_OFFSET": 1,
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
    "RWKV_RESUME_SKIP_GROUPS": "1",
    "RWKV_DECAY_RESUME_SCHED": "1",
    "RWKV_DECAY_SHAPE": "constant",
    "RWKV_ID_FEATURES": "0",
    "RWKV_REAL_CYCLES": "0",
    "RWKV_FINAL_LAYOUT": "0",
    "RWKV_E2S_PUBLISHED": "0",
    "RWKV_TA3": "1",
    "RWKV_TA3_ETA": "0.3",
    "RWKV_TA3_SCOPE": "cmix",
    "RWKV_TA3_FRAC": "0.5",
})
for key in ("RWKV_MUON", "RWKV_TA3_PROBE", "RWKV_TA3_CHECK"):
    env.pop(key, None)

log = DST / "train.log"
with log.open("w", encoding="utf-8", errors="replace") as handle:
    proc = subprocess.Popen([sys.executable, "-u", "-m", "rwkv.train_rwkv",
                             "--config", str(DST / "train.toml")],
                            cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
print("launched pid", proc.pid, "->", log)
