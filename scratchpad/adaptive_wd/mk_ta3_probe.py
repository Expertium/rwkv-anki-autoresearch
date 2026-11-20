"""Build a tiny TA3 probe config from the incumbent's real config and run the probe.

Probe mode measures, on the REAL training batch stream, whether a normalised downhill
trial step on half A lowers half Q's loss, against equal-norm zero/random/uphill controls.
It never takes an optimizer step, so it cannot change any run state.

Not accuracy evidence: it is a mechanism/plumbing check.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "adaptive_wd" / "adaptive_fullrun_92_20261105"
SRC = RUN / "attempt_11_epoch_05_wd_0.64"          # real config: right DBs, right settings
DST = ROOT / "scratchpad" / "ta3_probe_20261115"
DST.mkdir(parents=True, exist_ok=True)

STEPS = 4.0008            # total_steps = int(4.0003 * 43354) = 173,429 -> the last 13 steps of the epoch-5 slice
PROBE_STEPS = 30


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


# The trainer opens VALIDATE_USERS_FILE at startup, before any step.
state = json.loads((RUN / "adaptive_state.json").read_text(encoding="utf-8"))
screen = set(int(u) for u in state["users"])
excluded = set(int(u) for u in state.get("screen_excluded_multibatch_or_missing", []))
internal_val_users = [u for u in range(5001, 7501) if u not in screen and u not in excluded][:2]
(DST / "validation_users.json").write_text(json.dumps(internal_val_users), encoding="utf-8")

edit_toml(SRC / "train.toml", DST / "train.toml", {
    "EPOCHS": STEPS,
    "STEP_OFFSET": 173417,
    "VALIDATE_EVERY": 100000,
    "VALIDATE_USERS_START": min(internal_val_users),
    "VALIDATE_USERS_END": max(internal_val_users),
    "VALIDATE_USERS_FILE": str(DST / "validation_users.json"),
    "SAVE_MODEL_FOLDER": str(DST),
    "SAVE_MODEL_PREFIX": "ta3probe",
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
    "RWKV_TA3_PROBE": str(PROBE_STEPS),
    "RWKV_TA3_PROBE_ETAS": "0.03,0.1,0.3,1.0",
    "RWKV_TA3_SCOPE": os.environ.get("RWKV_TA3_SCOPE", "cmix"),
    "RWKV_TA3_FRAC": os.environ.get("RWKV_TA3_FRAC", "0.5"),
})
env.pop("RWKV_MUON", None)  # mirror the adaptive attempts: plain AdamW groups

log = DST / "ta3_probe.log"
with log.open("w", encoding="utf-8", errors="replace") as handle:
    proc = subprocess.run([sys.executable, "-u", "-m", "rwkv.train_rwkv",
                           "--config", str(DST / "train.toml")],
                          cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
text = log.read_text(encoding="utf-8", errors="replace")
print("returncode:", proc.returncode)
print("traceback:", "Traceback (most recent call last)" in text)
print("--- tail ---")
print("\n".join(text.splitlines()[-8:]))
