"""Build a tiny QAT smoke config that mirrors the FIXED controller QAT branch, then run it.

Purpose: the QAT stage of the adaptive controller died at startup twice
(missing final_qat/validation_users.json; then a VAL-range mismatch caused by
passing the screen roster instead of the internal validation roster). Validate
the fixed code path (trainer startup, validation-users file, QAT env flags,
learned-catalog exports, final checkpoint save) in ~86 steps before the full
43,354-step QAT epoch.

Plumbing evidence only: not accuracy evidence, not a research result.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "scratchpad" / "adaptive_wd" / "adaptive_fullrun_92_20261105"
SRC = RUN / "final_qat"
DST = ROOT / "scratchpad" / "adaptive_wd" / "qat_smoke_20261115"
DST.mkdir(parents=True, exist_ok=True)

STEPS = 0.002  # int(0.002 * 43354) = 86 optimizer steps

# The same internal roster the controller derives and writes.
state = json.loads((RUN / "adaptive_state.json").read_text(encoding="utf-8"))
screen = set(int(u) for u in state["users"])
excluded = set(int(u) for u in state.get("screen_excluded_multibatch_or_missing", []))
internal_val_users = [u for u in range(5001, 7501) if u not in screen and u not in excluded][:2]
print("internal_val_users:", internal_val_users)


def edit_toml(src, dst, values):
    lines = src.read_text(encoding="utf-8").splitlines()
    seen, out = set(), []
    for line in lines:
        raw = line.split("#", 1)[0]
        if "=" in raw:
            key = raw.split("=", 1)[0].strip()
            if key in values:
                if key not in seen:
                    v = values[key]
                    out.append(f"{key} = {json.dumps(v) if isinstance(v, str) else v}")
                    seen.add(key)
                continue
        out.append(line)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    missing = [k for k in values if k not in seen]
    if missing:
        raise SystemExit(f"keys not present in base toml: {missing}")


(DST / "validation_users.json").write_text(json.dumps(internal_val_users), encoding="utf-8")
edit_toml(SRC / "train.toml", DST / "train.toml", {
    "EPOCHS": STEPS,
    "VALIDATE_EVERY": 50,
    "VALIDATE_USERS_START": min(internal_val_users),
    "VALIDATE_USERS_END": max(internal_val_users),
    "VALIDATE_USERS_FILE": str(DST / "validation_users.json"),
    "SAVE_MODEL_FOLDER": str(DST),
    "SAVE_MODEL_PREFIX": "qatsmoke",
})

env = dict(os.environ)
env.update(json.loads((ROOT / "scratchpad" / "adaptive_wd" / "qat_env.json").read_text(encoding="utf-8")))
env.update({
    "RWKV_WEIGHT_DECAY": "0.01",
    "RWKV_WEIGHT_DECAY_CMIX": "0.01",
    "RWKV_WEIGHT_DECAY_HEAD": "0.01",
    "RWKV_VRAM_CAP_GB": "11",
})
env["RWKV_ID_FEATURES"] = "0"
env["RWKV_REAL_CYCLES"] = "0"
env["RWKV_FINAL_LAYOUT"] = "0"
env["RWKV_E2S_PUBLISHED"] = "0"

log = DST / "qat_smoke.log"
with log.open("w", encoding="utf-8", errors="replace") as fh:
    p = subprocess.run([sys.executable, "-u", "-m", "rwkv.train_rwkv",
                        "--config", str(DST / "train.toml")],
                       cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
text = log.read_text(encoding="utf-8", errors="replace")
print("returncode:", p.returncode)
print("MODEL SAVED.:", "MODEL SAVED." in text)
print("traceback:", "Traceback (most recent call last)" in text)
print("validation line:", "Mean ahead validation loss" in text)
print("catalog exports:", sorted(x.name for x in DST.glob("*.txt")))
print("checkpoints:", sorted(x.name for x in DST.glob("*.pth")))
print("--- tail ---")
print("\n".join(text.splitlines()[-10:]))
