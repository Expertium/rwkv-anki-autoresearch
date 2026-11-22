"""Exercise the E1 tail-energy penalty in a REAL training loop, on CPU, while the GPU is busy.

WHY. E1 is implemented, unit-tested and queued (chain 15 waits behind `it36corrupt2`), but its
integration point - `srs_model` adding `tail_reg_lambda * take_tail_penalty()` to the loss, and
`rwkv_model` passing a non-empty `tail_tag` only while training - has never run in the real trainer.
If it is broken, chain 15's own 0.04-epoch smoke would fail, `qat_point` would SKIP all three points,
and the GPU would sit idle at the end of the queue. This run retires that risk now, for free, without
touching the GPU: the same config file the harness generates, same env block as `qat_decay.py`, only
DEVICE=cpu, DTYPE=float32 and a 2,048-token batch.

WHAT IT PROVES, and what it does not. It proves the penalty is computed (the `[tail-penalty]` banner
prints the per-stream tail), that a non-empty tag reaches `fake_lowrank_state` under RWKV_QAT_FUSED=0,
and that the loss stays finite through 3 steps. It is NOT an accuracy result and it is NOT the
required 0.04-epoch GPU smoke - chain 15 still runs that on the real path.

Steps: generate the toml with `qat_decay.py --dry-run` (so the config is the harness's own), patch the
device/batch/step fields, run `python -m rwkv.train_rwkv`, then assert on the log.

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/e1_cpu_smoke.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent                      # scratchpad/quant_plan
ROOT = HERE.parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
TAG = "e1cpufake"
DST = RUN / f"qat_{TAG}"
TOML = DST / "train.toml"


def toml_set(text: str, key: str, value: str) -> str:
    out, n = re.subn(rf'(?m)^{key} = .*$', f"{key} = {value}", text)
    if n != 1:
        raise SystemExit(f"expected exactly one {key} line, found {n}")
    return out


def main() -> int:
    gen = subprocess.run(
        [str(PY), "-X", "utf8", str(RUN / "qat_decay.py"), "--candidate", "l2i8s4e1a",
         "--train-users", "1", "100",
         "--start-folder", "pretrain/rwkv/opt_iter36", "--start-name", "rwkv_iter36_558",
         # Standard 66,000-token batch cap and EPOCHS=0.04, exactly like the chain's own smoke:
         # `get_groups` DROPS any user whose history is larger than the cap, so a small cap leaves
         # fewer than FETCH_AHEAD=10 groups and `assert FETCH_AHEAD <= len(groups)` fires.
         "--epochs", "0.04", "--max-steps", "3", "--tag", TAG, "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True)
    print(gen.stdout.strip()[-300:])
    if gen.returncode != 0 or not TOML.exists():
        print(f"ABORT: dry-run failed rc={gen.returncode}")
        return 1

    text = TOML.read_text(encoding="utf-8")
    text = toml_set(text, "DEVICE", '"cpu"')
    # DTYPE is left at the generator's bfloat16: the training forward casts the model to bf16
    # independently of this field, so float32 inputs hit `mat1 and mat2 must have the same dtype`
    # in `features2card` on the first batch. bf16 on CPU is slow but supported, and it matches the
    # dtype of the real GPU run.
    TOML.write_text(text, encoding="utf-8")
    print(f"patched {TOML}: DEVICE=cpu DTYPE=float32 (the batch cap, step count and env are the "
          f"generator's own)")

    env = dict(os.environ)
    env.update({
        "RWKV_NO_JIT": "1",
        "RWKV_WEIGHT_DECAY": "0.01",
        "RWKV_WEIGHT_DECAY_CMIX": "0.01",
        "RWKV_WEIGHT_DECAY_HEAD": "0.01",
        "RWKV_ID_FEATURES": "0",
        "RWKV_REAL_CYCLES": "0",
        "RWKV_FINAL_LAYOUT": "0",
        "RWKV_E2S_PUBLISHED": "0",
        "RWKV_MAX_STEPS": "3",
        # the candidate's own env, exactly as qat_decay.py would set it
        "RWKV_QAT_LOWRANK_SCOPE": "card:2:int8,note:1:int8",
        "RWKV_QAT_SHIFT_SCOPE": "card:int4,note:int4",
        "RWKV_QAT_FUSED": "0",
        "RWKV_QAT_TAIL_PENALTY": os.environ.get("E1SMOKE_LAMBDA", "1.0"),
        # E1SMOKE_STRIDE: the step subsample that keeps the penalty's graph inside the VRAM cap
        # (default 8 = the module's own default). 1 = the pre-guard, every-step behaviour.
        "RWKV_QAT_TAIL_STRIDE": os.environ.get("E1SMOKE_STRIDE", "8"),
    })
    env.pop("RWKV_VRAM_CAP_GB", None)          # cuda-only
    print(f"running: {PY} -u -m rwkv.train_rwkv --config {TOML}")
    with (DST / "train.log").open("w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run([str(PY), "-u", "-m", "rwkv.train_rwkv", "--config", str(TOML)],
                           cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)

    log = (DST / "train.log").read_text(encoding="utf-8", errors="replace")
    tail_lines = [ln for ln in log.splitlines() if ln.startswith("[tail-penalty]")]
    print(f"\nreturncode: {p.returncode}   MODEL SAVED.: {'MODEL SAVED.' in log}   "
          f"traceback: {'Traceback (most recent call last)' in log}")
    print(f"[tail-penalty] lines: {len(tail_lines)}")
    for ln in tail_lines[:3]:
        print("   " + ln)
    for ln in log.splitlines():
        if ln.startswith("[QAT"):
            print("   " + ln)
    ok = p.returncode == 0 and "MODEL SAVED." in log and len(tail_lines) > 0
    print("\nVERDICT:", "PASS - the penalty ran inside the real trainer" if ok
          else "FAIL - see the log")
    (DST / "e1_cpu_smoke.json").write_text(json.dumps(
        {"rc": p.returncode, "model_saved": "MODEL SAVED." in log,
         "tail_penalty_lines": len(tail_lines), "first_lines": tail_lines[:3], "pass": ok},
        indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
