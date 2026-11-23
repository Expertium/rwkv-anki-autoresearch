"""Export the three deploy safetensors the row-55 bundle needs, into the bundle's weights/ dir.

Idempotent (skips an export whose output already exists) and single-instance (O_EXCL lock), because
the shell tool can start the same command twice.

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/_export_row55_weights.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "scratchpad" / "row55_l2i8s4e1c_bundle" / "weights"
LOCK = ROOT / "scratchpad" / "quant_plan" / "_export_row55.lock"

JOBS = [
    ("scratchpad/quant_plan/qat_it36l2i8s4e1c/qat_124.pth",
     "deploy_iter55_l2i8s4e1c.safetensors"),
    ("scratchpad/quant_plan/qat_it36ctl/ctl_124.pth", "deploy_ctl_fp32_124.safetensors"),
    ("pretrain/rwkv/opt_iter36/rwkv_iter36_558.pth", "deploy_pre_qat_iter36.safetensors"),
]


def main():
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"lock {LOCK} exists -- another copy is running, exiting")
        return 0
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        B.mkdir(parents=True, exist_ok=True)
        for src, out in JOBS:
            dst = B / out
            if dst.exists() and dst.stat().st_size > 100_000:
                print(f"skip   {out} (exists, {dst.stat().st_size:,} bytes)")
                continue
            r = subprocess.run([sys.executable, "scratchpad/quant_plan/export_weights_only.py",
                                src, str(B), out], cwd=str(ROOT))
            size = dst.stat().st_size if dst.exists() else -1
            print(f"export {out} rc={r.returncode} size={size:,}")
            if r.returncode != 0 or size < 100_000:
                print("FAILED")
                return 1
    finally:
        LOCK.unlink(missing_ok=True)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
