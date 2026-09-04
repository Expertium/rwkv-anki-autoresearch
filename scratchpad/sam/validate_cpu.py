"""Validate the RWKV_SAM_RHO hook by EXECUTION on CPU (2026-09-04), three arms on one tiny config:

  HEAD  : the committed train_rwkv.py (a git worktree of HEAD, i.e. BEFORE the SAM edits), SAM unset
  OFF   : the working tree, SAM unset           -> every printed per-step loss must EQUAL HEAD's
  ON    : the working tree, RWKV_SAM_RHO=0.05   -> losses finite, DIFFER from OFF after step 1
                                                  (step 1's printed loss is the unperturbed one and
                                                  must equal OFF's), and the "[sam] first pass:
                                                  weights restored bit-exactly" banner must appear.

User 101 of the published e2s db, 5 chunks, DEVICE=cpu, fp32, ~1-2 min/step. The arms share the
db (read-only) and write checkpoints into separate folders. No ambient RWKV_* is inherited.
"""
import os
import re
import shutil
import subprocess
import sys

REPO = r"C:\Users\Andrew\rwkv-anki-autoresearch"
WT = r"C:\Temp\claude\sam_head_worktree"
TOML = "scratchpad/sam/ws_cpu_validate.toml"
PY = os.path.join(REPO, r".venv\Scripts\python.exe")

BASE = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
BASE.update({
    "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "6",
    "RWKV_DETERMINISTIC": "1", "RWKV_AUGMENT_SEED": "4321", "RWKV_EMPTY_CACHE_EVERY": "0",
    "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
    "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
    "RWKV_PROBE_DENSITY": "0.08", "RWKV_PROBE_DUR": "0.0",
    "RWKV_MUON": "1", "RWKV_MUON_LR": "0.0025", "RWKV_MUON_MOMENTUM": "0.95", "RWKV_MUON_INCLUDE_LORA": "1",
    "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
    "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
    "RWKV_STRIP_CMIX": "user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1",
    "RWKV_WEIGHT_DECAY": "0.01", "RWKV_WEIGHT_DECAY_HEAD": "0.01", "RWKV_CLIP": "0.25",
    "RWKV_ADAMW_BETA2": "0.999", "RWKV_DROPOUT_SCALE": "0.5", "RWKV_MUON_BATCHED": "1",
    "RWKV_NO_JIT": "1", "RWKV_ZERO_FEATURES": "",
})
STEP_RE = re.compile(r"^0 (\d+) (\d+), all: ([\d.]+), ahead: ([\d.]+) \(([\d.]+)\), imm: ([\d.]+)", re.M)


def run(cwd, extra, tag):
    env = dict(BASE, **extra, PYTHONPATH=cwd)
    r = subprocess.run([PY, "-u", "-m", "rwkv.train_rwkv", "--config", TOML], cwd=cwd, env=env,
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    open(os.path.join(REPO, f"scratchpad/sam/validate_{tag}.log"), "w", encoding="utf-8").write(out)
    steps = [(int(m.group(2)), m.group(3), m.group(4), m.group(6)) for m in STEP_RE.finditer(out)]
    return r.returncode, steps, out


def main():
    os.chdir(REPO)
    if os.path.exists(WT):
        subprocess.run(["git", "worktree", "remove", "--force", WT], cwd=REPO)
        shutil.rmtree(WT, ignore_errors=True)
    subprocess.run(["git", "worktree", "add", "--detach", WT, "HEAD"], cwd=REPO, check=True)
    # the worktree has no built CUDA .pyd and no venv, but CPU training needs neither
    os.makedirs(os.path.join(WT, "scratchpad/sam"), exist_ok=True)
    shutil.copy(os.path.join(REPO, TOML), os.path.join(WT, TOML))
    if not os.path.exists(os.path.join(WT, "label_filter_db")):
        subprocess.run(["cmd", "/c", "mklink", "/J", os.path.join(WT, "label_filter_db"), os.path.join(REPO, "label_filter_db")], check=True)
    for d in ("scratchpad/sam/cpu_validate_out",):
        shutil.rmtree(os.path.join(REPO, d), ignore_errors=True); shutil.rmtree(os.path.join(WT, d), ignore_errors=True)

    rc_h, head, out_h = run(WT, {}, "HEAD")
    print(f"HEAD rc={rc_h} steps={len(head)}: {head}")
    shutil.rmtree(os.path.join(REPO, "scratchpad/sam/cpu_validate_out"), ignore_errors=True)
    rc_o, off, out_o = run(REPO, {}, "OFF")
    print(f"OFF  rc={rc_o} steps={len(off)}: {off}")
    shutil.rmtree(os.path.join(REPO, "scratchpad/sam/cpu_validate_out"), ignore_errors=True)
    rc_s, on, out_s = run(REPO, {"RWKV_SAM_RHO": "0.05"}, "ON")
    print(f"ON   rc={rc_s} steps={len(on)}: {on}")

    ok = True
    def check(c, msg):
        nonlocal ok
        print(("  PASS " if c else "  FAIL ") + msg); ok = ok and c
    check(rc_h == 0 and rc_o == 0 and rc_s == 0, "all three arms exit 0")
    check(len(head) >= 3 and len(head) == len(off) == len(on), f"same step count ({len(head)})")
    check(head == off, "OFF (working tree, flag unset) == HEAD: byte-identical printed losses")
    check(bool(on) and bool(off) and on[0] == off[0], "ON step 1 printed loss == OFF step 1 (the printed loss is the unperturbed one)")
    check(bool(on) and len(on) > 1 and on[1:] != off[1:], "ON diverges from OFF after step 1 (SAM changed the update)")
    check("[sam] Sharpness-Aware Minimization ON: rho=0.05" in out_s, "ON: SAM banner printed")
    check("[sam] first pass: weights restored bit-exactly" in out_s, "ON: weight restore asserted bit-exact")
    check("[sam]" not in out_o and "[sam]" not in out_h, "OFF/HEAD: no SAM banner")
    check(all(float(s[1]) == float(s[1]) for s in on), "ON: all losses finite")
    print("SAM_CPU_VALIDATE " + ("PASS" if ok else "FAIL"))
    subprocess.run(["git", "worktree", "remove", "--force", WT], cwd=REPO)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
