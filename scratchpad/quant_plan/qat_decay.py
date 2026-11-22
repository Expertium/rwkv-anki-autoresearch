"""Run a decay phase with quantization-aware training (QAT) or as its fp32 control.

This is the "redo the decay with QAT" option of Andrew's constraint 4. The recipe, data, optimizer,
schedule and weight decay come from a base train.toml (by default the controller's own final-QAT
recipe, `scratchpad/adaptive_wd/adaptive_fullrun_92_20261105/final_qat/train.toml`), and the ONLY
change is the state-quantization env taken from a candidate in `quant_sweep.CANDIDATES`.

The controller's QAT stage never ran (its qat_env.json sets flags no code reads). This script is the
fixed version of that stage: it writes the validation-users file the trainer opens at startup, picks
a 2-user internal validation roster that does NOT overlap the evaluation roster, and refuses to
launch if the quantization env is empty (unless --no-quant is given, which is a real control arm).

ANCHOR REDO (Andrew 2026-11-19: anchor = iter36). iter36's own decay is a 124-step phase that ran
from the WS-final checkpoint (see NOTES.md "Anchor investigation"), so the faithful replay is:

  QAT:     --candidate c2_nl1i8 --start-folder pretrain/rwkv/opt_iter36 --start-name rwkv_iter36_558
           --train-users 1 100 --max-train-len 66000 --max-steps 124 --prefix qat  --tag it36qat
  CONTROL: the same command with --no-quant and --prefix ctl --tag it36ctl

--max-steps sets RWKV_MAX_STEPS, which caps total_steps AND makes the LR anneal over exactly those
steps (train_rwkv.py:1029-1039, 1237-1259). The old 1-100 training db is GONE, so the replay trains
on the 5k db restricted to users 1-100 (probe_groups.py: 160 groups/epoch at MAX=66000, full
coverage). The control run is what tells us whether that substitution still reproduces the anchor.

Usage:
  .venv/Scripts/python.exe scratchpad/quant_plan/qat_decay.py --candidate l1i4 --epochs 0.04 \
      --tag l1i4smoke [--validate-every 50]
  .venv/Scripts/python.exe scratchpad/quant_plan/qat_decay.py --candidate c2_nl1i8 --epochs 2 \
      --max-steps 124 --train-users 1 100 --max-train-len 66000 \
      --start-folder pretrain/rwkv/opt_iter36 --start-name rwkv_iter36_558 --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratchpad" / "quant_plan"))
import quant_sweep as qs  # noqa: E402

BASE_TOML = (ROOT / "scratchpad" / "adaptive_wd" / "adaptive_fullrun_92_20261105" / "final_qat"
             / "train.toml")
ROSTER = ROOT / "scratchpad" / "quant_plan" / "roster750_seed20261119.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"


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
        raise SystemExit(f"keys not present in the base toml: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="name from quant_sweep.CANDIDATES")
    ap.add_argument("--epochs", type=float, required=True, help="1 = the full decay epoch, 0.04 = smoke")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--validate-every", type=int, default=0, help="0 = derive (500 full, 50 smoke)")
    ap.add_argument("--start-folder", default="",
                    help="checkpoint dir to train FROM (default: the base toml's LOAD_MODEL_FOLDER)")
    ap.add_argument("--start-name", default="", help="checkpoint file stem without .pth")
    ap.add_argument("--start-optim", default="", help="optimizer file name inside --start-folder")
    ap.add_argument("--peak-lr", type=float, default=0.0, help="0 = keep the base toml's PEAK_LR")
    ap.add_argument("--dry-run", action="store_true", help="write the toml + validation file, do not train")
    ap.add_argument("--no-quant", action="store_true",
                    help="CONTROL run: same recipe, quantization env NOT applied (not a QAT run)")
    ap.add_argument("--train-users", type=int, nargs=2, default=[0, 0],
                    help="override TRAIN_USERS_START/END (0 0 = keep the base toml)")
    ap.add_argument("--max-train-len", type=int, default=0,
                    help="override MAX_TRAIN_GLOBAL_LEN (0 = keep the base toml)")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="RWKV_MAX_STEPS: cap the run to N steps AND anneal the LR over exactly those N")
    ap.add_argument("--prefix", default="qat", help="SAVE_MODEL_PREFIX (default qat)")
    args = ap.parse_args()

    if args.candidate not in qs.CANDIDATES:
        raise SystemExit(f"unknown candidate {args.candidate}; known: {sorted(qs.CANDIDATES)}")
    env_extra = dict(qs.CANDIDATES[args.candidate][0] or {})
    if args.no_quant:
        # Andrew's control arm: identical recipe, start checkpoint, data and step budget, but NO
        # quantization. It is the unquantized twin the QAT run must beat, and it also shows whether
        # today's data still reproduces the anchor's own loss.
        print(f"CONTROL run (--no-quant): candidate {args.candidate}'s quantization env is NOT applied")
    else:
        if not env_extra:
            raise SystemExit("refusing to launch: this candidate sets no quantization env (that would "
                             "be a plain fp32 run; pass --no-quant if that is what you mean)")
        bits = qs.bits_for(env_extra, qs.CANDIDATES[args.candidate][2])
        over = [s for s in qs.LAYERS if bits[s] > qs.BAR_BITS[s]]
        print(f"candidate {args.candidate}: bits {bits}  "
              f"{'OVER BAR: ' + ','.join(over) if over else 'inside the bit bar'}")

    dst = ROOT / "scratchpad" / "quant_plan" / f"qat_{args.tag}"
    dst.mkdir(parents=True, exist_ok=True)

    roster = set(json.loads(ROSTER.read_text(encoding="utf-8"))["users"])
    excluded = {5626, 6630, 6701, 6810}
    internal = [u for u in range(5001, 7501) if u not in roster and u not in excluded][:2]
    if len(internal) < 2:
        raise SystemExit("could not pick two internal validation users")
    (dst / "validation_users.json").write_text(json.dumps(internal), encoding="utf-8")
    print(f"internal validation users: {internal}")

    steps = args.epochs
    edits = {
        "EPOCHS": steps,
        "VALIDATE_EVERY": args.validate_every or (50 if args.epochs < 1 else 500),
        "VALIDATE_USERS_START": min(internal),
        "VALIDATE_USERS_END": max(internal),
        "VALIDATE_USERS_FILE": str(dst / "validation_users.json"),
        "SAVE_MODEL_FOLDER": str(dst),
        "SAVE_MODEL_PREFIX": "qat",
    }
    if args.start_folder:
        folder = Path(args.start_folder).resolve()
        if not folder.is_dir():
            raise SystemExit(f"--start-folder does not exist: {folder}")
        name = args.start_name or ""
        if not name:
            raise SystemExit("--start-folder needs --start-name (checkpoint stem without .pth)")
        ckpt = folder / f"{name}.pth"
        if not ckpt.exists():
            candidates = sorted(p.stem for p in folder.glob("*.pth") if "optim" not in p.stem)
            raise SystemExit(f"no {ckpt.name} in {folder}; available: {candidates[-8:]}")
        optim = args.start_optim or ""
        if not optim:
            step = name.rsplit("_", 1)[-1]
            matches = sorted(folder.glob(f"*optim*{step}*.pth"))
            if len(matches) != 1:
                raise SystemExit(f"cannot pick the optimizer for {name}: matches={[p.name for p in matches]}; "
                                 f"pass --start-optim")
            optim = matches[0].name
        optim_path = folder / optim
        if not optim_path.exists():
            opts = sorted(p.name for p in folder.glob("*optim*.pth"))
            raise SystemExit(f"no optimizer {optim} in {folder}; available: {opts[-8:]}")
        edits["LOAD_MODEL_FOLDER"] = str(folder)
        edits["LOAD_MODEL_NAME"] = name
        edits["LOAD_OPTIM_PATH"] = str(optim_path)
        print(f"start checkpoint: {ckpt}\nstart optimizer:  {optim_path}")
    if args.peak_lr:
        edits["PEAK_LR"] = args.peak_lr
    if args.train_users[0]:
        edits["TRAIN_USERS_START"] = args.train_users[0]
        edits["TRAIN_USERS_END"] = args.train_users[1]
    if args.max_train_len:
        edits["MAX_TRAIN_GLOBAL_LEN"] = args.max_train_len
    if args.prefix:
        edits["SAVE_MODEL_PREFIX"] = args.prefix
    edit_toml(BASE_TOML, dst / "train.toml", edits)

    env = dict(os.environ)
    if not args.no_quant:
        env.update({k: str(v) for k, v in env_extra.items()})
    env.update({
        "RWKV_NO_JIT": "1",
        "RWKV_WEIGHT_DECAY": "0.01",
        "RWKV_WEIGHT_DECAY_CMIX": "0.01",
        "RWKV_WEIGHT_DECAY_HEAD": "0.01",
        "RWKV_VRAM_CAP_GB": "11",
        "RWKV_ID_FEATURES": "0",
        "RWKV_REAL_CYCLES": "0",
        "RWKV_FINAL_LAYOUT": "0",
        "RWKV_E2S_PUBLISHED": "0",
    })
    if not args.no_quant:
        # The fused kernel is the default for QAT runs. A CANDIDATE may ask for the Python per-step
        # loop instead (RWKV_QAT_FUSED=0), which is bit-parity with the fused kernels: the
        # tail-energy penalty needs it, because the fused rank-1 kernel truncates inside the kernel
        # where the pre-truncation state is invisible. Only a candidate can do this -- a stray value
        # in the parent shell must not silently slow every run down.
        env["RWKV_QAT_FUSED"] = str(env_extra.get("RWKV_QAT_FUSED", "1"))
        if env["RWKV_QAT_FUSED"] != "1":
            print(f"[qat_decay] RWKV_QAT_FUSED={env['RWKV_QAT_FUSED']} requested by candidate "
                  f"{args.candidate}: the Python per-step QAT loop (bit-parity twin)")
    if args.max_steps:
        env["RWKV_MAX_STEPS"] = str(args.max_steps)
        print(f"step cap: RWKV_MAX_STEPS={args.max_steps} (total_steps and the LR anneal use it)")
    (dst / "qat_env.json").write_text(json.dumps({**env_extra, "RWKV_NO_JIT": "1"}, indent=2),
                                      encoding="utf-8")
    log = dst / "train.log"
    if args.dry_run:
        print(f"[dry-run] wrote {dst / 'train.toml'} and {dst / 'validation_users.json'}")
        print((dst / "train.toml").read_text(encoding="utf-8"))
        return
    print(f"launching: {PY} -u -m rwkv.train_rwkv --config {dst / 'train.toml'}\nlog: {log}")
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run([str(PY), "-u", "-m", "rwkv.train_rwkv", "--config", str(dst / "train.toml")],
                           cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    text = log.read_text(encoding="utf-8", errors="replace")
    saved = "MODEL SAVED." in text
    tb = "Traceback (most recent call last)" in text
    print(f"returncode: {p.returncode}  MODEL SAVED.: {saved}  traceback: {tb}")
    qat_lines = [ln for ln in text.splitlines() if ln.startswith("[QAT")]
    print("quant env banners:")
    for ln in qat_lines:
        print("   " + ln)
    if args.no_quant:
        if qat_lines:
            raise SystemExit("CONTROL run printed [QAT..] banners -- quantization was NOT off")
        print("   none, as expected for a control run")
    elif not qat_lines:
        raise SystemExit("NONE -- quantization was INERT, this run is not a QAT run")
    print("--- tail ---")
    print("\n".join(text.splitlines()[-12:]))
    if not saved or tb:
        raise SystemExit("QAT epoch did not complete cleanly")


if __name__ == "__main__":
    main()
