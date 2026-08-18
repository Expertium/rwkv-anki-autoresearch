"""Build a mid-epoch resume config after a crash/reboot (2026-07-23).

Usage:
  python scratchpad/make_resume.py <run_dir> <prefix> <ws_toml> [out_toml]

Finds the newest checkpoint pair {prefix}_{N}.pth + {prefix}_optim_{N}.pth in
run_dir, copies the optimizer file to the loader's expected name
({prefix}_{N}_optim.pth), and writes a resume toml (LOAD_MODEL=true,
LOAD_MODEL_NAME={prefix}_{N}, STEP_OFFSET=N+1) next to the original.

The resume RUN must set RWKV_RESUME_SKIP_GROUPS=1 (train_rwkv then skips the
already-trained group prefix — without it a 1-epoch resume re-trains early data and
drops the tail). Craft the resume .cmd from the original run cmd: same full env +
RWKV_RESUME_SKIP_GROUPS=1, WS phase pointed at the resume toml, do NOT delete the
existing step-trace files (the trace continues), decay/eval/gate phases unchanged.

Caveats (documented in train_rwkv.py): the resumed tail's dropout draws differ from
an uninterrupted run (weights/optimizer exact; statistically equivalent); a
RWKV_GRAD_STATS json from a resumed WS covers only the tail steps.
"""

import shutil
import sys
from pathlib import Path


def main():
    run_dir = Path(sys.argv[1])
    prefix = sys.argv[2]
    ws_toml = Path(sys.argv[3])
    out_toml = Path(sys.argv[4]) if len(sys.argv) > 4 else ws_toml.with_name(
        ws_toml.stem + "_resume.toml")

    steps = []
    for p in run_dir.glob(f"{prefix}_*.pth"):
        stem = p.stem[len(prefix) + 1:]
        if stem.isdigit() and (run_dir / f"{prefix}_optim_{stem}.pth").exists():
            steps.append(int(stem))
    if not steps:
        sys.exit(f"no resumable checkpoint pairs for prefix '{prefix}' in {run_dir}")
    n = max(steps)

    src = run_dir / f"{prefix}_optim_{n}.pth"
    dst = run_dir / f"{prefix}_{n}_optim.pth"
    if not dst.exists():
        shutil.copyfile(src, dst)
        print(f"copied {src.name} -> {dst.name}")

    # REPLACE-OR-APPEND, not replace-only. A from-scratch WS toml -- which every runner in this
    # repo clones -- declares LOAD_MODEL and STEP_OFFSET but NOT LOAD_MODEL_FOLDER /
    # LOAD_MODEL_NAME, because it never loads anything. The original loop only rewrote keys that
    # were ALREADY PRESENT, so those two were silently omitted and the resume died on
    #     AttributeError: 'Namespace' object has no attribute 'LOAD_MODEL_FOLDER'
    # which train_rwkv then SWALLOWED, exiting 0 -- so the runner logged "WS OK" after 8 seconds
    # and marched on to decay from a half-trained checkpoint. Cost one aborted relaunch after the
    # 2026-08-18 power outage. Asserting the four keys are in the OUTPUT is what makes it safe.
    want = {
        "LOAD_MODEL": "LOAD_MODEL = true",
        "LOAD_MODEL_FOLDER": f'LOAD_MODEL_FOLDER = "{run_dir.as_posix()}"',
        "LOAD_MODEL_NAME": f'LOAD_MODEL_NAME = "{prefix}_{n}"',
        "STEP_OFFSET": f"STEP_OFFSET = {n + 1}",
    }
    text = ws_toml.read_text(encoding="utf-8")
    out_lines, seen = [], set()
    for line in text.splitlines():
        key = line.split("=")[0].strip() if "=" in line else ""
        if key in want:
            line = want[key]
            seen.add(key)
        out_lines.append(line)
    missing = [k for k in want if k not in seen]
    if missing:
        out_lines.append("")
        out_lines.append(f"# appended by make_resume.py -- absent from {ws_toml.name}")
        out_lines.extend(want[k] for k in missing)
    body = "\n".join(out_lines) + "\n"
    present = {l.split("=")[0].strip() for l in body.splitlines() if "=" in l}
    for k in want:
        assert k in present, f"resume toml lacks {k} -- train_rwkv would die AND still exit 0"
    out_toml.write_text(body, encoding="utf-8")
    if missing:
        print(f"appended keys absent from the source toml: {', '.join(missing)}")
    print(f"wrote {out_toml} (resume from step {n}, STEP_OFFSET={n + 1})")
    print("RUN WITH: RWKV_RESUME_SKIP_GROUPS=1 + the original run's full env; "
          "do NOT delete existing step-trace files.")


if __name__ == "__main__":
    main()
