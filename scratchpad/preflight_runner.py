"""Pre-flight a queued run `.cmd` BEFORE it is armed: would it actually get off the ground?

WHY THIS EXISTS (2026-08-17). Iters 53 and 54 sat armed and chained for six hours in a state where
BOTH would have died in their first seconds, for two independent reasons, and neither would have
produced a diagnosable error:

  1. mk53/mk54 built the runner as HEADER + `s[s.index("setlocal"):]`, and in the source runner the
     `cd /d` and the whole DIR/LOG/STAMP/DUMP/WSSTEPS/MAXSTEPS block sit BEFORE `setlocal`. Every
     %VAR% expanded to empty; `>> "%LOG%"` is a syntax error; the runner exits without ever writing
     a DONE_EXIT_ line, so a downstream waiter hangs forever.
  2. Both generators wrote the runner but never the WS toml it passes to `--config`, so even with
     (1) fixed the training phase would have died on a missing file.

Both were found by hand, one at a time, by comparing three runners. That is the wrong method: the
checks are mechanical, so they should be mechanical. Run this on every runner before arming it.

WHAT IT CHECKS
  * `cd /d` present -- Win32_Process.Create starts a detached job in System32, where
    `.venv\\Scripts\\python.exe` does not exist.
  * every %VAR% the runner uses is `set` before its first use (built-ins excepted).
  * every `--config <toml>` either exists on disk or is GENERATED earlier in the same runner by
    write_decay_setup.py / write_eval_toml.py (matched on that tool's output-path argument).
  * every checkpoint prefix handed to write_decay_setup.py has a real non-optim `{prefix}_{step}.pth`
    to decay from.
  * a KD dump named in RWKV_KD_MIX exists and actually contains the step it will be replayed to.
  * the runner can still write its terminal line: exactly one `DONE_EXIT_0`, and %LOG% resolvable.

It does NOT check semantics -- whether the lever is the one you meant, or the gate basis is right.
It checks that the thing can RUN. Usage:

    .venv/Scripts/python.exe scratchpad/preflight_runner.py scratchpad/iter55_rgate/run_iter55.cmd [...]
    .venv/Scripts/python.exe scratchpad/preflight_runner.py --all
"""
import glob
import io
import os
import re
import sys

BUILTIN = {"DATE", "TIME", "ERRORLEVEL", "RANDOM", "CD", "PATH", "TEMP", "TMP",
           "USERPROFILE", "COMPUTERNAME", "PROCESSOR_ARCHITECTURE"}


def preflight(path):
    problems = []
    notes = []
    text = io.open(path, encoding="ascii", errors="replace", newline="").read()
    lines = text.split("\r\n") if "\r\n" in text else text.split("\n")

    # ---- variables: declared before first use --------------------------------------------
    setpos, usepos = {}, {}
    for n, ln in enumerate(lines):
        m = re.match(r"\s*set\s+([A-Za-z_][A-Za-z0-9_]*)=", ln)
        if m and m.group(1).upper() not in setpos:
            setpos[m.group(1).upper()] = n
        for var in re.findall(r"%([A-Za-z_][A-Za-z0-9_]*)%", ln):
            if var.upper() not in usepos:
                usepos[var.upper()] = n
    for var, un in sorted(usepos.items()):
        if var in BUILTIN:
            continue
        if var not in setpos:
            problems.append(f"%{var}% used at line {un + 1} but never set")
        elif setpos[var] > un:
            problems.append(f"%{var}% used at line {un + 1} before it is set at line {setpos[var] + 1}")

    # resolve simple vars so path checks can expand them
    env = {}
    for n, ln in enumerate(lines):
        m = re.match(r"\s*set\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", ln)
        if m and m.group(1).upper() not in env:
            val = m.group(2)
            for k, v in env.items():
                val = val.replace(f"%{k}%", v)
            if "%" not in val:
                env[m.group(1).upper()] = val

    def expand(s):
        for k, v in env.items():
            s = s.replace(f"%{k}%", v)
        return s.replace("\\", "/")

    # ---- cd /d ---------------------------------------------------------------------------
    if not any(re.match(r"\s*cd /d ", ln) for ln in lines):
        problems.append("no `cd /d <repo>` -- a detached job starts in System32 and .venv is not there")

    # ---- configs: exist, or generated earlier in this same runner ------------------------
    generated = {}   # normalized out-path -> line number
    for n, ln in enumerate(lines):
        m = re.search(r"(write_decay_setup|write_eval_toml)\.py\s+(.+)$", ln)
        if not m:
            continue
        args = m.group(2).split(">")[0].split()
        for a in args:
            if a.endswith(".toml"):
                generated.setdefault(os.path.normpath(expand(a)).replace("\\", "/").lower(), n)
                break

    for n, ln in enumerate(lines):
        for cfg in re.findall(r"--config\s+(\S+)", ln):
            p = expand(cfg).strip('"')
            key = os.path.normpath(p).replace("\\", "/").lower()
            if os.path.isfile(p):
                continue
            if key in generated and generated[key] < n:
                notes.append(f"config {os.path.basename(p)} generated at line {generated[key] + 1}")
                continue
            problems.append(f"--config {p} (line {n + 1}) does not exist and is not generated earlier")

    # ---- decay source checkpoints ---------------------------------------------------------
    # A run that trains its own WS phase produces the checkpoint later-but-in-run, so "not on disk"
    # is only a failure if no EARLIER train_rwkv writes that prefix. Resolving it through the WS
    # toml also checks the pairing that matters: SAVE_MODEL_PREFIX must be exactly what
    # write_decay_setup will later search for, or the decay silently finds nothing.
    trained = {}    # (folder, prefix) -> line number of the train_rwkv call that writes it
    for n, ln in enumerate(lines):
        if "train_rwkv" not in ln:
            continue
        m = re.search(r"--config\s+(\S+)", ln)
        if not m:
            continue
        p = expand(m.group(1)).strip('"')
        if not os.path.isfile(p):
            continue
        try:
            import tomli
            d = tomli.load(open(p, "rb"))
            key = (os.path.normpath(d["SAVE_MODEL_FOLDER"]).replace("\\", "/").lower(),
                   d["SAVE_MODEL_PREFIX"])
            trained.setdefault(key, n)
        except Exception as e:                                  # noqa: BLE001
            notes.append(f"could not read {os.path.basename(p)}: {e}")

    for n, ln in enumerate(lines):
        m = re.search(r"write_decay_setup\.py\s+(\S+)\s+(\S+)\s+(\S+)", ln)
        if not m:
            continue
        folder, srcprefix = expand(m.group(1)).strip('"'), m.group(2)
        cands = [p for p in glob.glob(f"{folder}/{srcprefix}_*.pth") if "optim" not in os.path.basename(p)]
        if cands:
            best = max(cands, key=lambda p: int(re.search(r"_(\d+)\.pth$", p).group(1)))
            notes.append(f"decays from existing {os.path.basename(best)}")
            continue
        key = (os.path.normpath(folder).replace("\\", "/").lower(), srcprefix)
        if key in trained and trained[key] < n:
            notes.append(f"decay source {srcprefix}_<step>.pth is written in-run by the "
                         f"train_rwkv at line {trained[key] + 1}")
        else:
            problems.append(
                f"write_decay_setup (line {n + 1}) wants {srcprefix}_<step>.pth in {folder}: not on "
                f"disk, and no earlier train_rwkv writes that folder+prefix")

    # ---- KD dump --------------------------------------------------------------------------
    kd = env.get("RWKV_KD_MIX")
    if kd:
        kd = expand(kd)
        if ":" in kd[2:]:                       # skip the drive-letter colon
            head, _, steps = kd.rpartition(":")
            if not os.path.isdir(head):
                problems.append(f"KD dump dir missing: {head}")
            elif steps.isdigit() and not os.path.isfile(f"{head}/step_{steps}.pt"):
                problems.append(f"KD dump {head} has no step_{steps}.pt (it will be replayed to step {steps})")
            else:
                notes.append(f"KD dump ok through step_{steps}.pt")

    # ---- terminal line --------------------------------------------------------------------
    if sum(1 for ln in lines if "DONE_EXIT_0" in ln) != 1:
        problems.append("expected exactly one DONE_EXIT_0 line (the waiter's success signal)")

    return problems, notes


def main(argv):
    targets = argv[1:]
    if targets == ["--all"]:
        targets = sorted(glob.glob("scratchpad/*/run_iter*.cmd") + glob.glob("scratchpad/*/run_*.cmd"))
        targets = [t for t in targets if "wait_then" not in t]
    if not targets:
        print(__doc__)
        return 2
    bad = 0
    for t in targets:
        if not os.path.isfile(t):
            print(f"=== {t}\n    MISSING FILE")
            bad += 1
            continue
        problems, notes = preflight(t)
        status = "PASS" if not problems else "FAIL"
        print(f"=== {t}  [{status}]")
        for nt in notes:
            print(f"    - {nt}")
        for p in problems:
            print(f"    ! {p}")
        bad += bool(problems)
    print("\nPREFLIGHT_" + ("ALL_PASS" if not bad else f"FAILED ({bad})"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
