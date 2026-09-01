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
    # ⚠ MIXED LINE ENDINGS MUST BE REPORTED, NOT SILENTLY PARSED (fixed 2026-08-30).
    # The splitter below prefers "\r\n" whenever any CRLF is present. On a file that is mostly LF
    # with a few CRLF lines -- which is what you get by editing a Write-tool-authored runner from
    # PowerShell -- that collapses 96 real lines into 8 giant ones, and EVERY check downstream
    # then reports confident nonsense: variables "never set", a missing `cd /d` that is present,
    # and a DONE_EXIT_0 line that cannot be found. Three misleading findings and no hint of the
    # real cause. Mixed endings are also a genuine hazard in their own right: cmd.exe re-reads a
    # batch file from a saved BYTE OFFSET, which is what truncated iter 46's training log.
    n_crlf = text.count("\r\n")
    n_lone_lf = text.count("\n") - n_crlf
    if n_crlf and n_lone_lf:
        return [f"MIXED line endings ({n_crlf} CRLF, {n_lone_lf} lone LF) -- normalize the file "
                f"before trusting any other check; a mixed file parses as garbage here and is a "
                f"byte-offset hazard for cmd.exe"], []
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
        # ⚠ A REM line MENTIONING these tools is prose, not an invocation. Without this the
        # parser reads the comment's words as positional arguments and reports a missing
        # checkpoint that no runner ever asked for. Same false-positive class as the %VAR%
        # check, which already skips REM.
        if ln.strip().upper().startswith("REM"):
            continue
        m = re.search(r"(write_decay_setup|write_eval_toml)\.py\s+(.+)$", ln)
        if not m:
            continue
        args = m.group(2).split(">")[0].split()
        for a in args:
            if a.endswith(".toml"):
                generated.setdefault(os.path.normpath(expand(a)).replace("\\", "/").lower(), n)
                break

    for n, ln in enumerate(lines):
        # ⚠ REM lines mentioning these tools are prose, not invocations (see the note at the
        # config-generation loop above). Every parser over `lines` needs this guard, not just
        # the first one -- which is why the first fix did not stop the false positive.
        if ln.strip().upper().startswith("REM"):
            continue
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
        # ⚠ REM lines mentioning these tools are prose, not invocations (see the note at the
        # config-generation loop above). Every parser over `lines` needs this guard, not just
        # the first one -- which is why the first fix did not stop the false positive.
        if ln.strip().upper().startswith("REM"):
            continue
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
        # ⚠ REM lines mentioning these tools are prose, not invocations (see the note at the
        # config-generation loop above). Every parser over `lines` needs this guard, not just
        # the first one -- which is why the first fix did not stop the false positive.
        if ln.strip().upper().startswith("REM"):
            continue
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

    # ---- the training phase must keep the speed-stack env ---------------------------------
    # A phase-0 guard legitimately has to CLEAR RWKV_NO_JIT (a scripted eval is the whole point of
    # it), and forgetting to restore it silently trains a different configuration from the
    # champion's -- torch.compile cannot trace a ScriptModule, so RWKV_QAT_COMPILE quietly stops
    # doing anything. Nothing else in the pipeline notices. Checked by walking the file in order,
    # because only the LAST assignment before each train_rwkv call is the one that applies.
    for n, ln in enumerate(lines):
        if "train_rwkv" not in ln or "--config" not in ln:
            continue
        val = None
        for prev in lines[:n]:
            m = re.match(r"\s*set\s+RWKV_NO_JIT=(.*)$", prev)
            if m:
                val = m.group(1).strip()
        if val is None:
            notes.append(f"train_rwkv at line {n + 1}: RWKV_NO_JIT never set (plain-JIT run?)")
        elif val != "1":
            problems.append(
                f"train_rwkv at line {n + 1} runs with RWKV_NO_JIT='{val}' -- it was cleared "
                f"earlier and not restored; torch.compile/QAT_COMPILE cannot trace a ScriptModule")

    # ---- guard/value desync ---------------------------------------------------------------
    # A `findstr` guard that names a NUMBER must agree with the env var that sets it. Both
    # directions bit on 2026-08-18: iter 54 phase 2a set alpha 0.9 where 0.5 was meant (its
    # correct guard caught it, after 3.3 h of wasted decay), and `decayshape` set 0.5 but shipped
    # a guard testing 0.9, which would have rejected a GOOD 3.3 h run at the end. Checking only
    # one side catches only one of the two.
    for n, ln in enumerate(lines):
        m = re.search(r'findstr /C:"alpha FIXED at ([0-9.]+)"', ln)
        if not m:
            continue
        guard_val = m.group(1)
        env_val = None
        for prev in lines[:n]:
            mm = re.match(r"\s*set\s+RWKV_KD_ALPHA=(.*)$", prev)
            if mm and mm.group(1).strip():
                env_val = mm.group(1).strip()
        if env_val is None:
            problems.append(
                f"guard at line {n + 1} checks 'alpha FIXED at {guard_val}' but RWKV_KD_ALPHA "
                f"is never set before it")
        elif abs(float(env_val) - float(guard_val)) > 1e-9:
            problems.append(
                f"GUARD/VALUE DESYNC at line {n + 1}: the guard checks alpha {guard_val} but the "
                f"env sets RWKV_KD_ALPHA={env_val}. One of them is a leftover from the runner this "
                f"was cloned from; the run is wrong either way.")

    # An ECHOED line that names an alpha must name the one this run actually uses -- the log is
    # what a verdict gets read from months later. kdalpha025 announced 'alpha 0.9' for a run whose
    # alpha is 0.25 (found 2026-08-19; guards were correct, so the RUN was fine and the RECORD was
    # not). REM lines are exempt: they legitimately discuss the value being moved away from.
    _alphas = {mm.group(1).strip() for mm in
               (re.match(r"\s*set\s+RWKV_KD_ALPHA=(.*)$", l) for l in lines)
               if mm and mm.group(1).strip()}
    if _alphas:
        for n, ln in enumerate(lines):
            st = ln.strip()
            if not st.lower().startswith("echo"):
                continue
            named = re.findall(r"alpha[^0-9\r\n]{0,12}([0-9]+[.][0-9]+)", st, re.I)
            # The line may legitimately name the value moved FROM ("0.5 to 0.25"); flag it
            # only when THIS run's alpha appears nowhere on the line at all.
            if named and not any(a in st for a in _alphas):
                notes.append(
                    f"line {n + 1} ECHOES alpha {named} into the log, but this runner only ever "
                    f"sets {sorted(_alphas)} -- stale prose from a cloned runner")

    # ---- artifact gates -------------------------------------------------------------------
    # Every training phase must be followed by an `if not exist ...pth` before anything consumes
    # its output. train_rwkv swallowed a fatal AttributeError and still exited 0, so the runner
    # logged "WS OK" after 8 seconds and went on to decay a half-trained model (2026-08-18).
    for n, ln in enumerate(lines):
        if "train_rwkv --config" not in ln and "train_rwkv" not in ln:
            continue
        if "python" not in ln.lower():
            continue
        window = "\n".join(lines[n:n + 40])
        if not re.search(r"if not exist .*\.pth", window):
            problems.append(
                f"training phase at line {n + 1} has no `if not exist ...pth` artifact gate within "
                f"40 lines -- exit code 0 is not evidence the phase ran")

    # ---- terminal line --------------------------------------------------------------------
    # endlocal must NOT precede the terminal marker. endlocal restores the pre-setlocal
    # environment, so %LOG% is EMPTY after it and `>> "%LOG%"` appends to nothing -- the runner
    # exits 0, writes no marker, and every chained waiter polls forever. Cost 45 min of idle GPU
    # on 2026-08-18 after iter 53 had ALREADY finished cleanly. The vars-before-use check above
    # cannot catch it: endlocal invalidates variables mid-file, not at a line the parser sees.
    # ⚠ NESTING IS LEGITIMATE AND THE OLD CHECK COULD NOT SEE IT (fixed 2026-08-30). A runner that
    # isolates each phase's env -- `setlocal` ... teacher flags ... `endlocal` ... `setlocal` ...
    # student flags -- pops only the INNER scope, so a %LOG% set in the OUTER scope survives.
    # Verified by EXECUTION, not by reading: a nested endlocal leaves %LOG% intact and the marker
    # is written correctly. The old "any endlocal before any marker" heuristic fired on every such
    # runner, and a guard that cries wolf on correct files is a guard that gets ignored -- which
    # would eventually let the real bug through. Track DEPTH and complain only when an endlocal
    # closes the OUTERMOST scope (depth 1 -> 0) before the marker, which is the case that really
    # empties %LOG%.
    _depth = 0
    _fatal_el = None
    for n, ln in enumerate(lines):
        s = ln.strip().lower()
        if s == "setlocal" or s.startswith("setlocal "):
            _depth += 1
        elif s == "endlocal" or s.startswith("endlocal "):
            _depth -= 1
            if _depth <= 0 and _fatal_el is None:
                _fatal_el = n
    _de = [n for n, ln in enumerate(lines) if ln.strip().startswith("echo DONE_EXIT_")]
    if _fatal_el is not None and _de and _fatal_el < max(_de):
        problems.append(
            f"endlocal at line {_fatal_el + 1} closes the OUTERMOST scope before the DONE_EXIT_ "
            f"echo at line {max(_de) + 1}: %LOG% is out of scope there, so the marker is silently "
            f"never written")
    # Count only lines that EMIT the marker, not lines that READ it. A gated waiter legitimately
    # tests a predecessor's exit code with `findstr /B /C:"DONE_EXIT_0 "`, and the old substring
    # test counted those as extra emissions and failed a correct runner. The `endlocal` check three
    # lines up already used this predicate; this one had drifted from it.
    _emit = [ln for ln in lines if ln.strip().lower().startswith("echo done_exit_0")]
    if len(_emit) != 1:
        problems.append(
            "expected exactly one emitted DONE_EXIT_0 line (the runner's success signal), found "
            f"{len(_emit)}")

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
