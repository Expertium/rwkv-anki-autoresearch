"""Add two checks to preflight_runner.py, one per failure of 2026-08-18.

CHECK 1 -- GUARD/VALUE DESYNC. `findstr /C:"alpha FIXED at V"` must agree with the last
`set RWKV_KD_ALPHA=X` that precedes it. Two runs hit this in one day, in OPPOSITE directions:
iter 54 phase 2a set the wrong alpha and its correct guard caught it after 3.3 h of decay;
`decayshape` set the right alpha and shipped a guard testing the wrong one, which would have
REJECTED a good 3.3 h run at the very end. The pair is the argument -- checking only the env or
only the guard catches one of the two.

CHECK 2 -- MISSING ARTIFACT GATE. A training phase must be followed by an `if not exist ...pth`
before the next phase consumes its output. `train_rwkv` swallowed a fatal AttributeError and still
exited 0, so the resume runner logged `WS OK` after 8 seconds and marched on to decay a half-trained
model. Exit code 0 is not evidence a phase ran.
"""
import io
import re

P = "scratchpad/preflight_runner.py"
s = io.open(P, encoding="utf-8", newline="").read()

ANCHOR = "    # ---- terminal line ---"
assert ANCHOR in s, "anchor moved"

NEW = '''    # ---- guard/value desync ---------------------------------------------------------------
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
            mm = re.match(r"\\s*set\\s+RWKV_KD_ALPHA=(.*)$", prev)
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
               (re.match(r"\\s*set\\s+RWKV_KD_ALPHA=(.*)$", l) for l in lines)
               if mm and mm.group(1).strip()}
    if _alphas:
        for n, ln in enumerate(lines):
            st = ln.strip()
            if not st.lower().startswith("echo"):
                continue
            named = re.findall(r"alpha[^0-9\\r\\n]{0,12}([0-9]+[.][0-9]+)", st, re.I)
            if named and not (set(named) & _alphas):
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
        window = "\\n".join(lines[n:n + 40])
        if not re.search(r"if not exist .*\\.pth", window):
            problems.append(
                f"training phase at line {n + 1} has no `if not exist ...pth` artifact gate within "
                f"40 lines -- exit code 0 is not evidence the phase ran")

'''

s = s.replace(ANCHOR, NEW + ANCHOR, 1)
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("patched", P)
