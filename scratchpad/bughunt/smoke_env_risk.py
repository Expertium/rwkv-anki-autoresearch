"""Which smokes are at LIVE risk of a false-green control arm?

THE FAILURE (rgate, 2026-08-18). `run_iter55.cmd` does `set RWKV_RGATE=card` BEFORE calling the
smoke, and the smoke built its arms with `dict(os.environ, **extra)` -- so the OFF arm inherited the
lever and both arms were gated. Its inertness check then passed **vacuously at 0.000e+00**, comparing
two treated models to each other. A test that reads its CONTROL's configuration from the ambient
environment is not a control.

INHERITANCE ALONE IS NOT THE RISK, which is why counting `dict(os.environ` overstates it. Three
conditions must all hold:

  1. the smoke inherits os.environ without stripping its own lever, AND
  2. the smoke HAS a control arm (something it asserts is inert / unchanged / equal), AND
  3. some runner SETS that lever before invoking the smoke.

Condition 3 is what makes it live. A smoke whose lever no runner ever pre-sets can only be
contaminated by a human exporting the variable by hand.

Reports each smoke with the conditions it meets, so the list can be read as a risk ranking rather
than a raw count.
"""
import glob
import io
import os
import re

smokes = sorted(glob.glob("scratchpad/**/smoke*.py", recursive=True))
runners = sorted(glob.glob("scratchpad/**/*.cmd", recursive=True))

# Which runner sets which RWKV_ var before invoking which smoke?
preset = {}   # smoke basename -> set of vars the calling runner sets before the call
for r in runners:
    try:
        t = io.open(r, encoding="ascii", errors="replace", newline="").read()
    except OSError:
        continue
    lines = t.replace("\r\n", "\n").split("\n")
    for n, l in enumerate(lines):
        m = re.search(r"(smoke[A-Za-z0-9_]*\.py|smoke[A-Za-z0-9_]*\.sh)", l)
        if not m:
            continue
        before = set()
        for prev in lines[:n]:
            mm = re.match(r"\s*set\s+(RWKV_[A-Z0-9_]+)=(.+)$", prev)
            if mm and mm.group(2).strip():
                before.add(mm.group(1))
        preset.setdefault(m.group(1), set()).update(before)

rows = []
for s in smokes:
    base = os.path.basename(s)
    try:
        src = io.open(s, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    inherits = "dict(os.environ" in src or "os.environ.copy()" in src
    if not inherits:
        continue
    stripped = set(re.findall(r'pop\(\s*[\'"](RWKV_[A-Z0-9_]+)', src))
    stripped |= set(re.findall(r'[\'"](RWKV_[A-Z0-9_]+)[\'"]\s*,?\s*\)?\s*(?:#.*)?$', "\n".join(
        l for l in src.split("\n") if "_SMOKE_VARS" in l or "pop" in l), re.M))
    # the smoke's own lever(s): RWKV_ vars it sets in its arm dicts
    levers = set(re.findall(r'[\'"](RWKV_[A-Z0-9_]+)[\'"]\s*:', src))
    has_control = bool(re.search(r"inert|must be exactly 0|0\.000e\+00|OFF|control", src, re.I))
    risky = sorted((levers - stripped) & preset.get(base, set()))
    rows.append((base, sorted(levers), sorted(stripped), has_control, risky))

print("%-30s %-7s %-8s %s" % ("smoke", "control", "strips", "LIVE RISK (lever pre-set by a runner)"))
print("-" * 100)
live = 0
for base, levers, stripped, ctl, risky in sorted(rows, key=lambda r: (not r[4], r[0])):
    if risky:
        live += 1
    print("%-30s %-7s %-8s %s" % (base, "yes" if ctl else "-", len(stripped),
                                  ", ".join(risky) if risky else "-"))
print("\n%d smokes inherit os.environ; %d are a LIVE risk (all three conditions)" % (len(rows), live))
