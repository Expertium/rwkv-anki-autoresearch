"""Restore the setup block that mk53/mk54's `s[s.index("setlocal"):]` slice discarded.

Both generators built the runner as HEADER + everything-from-`setlocal`. In iter 45's runner the
`cd /d` and the DIR/LOG/STAMP/DUMP/WSSTEPS/MAXSTEPS block sit BEFORE `setlocal`, so the slice threw
them away. Every assert in those generators checks that stale text did not leak IN; none checks
that required setup survived. Result: %LOG% empty -> `>> ""` is a syntax error at phase 0, and the
runner exits without ever writing DONE_EXIT_, so the chain silently stops.
"""
import io, sys

RUNS = [
    ("scratchpad/iter53_muonlora/run_iter53.cmd", "iter53_muonlora", "iter53",
     "ITER 53 (Muon on the LoRA matrices)"),
    ("scratchpad/iter54_cmixpow/run_iter54.cmd", "iter54_cmixpow", "iter54",
     "ITER 54 (learnable channel-mixer exponent)"),
]
ROOT = r"C:\Users\Andrew\rwkv-anki-autoresearch"

for path, dirname, tag, title in RUNS:
    raw = io.open(path, "rb").read()
    text = raw.decode("ascii")
    assert "\r\n" in text, path
    assert "set DIR=" not in text, f"{path} already has DIR -- refusing to double-patch"
    lines = text.split("\r\n")
    i = lines.index("setlocal")
    block = [
        f"cd /d {ROOT}",
        f"set DIR={ROOT}\scratchpad\{dirname}",
        f"set LOG=%DIR%\{tag}.log",
        "set STAMP=%RANDOM%%RANDOM%",
        r"set DUMP=C:\rwkv_kd_dump\t128_seedpair_65k",
        "set WSSTEPS=10935",
        "REM BUDGET: 0 = full budget",
        "set MAXSTEPS=0",
        f"echo ===== {title} START %DATE% %TIME% ===== >> \"%LOG%\"",
    ]
    out = "\r\n".join(lines[:i + 1] + block + lines[i + 1:])

    # guards on the RESULT -- the class of check both generators lacked
    assert out.count("set DIR=") == 1 and out.count("set LOG=") == 1
    assert out.count("cd /d ") == 1
    for var in ("DIR", "LOG", "STAMP", "DUMP", "WSSTEPS", "MAXSTEPS"):
        used = out.index("%" + var + "%")
        declared = out.index("set " + var + "=")
        assert declared < used, f"{path}: %{var}% used before it is set"
    # the log line the DOWNSTREAM waiter polls must not be creatable by anything but a real finish
    for ln in out.split("\r\n"):
        if ln.strip().upper().startswith("REM"):
            assert not [c for c in "<>&|^" if c in ln], "redirection char in REM: " + ln
    io.open(path, "wb").write(out.encode("ascii"))
    print(f"patched {path}: +{len(block)} lines, LOG={tag}.log")
