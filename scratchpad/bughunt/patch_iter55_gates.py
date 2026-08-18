"""Add the two missing STEP-VERIFICATION gates to run_iter55.cmd (the rgate retry, still queued).

THE GAP. Neither training phase checks that it reached its final step. The runner tests the exit
code and greps the phase log, then hands off. That is precisely the hole the 2026-08-18 power-outage
recovery fell through: `train_rwkv` hit a fatal AttributeError, SWALLOWED it, and exited 0, so the
runner logged `WS OK` after 8 seconds and went on to decay a half-trained model.

AND AN `if not exist` ALONE WOULD NOT HAVE CAUGHT THAT ONE. A checkpoint DID exist -- the step-8000
one from the interrupted run -- so an existence test would have passed and `write_decay_setup.py`
would have happily decayed from it. The check has to name the EXPECTED STEP. That is what
kdalpha025 does correctly (`findstr /C:"i45_ws_%WSSTEPS%"` on the dsetup log) and what this runner
never had.

Two gates, one per phase, both naming %WSSTEPS% rather than testing for any .pth:
  * after WS    -- i55_ws_10935.pth must exist before the decay setup reads the directory;
  * after decay -- i55_d_10935.pth must exist before the eval toml is written.
Both land in scratchpad/iter55_rgate: write_decay_setup takes the folder holding the WS checkpoint,
so a decay-phase checkpoint is written beside its source, not into a run's own new directory.

SAFE TO PATCH: this runner is queued behind three others and its waiter log reads "waiter armed"
only, so cmd.exe does not yet hold the file open. A byte-exact backup is taken first.
"""
import io

P = "scratchpad/iter55_rgate/run_iter55.cmd"
s = io.open(P, encoding="ascii", newline="").read()
orig_len = len(s)

# ---- gate 1: the WS phase reached its final step -----------------------------------------
DSETUP = 'echo === DECAY SETUP %TIME% === >> "%LOG%"\r\n'
assert s.count(DSETUP) == 1, "decay-setup marker not unique"
GATE1 = (
    'REM Exit code 0 is not evidence the phase ran: train_rwkv has swallowed a fatal error and\r\n'
    'REM still exited 0 (2026-08-18). Nor is "a checkpoint exists" -- an INTERRUPTED run leaves\r\n'
    'REM one at the step it died on, which is what made the outage recovery decay a half-trained\r\n'
    'REM model. The gate has to name the expected FINAL step.\r\n'
    'if not exist "%DIR%\\i55_ws_%WSSTEPS%.pth" (\r\n'
    '  echo ITER55 WS_SHORT no i55_ws_%WSSTEPS%.pth %DATE% %TIME% >> "%LOG%"\r\n'
    '  echo DONE_EXIT_WSSHORT %DATE% %TIME% >> "%LOG%"\r\n'
    '  exit /b 27\r\n'
    ')\r\n'
)
s = s.replace(DSETUP, GATE1 + DSETUP, 1)

# ---- gate 2: the decay phase reached its final step ---------------------------------------
ETOML = 'echo === EVAL TOML rect %TIME% === >> "%LOG%"\r\n'
assert s.count(ETOML) == 1, "eval-toml marker not unique"
GATE2 = (
    'REM Same reasoning for the decay phase: name the step, do not merely test for a .pth.\r\n'
    'if not exist "%DIR%\\i55_d_%WSSTEPS%.pth" (\r\n'
    '  echo ITER55 DECAY_SHORT no i55_d_%WSSTEPS%.pth %DATE% %TIME% >> "%LOG%"\r\n'
    '  echo DONE_EXIT_DECAYSHORT %DATE% %TIME% >> "%LOG%"\r\n'
    '  exit /b 28\r\n'
    ')\r\n'
)
s = s.replace(ETOML, GATE2 + ETOML, 1)

# ---- output guards: the patch must not have broken anything ------------------------------
lines = s.split("\r\n")
assert s.count("DONE_EXIT_0") == 1, "terminal marker count changed"
_bare_el = [n for n, l in enumerate(lines) if l.strip().lower() == "endlocal"]
_de = [n for n, l in enumerate(lines) if l.strip().startswith("echo DONE_EXIT_")]
assert not (_bare_el and _de and min(_bare_el) < max(_de)), "endlocal now precedes a marker"
for l in lines:
    if l.strip().upper().startswith("REM"):
        bad = [c for c in "<>&|^" if c in l]
        assert not bad, "redirection char " + str(bad) + " in a REM line: " + l
assert len(s) > orig_len, "nothing was inserted"

io.open(P, "w", encoding="ascii", newline="").write(s)
print("patched", P, orig_len, "->", len(s), "bytes")
