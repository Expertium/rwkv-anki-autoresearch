"""Is arm 2's decay FINISHED, still RUNNING, or DEAD?  Used by wait_decay_then_qateval.cmd.

Exit 0  = FINISHED: no trainer process left AND all three step-21870 artifacts exist
          (the checkpoint and BOTH learned catalogs -- the catalogs are written after the .pth
          in the same save block, so requiring them is what proves the save completed).
Exit 10 = RUNNING: a python.exe running `rwkv.train_rwkv --config <...>/w10qat/decay.toml` lives.
Exit 20 = DEAD: the trainer is gone and at least one artifact is missing.

Why a process check and not the log: the decay's runner cmd was stopped on purpose (its phase C
would have evaluated with the START catalogs), so no line in any log announces the end of the
decay. Two witnesses instead -- the process AND the artifacts -- so that a crash (gone, no
artifacts) cannot read as success and a slow final save (alive, artifacts partial) cannot either.
Only python.exe is inspected: a shell that merely TYPES the pattern must not count as a trainer
(the 2026-09-02 probe matched its own powershell line and could never reach zero).

Test hooks (never set them in the waiter): W10Q_STEPS picks the step, W10Q_TOML the toml suffix.
No backslashes anywhere in this file: the tool chain that writes it collapses them.
"""
import os
import sys

import psutil

WS10 = os.path.join("C:/Users/Andrew/rwkv-anki-autoresearch", "scratchpad", "ws10")
STEPS = int(os.environ.get("W10Q_STEPS", "21870"))
TOML = os.environ.get("W10Q_TOML", "w10qat/decay.toml").lower()
NEED = [f"w10q_d_{STEPS}.pth", f"w10q_d_wkvcb_{STEPS}.txt", f"w10q_d_shiftcb_{STEPS}.txt"]


def trainers():
    pids = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower() != "python.exe":
                continue
            cl = " ".join(p.info["cmdline"] or []).replace(chr(92), "/").lower()
            if "rwkv.train_rwkv" in cl and TOML in cl:
                pids.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return pids


def main():
    pids = trainers()
    have = {f: os.path.exists(os.path.join(WS10, f)) for f in NEED}
    n = sum(have.values())
    if pids:
        print(f"RUNNING: trainer pids {pids}; step-{STEPS} artifacts {n}/3")
        return 10
    if n == 3:
        print(f"FINISHED: no trainer; all 3 step-{STEPS} artifacts present")
        return 0
    print(f"DEAD: no trainer; missing {[f for f, ok in have.items() if not ok]}")
    return 20


if __name__ == "__main__":
    sys.exit(main())
