"""Is the grafted checkpoint BIT-IDENTICAL to ws10's on the deploy path? CPU, ~1 min, no GPU.

The whole design rests on one claim: restoring six channel mixers whose output projection `W_v` is
zero cannot change the forward pass, because a mixer returns `in_BTC + dropout(W_v(...))`. If that
is true, the capacity arm starts exactly where ws10 ended and needs no new 38 h WS phase; if it is
false, the arm is a bundle of "capacity" and "a perturbed starting point" and its number means
nothing. So it is checked by EXECUTION, in two separate processes -- the arch flags are read at
import and old-style ScriptModule bakes the first construction's flags into the class, so one
process cannot legitimately hold both configurations.

NON-VACUITY, which is the half that is easy to skip and is the reason this file exists: a smoke
that only compares two identical things passes when the graft did nothing at all. So it ALSO runs
a PERTURBED arm -- the same grafted checkpoint with one restored `W_v` set to a small non-zero
value -- and REQUIRES that arm to DIFFER. If the perturbed arm matches too, the restored mixers are
not being executed and the comparison proves nothing.

Usage: python scratchpad/cmixgraft/smoke_graft.py [users...]
"""
import json
import os
import subprocess
import sys

os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch")

ARM1_STRIP = ("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,"
              "deck_id:1,deck_id:2,card_id:1")
GRAFT_STRIP = "deck_id:1,deck_id:2,card_id:1"
WS = "scratchpad/ws10/w10_ws_109350.pth"
GRAFT = "scratchpad/ws10/w10g_ws_109350.pth"
ROWS = 300

CHILD = r'''
import os, sys, json
os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch"); sys.path.insert(0, os.getcwd())
_E = {
  "RWKV_ARCH_MODULE": "scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
  "RWKV_INTERLEAVE": "1", "RWKV_GRU_HEAD": "3", "RWKV_PAVA_LAMBDA": "0.2",
  "RWKV_NO_AHEAD_RESIDUAL": "1", "RWKV_STRIP_L0_VLORA": "1",
  "RWKV_STATE_CLAMP_TAU": "300", "RWKV_STATE_CLAMP_WINDOW": "32768",
  "RWKV_ID_FEATURES": "1", "RWKV_REAL_CYCLES": "1", "RWKV_ZERO_FEATURES": "",
  "RWKV_DROPOUT_SCALE": "0.5", "RWKV_NO_JIT": "1",
  "RWKV_STRIP_CMIX": sys.argv[1],
}
for k, v in _E.items(): os.environ[k] = v
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
import rwkv.run_as_rnn as rnn_mod
from rwkv.data_processing import get_rwkv_data
ckpt, users, rows, perturb = sys.argv[2], [int(u) for u in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5] == "1"
proc = rnn_mod.RNNProcess(path=ckpt, device=torch.device("cpu"), dtype=torch.float32)
if perturb:
    n = 0
    for nm, p in proc.rnn.named_parameters():
        if nm.endswith("W_v.weight") and p.shape[0] > 1 and p.abs().max().item() == 0.0:
            with torch.no_grad(): p[0, 0] = 1e-3
            n += 1
    assert n >= 1, "perturbation found no zero W_v to perturb -- the smoke would be vacuous"
    print(f"[perturb] {n} restored W_v perturbed", file=sys.stderr)
out = []
for uid in users:
    torch.manual_seed(uid)
    df = get_rwkv_data(Path(r"C:\Users\Andrew\anki-revlogs-10k-id"), uid).sort_values(
        "review_th", kind="stable").reset_index(drop=True)
    with torch.inference_mode():
        for row in df.to_dict("records")[:rows]:
            r = proc.run(row, skip=False)
            if r is None: continue
            (ahead, w, s, d), imm = r
            out.append([float(np.asarray(ahead).ravel()[0]), float(np.asarray(imm).ravel()[0])])
print("RESULT " + json.dumps(out))
'''


def run(strip, ckpt, users, perturb=False):
    src = os.path.join(os.environ.get("TEMP", "."), "cmixgraft_child.py")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(CHILD)
    cp = subprocess.run(
        [r".venv\Scripts\python.exe", src, strip, ckpt, ",".join(str(u) for u in users),
         str(ROWS), "1" if perturb else "0"],
        capture_output=True, text=True)
    for ln in cp.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[7:])
    sys.exit(f"child failed (rc {cp.returncode})\n{cp.stdout[-2000:]}\n{cp.stderr[-3000:]}")


def main():
    users = [int(u) for u in sys.argv[1:]] or [107, 136]
    print(f"users {users}, {ROWS} rows each\n")
    a = run(ARM1_STRIP, WS, users)
    print(f"arm 1  (stripped, ws10 ckpt)      {len(a)} predictions")
    b = run(GRAFT_STRIP, GRAFT, users)
    print(f"graft  (restored, grafted ckpt)   {len(b)} predictions")
    if len(a) != len(b):
        sys.exit(f"FAIL: {len(a)} vs {len(b)} predictions -- not comparable")
    diff = max(max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(a, b))
    print(f"\nmax |graft - arm1| over {len(a)} predictions x 2 heads: {diff:.3e}")

    c = run(GRAFT_STRIP, GRAFT, users, perturb=True)
    pdiff = max(max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(a, c))
    print(f"max |perturbed - arm1|:                                 {pdiff:.3e}")

    ok = True
    if diff != 0.0:
        print("\nFAIL: the graft is NOT bit-identical -- it is not function-preserving")
        ok = False
    if pdiff == 0.0:
        print("\nFAIL: the PERTURBED arm is also identical -- the restored mixers are not being "
              "executed, so the comparison above proves nothing (VACUOUS)")
        ok = False
    print("\nSMOKE PASS" if ok else "\nSMOKE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
