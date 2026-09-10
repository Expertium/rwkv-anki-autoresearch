"""Build the grafted OPTIMIZER state that cmixgraft's decay loads. CPU, ~1 min, no GPU.

WHY IT IS NEEDED (found 2026-09-10 12:40, before any GPU was spent). A decay branch loads the WS
checkpoint's optimizer state as well as its weights: write_decay_setup.py copies
`<ws>_optim_<step>.pth` to `<ws>_<step>_optim.pth`, and train_rwkv.py:709 then loads that file with
no existence check. expand_ckpt.py wrote only the WEIGHTS, so cmixgraft's decay would have died at
load. Copying ws10's optimizer file would not have worked either, for a less obvious reason:

get_optimizer groups by `len(param.squeeze().shape) >= 2`. A stripped channel mixer is a
d_model=1 dummy, so its W_k/W_v squeeze below 2-D and sit in the AdamW `other` group; in the graft
the same names are real matrices and go to the Muon `channel_mixer` group. The group SIZES differ,
so optimizer.load_state_dict refuses ws10's state outright -- and a hand-copied state could have
silently attached moments to the wrong parameters, since the state dict is keyed by POSITION.

WHAT THIS DOES. Rebuild both optimizers with the REAL get_optimizer (two child processes, because
the arch flags are read at import), each under the env parsed from its own runner file -- never the
ambient env, which is the rgate lesson. Map ws10's saved state to the graft BY NAME: every
parameter whose shape is unchanged keeps its moments bit-for-bit, and the 30 grown tensors get NO
state entry, which both optimizers treat as fresh (MuonAdamW creates momentum_buffer and exp_avg
lazily -- rwkv/muon.py). That is the single-variable version: arm 1 decayed from exactly these
moments.

VERIFIED IN A THIRD CHILD, graft env: load the grafted weights, load this state (must not raise),
compare every carried entry to ws10's by name (bit-identical), require the grown ones empty, then
take one synthetic optimizer step and require the grown parameters to acquire state of the right
shape.

Usage: python scratchpad/cmixgraft/expand_optim.py
"""
import json
import math
import os
import re
import subprocess
import sys

REPO = r"C:\Users\Andrew\rwkv-anki-autoresearch"
os.chdir(REPO)

OLD_OPTIM = "scratchpad/ws10/w10_ws_optim_109350.pth"
NEW_OPTIM = "scratchpad/ws10/w10g_ws_optim_109350.pth"   # the name write_decay_setup copies from
GRAFT_CKPT = "scratchpad/ws10/w10g_ws_109350.pth"
RUNNER_STRIP = "scratchpad/w10plain/run_w10plain.cmd"
RUNNER_GRAFT = "scratchpad/cmixgraft/run_cmixgraft.cmd"

CHILD = r'''
import os, sys, json
os.chdir(r"C:\Users\Andrew\rwkv-anki-autoresearch"); sys.path.insert(0, os.getcwd())
from types import SimpleNamespace
import torch
torch.set_num_threads(2)
from rwkv.train_rwkv import get_optimizer, SrsRWKV, DEFAULT_ANKI_RWKV_CONFIG
mode = sys.argv[1]
model = SrsRWKV(anki_rwkv_config=DEFAULT_ANKI_RWKV_CONFIG)
opt = get_optimizer(SimpleNamespace(PEAK_LR=1e-3), model)
name_of = {id(p): n for n, p in model.named_parameters()}
groups = [[[name_of[id(p)], list(p.shape)] for p in g["params"]] for g in opt.param_groups]
keys = [sorted(k for k in g if k != "params") for g in opt.param_groups]
if mode == "names":
    print("RESULT " + json.dumps({"groups": groups, "keys": keys}))
    sys.exit(0)
# ---- verify ----
strip = json.load(open(sys.argv[2]))
grown = set(json.load(open(sys.argv[3])))
old = torch.load(sys.argv[4], weights_only=False)
new = torch.load(sys.argv[5], weights_only=False)
model.load_state_dict(torch.load(sys.argv[6], weights_only=True))
opt.load_state_dict(new)
old_idx = {n: i for i, n in enumerate(n for g in strip["groups"] for n, _ in g)}
params = dict(model.named_parameters())
carried = empty = 0
for n, p in params.items():
    st = opt.state.get(p, {})
    if n in grown:
        assert len(st) == 0, "grown param has state: " + n
        empty += 1
        continue
    ost = old["state"].get(old_idx[n], {})
    assert set(st) == set(ost), "state keys differ for " + n
    for k, v in st.items():
        if torch.is_tensor(v):
            assert torch.equal(v.cpu(), ost[k].cpu()), "state %s differs for %s" % (k, n)
        else:
            assert v == ost[k], "state %s differs for %s" % (k, n)
    carried += 1
torch.manual_seed(0)
for p in model.parameters():
    p.grad = torch.randn_like(p) * 1e-3
opt.step()
fresh = 0
for n in grown:
    st = opt.state.get(params[n], {})
    assert len(st) > 0, "grown param got no state after a step: " + n
    for k, v in st.items():
        if torch.is_tensor(v) and v.dim() > 0:
            assert v.numel() == params[n].numel(), "wrong-size state %s for %s" % (k, n)
    fresh += 1
print("RESULT " + json.dumps({"carried": carried, "empty_before_step": empty, "fresh_after_step": fresh}))
'''


def runner_env(path):
    """The RWKV_* / OMP env a runner sets, parsed from its non-REM `set` lines."""
    env = {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if ln.upper().startswith("REM"):
            continue
        m = re.match(r"set (RWKV_[A-Z0-9_]+|OMP_NUM_THREADS)=(.*)$", ln)
        if m and "%" not in m.group(2):
            env[m.group(1)] = m.group(2)
    return env


def run_child(mode, runner, *args):
    src = os.path.join(os.environ.get("TEMP", "."), "cmixgraft_optim_child.py")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(CHILD)
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
    env.update(runner_env(runner))
    env["RWKV_NO_JIT"] = "1"
    cp = subprocess.run([r".venv\Scripts\python.exe", src, mode, *args], env=env,
                        capture_output=True, text=True)
    for ln in cp.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[7:])
    sys.exit(f"child {mode} ({runner}) failed rc {cp.returncode}\n{cp.stdout[-2000:]}\n{cp.stderr[-3000:]}")


def main():
    import torch

    es, eg = runner_env(RUNNER_STRIP), runner_env(RUNNER_GRAFT)
    diff = sorted(k for k in set(es) | set(eg) if es.get(k) != eg.get(k))
    print(f"runner env keys that differ: {diff}")
    if diff != ["RWKV_STRIP_CMIX"]:
        sys.exit("REFUSING: the two runners differ in more than RWKV_STRIP_CMIX")

    s = run_child("names", RUNNER_STRIP)
    g = run_child("names", RUNNER_GRAFT)
    if len(s["groups"]) != len(g["groups"]) or s["keys"] != g["keys"]:
        sys.exit("REFUSING: the two optimizers have different group structures")
    sn = {n: tuple(sh) for grp in s["groups"] for n, sh in grp}
    gn = {n: tuple(sh) for grp in g["groups"] for n, sh in grp}
    if set(sn) != set(gn):
        sys.exit("REFUSING: the parameter NAME sets differ")
    sgrp = {n: i for i, grp in enumerate(s["groups"]) for n, _ in grp}
    ggrp = {n: i for i, grp in enumerate(g["groups"]) for n, _ in grp}
    grown = sorted(n for n in gn if sn[n] != gn[n])
    moved = sorted(n for n in gn if sgrp[n] != ggrp[n])
    print(f"grown tensors: {len(grown)}   group moves: {len(moved)}")
    for n in moved:
        print(f"  moves group {sgrp[n]} -> {ggrp[n]}: {n}  {sn[n]} -> {gn[n]}")
    if len(grown) != 30 or not all(".channel_mixer." in n for n in grown):
        sys.exit("REFUSING: expected exactly 30 grown channel-mixer tensors")
    if not set(moved) <= set(grown):
        sys.exit("REFUSING: a parameter whose shape did NOT change moved group")

    old = torch.load(OLD_OPTIM, weights_only=False)
    if [len(x["params"]) for x in old["param_groups"]] != [len(x) for x in s["groups"]]:
        sys.exit("REFUSING: ws10's saved groups do not match the rebuilt stripped optimizer -- the "
                 "name->position map would be wrong")
    old_names = [n for grp in s["groups"] for n, _ in grp]
    for i, n in enumerate(old_names):
        for k, v in old["state"].get(i, {}).items():
            if torch.is_tensor(v) and v.dim() > 0 and v.numel() != math.prod(sn[n]):
                sys.exit(f"REFUSING: saved state {k} at position {i} does not fit {n} {sn[n]}")
    print(f"ws10 optimizer: {len(old['state'])} state entries over {len(old_names)} params; "
          f"every entry fits the name it maps to")

    old_idx = {n: i for i, n in enumerate(old_names)}
    new_names = [n for grp in g["groups"] for n, _ in grp]
    new_state = {}
    for j, n in enumerate(new_names):
        if n in grown:
            continue
        i = old_idx[n]
        if i in old["state"]:
            new_state[j] = old["state"][i]
    new_groups, k = [], 0
    for gi, grp in enumerate(g["groups"]):
        hg = {kk: vv for kk, vv in old["param_groups"][gi].items() if kk != "params"}
        hg["params"] = list(range(k, k + len(grp)))
        k += len(grp)
        new_groups.append(hg)
    torch.save({"state": new_state, "param_groups": new_groups}, NEW_OPTIM)
    print(f"wrote {NEW_OPTIM}: {len(new_state)} carried state entries, "
          f"{len(grown)} grown params left fresh")

    tmp = os.environ.get("TEMP", ".")
    sp, gp = os.path.join(tmp, "cmixgraft_strip.json"), os.path.join(tmp, "cmixgraft_grown.json")
    json.dump(s, open(sp, "w"))
    json.dump(grown, open(gp, "w"))
    v = run_child("verify", RUNNER_GRAFT, sp, gp, OLD_OPTIM, NEW_OPTIM, GRAFT_CKPT)
    print(f"verify: {v}")
    if v["empty_before_step"] != 30 or v["fresh_after_step"] != 30 or v["carried"] != len(gn) - 30:
        sys.exit("VERIFY FAILED")
    print("OPTIM GRAFT PASS")


if __name__ == "__main__":
    main()
