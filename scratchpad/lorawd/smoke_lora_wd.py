#!/usr/bin/env python
"""RWKV_MUON_LORA_WD: inert when unset, and changes EXACTLY ONE group's weight decay when set. CPU, ~30 s.

The lever (lorawd, adopted from Moonlight arXiv 2502.16982) gives the LoRA Muon group -- the one
iter 53 created at wd=0.0 -- a decoupled weight decay of 0.05. What a param-group bug looks like is
silence, so this proves, by param IDENTITY not by count:

 1. UNSET == iter 53's partition exactly (same groups, same params per group, same wd per group,
    LoRA group at wd 0.0).
 2. SET (0.05): the partition is IDENTICAL to (1) -- no param moves -- and the ONLY field that differs
    anywhere is the LoRA group's weight_decay, which equals the env value.
 3. The LoRA group is a Muon group (use_muon True), so muon.py's wd_eff = lr * wd_lr_scale * wd path
    is the one that consumes it (it is generic over groups; nothing else needs to change).
 4. NON-VACUITY: a THIRD arm at a different value (0.02) must differ from the 0.05 arm in that one
    field -- otherwise the check could pass on a flag that is never read.

One subprocess per flag value (the arch env is baked at import by the ScriptModule machinery).
Reuses iter 53's child verbatim, plus use_muon.
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHILD = r"""
import os, sys, types, json
sys.path.insert(0, os.environ["PYTHONPATH"])
import torch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.model.srs_model import SrsRWKV
import rwkv.train_rwkv as T

model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
cfg = types.SimpleNamespace(PEAK_LR=1e-3)
opt = T.get_optimizer(cfg, model)

by_id = {id(p): n for n, p in model.named_parameters()}
groups = []
for g in opt.param_groups:
    groups.append({
        "n": sum(p.numel() for p in g["params"]),
        "wd": g.get("weight_decay"),
        "muon": bool(g.get("use_muon", False)),
        "names": sorted(by_id.get(id(p), "?") for p in g["params"]),
    })
print("GROUPS_JSON " + json.dumps(groups))
"""


def run(wd):
    env = dict(os.environ, PYTHONPATH=REPO, RWKV_MUON_INCLUDE_LORA="1",
               RWKV_MUON="1", RWKV_MUON_LR="0.0025",
               RWKV_ARCH_MODULE="scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
               RWKV_INTERLEAVE="1", RWKV_GRU_HEAD="3", RWKV_PAVA_LAMBDA="0.2",
               RWKV_STRIP_L0_VLORA="1",
               RWKV_STRIP_CMIX=("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                                "preset_id:2,deck_id:1,deck_id:2,card_id:1"))
    env.pop("RWKV_MUON_LORA_WD", None)
    if wd is not None:
        env["RWKV_MUON_LORA_WD"] = wd
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env, capture_output=True, text=True)
    for ln in (p.stdout + p.stderr).splitlines():
        if ln.startswith("GROUPS_JSON "):
            return json.loads(ln[len("GROUPS_JSON "):])
    print((p.stdout + p.stderr)[-1500:])
    raise SystemExit("child failed")


def main():
    unset, on, alt = run(None), run("0.05"), run("0.02")
    fails = []

    def check(name, ok, detail=""):
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""))
        if not ok:
            fails.append(name)

    lora_idx = [i for i, g in enumerate(unset) if g["names"] and all("lora" in n for n in g["names"])]
    check("exactly one all-LoRA group exists", len(lora_idx) == 1, str(lora_idx))
    li = lora_idx[0]
    check("UNSET: LoRA group wd is 0.0 (iter 53's value)", unset[li]["wd"] == 0.0, str(unset[li]["wd"]))
    check("UNSET: LoRA group is on Muon", unset[li]["muon"])
    part = lambda gs: [(g["names"], g["n"], g["muon"]) for g in gs]
    check("SET: partition identical to UNSET (no param moves)", part(on) == part(unset))
    check("SET: LoRA group wd == 0.05", on[li]["wd"] == 0.05, str(on[li]["wd"]))
    others_same = all(on[i]["wd"] == unset[i]["wd"] for i in range(len(on)) if i != li)
    check("SET: every OTHER group's wd unchanged", others_same,
          str([(unset[i]["wd"], on[i]["wd"]) for i in range(len(on)) if i != li]))
    check("NON-VACUITY: 0.02 arm differs from 0.05 arm in that one field only",
          alt[li]["wd"] == 0.02 and part(alt) == part(on)
          and all(alt[i]["wd"] == on[i]["wd"] for i in range(len(on)) if i != li))
    print("\n" + ("LORA_WD_SMOKE_PASS" if not fails else "LORA_WD_SMOKE_FAIL: " + ", ".join(fails)))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
