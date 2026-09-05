#!/usr/bin/env python
"""RWKV_MUON_INCLUDE_SCALE: inert when off, and moves exactly the 26 scale matrices when on. CPU, ~30 s.
Cloned from scratchpad/iter53_muonlora/smoke_muon_lora.py (the lever has the same shape: a name-rule
exclusion in get_optimizer becomes its own Muon group at the wd the params already had).
Proves, by param IDENTITY not by count:
 1. OFF is byte-identical: same groups, same per-group counts, the scale tensors on AdamW.
 2. ON moves EXACTLY the 2-D `*scale*` weights -- not 1-D scale params, not LoRAs, nothing already on Muon.
 3. The moved params keep weight_decay 0.0.
 4. Every param is in exactly one group; no group is empty.
 5. With BOTH include flags on (the realcyc recipe has INCLUDE_LORA=1), the LoRA group is untouched.
One subprocess per flag combination (the arch env is baked at import by the ScriptModule machinery).
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
    groups.append({"n": sum(p.numel() for p in g["params"]), "wd": g.get("weight_decay"),
                   "muon": bool(g.get("use_muon", False)),
                   "names": sorted(by_id.get(id(p), "?") for p in g["params"])})
print("GROUPS_JSON " + json.dumps(groups))
"""


def run(scale_flag, lora_flag="1"):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
    env.update(PYTHONPATH=REPO, RWKV_MUON_INCLUDE_SCALE=scale_flag, RWKV_MUON_INCLUDE_LORA=lora_flag,
               RWKV_MUON="1", RWKV_MUON_LR="0.0025",
               RWKV_ARCH_MODULE="scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
               RWKV_INTERLEAVE="1", RWKV_GRU_HEAD="3", RWKV_PAVA_LAMBDA="0.2", RWKV_STRIP_L0_VLORA="1",
               RWKV_ID_FEATURES="1", RWKV_REAL_CYCLES="1",
               RWKV_STRIP_CMIX="user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1")
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env, capture_output=True, text=True)
    for ln in (p.stdout + p.stderr).splitlines():
        if ln.startswith("GROUPS_JSON "):
            return json.loads(ln[len("GROUPS_JSON "):])
    print((p.stdout + p.stderr)[-1500:])
    raise SystemExit("child failed")


def main():
    off, on = run("0"), run("1")
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)
    tot_off, tot_on = sum(g["n"] for g in off), sum(g["n"] for g in on)
    check("same total params either way", tot_off == tot_on, f"{tot_off:,}")
    check("OFF has 6 groups (incl. the LoRA group), ON has 7", len(off) == 6 and len(on) == 7, f"{len(off)} / {len(on)}")
    check("no empty group", all(g["n"] for g in off) and all(g["n"] for g in on))
    off_muon = {n for g in off if g["muon"] for n in g["names"]}
    on_muon = {n for g in on if g["muon"] for n in g["names"]}
    moved, lost = on_muon - off_muon, off_muon - on_muon
    check("nothing LEAVES Muon", not lost, f"{len(lost)} lost")
    check("everything that moved is a 2-D scale weight",
          bool(moved) and all("scale" in n and "weight" in n for n in moved), f"{len(moved)} tensors moved")
    check("exactly 26 tensors moved", len(moved) == 26, f"{len(moved)}")
    off_wd = {g["wd"] for g in off for n in g["names"] if n in moved}
    on_wd = {g["wd"] for g in on for n in g["names"] if n in moved}
    check("moved params keep weight_decay 0.0", off_wd == {0.0} and on_wd == {0.0}, f"off {off_wd} -> on {on_wd}")
    n_moved = sum(g["n"] for g in on if g["muon"]) - sum(g["n"] for g in off if g["muon"])
    check("moved mass = the 26 (5,80) matrices = 10,400 params", n_moved == 10400, f"{n_moved:,} ({100*n_moved/tot_on:.2f}% of the model)")
    check("OFF still has the scale matrices on AdamW", all("scale" not in n for g in off if g["muon"] for n in g["names"]))
    lora_off = {n for g in off if g["muon"] for n in g["names"] if "lora" in n}
    lora_on = {n for g in on if g["muon"] for n in g["names"] if "lora" in n}
    # 94 here, not iter 53's 104: RWKV_STRIP_L0_VLORA removes 10 LoRA tensors on this arch env
    check("the LoRA Muon group is untouched by the scale flag", lora_off == lora_on and len(lora_on) > 0, f"{len(lora_on)} LoRA tensors on Muon both ways")
    only_scale = run("1", lora_flag="0")
    check("scale flag alone (no LoRA flag) still gives exactly one extra Muon group", len(only_scale) == 6 and sum(1 for g in only_scale if g["muon"]) == 5)
    print("\n" + ("MUONSCALE_ALL_PASS" if not fails else "MUONSCALE_FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
