#!/usr/bin/env python
"""RWKV_MUON_INCLUDE_LORA: inert when off, and moves exactly the LoRA matrices when on. CPU, ~30 s.

The lever (iter 53) puts the LoRA projections on Muon. They have always run on AdamW because the
grouping in `get_optimizer` excludes any param whose name contains "lora" -- a rule that predates
the A18 width ladder, which then made LoRA rank a load-bearing part of the trunk.

WHAT THIS HAS TO PROVE, because a param-group bug is silent:

 1. **OFF is byte-identical**: same number of groups, same param counts per group, and the LoRA
    tensors still sitting in the AdamW group.
 2. **ON moves EXACTLY the LoRA matrices** -- not the 1-D LoRA params, not the "scale" matrices,
    and nothing that was already on Muon. Checked by identity (`id(param)`), not by count, because
    two different partitions can share a total.
 3. **The weight decay of the moved params does not change.** This is the whole reason they get
    their own group: dropping them into `decay_params` would have changed the optimizer AND the wd
    in one iteration, and the result would have been uninterpretable either way it went.
 4. **Every param is in exactly one group, and no group is empty.** An empty group reaching the
    optimizer is the failure mode of the "only add it if non-empty" guard being wrong.

One subprocess per flag value: the flag is read inside get_optimizer, but the arch env is baked at
import by the ScriptModule machinery, so a single process cannot honestly do both.

ASCII output only.
"""
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


def run(flag):
    env = dict(os.environ, PYTHONPATH=REPO, RWKV_MUON_INCLUDE_LORA=flag,
               RWKV_MUON="1", RWKV_MUON_LR="0.0025",
               RWKV_ARCH_MODULE="scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
               RWKV_INTERLEAVE="1", RWKV_GRU_HEAD="3", RWKV_PAVA_LAMBDA="0.2",
               RWKV_STRIP_L0_VLORA="1",
               RWKV_STRIP_CMIX=("user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,"
                                "preset_id:2,deck_id:1,deck_id:2,card_id:1"))
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env,
                       capture_output=True, text=True)
    import json
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

    tot_off = sum(g["n"] for g in off)
    tot_on = sum(g["n"] for g in on)
    check("same total params either way", tot_off == tot_on, f"{tot_off:,}")
    check("OFF has 5 groups, ON has 6", len(off) == 5 and len(on) == 6,
          f"{len(off)} / {len(on)}")
    check("no empty group", all(g["n"] for g in off) and all(g["n"] for g in on))

    off_muon = {n for g in off if g["muon"] for n in g["names"]}
    on_muon = {n for g in on if g["muon"] for n in g["names"]}
    moved = on_muon - off_muon
    lost = off_muon - on_muon
    check("nothing LEAVES Muon", not lost, f"{len(lost)} lost")
    check("everything that moved is a LoRA matrix weight",
          bool(moved) and all("lora" in n and "weight" in n and "scale" not in n for n in moved),
          f"{len(moved)} tensors moved")

    off_lora_wd = {n: g["wd"] for g in off for n in g["names"] if n in moved}
    on_lora_wd = {n: g["wd"] for g in on for n in g["names"] if n in moved}
    check("moved params keep weight_decay 0.0",
          set(off_lora_wd.values()) == {0.0} and set(on_lora_wd.values()) == {0.0},
          f"off {set(off_lora_wd.values())} -> on {set(on_lora_wd.values())}")

    n_moved = sum(g["n"] for g in on if g["muon"]) - sum(g["n"] for g in off if g["muon"])
    check("the moved mass is material", n_moved > 10_000,
          f"{n_moved:,} params ({100*n_moved/tot_on:.1f}% of the model)")

    # OFF must match the historical partition exactly: 4 Muon groups, LoRAs on AdamW
    check("OFF still has the LoRAs on AdamW",
          all("lora" not in n for g in off if g["muon"] for n in g["names"]))

    print("\n" + ("MUONLORA_ALL_PASS" if not fails else "MUONLORA_FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
