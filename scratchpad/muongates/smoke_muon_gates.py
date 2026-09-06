#!/usr/bin/env python
"""RWKV_MUON_INCLUDE_GATES: inert when off, and moves EXACTLY the 13 rkvdag_lerp tensors when on.
CPU, ~30 s. Cloned from scratchpad/muonscale/smoke_muon_scale.py (same shape of lever), with one
extra class of check that iter 67 taught: the DEGENERATE-RESHAPE guard must be proved, not assumed.

Proves, by param IDENTITY and by the reshape MuonAdamW actually performs:
 1. OFF is unchanged: same groups, same counts, no gate tensor on Muon.
 2. ON moves EXACTLY the 13 `rkvdag_lerp` tensors (8,320 params) -- not `bonus`, not 1-D, not a
    tensor already on Muon.
 3. `bonus` is EXCLUDED, and the smoke proves the exclusion is EARNED: its reshape(size(0), -1) is
    (1, 80), on which Newton-Schulz is a normalisation rather than an orthogonalisation. The check
    asserts every ADMITTED tensor reshapes to a matrix with both dims >= 2 and every tensor the
    guard REJECTED for degeneracy really is degenerate -- so the rule cannot silently admit a row
    vector later.
 4. The moved params keep weight_decay 0.0; every param is in exactly one group; no empty group.
 5. With the other include flags on (the realcyc recipe has INCLUDE_LORA=1), those groups are
    untouched.
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
shape_of = {n: tuple(p.shape) for n, p in model.named_parameters()}
groups = []
for g in opt.param_groups:
    groups.append({"n": sum(p.numel() for p in g["params"]), "wd": g.get("weight_decay"),
                   "muon": bool(g.get("use_muon", False)),
                   "names": sorted(by_id.get(id(p), "?") for p in g["params"])})
print("GROUPS_JSON " + json.dumps({"groups": groups, "shapes": shape_of}))
"""


def run(**flags):
    env = {k: v for k, v in os.environ.items() if not k.startswith("RWKV_")}
    env.update(PYTHONPATH=REPO, RWKV_MUON="1", RWKV_MUON_LR="0.0025",
               RWKV_ARCH_MODULE="scratchpad/track2_a18/architecture_d80_lora4_cnd.py",
               RWKV_INTERLEAVE="1", RWKV_GRU_HEAD="3", RWKV_PAVA_LAMBDA="0.2", RWKV_STRIP_L0_VLORA="1",
               RWKV_ID_FEATURES="1", RWKV_REAL_CYCLES="1",
               RWKV_STRIP_CMIX="user_id:0,user_id:1,user_id:2,preset_id:0,preset_id:1,preset_id:2,deck_id:1,deck_id:2,card_id:1")
    env.update({k: v for k, v in flags.items()})
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=REPO, env=env, capture_output=True, text=True)
    for ln in (p.stdout + p.stderr).splitlines():
        if ln.startswith("GROUPS_JSON "):
            return json.loads(ln[len("GROUPS_JSON "):])
    print((p.stdout + p.stderr)[-1500:])
    raise SystemExit("child failed")


def main():
    off = run(RWKV_MUON_INCLUDE_LORA="1")
    on = run(RWKV_MUON_INCLUDE_LORA="1", RWKV_MUON_INCLUDE_GATES="1")
    shapes = on["shapes"]
    og, ng = off["groups"], on["groups"]
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    tot_off, tot_on = sum(g["n"] for g in og), sum(g["n"] for g in ng)
    check("same total params either way", tot_off == tot_on, f"{tot_off:,}")
    check("ON has exactly one more group than OFF", len(ng) == len(og) + 1, f"{len(og)} -> {len(ng)}")
    check("no empty group", all(g["n"] for g in og) and all(g["n"] for g in ng))
    off_muon = {n for g in og if g["muon"] for n in g["names"]}
    on_muon = {n for g in ng if g["muon"] for n in g["names"]}
    moved, lost = on_muon - off_muon, off_muon - on_muon
    check("nothing LEAVES Muon", not lost, f"{len(lost)} lost")
    check("everything that moved is an rkvdag_lerp tensor",
          bool(moved) and all("rkvdag_lerp" in n for n in moved), f"{len(moved)} moved")
    check("exactly 13 tensors / 8,320 params moved",
          len(moved) == 13 and sum(g["n"] for g in ng if g["muon"]) - sum(g["n"] for g in og if g["muon"]) == 8320,
          f"{len(moved)} tensors, {sum(g['n'] for g in ng if g['muon']) - sum(g['n'] for g in og if g['muon']):,} params")
    on_wd = {g["wd"] for g in ng for n in g["names"] if n in moved}
    check("moved params keep weight_decay 0.0", on_wd == {0.0}, str(on_wd))
    # THE GUARD IS EARNED, NOT ASSUMED (iter 67's lesson: validate against what CANNOT change).
    def reshaped(n):
        s = shapes[n]
        return (s[0], max(1, int(__import__("math").prod(s)) // s[0]))
    bad = [(n, reshaped(n)) for n in moved if min(reshaped(n)) < 2]
    check("every ADMITTED tensor reshapes to a real matrix (both dims >= 2)", not bad, str(bad[:3]))
    check("admitted tensors reshape to (8, 80)", {reshaped(n) for n in moved} == {(8, 80)},
          str(sorted({reshaped(n) for n in moved})))
    adam_names = {n for g in ng if not g["muon"] for n in g["names"]}
    bonus = sorted(n for n in adam_names if "bonus" in n)
    check("`bonus` stayed on AdamW", len(bonus) == 13, f"{len(bonus)} bonus tensors on AdamW")
    check("and its exclusion is EARNED: it reshapes to a degenerate row vector",
          bool(bonus) and all(min(reshaped(n)) < 2 for n in bonus),
          f"{reshaped(bonus[0]) if bonus else '-'}")
    # nothing squeeze-2-D and non-degenerate is left behind on AdamW
    import math
    left = [n for n in adam_names
            if "weight" not in n and shapes[n][0] >= 2 and math.prod(shapes[n]) // shapes[n][0] >= 2]
    check("no non-degenerate gate tensor is left on AdamW", not left, str(left[:3]))
    lora_off = {n for g in og if g["muon"] for n in g["names"] if "lora" in n}
    lora_on = {n for g in ng if g["muon"] for n in g["names"] if "lora" in n}
    check("the LoRA Muon group is untouched by the gate flag", lora_off == lora_on and len(lora_on) > 0,
          f"{len(lora_on)} LoRA tensors on Muon both ways")
    both = run(RWKV_MUON_INCLUDE_LORA="1", RWKV_MUON_INCLUDE_SCALE="1", RWKV_MUON_INCLUDE_GATES="1")
    check("composes with INCLUDE_SCALE (two extra Muon groups over OFF)",
          len(both["groups"]) == len(og) + 2 and sum(1 for g in both["groups"] if g["muon"]) == sum(1 for g in og if g["muon"]) + 2,
          f"{len(both['groups'])} groups")
    print("\n" + ("MUONGATES_ALL_PASS" if not fails else "MUONGATES_FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
