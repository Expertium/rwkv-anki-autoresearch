"""Unit test for rwkv/grad_stats.py after the 2026-07-17 per-param None fix.

The A2 WS recording came out all-zero because accumulate() skipped the WHOLE step when
ANY param had grad None -- and structurally-unused params (layer-0 v_lora_simple.A) never
receive grads. Now: present-grad subset accumulates, never-grad params keep
steps_counted=0, and grad_stats_report separates them from the ranking.
"""

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from rwkv.grad_stats import GradStats


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(4) * 2.0)
        self.dead = torch.nn.Parameter(torch.zeros(3))  # never in the graph
        self.b = torch.nn.Parameter(torch.full((2,), 0.5))


m = Toy()
path = os.path.join(tempfile.gettempdir(), "gs_test.json")
gs = GradStats(path, m)

gs.accumulate()  # pre-backward: all grads None -> no-op
assert int(gs.count.sum()) == 0

for step in range(3):
    m.zero_grad()
    loss = (m.a * 1.0).sum() + (m.b * 3.0).sum()
    loss.backward()
    gs.accumulate()

gs.dump()
d = json.load(open(path))
assert d["a"]["steps_counted"] == 3, d["a"]
assert d["b"]["steps_counted"] == 3, d["b"]
assert d["dead"]["steps_counted"] == 0, d["dead"]
assert abs(d["a"]["mean_abs_grad"] - 1.0) < 1e-9
assert abs(d["b"]["mean_abs_grad"] - 3.0) < 1e-9
assert abs(d["a"]["mean_abs_grad_x_w"] - 2.0) < 1e-9   # |g*w| = 1*2
assert abs(d["b"]["mean_abs_grad_x_w"] - 1.5) < 1e-9   # 3*0.5
assert d["dead"]["mean_abs_grad"] == 0.0
print("PASS: subset accumulation, never-grad kept at steps_counted=0, means exact")

# NaN masking still works per-param
m.zero_grad()
(m.a.sum() + m.b.sum()).backward()
m.a.grad[0] = float("nan")
gs.accumulate()
d2 = gs.count.tolist()
assert d2[0] == 3 and d2[2] == 4, d2  # a masked this step, b counted
print("PASS: NaN step masked per-param")
print("ALL_PASS")
