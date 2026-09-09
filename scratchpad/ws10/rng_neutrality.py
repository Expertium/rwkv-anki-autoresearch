"""Does a VALIDATION pass perturb the training RNG? Decides whether the endgame decay can be
given real checkpoints for free.

WHY IT MATTERS. `write_decay_setup.py` writes VALIDATE_EVERY = 100000, so a 21,870-step decay
checkpoints at step 50 and at the end and NOTHING else -- a crash at step 21,000 costs 6.2 h.
CLAUDE.md flags this as a MUST-FIX for the endgame decay (a PC restart already killed one at
10,681/10,935). The fix is trivial, but only FREE if raising the validation frequency leaves the
training trajectory alone. If validation draws from the main-process RNG, arm 1's dropout draws
would diverge from realcyc's decay and the one CLEAN comparison in PREREG.md picks up a confound.

The claim from reading the code is that it does not: validation runs under model.eval() inside
torch.no_grad(), so the dropout modules do not draw, and the per-batch content is seeded inside
the fetch WORKERS. This executes that claim instead of trusting it.
"""
import torch
from torch import nn

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(16, 16), nn.Dropout(0.5), nn.Linear(16, 16))
x = torch.randn(8, 16)


def state():
    return torch.get_rng_state().clone()


net.train()
_ = net(x)                      # a training step DOES draw
s_before = state()

net.eval()
with torch.no_grad():
    for _ in range(25):         # stand in for a whole validation pass
        _ = net(x)
net.train()
s_after = state()

same = torch.equal(s_before, s_after)
print("RNG state unchanged across an eval-mode pass:", same)

# Non-vacuity: the same loop in TRAIN mode must CHANGE the state, or this proves nothing.
net.train()
s0 = state()
for _ in range(25):
    _ = net(x)
changed = not torch.equal(s0, state())
print("RNG state changed across a train-mode pass:  ", changed, " (must be True)")

if same and changed:
    print("\n=> validation is RNG-NEUTRAL: raising VALIDATE_EVERY costs the trajectory nothing")
else:
    print("\n=> NOT neutral (or the check is vacuous) -- do not change the decay cadence")
    raise SystemExit(1)
