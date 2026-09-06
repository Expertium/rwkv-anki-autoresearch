"""eqw P2 (engagement): by-user ahead BCE on the never-scored FIRST SIXTH of each train user's labelled
history vs the rest, from a screen_pass.py records file. Prediction (PREREG P2): on the candidate the
first-sixth loss RISES relative to realcyc's (the fit moved away from those rows) while the rest FALLS.
Usage: p2_history_split.py <records.npz> [<control records.npz>]
"""
import sys

import numpy as np


def split(path):
    z = np.load(path)
    p, y, u = z["p"], z["y"], z["u"]
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    bce = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    first, rest = [], []
    for uid in np.unique(u):
        m = np.where(u == uid)[0]
        q = np.arange(len(m)) / len(m)
        first.append(bce[m[q < 1 / 6]].mean())
        rest.append(bce[m[q >= 1 / 6]].mean())
    return float(np.mean(first)), float(np.mean(rest)), len(np.unique(u))


a = split(sys.argv[1])
print(f"candidate {sys.argv[1]}: by-user ahead BCE first sixth {a[0]:.5f}  rest {a[1]:.5f}  (users {a[2]})")
if len(sys.argv) > 2:
    b = split(sys.argv[2])
    print(f"control   {sys.argv[2]}: by-user ahead BCE first sixth {b[0]:.5f}  rest {b[1]:.5f}  (users {b[2]})")
    d_first, d_rest = a[0] - b[0], a[1] - b[1]
    print(f"delta (candidate - control): first sixth {d_first:+.5f}   rest {d_rest:+.5f}")
    engaged = d_first > 0 and d_rest < 0
    print("P2 ENGAGED (first sixth up, rest down)" if engaged else "P2 NOT as predicted -- the weighting did not move the fit the way the PREREG expects; interpret with care")
