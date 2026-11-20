"""Unit test for the TA3 A/Q split on synthetic label layouts (no data, no GPU, no torch model).

Covers the failure that killed epoch 1 on 2026-11-15 (`mode 'ahead' has no labelled rows in this
phase`) and the invariants the merge depends on:

  1. exact partition: for every group, phase-A count + phase-Q count == whole-batch count;
  2. non-empty phases: a group with >= 2 labelled rows has at least one row in EACH phase;
  3. degenerate groups: a group with < 2 labelled rows lands whole in phase A and carries no Q rows
     (the merge then gives it weight 0 in Q, i.e. exactly its full-batch gradient);
  4. the split stays near the requested fraction of each group's rows;
  5. time order inside a group: every phase-A row of a group is at or before every phase-Q row of
     that group (earlier rows train the direction, later rows test it);
  6. multi-batch shapes (B > 1, so several rows share one time position) and tiny sequences (T == 2)
     are handled.

Run: .venv/Scripts/python.exe scratchpad/ta3/test_split.py
"""
import dataclasses
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rwkv import ta3  # noqa: E402

HAS, ISQ = ta3._HAS_LABEL, ta3._IS_QUERY


@dataclasses.dataclass
class FakeBatch:
    labels: torch.Tensor
    probe_rows: object = None


def make_labels(B, T, ahead_positions, imm_positions, extra_noise=True):
    """labels[..., HAS]/[..., ISQ] set at the given (batch, time) positions; other columns noise."""
    labels = torch.zeros(B, T, 8, dtype=torch.float32)
    if extra_noise:
        labels[..., 0] = torch.rand(B, T)
        labels[..., 5] = (torch.rand(B, T) < 0.3).float()  # label_is_equalize, must be ignored
    for b, t in ahead_positions:
        labels[b, t, HAS] = 1.0
    for b, t in imm_positions:
        labels[b, t, HAS] = 1.0
        labels[b, t, ISQ] = 1.0
    return labels


def check(labels, frac, tag):
    batch = FakeBatch(labels=labels)
    a, q = ta3.split_batch(batch, frac=frac)
    n_a_full, n_i_full = ta3._mode_counts(batch)
    a_cnt, i_cnt = ta3._mode_counts(a)
    q_cnt, j_cnt = ta3._mode_counts(q)
    # 1. exact partition
    assert abs((a_cnt + q_cnt) - n_a_full) < 1e-9, f"{tag}: ahead {a_cnt}+{q_cnt} != {n_a_full}"
    assert abs((i_cnt + j_cnt) - n_i_full) < 1e-9, f"{tag}: imm {i_cnt}+{j_cnt} != {n_i_full}"
    # 2/3. non-empty phases, or the documented degenerate case
    for name, full, ca, cq in (("ahead", n_a_full, a_cnt, q_cnt), ("imm", n_i_full, i_cnt, j_cnt)):
        if full >= 2:
            assert ca >= 1 and cq >= 1, f"{tag}: {name} full={full} but phases {ca}/{cq}"
        elif full == 1:
            assert ca == 1 and cq == 0, f"{tag}: single {name} row must land in phase A ({ca}/{cq})"
        else:
            assert ca == 0 and cq == 0, f"{tag}: absent {name} group got rows ({ca}/{cq})"
        # 4. near the requested fraction
        if full >= 2:
            assert abs(ca / full - frac) <= 0.5, f"{tag}: {name} A fraction {ca}/{full} vs {frac}"
    # 5. time order inside each group: all phase-A rows are at or before all phase-Q rows
    for name, mask in (("ahead", (labels[..., HAS] > 0.5) & (labels[..., ISQ] <= 0.5)),
                       ("imm", (labels[..., HAS] > 0.5) & (labels[..., ISQ] > 0.5))):
        rows_a = mask & (a.labels[..., HAS] > 0.5)
        rows_q = mask & (q.labels[..., HAS] > 0.5)
        if int(rows_a.sum()) and int(rows_q.sum()):
            t_a = rows_a.nonzero(as_tuple=False)[:, 1].max()
            t_q = rows_q.nonzero(as_tuple=False)[:, 1].min()
            assert int(t_a) <= int(t_q), f"{tag}: {name} A time {int(t_a)} > Q time {int(t_q)}"
    return a_cnt, i_cnt, q_cnt, j_cnt


def main():
    random.seed(0)
    torch.manual_seed(0)

    # T == 2, one row of each group: the exact shape that used to abort (cut had to be position 1,
    # so the Q phase held only the imm row and the ahead group had no Q row at all). The ahead row
    # now lands whole in phase A.
    labels = make_labels(1, 2, [(0, 0)], [(0, 1)])
    a_cnt, i_cnt, q_cnt, j_cnt = check(labels, 0.5, "T2 mixed")
    assert (a_cnt, i_cnt, q_cnt, j_cnt) == (1.0, 1.0, 0.0, 0.0), (a_cnt, i_cnt, q_cnt, j_cnt)

    # single futher degenerate shapes
    check(make_labels(1, 3, [(0, 1)], [(0, 0)]), 0.5, "T3 single ahead late")
    check(make_labels(1, 8, [(0, 0)], [(0, 7)]), 0.5, "T8 far apart singles")
    check(make_labels(1, 2, [(0, 0)], [(0, 0)]), 0.5, "T2 one row both groups")

    # random layouts, single and multi batch, several fractions
    for trial in range(400):
        T = random.choice([2, 3, 4, 8, 64, 512, 8192])
        B = random.choice([1, 1, 2, 4])
        n_a = random.choice([0, 1, 1, 2, 3, 7, 40, 300])
        n_i = random.choice([0, 1, 1, 2, 3, 7, 40, 300])
        pos = [(b, t) for b in range(B) for t in range(T)]
        random.shuffle(pos)
        ahead, imm = pos[:min(n_a, len(pos))], pos[min(n_a, len(pos)):]
        imm = imm[:min(n_i, len(imm))]
        labels = make_labels(B, T, ahead, imm)
        for frac in (0.25, 0.5, 0.75):
            check(labels, frac, f"trial{trial} T={T} B={B} na={len(ahead)} ni={len(imm)} f={frac}")

    # an extreme shape: all ahead rows early, all imm rows late (what a single cut cannot split)
    labels = make_labels(1, 256, [(0, t) for t in range(0, 8)], [(0, t) for t in range(248, 256)])
    for frac in (0.1, 0.5, 0.9):
        check(labels, frac, f"disjoint blocks f={frac}")

    print("SPLIT_TESTS_OK")


if __name__ == "__main__":
    main()
