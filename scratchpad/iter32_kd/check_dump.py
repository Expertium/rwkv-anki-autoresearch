"""Validate a KD teacher dump before a night of training is staked on it.

The student side already hard-exits 43 on a batch-stream mismatch (labels checksum + shape), so
ALIGNMENT is covered. What no existing check covers is whether the dumped tensors are the right
QUANTITY at all -- a teacher loaded into the wrong architecture, or run with a flag it was never
trained under, produces perfectly well-aligned garbage, and the student would train on it happily
for ten hours. So this checks the two things that are true of teacher outputs and nothing else:

  * p_curve is a PROBABILITY   -> strictly inside (0,1)
  * p_imm_all is a DISTRIBUTION over the 4 ratings -> rows sum to 1

and it projects the full dump's disk footprint from the smoke files, because the per-step size
depends on the padded batch shape (B*T), which is not something to discover at 90% disk usage.

Usage:
  python scratchpad/iter32_kd/check_dump.py <dump_dir> --expect-steps N [--planned-steps M]
                                            [--max-gb 60]
"""
import argparse
import glob
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--expect-steps", type=int, required=True,
                    help="how many step_*.pt files must be present NOW")
    ap.add_argument("--planned-steps", type=int, default=0,
                    help="steps the full dump will have; enables the disk projection")
    ap.add_argument("--max-gb", type=float, default=60.0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "step_*.pt")))
    print(f"dump dir: {args.dump_dir}")
    print(f"files: {len(files)} (expected {args.expect_steps})")
    if len(files) < args.expect_steps:
        print("DUMP_CHECK_FAIL: too few step files")
        return 1

    sizes = [os.path.getsize(f) for f in files]
    mean_mb = sum(sizes) / len(sizes) / 1e6
    print(f"mean file size: {mean_mb:.2f} MB")
    if args.planned_steps:
        proj_gb = mean_mb * args.planned_steps / 1000.0
        print(f"projected full dump: {proj_gb:.1f} GB for {args.planned_steps} steps "
              f"(limit {args.max_gb} GB)")
        if proj_gb > args.max_gb:
            print("DUMP_CHECK_FAIL: projected dump exceeds the disk budget -- do NOT proceed")
            return 2

    bad = 0
    for f in files:
        rec = torch.load(f, weights_only=True)
        pc = rec["p_curve"].float()
        pi = rec["p_imm_all"].float()
        if not torch.isfinite(pc).all() or not torch.isfinite(pi).all():
            print(f"  {os.path.basename(f)}: NON-FINITE values"); bad += 1; continue
        if pc.min() <= 0.0 or pc.max() >= 1.0:
            print(f"  {os.path.basename(f)}: p_curve outside (0,1): "
                  f"[{pc.min():.6f}, {pc.max():.6f}]"); bad += 1; continue
        rowsum = pi.sum(dim=-1)
        # fp16 storage, so allow a loose tolerance -- this is a "is it a distribution at all"
        # test, not a numerics test.
        if (rowsum - 1.0).abs().max() > 5e-2:
            print(f"  {os.path.basename(f)}: p_imm_all rows do not sum to 1 "
                  f"(worst {(rowsum - 1.0).abs().max():.4f})"); bad += 1; continue
        print(f"  {os.path.basename(f)}: shape {list(pc.shape)}  "
              f"p_curve [{pc.min():.4f},{pc.max():.4f}] mean {pc.mean():.4f}  "
              f"p_again mean {pi[..., 0].mean():.4f}")

    if bad:
        print(f"DUMP_CHECK_FAIL: {bad}/{len(files)} files invalid")
        return 3
    print("DUMP_CHECK_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
