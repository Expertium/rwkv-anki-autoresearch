"""Unit test for the fused rwkv7_pq_argmin CUDA kernel (tier-1 shift-PQ search).

vs torch.cdist().argmin() reference:
- exact-tie adversarial cases (rows == centroids, duplicated centroids): indices must match
  EXACTLY (both paths compute exact-zero/equal distances -> first-min tie-break).
- random data: distances round differently (direct-form vs mm-form), so an index may flip only
  when two centroids are near-equidistant (< ~1e-6 relative). Report the flip count; gate at
  < 0.01% of rows. Also print a micro-benchmark on the real per-call shape.
"""
import sys
import time

sys.path.insert(0, r"C:\Users\Andrew\rwkv-anki-autoresearch")
import torch  # noqa: E402
import rwkv.model.rwkv_ops  # noqa: E402,F401  (loads the RWKV_CUDA extension)

op = torch.ops.rwkv.rwkv7_pq_argmin
dev = "cuda"
torch.manual_seed(0)
fails = 0


def ref_idx(a, b):
    outs = [torch.cdist(a[s:s + 8192], b).argmin(dim=1) for s in range(0, a.shape[0], 8192)]
    return torch.cat(outs)


def check(a, b, tag, exact):
    global fails
    ik = op(a, b)
    ir = ref_idx(a, b)
    n_diff = (ik != ir).sum().item()
    frac = n_diff / a.shape[0]
    ok = (n_diff == 0) if exact else (frac < 1e-4)
    print(f"{tag:42s} mismatches={n_diff}/{a.shape[0]} ({frac:.2e})  {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails += 1
        if n_diff:
            i = (ik != ir).nonzero()[0].item()
            da = torch.cdist(a[i:i + 1], b)[0]
            print(f"   first flip row {i}: kernel={ik[i].item()} (d={da[ik[i]].item():.9e}) "
                  f"ref={ir[i].item()} (d={da[ir[i]].item():.9e})")


# random q72u shapes (full call size: ~110k rows x 4096 x 16)
for t in range(3):
    a = torch.randn(110000, 16, device=dev)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-20)
    b = torch.randn(4096, 16, device=dev) * 0.3
    check(a, b, f"random 110000x16 vs 4096 (t{t})", exact=False)

# small-catalog + ragged shapes
a = torch.randn(3313, 16, device=dev)
a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-20)
check(a, torch.randn(256, 16, device=dev) * 0.3, "ragged 3313x16 vs 256", exact=False)

# adversarial exact ties
b = torch.randn(4096, 16, device=dev) * 0.3
a = b[torch.randint(0, 4096, (8192,), device=dev)].clone()
check(a, b, "rows == centroids (exact zeros)", exact=True)
b_dup = b.clone()
b_dup[2048:] = b_dup[:2048]
check(a, b_dup, "duplicated centroids (exact ties)", exact=True)

# NaN row -> masked upstream; kernel must not crash and returns 0
a_nan = a.clone()
a_nan[5] = float("nan")
i = op(a_nan, b)
print(f"NaN row -> idx {i[5].item()} (expect 0)  {'PASS' if i[5].item() == 0 else 'FAIL'}")
if i[5].item() != 0:
    fails += 1

# micro-benchmark: kernel vs the eager sq-mm path vs cdist, real call shape
a = torch.randn(110000, 16, device=dev)
a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-20)
b = torch.randn(4096, 16, device=dev) * 0.3


def bench(fn, n=20):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


from rwkv.model.rwkv_model import _sq_dist_rows  # noqa: E402

t_k = bench(lambda: op(a, b))
t_sq = bench(lambda: torch.cat([_sq_dist_rows(a[s:s + 8192], b).clamp_min_(0).argmin(dim=1)
                                for s in range(0, a.shape[0], 8192)]))
t_cd = bench(lambda: ref_idx(a, b))
print(f"per-call (110k x 4096 x 16): kernel {t_k:.2f} ms | sq-mm {t_sq:.2f} ms | cdist {t_cd:.2f} ms")

print("ALL_PASS" if fails == 0 else f"FAILURES={fails}")
sys.exit(0 if fails == 0 else 1)
