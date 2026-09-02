"""Are the 7 pseudo day-offset cycles EXACTLY dims 86..113 of the 114-dim input? Checked on a
real batch, not by arithmetic -- a wrong range would zero the wrong thing and the surgery's own
non-vacuity check (norm > 0) would still pass.

Structure test: dims 46..85 must be ID codes (values in {-1.5,-0.5,0.5,1.5}); dims 86..113 must be
14 (sin,cos) pairs on the unit circle. The inputs are stored bf16, whose unit roundoff is 2^-8, so
sin^2+cos^2 lands within ~0.01 of 1 -- the first version of this check used 1e-3 and "failed" on
roundoff. NEGATIVE CONTROL: the same pair test on a window shifted by ONE dim must FAIL, or the
test does not discriminate position and a pass means nothing.
"""
import json, lmdb, torch, sys
from rwkv.prepare_batch import get_data, prepare

TOL = 0.02   # bf16: each of sin, cos carries ~0.4% relative error; the sum of squares ~0.01

env = lmdb.open("F:/rwkv_lmdb/test_db_5k_id3", map_size=250_000_000_000, readonly=True, lock=False)
with env.begin(write=False) as txn:
    b = json.loads(txn.get(b"5001_batches"))[0]
    d = get_data(txn, (5001, b[0], b[1], b[2]), device="cpu")
pb = prepare([d], target_len=int(d.length), seed=4321)
x = pb.start.float().reshape(-1, 114)
real = x[x.abs().sum(1) > 0]

def pairs_on_circle(lo):
    seg = real[:, lo:lo + 28].reshape(-1, 14, 2)
    r2 = (seg ** 2).sum(-1)
    return float((r2 - 1).abs().max())

codes = torch.tensor([-1.5, -0.5, 0.5, 1.5])
id_ok = torch.isin(real[:, 46:86], codes).float().mean().item()
at86 = pairs_on_circle(86)
at85 = pairs_on_circle(85)   # shifted by one: pairs straddle (cos_k, sin_k+1) -- must break
print("rows %d   dims 46..85 in ID-code set: %.4f" % (real.shape[0], id_ok))
print("dims 86..113 as (sin,cos) pairs: max |sin^2+cos^2 - 1| = %.4f   (bf16 tol %.2f)" % (at86, TOL))
print("NEGATIVE CONTROL, window 85..112: max |sin^2+cos^2 - 1| = %.4f   (must exceed tol)" % at85)
ok = id_ok > 0.999 and at86 < TOL and at85 > TOL
print("LAYOUT CHECK", "PASS -- cycles are dims 86..113 and the test discriminates position"
      if ok else "*** FAIL")
sys.exit(0 if ok else 1)
