"""Screen for the ENSEMBLE TEACHER proposal (queued as iter 55) -- zero GPU, reads two dumps.

The proposal averages two teachers (the d=128 pretrain model + a frozen past champion) on the
theory that KD pays here through target-VARIANCE reduction, so averaging two INDEPENDENT teachers
reduces it further. That mechanism needs the two teachers to actually DISAGREE. If they predict
nearly the same thing, 0.5*(p_A+p_B) ~ p_A and the intervention is inert -- the same shape as the
decay-floor probe, which killed a plausible lever by measuring that its bound was not binding.

Both dumps were produced over the IDENTICAL batch stream (same seed/db/MAX), which `labels_sum`
verifies per step, so rows correspond one-for-one and no forward pass is needed.

⚠ The dumps predate RWKV_KD_DUMP_LABELS, so there are no targets here and this CANNOT say which
teacher is more accurate or whether their ERRORS decorrelate. It answers only the prior question --
is the intervention non-trivial at all -- which is one-sided but decisive when the answer is "no".
"""
import sys
import numpy as np
import torch

A_DIR = 'C:/rwkv_kd_dump/t128_seedpair_65k'      # d=128 pretrain teacher (KD teacher today)
B_DIR = 'C:/rwkv_kd_dump/ours_i45_full'          # frozen plain iter-45 champion

STEPS = [1, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 10900]

rows_tot = 0
mis = 0
d_curve, d_imm = [], []
pa_all, pb_all = [], []
conf_a, conf_b, conf_e = [], [], []

for st in STEPS:
    a = torch.load(f'{A_DIR}/step_{st}.pt', map_location='cpu')
    b = torch.load(f'{B_DIR}/step_{st}.pt', map_location='cpu')
    if a['labels_sum'] != b['labels_sum'] or a['shape'] != b['shape']:
        mis += 1
        print(f'  step {st}: MISALIGNED  {a["labels_sum"]} vs {b["labels_sum"]}')
        continue
    pa = a['p_curve'].float().numpy().ravel()
    pb = b['p_curve'].float().numpy().ravel()
    # Padding rows carry no prediction; keep only rows where SOMETHING was predicted.
    m = (pa > 0) & (pa < 1) & (pb > 0) & (pb < 1)
    pa, pb = pa[m], pb[m]
    rows_tot += pa.size
    d_curve.append(np.abs(pa - pb))
    pa_all.append(pa); pb_all.append(pb)

    ia = a['p_imm_all'].float().numpy().reshape(-1, 4)
    ib = b['p_imm_all'].float().numpy().reshape(-1, 4)
    mi = (ia.sum(1) > 0.99) & (ia.sum(1) < 1.01) & (ib.sum(1) > 0.99) & (ib.sum(1) < 1.01)
    ia, ib = ia[mi], ib[mi]
    d_imm.append(np.abs(ia - ib).sum(1) / 2.0)          # total variation distance
    # Confidence = distance from the maximally-uninformative 0.5; averaging can only blur.
    conf_a.append(np.abs(pa - 0.5)); conf_b.append(np.abs(pb - 0.5))
    conf_e.append(np.abs(0.5 * (pa + pb) - 0.5))

if mis:
    print(f'!! {mis}/{len(STEPS)} steps misaligned -- dumps are NOT the same batch stream')
    sys.exit(1)

dc = np.concatenate(d_curve); di = np.concatenate(d_imm)
pa = np.concatenate(pa_all); pb = np.concatenate(pb_all)
ca, cb, ce = map(np.concatenate, (conf_a, conf_b, conf_e))

print(f'\nALIGNED on all {len(STEPS)} sampled steps; {rows_tot:,} predicted ahead rows\n')
print('--- p_curve (the ahead/KD target) ---')
print(f'  teacher A (d=128)  mean {pa.mean():.4f}  std {pa.std():.4f}')
print(f'  teacher B (i45)    mean {pb.mean():.4f}  std {pb.std():.4f}')
print(f'  Pearson r(A,B)                 {np.corrcoef(pa, pb)[0,1]:.4f}')
print(f'  mean |A-B|                     {dc.mean():.4f}')
print(f'  RMS  |A-B|                     {np.sqrt((dc**2).mean()):.4f}')
print(f'  median / p90 / p99 |A-B|       {np.median(dc):.4f} / '
      f'{np.percentile(dc,90):.4f} / {np.percentile(dc,99):.4f}')
print(f'  DISAGREEMENT RATIO  mean|A-B| / std(A)   {dc.mean()/pa.std():.4f}')
print(f'  rows where |A-B| > 0.05        {100.0*(dc>0.05).mean():.2f}%')
print(f'  rows where |A-B| > 0.10        {100.0*(dc>0.10).mean():.2f}%')
print('\n--- p_imm_all (4-way rating dist) ---')
print(f'  mean total-variation distance  {di.mean():.4f}')
print(f'  p90 / p99 TV                   {np.percentile(di,90):.4f} / {np.percentile(di,99):.4f}')
print('\n--- what averaging does to target confidence ---')
print(f'  mean |p-0.5|  A {ca.mean():.4f}   B {cb.mean():.4f}   ensemble {ce.mean():.4f}')
print(f'  ensemble vs the more confident teacher: '
      f'{100.0*(ce.mean()/max(ca.mean(), cb.mean()) - 1):+.2f}%')

# ---------------------------------------------------------------------------
# CALIBRATION: how big is this intervention next to a KD change we already priced?
# The mix is linear in probability space (srs_model.py:1189):
#     label_y = alpha*teacher + (1-alpha)*hard
# so swapping teacher A for 0.5*(A+B) shifts the target by alpha*0.5*|A-B|, while iter 39's
# accepted alpha 0.5 -> 0.9 shifted it by 0.4*|teacher-hard|. Both are the same units.
ALPHA = 0.9
# No hard labels in these dumps (they predate RWKV_KD_DUMP_LABELS), so E|p-y| is estimated
# from the teacher's own distribution assuming calibration: E|p-y| = 2*E[p(1-p)].
e_p1p = float((pa * (1.0 - pa)).mean())
e_abs_err = 2.0 * e_p1p
shift_ens = ALPHA * 0.5 * float(dc.mean())
shift_a39 = 0.4 * e_abs_err
print('\n--- magnitude vs the accepted iter-39 KD change (alpha 0.5 -> 0.9) ---')
print(f'  E|teacher-hard| (calibration estimate)   {e_abs_err:.4f}')
print(f'  target shift, iter 39 (0.4*|p-y|)        {shift_a39:.4f}')
print(f'  target shift, ensemble (0.9*0.5*|A-B|)   {shift_ens:.4f}')
print(f'  RATIO ensemble / iter39                  {shift_ens/shift_a39:.3f}')
print(f'  iter 39 measured +0.000158 / +0.000153 -> linear projection '
      f'{0.000158*shift_ens/shift_a39:+.6f} / {0.000153*shift_ens/shift_a39:+.6f}')
print('  (bar is 0.0001 raw in BOTH modes; same-capacity noise floor is 7.5e-5)')

# Is the disagreement concentrated on the uncertain rows, where the loss gradient lives?
unc = pa < 0.95
print('\n--- is the disagreement where the loss is? ---')
print(f'  rows with teacher p < 0.95      {100.0*unc.mean():.1f}%')
print(f'  mean |A-B| there                {dc[unc].mean():.4f}')
print(f'  mean |A-B| on confident rows    {dc[~unc].mean():.4f}')
print(f'  share of total |A-B| mass on uncertain rows  {100.0*dc[unc].sum()/dc.sum():.1f}%')
