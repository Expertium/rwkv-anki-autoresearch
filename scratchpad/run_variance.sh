#!/usr/bin/env bash
# Measure run-to-run variance of the tuner objective: train the SAME from-scratch WS config TWICE
# (current code = stochastic per-batch augmentation kept) and eval each on users 101-200, comparing
# the by-user-mean LogLoss (ahead + imm). JIT-on (the @torch.jit.ignore fix). Runs ~7 min total.
set -e
cd /c/Users/Andrew/rwkv-anki-autoresearch
PY=.venv/Scripts/python.exe
for rep in 1 2; do
  rm -rf scratchpad/var_run
  echo "=== TRAIN rep $rep (from scratch, stochastic augmentation, JIT-on) ==="
  OMP_NUM_THREADS=7 $PY -u -m rwkv.train_rwkv --config scratchpad/var_ws.toml 2>&1 | tail -2
  cp scratchpad/var_run/varws_558.pth scratchpad/var_ckpt_$rep.pth
  sed "s#MODEL_PATH = .*#MODEL_PATH = \"scratchpad/var_ckpt_$rep.pth\"#; s/RWKV-iter36/RWKV-var$rep/; s/RWKV-P-iter36/RWKV-P-var$rep/" rwkv/get_result_config_iter36.toml > scratchpad/var_eval_$rep.toml
  echo "=== EVAL rep $rep on 101-200 ==="
  OMP_NUM_THREADS=7 $PY -m rwkv.get_result --config scratchpad/var_eval_$rep.toml 2>&1 | tail -1
done
$PY -c "
import json
def m(f):
    r=[json.loads(l) for l in open(f)]
    return sum(x['metrics']['LogLoss'] for x in r)/len(r)
a1=m('result/RWKV-var1.jsonl'); a2=m('result/RWKV-var2.jsonl')
i1=m('result/RWKV-P-var1.jsonl'); i2=m('result/RWKV-P-var2.jsonl')
print('=== RUN-TO-RUN VARIANCE (2 from-scratch trainings, stochastic augmentation, 100-user by-user-mean) ===')
print(f'  ahead: run1 {a1:.6f}  run2 {a2:.6f}  |diff| {abs(a1-a2):.6f}')
print(f'  imm:   run1 {i1:.6f}  run2 {i2:.6f}  |diff| {abs(i1-i2):.6f}')
"
echo "VARIANCE_DONE"
