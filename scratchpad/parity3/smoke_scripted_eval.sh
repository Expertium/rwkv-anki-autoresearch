#!/usr/bin/env bash
# ONE-USER SCRIPTED EVAL -- the pre-flight that iter 48 needed and did not have (~90 s GPU).
#
# WHY IT EXISTS. Training always runs RWKV_NO_JIT=1, and QAT evals do too, so the ONLY path that
# scripts the model is a PLAIN eval. A bug reachable only under TorchScript is therefore invisible
# to every smoke, every training run and every QAT eval -- and surfaces at the eval phase, i.e.
# AFTER the full training spend. That is exactly how iter 48 lost its eval after 6.5 h: an
# unannotated `@torch.jit.ignore` returning None handed scripted code an undefined tensor.
#
# ⚠ COMPILING IS NOT RUNNING. `torch.jit.script(model)` succeeding proves nothing about ignored
# bodies -- their return types are only exercised when a tensor actually flows through. This script
# runs the real get_result path on ONE user, which does exercise them.
#
# RUN THIS BEFORE ANY LAUNCH THAT TOUCHES srs_model.py / rwkv_model.py, and after the checkpoint
# exists it doubles as a fast sanity read (`size` must match the champion's for the same user).
#
# Usage:  bash scratchpad/parity3/smoke_scripted_eval.sh <eval_toml> [user]
#   The toml is copied with USER_END clamped to one user and the FILE_* tags redirected, so it
#   never touches the real run's result jsonls.
set -euo pipefail
cd "$(dirname "$0")/../.."
TOML="${1:?usage: smoke_scripted_eval.sh <eval_toml> [user]}"
USER="${2:-}"
[ -z "$USER" ] && USER=$(grep -E '^USER_START' "$TOML" | tr -dc '0-9')
DBG=scratchpad/parity3/_smoke_scripted.toml
sed -e "s/^USER_START = .*/USER_START = $USER/" -e "s/^USER_END = .*/USER_END = $USER/" \
    -e 's/^FILE_AHEAD = .*/FILE_AHEAD = "RWKV-smokejit"/' \
    -e 's/^FILE_IMM = .*/FILE_IMM = "RWKV-P-smokejit"/' \
    -e 's/^NUM_FETCH_PROCESSES = .*/NUM_FETCH_PROCESSES = 2/' "$TOML" > "$DBG"
rm -f result/RWKV-smokejit.jsonl result/RWKV-P-smokejit.jsonl
# NOTE: the caller must already have the run's arch env exported. RWKV_NO_JIT must be UNSET --
# that is the whole point; setting it here would make the test vacuous.
if [ "${RWKV_NO_JIT:-}" = "1" ]; then echo "REFUSING: RWKV_NO_JIT=1 makes this test vacuous"; exit 2; fi
.venv/Scripts/python.exe -u -m rwkv.get_result --config "$DBG" > scratchpad/parity3/_smoke_scripted.log 2>&1 || {
  echo "SCRIPTED EVAL FAILED -- tail:"; tail -20 scratchpad/parity3/_smoke_scripted.log; exit 1; }
echo "SCRIPTED EVAL OK  user=$USER"
cat result/RWKV-smokejit.jsonl result/RWKV-P-smokejit.jsonl
