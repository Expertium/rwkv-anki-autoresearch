#!/bin/bash
# Chain monitor for the queued iteration chain. Replaces the inline monitor that cried
# "BOX HANG SUSPECTED" at 00:10 on 2026-08-18 while training was advancing at 0.967 steps/s.
#
# ★ WHY THAT FIRED, and it would have fired EVERY night: the flight recorder writes
# `flight_YYYYMMDD.csv`, so at 23:59:47 it stops appending to yesterday's file and starts
# today's. The old monitor resolved the filename ONCE at launch (`REC=...flight_20260817.csv`)
# and then watched a file that nothing would ever write to again. 23:59:47 + 624 s is exactly
# when the first alert arrived. Chains here run ~25 h, so every one of them crosses midnight.
#
# TWO FIXES, and the second is the general one:
#   1. re-resolve the newest flight_*.csv on every poll, so the rollover is invisible.
#   2. NEVER declare a hang from the recorder alone. A hang means the BOX stopped, which implies
#      training stopped too -- so require BOTH the recorder silent AND the training log not
#      growing. One signal failing (recorder crashed, file rotated, permissions) then costs a
#      log line instead of a false alarm. Corroboration is what makes an alert trustworthy;
#      CLAUDE.md already says the reliable signal is the ABSENCE OF FORWARD PROGRESS, and the
#      training log is the more direct witness to that than the recorder is.
#
# Notifications are one-shot per event (marker files), so a persistent condition does not spam.

cd /c/Users/Andrew/rwkv-anki-autoresearch || exit 1
RECDIR=/c/Users/Andrew/hang-monitor
MARK=/c/Temp/claude/chainmon
mkdir -p "$MARK"

DIRS="iter53_muonlora iter54_cmixpow iter52_kdalpha iter55_rgate iter57_decayshape"
stall=0
prev_ws=0

newest_train_log() {
  ls -t scratchpad/iter5*/ws_*.log scratchpad/iter5*/decay_*.log 2>/dev/null | head -1
}

while true; do
  # --- 1. chain-phase completions (anchored: an unanchored match hits prose mentioning the token)
  for d in $DIRS; do
    L="scratchpad/$d/${d%%_*}.log"
    [ -f "$L" ] || continue
    if grep -q "^DONE_EXIT_" "$L" 2>/dev/null && [ ! -f "$MARK/done_$d" ]; then
      touch "$MARK/done_$d"
      echo "$d TERMINATED: $(grep -h '^DONE_EXIT_' "$L" | tail -1)"
    fi
  done

  NEW=$(newest_train_log)
  if [ -n "$NEW" ]; then
    # --- 2. failure signatures. tr -d '\0' because a dirty shutdown pads the log with NULs.
    if tr -d '\0' < "$NEW" 2>/dev/null \
        | grep -qE "Nan from RWKV-7|Traceback|CUDA out of memory|INTERNAL ASSERT" \
        && [ ! -f "$MARK/err_$(basename "$NEW")" ]; then
      touch "$MARK/err_$(basename "$NEW")"
      echo "ERROR SIGNATURE in $(basename "$NEW"): $(tr -d '\0' < "$NEW" \
        | grep -oE 'Nan from RWKV-7|Traceback|CUDA out of memory|INTERNAL ASSERT' | tail -1)"
    fi
    # --- 3. stall on the training log itself
    cur_ws=$(stat -c %s "$NEW" 2>/dev/null || echo 0)
    if [ "$cur_ws" = "$prev_ws" ]; then stall=$((stall + 1)); else stall=0; fi
    prev_ws=$cur_ws
    if [ "$stall" -ge 30 ]; then
      echo "STALL: $(basename "$NEW") has not grown in ~30 min"
      stall=0
    fi
  fi

  # --- 4. box hang: BOTH witnesses must agree (see the header)
  REC=$(ls -t "$RECDIR"/flight_*.csv 2>/dev/null | head -1)
  if [ -n "$REC" ]; then
    now=$(date +%s)
    rec_age=$((now - $(stat -c %Y "$REC")))
    if [ "$rec_age" -gt 600 ]; then
      if [ "$stall" -ge 8 ]; then
        echo "BOX HANG SUSPECTED: recorder silent ${rec_age}s AND $(basename "$NEW") not growing"
      else
        # recorder-only silence: log it, do not alert. This is the midnight-rollover case and
        # the recorder-crashed case, neither of which is a hang.
        echo "note: recorder $(basename "$REC") silent ${rec_age}s but training IS advancing"
      fi
    fi
  fi
  sleep 60
done
