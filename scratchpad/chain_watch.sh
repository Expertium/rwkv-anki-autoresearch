#!/bin/bash
# Follow WHICHEVER arm is currently running, and alarm on a STALLED run as well as a dead one.
#
# ⚠ WHY THIS EXISTS -- two failures on 2026-08-20/21, same family, one level apart:
#   1. the first featA watcher pinned a PID. A runner is a CHAIN (WS -> decay -> eval) and each
#      phase is a new process, so a normal transition read as death;
#   2. the fixed watcher followed featA BY NAME and exited when featA completed -- so featB
#      launched into an unmonitored world, deadlocked at step ~950, and sat with an idle GPU for
#      69 minutes before a human-initiated check found it.
# Fixing the witness for the arm you are watching is not the same as having a witness for the arm
# that replaces it. This one takes a LIST of logs and follows whichever is live.
#
# ⚠ AND IT ALARMS ON A STALL, NOT ONLY ON DEATH. featB's fetch worker died while the PARENT stayed
# alive, so every liveness check based on "is the process there" said yes. The signal that would
# have caught it in minutes is the one the flight-recorder rule already names: ABSENCE OF FORWARD
# PROGRESS. Here that is the training log's byte count not growing.
#
# Two witnesses before declaring anything, as before:
#   dead  = no runner process AND no terminal marker in its log
#   stall = runner process alive AND its newest phase log has not grown for STALL_MIN minutes
# It only REPORTS. It never kills.
#
# Usage: bash scratchpad/chain_watch.sh <arm_dir> [<arm_dir> ...]
STALL_MIN=${STALL_MIN:-12}
LOG=scratchpad/chain_watch.log
: > "$LOG"

note () { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
note "watching: $* (stall threshold ${STALL_MIN} min)"

declare -A lastsize
declare -A lastmove

for i in $(seq 1 2000); do
  live_any=0
  for d in "$@"; do
    arm=$(basename "$d")
    marker=no
    [ -f "$d/$arm.log" ] && grep -qE "^DONE_EXIT_" "$d/$arm.log" 2>/dev/null && marker=yes
    alive=$(powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*run_${arm}*' }) {'up'} else {'gone'}" 2>/dev/null | tr -d '\r')

    if [ "$alive" = "gone" ] && [ "$marker" = "yes" ]; then
      continue                      # finished cleanly; nothing to watch
    fi
    if [ "$alive" = "gone" ] && [ "$marker" = "no" ]; then
      note "ALERT $arm: runner gone with NO terminal marker -- real death"
      echo "ALERT $arm runner gone with no terminal marker at $(date +%H:%M:%S)"
      exit 2
    fi

    live_any=1
    # newest phase log for this arm
    newest=$(ls -t "$d"/{ws,decay,eval}_*.log 2>/dev/null | head -1)
    if [ -z "$newest" ]; then
      note "$arm: alive, no phase log yet"
      continue
    fi
    sz=$(stat -c%s "$newest" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "${lastsize[$arm]:-}" != "$sz" ]; then
      lastsize[$arm]=$sz
      lastmove[$arm]=$now
    fi
    idle=$(( (now - ${lastmove[$arm]:-$now}) / 60 ))
    note "$arm: alive, $(basename "$newest") ${sz}B, idle ${idle} min"
    if [ "$idle" -ge "$STALL_MIN" ]; then
      note "ALERT $arm: STALLED -- $(basename "$newest") has not grown for ${idle} min while the runner is alive"
      echo "ALERT $arm STALLED ${idle} min with the runner alive at $(date +%H:%M:%S)"
      exit 3
    fi
  done
  if [ "$live_any" = "0" ]; then
    note "no live arm remains -- standing down"
    exit 0
  fi
  sleep 60
done
