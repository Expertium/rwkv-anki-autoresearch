#!/bin/bash
# Exit (and therefore notify) the moment free RAM crosses a floor, so a slow slide toward the
# generation-1 OOM is caught while there is still time to act.
#
# It does NOT kill anything. The gen-1 OOM took the TEST build, not the training run, so the
# correct response is to reduce the rebuild's PROCESSES and resume it -- the rebuild is resumable
# and featA is not. An auto-killer would be one bad threshold away from getting that backwards.
FLOOR_GB=2.5
LOG=scratchpad/features_rebuild/ram_alarm.log
: > "$LOG"
for i in $(seq 1 900); do
  ram=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" 2>/dev/null | tr -d '\r' | tr ',' '.')
  ok=$(awk -v r="$ram" -v f="$FLOOR_GB" 'BEGIN{print (r+0 < f+0) ? "LOW" : "ok"}')
  echo "$(date +%H:%M:%S) ram=${ram}GB $ok" >> "$LOG"
  if [ "$ok" = "LOW" ]; then
    echo "ALARM free RAM ${ram}GB below ${FLOOR_GB}GB" >> "$LOG"
    echo "ALARM free RAM ${ram}GB below ${FLOOR_GB}GB at $(date +%H:%M:%S)"
    exit 3
  fi
  if grep -qE "^DONE_EXIT_" scratchpad/features_rebuild/rebuild2.log 2>/dev/null; then
    echo "rebuild finished -- alarm standing down" >> "$LOG"; exit 0
  fi
  sleep 40
done
