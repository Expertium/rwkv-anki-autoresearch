#!/bin/bash
# Watch the gen-2 rebuild for the two things that can go wrong beside a live training run:
# free RAM collapsing (the PROCESSES=6 OOM that killed the gen-1 test build) and featA dying.
# It only REPORTS -- it never kills anything. An auto-killer here would be one bad threshold
# away from taking the 7.75 h run it exists to protect.
LOG=scratchpad/features_rebuild/watch2.log
: > "$LOG"
low=999
for i in $(seq 1 600); do
  ram=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r')
  feta=$(powershell -NoProfile -Command "if (Get-Process -Id 3356 -ErrorAction SilentlyContinue) {'up'} else {'DOWN'}" 2>/dev/null | tr -d '\r')
  prog=$(grep -o "Generating Data: *[0-9]*%[^ ]* *[0-9]*/5000" scratchpad/features_rebuild/train2_*.log 2>/dev/null | tail -1)
  echo "$(date +%H:%M:%S) ram=${ram}GB featA=${feta} ${prog}" >> "$LOG"
  case "$feta" in DOWN) echo "ALERT featA DOWN" >> "$LOG"; exit 2;; esac
  if [ -f scratchpad/features_rebuild/rebuild2.log ] && grep -qB0 "^DONE_EXIT_" scratchpad/features_rebuild/rebuild2.log 2>/dev/null; then
    echo "REBUILD FINISHED" >> "$LOG"; exit 0
  fi
  sleep 60
done
