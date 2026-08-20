#!/bin/bash
# Watch the gen-2 rebuild and the featA chain. REPORTS ONLY -- never kills anything.
#
# ⚠ THE FIRST VERSION OF THIS FILE FIRED A FALSE ALARM AT 20:38 AND THE CAUSE IS INSTRUCTIVE.
# It watched a HARDCODED PID (featA's WS process). featA's runner is a CHAIN -- WS, then decay,
# then eval -- and each phase is a NEW process. So a perfectly normal phase transition, two
# minutes after `featA WS_OK`, read as "featA DOWN".
#
# That is the flight-recorder-at-midnight failure repeated: an alert built on a SINGLE witness
# reports the WITNESS's health, not the system's. Two fixes, both from that rule:
#   1. identify the runner by COMMAND LINE (the cmd.exe wrapper spans every phase), not by a pid
#      that is only valid for one of them;
#   2. require TWO witnesses to fail before alarming -- the runner process gone AND no terminal
#      marker in its log. A finished chain writes the marker, so "gone + marker" is success, and
#      only "gone + no marker" is a death.
LOG=scratchpad/features_rebuild/watch2.log
FEATALOG=scratchpad/features_ab/featA/featA.log
: > "$LOG"
for i in $(seq 1 700); do
  ram=$(powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" 2>/dev/null | tr -d '\r')
  alive=$(powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*run_featA*' }) {'up'} else {'gone'}" 2>/dev/null | tr -d '\r')
  marker=no
  grep -qE "^DONE_EXIT_" "$FEATALOG" 2>/dev/null && marker=yes
  phase=$(grep -oE "featA (WS|DECAY|EVAL)_OK" "$FEATALOG" 2>/dev/null | tail -1)
  prog=$(grep -o "Generating Data: *[0-9]*%[^ ]* *[0-9]*/5000" scratchpad/features_rebuild/test2_*.log 2>/dev/null | tail -1)
  echo "$(date +%H:%M:%S) ram=${ram}GB featA=${alive}/${marker} ${phase} ${prog}" >> "$LOG"

  # TWO witnesses must agree before this is a death.
  if [ "$alive" = "gone" ] && [ "$marker" = "no" ]; then
    echo "ALERT featA runner gone with NO terminal marker -- real death" >> "$LOG"
    echo "ALERT featA runner gone with NO terminal marker at $(date +%H:%M:%S)"
    exit 2
  fi
  if [ "$alive" = "gone" ] && [ "$marker" = "yes" ]; then
    echo "featA chain COMPLETE (marker present) -- watcher standing down" >> "$LOG"
    exit 0
  fi
  sleep 60
done
