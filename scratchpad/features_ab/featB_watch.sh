#!/bin/bash
# featB watcher. Exits when the run finishes, dies, or stalls -- which re-invokes Claude.
#
# ★ TWO WITNESSES, ALWAYS. A chain is WS -> decay -> eval and every phase is a NEW process, so
# "the pid I saw is gone" means a normal phase transition far more often than it means death.
# That exact mistake produced a false "featA DOWN" on 2026-08-20, from a watcher written the
# same day by someone who knew the rule. So:
#     process gone AND a terminal marker  -> SUCCESS (a finished chain writes its marker)
#     process gone AND no marker          -> DEATH
#     process alive AND no output growth  -> STALL (the featB failure mode: a fetch worker dies,
#                                            the trainer deadlocks, GPU sits at 0%)
# The runner is identified by COMMAND LINE, never by pid, because the cmd.exe wrapper spans all
# three phases while every python under it is replaced.
#
# The stall witness is the one that matters here. featB deadlocked for 69 min on 2026-08-21 and
# for ~60 min on 2026-09-01 with nothing in any log to say so; only the absence of forward
# progress showed it.

cd /c/Users/Andrew/rwkv-anki-autoresearch || exit 1
DIR=scratchpad/features_ab/featB
LOG=$DIR/featB.log
STALL_LIMIT=2700          # 45 min without any file in the run dir being written
POLL=300

# ⚠ THE WITNESS MUST COVER EVERY PHASE, AND ORIGINALLY IT DID NOT. This watched only $DIR, which
# WS and decay write to -- but the EVAL phase writes to scratchpad/eval_shards/ and result/, so it
# reported "STALLED, nothing written for 45 min" at 03:29 on 2026-09-02 while the eval was
# advancing normally through a memory-bound giant user (the shard log grew two minutes later and
# power went 18 W -> 90 W).
#
# That is the single-witness failure this file's own header warns about, committed in the file
# that warns about it: the alert measured the health of the place it happened to be looking, not
# the health of the run. A chain that changes which files it writes needs a witness that follows
# it -- so take the newest mtime across ALL the places featB writes, in any phase.
newest_mtime() {
  find "$DIR" scratchpad/eval_shards result -type f \
       \( -name '*featB*' -o -path '*/featB/*' -o -name 'shard_*' \) \
       -printf '%T@\n' 2>/dev/null | sort -rn | head -1
}
runner_alive() {
  powershell.exe -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | Where-Object { \$_.CommandLine -like '*run_featB*' }).Count" \
    2>/dev/null | tr -d '\r\n '
}

last_change=$(date +%s)
prev=$(newest_mtime)

while true; do
  sleep $POLL

  # `grep -c` PRINTS 0 and EXITS 1 when nothing matches, so `|| echo 0` appended a second line
  # and every poll then died on `[: 0\n0: integer expected`. The logic still fell through
  # correctly, which is exactly why it went unnoticed -- a monitor that errors on its own normal
  # path is one whose silence means nothing.
  marker=$(grep -c '^DONE_EXIT_' "$LOG" 2>/dev/null)
  case "$marker" in ''|*[!0-9]*) marker=0 ;; esac
  alive=$(runner_alive)
  case "$alive" in ''|*[!0-9]*) alive=0 ;; esac
  cur=$(newest_mtime)
  now=$(date +%s)
  [ "$cur" != "$prev" ] && { prev=$cur; last_change=$now; }
  idle=$(( now - last_change ))

  if [ "$marker" -gt 0 ]; then
    echo "FEATB FINISHED -- terminal marker present"
    tail -12 "$LOG"
    exit 0
  fi

  if [ "$alive" = "0" ]; then
    echo "FEATB DIED -- runner process gone and NO terminal marker in $LOG"
    echo "last lines of featB.log:"; tail -8 "$LOG"
    echo "newest phase log:"; ls -t "$DIR"/*.log 2>/dev/null | head -1 | xargs -r tail -25
    exit 1
  fi

  if [ "$idle" -gt "$STALL_LIMIT" ]; then
    echo "FEATB STALLED -- runner alive but nothing in $DIR written for $((idle/60)) min"
    echo "This is the deadlock signature (fetch worker died, trainer waiting). Check GPU util."
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
    echo "newest phase log:"; ls -t "$DIR"/*.log 2>/dev/null | head -1 | xargs -r tail -25
    exit 2
  fi
done
