# Stop featA2 cleanly AFTER its decay phase, so the PC is free for Andrew's FSRS-7 benchmark.
# Andrew 2026-08-21: "After decay is done, let's pause. I want to free the PC to finish the FSRS-7
# benchmark".
#
# WHY A WATCHER AND NOT AN EDIT. featA2's runner is one cmd.exe doing WS -> decay -> eval, and it
# starts the eval the instant decay returns. The runner CANNOT be edited while it runs: cmd.exe
# re-reads a batch file from a saved byte offset, so any edit that shifts bytes makes it resume
# mid-garbage (the trap that cost iters 43 and 46). So the only safe stop is external.
#
# WHERE IT STOPS, and why that point is safe. It waits for `featA2 DECAY_OK` in featA2.log. The
# runner writes that line only AFTER the decay phase exits 0 AND its artifact gate confirms the
# final checkpoint exists -- so by the time this fires, featA2_d_10935.pth is complete on disk.
# Killing during the eval loses nothing: the eval writes per-user rows and can simply be re-run.
#
# It kills the whole featA2 tree (its waiter cmd, its python processes and their children) so no
# half-dead cmd.exe writes a spurious terminal marker that a future waiter could latch onto.
# It touches NOTHING else -- not Andrew's FSRS benchmark, not the Telegram bridge, not the Reddit
# bot. Targets are resolved by command line each time rather than by remembered pids.
$repo = "C:\Users\Andrew\rwkv-anki-autoresearch"
$log  = "$repo\scratchpad\features_ab\stop_after_decay.log"
$fa   = "$repo\scratchpad\features_ab\featA2\featA2.log"
"$(Get-Date -Format 'HH:mm:ss') armed -- waiting for featA2 DECAY_OK" | Out-File -FilePath $log -Encoding utf8

for ($i = 0; $i -lt 1440; $i++) {
    if (Test-Path $fa) {
        $txt = Get-Content $fa -Raw -ErrorAction SilentlyContinue
        if ($txt -and $txt -match 'DECAY_OK') {
            "$(Get-Date -Format 'HH:mm:ss') DECAY_OK seen" | Out-File -FilePath $log -Append -Encoding utf8

            $ckpt = "$repo\scratchpad\features_ab\featA2\featA2_d_10935.pth"
            if (Test-Path $ckpt) {
                $sz = [math]::Round((Get-Item $ckpt).Length / 1MB, 2)
                "$(Get-Date -Format 'HH:mm:ss') decay checkpoint present, $sz MB" | Out-File -FilePath $log -Append -Encoding utf8
            } else {
                "$(Get-Date -Format 'HH:mm:ss') WARNING decay checkpoint NOT found -- killing anyway, but check before resuming" | Out-File -FilePath $log -Append -Encoding utf8
            }

            # Resolve the tree fresh. Never kill by a pid remembered from arming time.
            $targets = @()
            $cmds = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
                    Where-Object { $_.CommandLine -like '*featA2*' }
            foreach ($c in $cmds) { $targets += $c.ProcessId }
            $pys = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                   Where-Object { $_.CommandLine -like '*featA2*' }
            foreach ($p in $pys) {
                $targets += $p.ProcessId
                $kids = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                        Where-Object { $_.ParentProcessId -eq $p.ProcessId }
                foreach ($k in $kids) { $targets += $k.ProcessId }
            }
            $targets = $targets | Sort-Object -Unique
            foreach ($id in $targets) {
                try {
                    $pr = Get-Process -Id $id -ErrorAction Stop
                    Stop-Process -Id $id -Force
                    "$(Get-Date -Format 'HH:mm:ss') killed $id ($($pr.ProcessName))" | Out-File -FilePath $log -Append -Encoding utf8
                } catch { }
            }
            Start-Sleep -Seconds 10
            $left = (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*featA2*' } | Measure-Object).Count
            "$(Get-Date -Format 'HH:mm:ss') PAUSED. featA2 processes remaining: $left" | Out-File -FilePath $log -Append -Encoding utf8
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader |
                Out-File -FilePath $log -Append -Encoding utf8
            break
        }
    }
    Start-Sleep -Seconds 30
}
