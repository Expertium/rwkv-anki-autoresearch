# Keep the rebuild at BelowNormal for its whole life.
#
# WHY: featA is the CRITICAL PATH (featB cannot start until it ends), and the rebuild has hours of
# slack. With both at Normal priority the rebuild starved featA's fetch workers -- GPU utilization
# fell to 4-18% and the step rate went 1.25 -> 0.77 steps/s, i.e. the rebuild was buying its own
# ~2 h by spending ~3 h of featA's. Lowering priority costs the slack task nothing that matters.
#
# It re-applies on a loop because PHASE 2 (the test db) spawns FRESH worker processes at Normal
# priority; a one-shot fix silently lapses the moment the train phase ends.
# It only ever sets priority. It never kills anything.
$log = "C:\Users\Andrew\rwkv-anki-autoresearch\scratchpad\features_rebuild\nice.log"
"$(Get-Date -Format HH:mm:ss) nice loop started" | Out-File -FilePath $log -Encoding utf8
for ($i = 0; $i -lt 480; $i++) {
    $t = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -like '*data_processing*' }
    $ids = @()
    foreach ($x in $t) { $ids += $x.ProcessId }
    if ($ids.Count -gt 0) {
        $kids = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                Where-Object { $ids -contains $_.ParentProcessId }
        foreach ($k in $kids) { $ids += $k.ProcessId }
    }
    $n = 0
    foreach ($id in $ids) {
        try {
            $pr = Get-Process -Id $id -ErrorAction Stop
            if ($pr.PriorityClass -ne 'BelowNormal') { $pr.PriorityClass = 'BelowNormal'; $n++ }
        } catch { }
    }
    if ($n -gt 0) { "$(Get-Date -Format HH:mm:ss) renice $n proc(s)" | Out-File -FilePath $log -Append -Encoding utf8 }
    if ($ids.Count -eq 0) { "$(Get-Date -Format HH:mm:ss) no rebuild processes -- exiting" | Out-File -FilePath $log -Append -Encoding utf8; break }
    Start-Sleep -Seconds 30
}
