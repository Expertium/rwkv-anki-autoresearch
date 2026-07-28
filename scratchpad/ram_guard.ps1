# Keep free RAM above a floor by trimming python working sets.
# Non-destructive: the bulk of the growth is CLEAN, file-backed LMDB mmap pages, so Windows
# drops them and re-faults on demand. Costs a brief slowdown; prevents the 56-63 GB hang band.
# Measured 2026-07-29 00:02 during iter 33's eval: free 4.7 GB -> 23.6 GB, 19 GB reclaimed.
param([double]$FloorGB = 14, [int]$IntervalSec = 60, [int]$Minutes = 600)
Add-Type -Namespace W -Name Mem -MemberDefinition '[DllImport("psapi.dll")] public static extern bool EmptyWorkingSet(IntPtr hProc);'
$end = (Get-Date).AddMinutes($Minutes)
while((Get-Date) -lt $end){
  $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
  if($free -lt $FloorGB){
    $n=0
    foreach($t in (Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.WorkingSetSize -gt 1GB })){
      try { $p=[System.Diagnostics.Process]::GetProcessById($t.ProcessId); $null=[W.Mem]::EmptyWorkingSet($p.Handle); $n++ } catch {}
    }
    Start-Sleep -Seconds 3
    $after = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    "{0}  free {1:N1} GB -> {2:N1} GB after trimming {3} procs" -f (Get-Date -Format 'HH:mm:ss'),$free,$after,$n
  }
  Start-Sleep -Seconds $IntervalSec
}
