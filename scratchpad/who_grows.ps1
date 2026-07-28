# Which process is actually growing? Sample working sets twice, 15 min apart, and report deltas.
param([int]$WaitSec = 900)
function snap {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -or $_.Name -like 'chrome*' -or $_.Name -like 'msedge*' } |
    ForEach-Object { [pscustomobject]@{ pid=$_.ProcessId; name=$_.Name; ws=$_.WorkingSetSize
      cmd=$_.CommandLine.Substring(0,[Math]::Min(70,$_.CommandLine.Length)) } }
}
$a = snap
Start-Sleep -Seconds $WaitSec
$b = snap
$map = @{}; foreach($p in $a){ $map[$p.pid] = $p.ws }
"{0,-7} {1,10} {2,10} {3,9}  {4}" -f 'pid','start_MB','end_MB','delta_MB','cmd'
$b | ForEach-Object {
  if($map.ContainsKey($_.pid)) {
    $d = ($_.ws - $map[$_.pid])/1MB
    [pscustomobject]@{ pid=$_.pid; s=[math]::Round($map[$_.pid]/1MB,0); e=[math]::Round($_.ws/1MB,0); d=[math]::Round($d,0); cmd=$_.cmd }
  }
} | Sort-Object d -Descending | Select-Object -First 12 | ForEach-Object {
  "{0,-7} {1,10} {2,10} {3,9}  {4}" -f $_.pid,$_.s,$_.e,$_.d,$_.cmd
}
