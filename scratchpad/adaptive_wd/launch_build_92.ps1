$ErrorActionPreference = 'Stop'
$root = 'C:\Users\Andrew\rwkv-anki-autoresearch'
$work = Join-Path $root 'scratchpad\adaptive_wd'
$log = Join-Path $work 'build_92_launcher.log'
Set-Location $root
$proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', 'scratchpad\adaptive_wd\build_92.cmd' -WorkingDirectory $root -WindowStyle Hidden -PassThru
"BUILD_92_PID=$($proc.Id)" | Set-Content (Join-Path $work 'build_92.pid')
"Started adaptive 92-input LMDB builds at $(Get-Date -Format o), pid $($proc.Id)" | Set-Content $log
