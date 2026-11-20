# detach.ps1 - start a long command in a hidden window that survives this console.
#
# AGENTS.md tells long runs to use this file. It was missing from the checkout on 2026-11-15 and is
# recreated here. The child is a plain hidden cmd.exe window, so closing or resetting the launching
# terminal does NOT stop it. Always give the command its own log redirection.
#
# Usage:
#   powershell -NoProfile -File scratchpad/detach.ps1 -Command ".venv\Scripts\python.exe -u scratchpad\ta3\ta3_train.py --run-dir scratchpad\ta3_fullrun_20261115 > scratchpad\ta3\chain.log 2>&1"
#
# Check on it with the log file and Get-Process; there is no console window to watch.
param(
    [Parameter(Mandatory = $true)][string]$Command,
    [string]$WorkDir = "C:\Users\Andrew\rwkv-anki-autoresearch"
)
# Remember how to find the child later.
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$record = Join-Path $WorkDir "scratchpad\detach_launches.log"
Add-Content -Path $record -Value "$stamp :: $Command"
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "cd /d $WorkDir && $Command" `
    -WorkingDirectory $WorkDir -WindowStyle Hidden
Write-Output "detached: $Command"
