# Launch a command FULLY DETACHED from Claude's process tree so it survives Esc / session
# interruption. Claude tree-kills its background jobs on Esc; a process created via WMI
# Win32_Process.Create is parented to WmiPrvSE (a system service), NOT Claude, so it keeps running.
#
# Usage:  powershell -NoProfile -File scratchpad/detach.ps1 -Script <abs path to .cmd>
# Returns: detached_pid=<pid>  (poll it / its log via OS truth; there is NO tool-completion event).
param([Parameter(Mandatory = $true)][string]$Script)
$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c `"$Script`"" }
if ($res.ReturnValue -ne 0) { Write-Output "FAILED returnvalue=$($res.ReturnValue)"; exit 1 }
$pid_ = $res.ProcessId
$parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$pid_" -ErrorAction SilentlyContinue).ParentProcessId
$pname = (Get-Process -Id $parent -ErrorAction SilentlyContinue).ProcessName
Write-Output "detached_pid=$pid_  parent_pid=$parent ($pname)"
