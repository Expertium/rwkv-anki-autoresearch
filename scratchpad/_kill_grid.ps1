$targets = Get-CimInstance Win32_Process -Filter "name like 'python%' or name like 'rwkv-infer%'"
foreach ($p in $targets) {
    $cl = ""
    if ($p.CommandLine) { $cl = $p.CommandLine }
    $isGrid = ($cl -like '*rust_grid.py*')
    $isEngine = ($p.Name -eq 'rwkv-infer.exe')
    if ($isGrid -or $isEngine) {
        Write-Output ("kill " + $p.ProcessId + " " + $p.Name)
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 3
Write-Output "--- remaining"
$left = Get-CimInstance Win32_Process -Filter "name like 'python%' or name like 'rwkv-infer%'"
foreach ($p in $left) {
    $cl = ""
    if ($p.CommandLine) { $cl = $p.CommandLine }
    if (($cl -like '*rust_grid.py*') -or ($p.Name -eq 'rwkv-infer.exe')) {
        Write-Output ("STILL " + $p.ProcessId + " " + $p.Name)
    }
}
Write-Output "done"
