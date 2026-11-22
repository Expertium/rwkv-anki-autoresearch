# Wait until the active chain's real GPU work processes are gone (the queue hold keeps the waiters
# away, so this only has to notice when the running chain stops).
$pat = 'qat_decay.py|quant_sweep.py|rwkv.train_rwkv|eval_sharded.py|get_result'
for ($i = 0; $i -lt 40; $i++) {
    $m = (Get-CimInstance Win32_Process |
          Where-Object { $_.Name -match 'python|rwkv-infer' -and $_.CommandLine -match $pat } |
          Measure-Object).Count
    Write-Output ((Get-Date -Format 'HH:mm:ss') + '  real gpu work = ' + $m)
    if ($m -eq 0) { Write-Output 'FREE'; break }
    Start-Sleep -Seconds 60
}
Get-Content "$PSScriptRoot\..\it36equalbits_chain.log" -Tail 4
