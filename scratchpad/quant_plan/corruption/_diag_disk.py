"""Is the F: drive (the LMDB store) healthy right now? (2026-11-22 11:16)

Times a real read of both LMDB envs and dumps the recent System-log disk errors.
"""
import pathlib
import subprocess
import time

import lmdb

print("=== timed LMDB reads ===")
for name, path, size in (("train_db_5k_h1_adaptive92", "F:/rwkv_lmdb/train_db_5k_h1_adaptive92", 400_000_000_000),
                         ("test_db_5k_adaptive92", "F:/rwkv_lmdb/test_db_5k_adaptive92", 250_000_000_000)):
    t0 = time.time()
    env = lmdb.open(path, map_size=size, readonly=True, lock=False, readahead=False, max_readers=512)
    t_open = time.time() - t0
    t1 = time.time()
    with env.begin() as txn:
        n = 0
        total = 0
        cur = txn.cursor()
        for k, v in cur:
            n += 1
            total += len(v)
            if n >= 200:
                break
    t_read = time.time() - t1
    env.close()
    print(f"{name}: open {t_open:.2f} s, first 200 keys {t_read:.2f} s, {total/1e6:.1f} MB")

t0 = time.time()
free = pathlib.Path("F:/").stat()
print(f"F:/ stat {time.time()-t0:.2f} s")

print("\n=== free space on F: ===")
out = subprocess.run(["powershell", "-NoProfile", "-Command",
                      "Get-PSDrive F | Select-Object Used,Free | Format-List"], capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip())

print("\n=== System log, level 1-3, last 12 h, disk-related ===")
ps = ("Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddHours(-12); Level=1,2,3} "
      "-ErrorAction SilentlyContinue | Where-Object { $_.ProviderName -match 'disk|Ntfs|volmgr|storahci|"
      "stornvme|partmgr' } | Select-Object -First 20 | ForEach-Object { \"$($_.TimeCreated) "
      "[$($_.ProviderName)] $($_.Id): $($_.Message)\" }")
out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
print(out.stdout.strip()[:3000] or "(no disk events)")
