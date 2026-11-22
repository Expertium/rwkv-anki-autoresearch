"""Append GPU_RUN_BOARD.md amendment 16 (2026-11-22 11:35): the F: disk wedge, the watchdog, and
Andrew's stop instruction. Appends with LF endings, like the last line of the file."""
from pathlib import Path

BOARD = Path(__file__).resolve().parents[3] / "optimization" / "GPU_RUN_BOARD.md"

TEXT = """
Row 11 amendment 16 (2026-11-22 11:35, Moscow). **The 02:33-11:15 stall was an F: disk I/O error, and the corruption check now owns the GPU alone.**

- **Root cause of the 8.7 h gap - hardware, not code.** `it36shrink`'s `l3i4s4n4` QAT decay started 02:32:04 and its `train.log` stopped at 02:33:24. `corruption/_diag_stall.py` then measured `rwkv.train_rwkv` (pid 29396) and its fetch worker (26952) at **0.000 s user and 0.000 s system CPU over a 6 s window** while the process was `running` with 50 threads and rss 2.23 GB; `corruption/_diag_disk.py` found the System-log events `[disk] 153` ("The IO operation at logical block address 0x131488190 for Disk 1 (PDO name: \\Device\\00000060) was retried", 02:33:45) and `[disk] 51` ("An error was detected on device \\DR1 during a paging operation", 02:33:46). `Get-Partition` confirms F: = Disk 1 (2.49 TB free). So an LMDB read on F: wedged the training loop while every guard stayed green.
- **Fix 1 - the watchdog.** `it36ladder_lib.run` now logs a heartbeat, tracks log growth AND the total CPU of the child's whole process tree, and if neither changes for `STALL_MIN=20` minutes it kills the tree with `taskkill /F /T` and retries once (`STALL_RC=-9`, `STALL_RETRIES=1`, `POLL_S=30`). `it36corrupt_chain.py` and `it36corrupt2_chain.py` delegate to it.
- **Fix 2 - the report can see a wedge.** `status_report.py` prints `CPU over 5 s: <pid>=<seconds> ... -> CPU alive (n/m busy)`, with `WEDGE? live workload processes burned no CPU` for a stuck tree, `gone` for a process that exited during the sample, and "all sampled processes exited during the sample" when a chain changed point between the two samples. A healthy 4-process tree reads about 14 s of CPU per 5 s of wall time (~2.8 cores).
- **Andrew's stop instruction (11:21).** "Let's stop further runs for now and ONLY do corruption checks for (as of right now) passed runs." The queue was reordered so `it36corrupt` runs FIRST (it no longer waits for `it36shrink`), and at 11:24 the waiters `it36shrink`, `it36equalbits`, `it36combo` and `it36corrupt2` were killed with their lock files deleted. `QUEUE.md` sections 2 and 5 now mark them PAUSED and forbid a restart: their logs have no end marker, so the supervisor's restart rule would otherwise fire as soon as the GPU is idle and their logs are 45 min old.
- **Rows that pass as of now: 15, 16, 17 and 18 only.** All five overnight sweeps FAIL: 24 `l4i4s4` +0.000218/+0.000463, 25 `l7i4s4` +0.000209/+0.000466, 26 `n2i4s4` +0.000180/+0.000492, 27 `l2i8496` +0.000715/+0.000793, 28 `l2i8992` +0.000554/+0.000829.
- **Corruption point 1 (`l2i8s4`, row 17) PASSES** at 11:25:24: rc=0, 50 of 50 users, **158,986 raw state observations and 150 p(recall) tensors, 0 non-finite**, 0 NaN skips, 4.8 min on the `corrupt50` roster with histories NOT truncated, on the Python QAT path (`RWKV_QAT_FUSED=0`) where the raw state is visible before `_sanitize_state` caps it. Verdict in `corruption/corruption_corrupt_l2i8s4.json`; the RESULTS.md cell is written by `corruption/record_corruption.py`. Points left: `l2i8` (row 15), `l3i8` (16), `l3i8s4` (18), and the quantization-OFF control `ctl_fp32`.
"""

with BOARD.open("a", encoding="utf-8", newline="") as fh:
    fh.write(TEXT.lstrip("\n").replace("\n", "\n"))
print("appended", len(TEXT), "chars to", BOARD)
