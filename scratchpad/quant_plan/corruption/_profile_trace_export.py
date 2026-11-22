"""Profile one corrupt50 trace export, to find where the 30 ms/review goes.

The full-roster export is ~16 h at the measured 34 rev/s, which is too slow to be worth running
unattended. The per-review loop in export_rnn_trace.export_user is the same code the certified
parity traces come from, so before touching it we need to know which call owns the time: pandas
per-row work (iterrows / Series.copy / Series.drop), the RNN forwards, or the per-row feature
engineering inside process_row.

Usage: .venv/Scripts/python.exe scratchpad/quant_plan/corruption/_profile_trace_export.py <user>
"""
import cProfile
import io
import os
import pstats
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_corrupt50_traces as drv  # noqa: E402  (sets the arch env before the exporter imports)

ROOT = drv.ROOT
import export_rnn_trace as e  # noqa: E402

e.MODEL_PATH = str(drv.DEFAULT_CKPT)
e.OUT_DIR = ROOT / "scratchpad" / "quant_plan" / "trace_corrupt50_profile"
e.OUT_DIR.mkdir(parents=True, exist_ok=True)
e.LABEL_DB = os.environ["RWKV_EXPORT_LABEL_DB"]
e.LABEL_DB_SIZE = drv.LABEL_DB_SIZE

user = int(sys.argv[1]) if len(sys.argv) > 1 else 7325
pr = cProfile.Profile()
pr.enable()
e.export_user(user)
pr.disable()
st = pstats.Stats(pr)
st.sort_stats("tottime")
buf = io.StringIO()
st.stream = buf
st.print_stats(28)
print(buf.getvalue())
print("=== by cumtime ===")
st.sort_stats("cumtime")
buf = io.StringIO()
st.stream = buf
st.print_stats(28)
print(buf.getvalue())
