import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "qs", pathlib.Path("scratchpad/quant_plan/quant_sweep.py"))
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)

print("candidates:", len(qs.CANDIDATES))
for name in ("l1i8", "l2i8", "l3i8", "l2i8s4", "l3i8s4", "l1i8s4", "cn8"):
    env = qs.CANDIDATES[name][0]
    print(f"  {name:8s} bits={qs.bits_for(env)}")
print("compile ok")
