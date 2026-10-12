import os as _os

# RWKV_CUDA_PYD=<path to a .pyd> loads an ALTERNATE build of the extension for this process and its
# children (2026-09-11: the b13 PQ-catalog-capacity build, scratchpad/pqcap). Unset -> the plain
# relative import below, which is the only path every existing run takes.
_ALT_PYD = _os.environ.get("RWKV_CUDA_PYD", "")
if _ALT_PYD:
    import importlib.machinery as _mach
    import importlib.util as _iu
    import sys as _sys

    import torch as _torch  # noqa: F401 -- the extension links against torch's DLLs

    if not _os.path.isfile(_ALT_PYD):
        raise ImportError(f"RWKV_CUDA_PYD points at a missing file: {_ALT_PYD}")
    _loader = _mach.ExtensionFileLoader("rwkv.model.RWKV_CUDA", _ALT_PYD)
    _spec = _iu.spec_from_file_location("rwkv.model.RWKV_CUDA", _ALT_PYD, loader=_loader)
    RWKV_CUDA = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(RWKV_CUDA)
    _sys.modules["rwkv.model.RWKV_CUDA"] = RWKV_CUDA
    print(f"[RWKV_CUDA] ALTERNATE build loaded: {_ALT_PYD}", flush=True)
else:
    try:
        from . import RWKV_CUDA  # type: ignore
    except ImportError:
        RWKV_CUDA = None
