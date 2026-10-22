"""ctypes binding to the C++ filter (core/capi.cpp)."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_NAMES = ["libtrackfuse.so", "libtrackfuse.dylib", "trackfuse.dll"]


def _candidates() -> list[Path]:
    root = Path(__file__).resolve().parent.parent / "core" / "build"
    env = os.environ.get("TRACKFUSE_LIB")
    return ([Path(env)] if env else []) + [root / n for n in _NAMES]


def _load() -> ctypes.CDLL | None:
    for path in _candidates():
        if path.exists():
            lib = ctypes.CDLL(str(path))
            dp = ctypes.POINTER(ctypes.c_double)
            ip = ctypes.POINTER(ctypes.c_int32)
            lib.tf_run.restype = ctypes.c_long
            lib.tf_run.argtypes = [
                ctypes.c_int, ip, dp, dp, dp,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, dp, dp,
            ]  # fmt: skip
            return lib
    return None


_LIB = _load()


def available() -> bool:
    return _LIB is not None


def run(
    types: np.ndarray,
    vals: np.ndarray,
    x0: np.ndarray,
    p0_diag: np.ndarray,
    sigma_a: float,
    sigma_w: float,
    gate_chi2: float = 0.0,
    want_cov: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if _LIB is None:
        raise RuntimeError("C++ core not built; run `make -C core build/libtrackfuse.so`")
    n = len(types)
    t = np.ascontiguousarray(types, dtype=np.int32)
    v = np.ascontiguousarray(vals, dtype=np.float64)
    x = np.ascontiguousarray(x0, dtype=np.float64)
    p = np.ascontiguousarray(p0_diag, dtype=np.float64)
    out_x = np.empty((n, 4), dtype=np.float64)
    out_p = np.empty((n, 4, 4), dtype=np.float64) if want_cov else None
    dp = ctypes.POINTER(ctypes.c_double)
    rejected = _LIB.tf_run(
        n,
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        v.ctypes.data_as(dp),
        x.ctypes.data_as(dp),
        p.ctypes.data_as(dp),
        sigma_a,
        sigma_w,
        gate_chi2,
        out_x.ctypes.data_as(dp),
        out_p.ctypes.data_as(dp) if out_p is not None else None,
    )
    if rejected < 0:
        raise ValueError("tf_run rejected its arguments")
    return out_x, out_p, int(rejected)
