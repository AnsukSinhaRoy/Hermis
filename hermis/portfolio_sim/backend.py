# portfolio_sim/backend.py
"""
Backend abstraction: returns xp (numpy/cupy equivalent) and helpers.
If use_gpu=True and cupy is available, xp is cupy; otherwise xp is numpy.

Functions:
- get_backend(use_gpu: bool) -> (xp, info_dict)
- to_numpy(xp_array) -> numpy array (no-op for numpy)
"""
from typing import Tuple, Dict
import numpy as _np

def get_backend(use_gpu: bool = False) -> Tuple[object, Dict]:
    """
    Returns (xp, info) where xp is the array module (numpy or cupy)
    and info contains backend name and availability.
    """
    info = {"use_gpu": False, "backend": "numpy"}
    if use_gpu:
        try:
            import cupy as cp  # type: ignore
            info["use_gpu"] = True
            info["backend"] = "cupy"
            return cp, info
        except Exception:
            # cupy not available; fallback to numpy
            info["use_gpu"] = False
            info["backend"] = "numpy"
            return _np, info
    else:
        return _np, info

def to_numpy(arr):
    """Convert xp array to numpy if needed."""
    try:
        # cupy arrays have .get() or cupy.asnumpy
        import cupy as cp  # type: ignore
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    # otherwise assume numpy
    return _np.asarray(arr)
