from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _normalize_box(box):
    """Accept box as dict {'min':..., 'max':...} or tuple (min, max) or None."""
    if box is None:
        return None
    if isinstance(box, dict):
        return {"min": box.get("min", None), "max": box.get("max", None)}
    if isinstance(box, (list, tuple)) and len(box) >= 2:
        return {"min": box[0], "max": box[1]}
    return None


def _clip_box_and_long_only(w, box: Optional[dict], long_only: bool):
    """Apply long-only / box constraints by clipping and renormalizing.

    Accepts numpy array-like OR pd.Series. Returns same kind.
    """
    w_is_series = isinstance(w, pd.Series)
    w_arr = np.asarray(w, dtype=float).copy()

    low, high = None, None
    if box is not None:
        low, high = box.get("min", None), box.get("max", None)

    if long_only:
        low = 0.0 if low is None else max(0.0, float(low))

    if low is not None:
        w_arr = np.maximum(w_arr, float(low))
    if high is not None:
        w_arr = np.minimum(w_arr, float(high))

    s = float(np.nansum(w_arr))
    if s > 0:
        w_arr = w_arr / s

    if w_is_series:
        return pd.Series(w_arr, index=w.index)
    return w_arr


def project_to_k_set(w, k: int):
    """Project weights onto a top-k simplex (keep only k largest, renormalize)."""
    w_arr = np.asarray(w, dtype=float).copy()
    n = w_arr.shape[0]
    if k is None or int(k) <= 0 or int(k) >= n:
        s = float(np.sum(w_arr))
        return w_arr / s if s > 0 else np.ones(n) / n

    k = int(k)
    out = np.zeros_like(w_arr)
    top_k = np.argsort(w_arr)[-k:]
    out[top_k] = w_arr[top_k]
    s = float(out.sum())
    if s > 0:
        out /= s
    else:
        out[top_k] = 1.0 / k
    return out
