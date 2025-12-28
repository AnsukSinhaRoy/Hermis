from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .common import _clip_box_and_long_only, project_to_k_set


def _price_relatives_from_prices_window(prices_window: pd.DataFrame) -> Optional[np.ndarray]:
    """Compute last-step price relatives r_t = p_t / p_{t-1} from a price window."""
    if prices_window is None or not isinstance(prices_window, pd.DataFrame) or len(prices_window) < 2:
        return None

    p_t = prices_window.iloc[-1].astype(float)
    p_prev = prices_window.iloc[-2].astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        r = p_t / p_prev

    r = r.replace([np.inf, -np.inf], 1.0)
    r = r.fillna(1.0)
    r = r.clip(lower=1e-12)
    return r.values.astype(float)


def omd_step(
    w_prev: np.ndarray,
    r_t: np.ndarray,
    eta: float,
    v_target: float = 1.01,
    k_cardinality: Optional[int] = None,
    box: Optional[dict] = None,
    long_only: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Online Mirror Descent (Exponentiated Gradient) step using asymmetric shortfall loss."""
    w_prev = np.asarray(w_prev, dtype=float)
    r_t = np.asarray(r_t, dtype=float)

    daily = float(np.dot(w_prev, r_t))
    shortfall = float(v_target - daily)
    loss = float(max(0.0, shortfall) ** 2)

    grad = (-2.0 * shortfall * r_t) if shortfall > 0 else np.zeros_like(r_t)

    w_new = w_prev * np.exp(-float(eta) * grad)
    s = float(w_new.sum())
    w_new = (w_new / s) if s > 0 else np.ones_like(w_new) / len(w_new)

    if k_cardinality is not None and int(k_cardinality) > 0:
        w_new = project_to_k_set(w_new, int(k_cardinality))

    w_new = _clip_box_and_long_only(w_new, box=box, long_only=long_only)

    if k_cardinality is not None and int(k_cardinality) > 0:
        w_new = project_to_k_set(w_new, int(k_cardinality))

    info = {"daily_return": daily, "loss": loss, "shortfall": shortfall}
    return w_new, info


def ftrl_step(
    w_prev: np.ndarray,
    r_t: np.ndarray,
    B_prev: np.ndarray,
    v_prev: np.ndarray,
    lambda_2: float,
    gamma: float,
    v_target: float = 1.01,
    k_cardinality: Optional[int] = None,
    box: Optional[dict] = None,
    long_only: bool = True,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], str]:
    """FTRL with entropy regularization + forgetting (SLSQP solve each step)."""
    from scipy.optimize import minimize

    w_prev = np.asarray(w_prev, dtype=float)
    r_t = np.asarray(r_t, dtype=float)

    B_t = (float(gamma) * B_prev) + np.outer(r_t, r_t)
    v_t = (float(gamma) * v_prev) + r_t

    daily = float(np.dot(w_prev, r_t))
    shortfall = float(v_target - daily)
    loss = float(max(0.0, shortfall) ** 2)

    n = len(w_prev)

    def obj(w):
        w = np.asarray(w, dtype=float)
        w_safe = np.maximum(w, 1e-12)
        quad = float(w @ B_t @ w)
        lin = float(-2.0 * float(v_target) * (v_t @ w))
        reg = float(lambda_2) * float(np.sum(w_safe * np.log(w_safe)))
        return quad + lin + reg

    cons = ({'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)})

    bounds = [(None, None)] * n
    if long_only:
        bounds = [(0.0, 1.0)] * n
    if box is not None:
        low = box.get('min', None)
        high = box.get('max', None)
        bounds = [(low if low is not None else b0, high if high is not None else b1) for (b0, b1) in bounds]

    x0 = w_prev.copy()
    x0 = np.clip(x0, 1e-12, None)
    x0 = x0 / x0.sum()

    status = "ok"
    try:
        res = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': int(max_iter), 'ftol': float(tol)},
        )
        if res.success and res.x is not None:
            w_dense = np.asarray(res.x, dtype=float)
        else:
            w_dense = x0
            status = "solver_failed"
    except Exception:
        w_dense = x0
        status = "solver_exception"

    w_new = w_dense

    if k_cardinality is not None and int(k_cardinality) > 0:
        w_new = project_to_k_set(w_new, int(k_cardinality))

    w_new = _clip_box_and_long_only(w_new, box=box, long_only=long_only)

    if k_cardinality is not None and int(k_cardinality) > 0:
        w_new = project_to_k_set(w_new, int(k_cardinality))

    info = {"daily_return": daily, "loss": loss, "shortfall": shortfall}
    return np.asarray(w_new, dtype=float), B_t, v_t, info, status
