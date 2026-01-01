from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .common import _clip_box_and_long_only, _normalize_box


def mv_optimize(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    target_return: Optional[float] = None,
    box: Optional[object] = None,
    long_only: bool = True,
    **kwargs,
) -> Dict:
    """Compatibility wrapper for older code that calls mv_optimize(...).

    Forwards to `mv_reg_optimize` using a default lambda sweep.

    Notes:
    - `target_return` is currently ignored (this wrapper optimizes regularized MV).
    - `box` accepted as dict or tuple.
    """
    if expected_returns is None or cov is None:
        return {"weights": None, "status": "invalid_inputs"}

    box_norm = _normalize_box(box)
    lambdas = kwargs.get("lambdas", [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0])
    return mv_reg_optimize(expected_returns, cov, lambdas=lambdas, box=box_norm, long_only=long_only)


def mv_reg_optimize(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    lambdas: Optional[List[float]] = None,
    lambda_reg: Optional[float] = None,
    box: Optional[dict] = None,
    long_only: bool = True,
    solver: Optional[str] = None,
    **kwargs,
) -> Dict:
    """Regularized mean-variance optimizer without requiring CVXPY.

    Objective (per lambda):
        maximize   mu^T w - lambda * w^T Sigma w
        subject to sum(w)=1 and optional bounds.

    Implementation notes:
      - Uses SciPy SLSQP (works in environments where cvxpy is not installed).
      - If `lambdas` is provided, it performs a sweep and selects the best in-sample Sharpe.
      - If only `lambda_reg` is provided, it is treated as a single-element sweep.
    """
    from scipy.optimize import minimize

    if lambdas is None and lambda_reg is not None:
        lambdas = [float(lambda_reg)]
    if lambdas is None:
        lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]

    if lambdas is not None and not isinstance(lambdas, (list, tuple)):
        lambdas = [float(lambdas)]

    assets = list(mu.index)
    n = len(assets)
    if n == 0:
        return {"weights": None, "status": "no_assets"}

    mu_np = mu.reindex(assets).fillna(0.0).values.astype(float)
    Sigma_np = Sigma.reindex(index=assets, columns=assets).fillna(0.0).values.astype(float)

    bounds = [(None, None)] * n
    if long_only:
        bounds = [(0.0, 1.0)] * n
    if box is not None:
        low = box.get('min', None)
        high = box.get('max', None)
        bounds = [(low if low is not None else b0, high if high is not None else b1) for (b0, b1) in bounds]

    cons = ({'type': 'eq', 'fun': lambda w: float(w.sum() - 1.0)})

    x0 = np.ones(n) / n
    x0 = _clip_box_and_long_only(x0, box=box, long_only=long_only)
    x0 = np.asarray(x0, dtype=float)

    best_res = None
    best_sharpe = -np.inf

    for lam in lambdas:
        lam = float(lam)

        def obj(w):
            w = np.asarray(w, dtype=float)
            return float(lam * (w @ Sigma_np @ w) - (mu_np @ w))

        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
            if not res.success or res.x is None:
                continue
            w_np = np.asarray(res.x, dtype=float)
            if long_only:
                w_np = np.clip(w_np, 0.0, None)
            s = float(w_np.sum())
            if s <= 0:
                continue
            w_np /= s

            port_mean = float(mu_np @ w_np)
            port_var = float(w_np @ Sigma_np @ w_np)
            port_vol = np.sqrt(port_var) if port_var > 0 else 1e-9
            sharpe = port_mean / port_vol if port_vol > 0 else -np.inf

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_res = {
                    "weights": pd.Series(w_np, index=assets),
                    "status": "ok",
                    "lambda": lam,
                    "sharpe_insample": float(sharpe),
                }
        except Exception:
            continue

    if best_res is None:
        w0 = pd.Series(np.ones(n) / n, index=assets)
        if box or long_only:
            w0 = pd.Series(_clip_box_and_long_only(w0.values, box, long_only), index=assets)
        return {"weights": w0, "status": "fallback_equal", "lambda": None, "sharpe_insample": None}

    return best_res
