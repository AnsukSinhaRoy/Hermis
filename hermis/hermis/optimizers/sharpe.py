from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common import _clip_box_and_long_only


def sharpe_optimize(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    box: Optional[dict] = None,
    long_only: bool = True,
    solver: Optional[str] = None,
) -> Dict:
    """Maximize Sharpe ratio using SciPy SLSQP.

    Maximize: (mu^T w) / sqrt(w^T Sigma w)
    Subject to: sum(w)=1 and optional bounds.

    Note: Non-convex; this is a single-start local optimizer.
    """
    from scipy.optimize import minimize

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

    cons = ({'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)})

    x0 = np.ones(n) / n
    x0 = _clip_box_and_long_only(x0, box=box, long_only=long_only)
    x0 = np.asarray(x0, dtype=float)

    def obj(w):
        w = np.asarray(w, dtype=float)
        denom = float(w @ Sigma_np @ w)
        denom = max(denom, 1e-12)
        return float(-(mu_np @ w) / (denom ** 0.5))

    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
        if not res.success or res.x is None:
            raise RuntimeError(res.message if hasattr(res, 'message') else 'solver_failed')
        wv = np.asarray(res.x, dtype=float)
        if long_only:
            wv = np.clip(wv, 0.0, None)
        s = float(wv.sum())
        wv = (wv / s) if s > 0 else (np.ones(n) / n)
        port_mean = float(mu_np @ wv)
        port_var = float(wv @ Sigma_np @ wv)
        port_vol = float(port_var ** 0.5) if port_var > 0 else 1e-9
        sharpe = port_mean / port_vol if port_vol > 0 else 0.0
        return {"weights": pd.Series(wv, index=assets), "status": "ok", "sharpe": sharpe}
    except Exception as e:
        return {"weights": pd.Series(np.ones(n) / n, index=assets), "status": f"solver_failed:{repr(e)}"}
