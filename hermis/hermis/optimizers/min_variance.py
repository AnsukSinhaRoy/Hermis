from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common import _clip_box_and_long_only


def min_variance_optimize(
    Sigma: pd.DataFrame,
    box: Optional[dict] = None,
    long_only: bool = True,
    solver: Optional[str] = None,
    **kwargs,
) -> Dict:
    """Minimum variance portfolio using SciPy SLSQP (no CVXPY dependency).

    Minimize: w^T Sigma w
    Subject to: sum(w)=1 and optional bounds.
    """
    from scipy.optimize import minimize

    assets = list(Sigma.index)
    n = len(assets)
    if n == 0:
        return {"weights": None, "status": "no_assets"}

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
        return float(w @ Sigma_np @ w)

    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
        if not res.success or res.x is None:
            raise RuntimeError(res.message if hasattr(res, 'message') else 'solver_failed')
        w_np = np.asarray(res.x, dtype=float)
        if long_only:
            w_np = np.clip(w_np, 0.0, None)
        s = float(w_np.sum())
        w_np = (w_np / s) if s > 0 else (np.ones(n) / n)
        return {"weights": pd.Series(w_np, index=assets), "status": "ok"}
    except Exception as e:
        w0 = np.ones(n) / n
        return {"weights": pd.Series(w0, index=assets), "status": f"solver_failed:{repr(e)}"}
