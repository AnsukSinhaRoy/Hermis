"""Small, self-contained optimizers used by Prism UI.

These functions are convenient and work directly on a `returns` DataFrame.
They are kept here so the Prism UI doesn't have to depend on the full
simulation pipeline.

They were previously embedded in `portfolio_sim.optimizer`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _to_mu_sigma(returns: pd.DataFrame):
    """Convert input returns (Series/DataFrame/ndarray) to mu (n,) and Sigma (n,n)."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    if isinstance(returns, np.ndarray):
        arr = returns
        if arr.ndim == 1:
            arr = arr[:, None]
        mu = np.nanmean(arr, axis=0)
        sigma = np.cov(arr, rowvar=False, ddof=1)
        return mu, sigma
    if isinstance(returns, pd.DataFrame):
        df = returns.dropna(how='all')
        mu = df.mean(axis=0).values
        sigma = df.cov(ddof=1).values
        return mu, sigma
    raise ValueError("Unsupported returns input type")


def global_minimum_variance(returns: pd.DataFrame, allow_short: bool = True) -> pd.Series:
    return min_variance_weights(returns, target_return=None, allow_short=allow_short)


def min_variance_weights(returns, target_return: Optional[float] = None, allow_short: bool = True):
    mu, sigma = _to_mu_sigma(returns)
    n = len(mu)

    # regularize
    eps = 1e-8
    sigma = sigma + np.eye(n) * eps

    inv_sigma = np.linalg.inv(sigma)

    ones = np.ones(n)
    A = ones @ inv_sigma @ ones
    B = ones @ inv_sigma @ mu
    C = mu @ inv_sigma @ mu

    if target_return is None and allow_short:
        w = (inv_sigma @ ones) / A
    elif target_return is not None and allow_short:
        M = np.array([[B, A], [C, B]])
        rhs = np.array([1.0, float(target_return)])
        alpha, beta = np.linalg.solve(M, rhs)
        w = alpha * (inv_sigma @ mu) + beta * (inv_sigma @ ones)
    else:
        try:
            from scipy.optimize import minimize

            def port_var(x):
                return float(x @ sigma @ x)

            constraints = [{'type': 'eq', 'fun': lambda x: float(np.sum(x) - 1.0)}]
            if target_return is not None:
                constraints.append({'type': 'eq', 'fun': lambda x: float(mu @ x - target_return)})

            bounds = [(0.0, 1.0) for _ in range(n)]
            x0 = np.ones(n) / n
            res = minimize(port_var, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if not res.success:
                w = np.clip(x0, 0, 1)
                w = w / w.sum()
            else:
                w = res.x
        except Exception:
            try:
                if target_return is None:
                    w_un = (inv_sigma @ ones) / A
                else:
                    M = np.array([[B, A], [C, B]])
                    rhs = np.array([1.0, float(target_return)])
                    alpha, beta = np.linalg.solve(M, rhs)
                    w_un = alpha * (inv_sigma @ mu) + beta * (inv_sigma @ ones)
                w = np.clip(w_un, 0.0, None)
                s = w.sum()
                w = np.ones(n) / n if s <= 0 else w / s
            except Exception:
                w = np.ones(n) / n

    if isinstance(returns, pd.DataFrame):
        return pd.Series(w, index=returns.columns)
    return pd.Series(w)


def risk_parity_weights(returns, allow_short: bool = False, maxiter: int = 1000, tol: float = 1e-8):
    """Estimate equal risk contribution weights (simple iterative algorithm)."""
    _, sigma = _to_mu_sigma(returns)
    n = sigma.shape[0]
    x = np.ones(n) / n

    for _ in range(int(maxiter)):
        sigma_x = sigma @ x
        rc = x * sigma_x
        total_rc = rc.sum()
        target = total_rc / n
        x_new = x * np.sqrt(target / (rc + 1e-12))
        x_new = np.clip(x_new, 1e-12, None)
        x_new = x_new / x_new.sum()
        if np.linalg.norm(x_new - x) < float(tol):
            x = x_new
            break
        x = x_new

    if isinstance(returns, pd.DataFrame):
        return pd.Series(x, index=returns.columns)
    return pd.Series(x)
