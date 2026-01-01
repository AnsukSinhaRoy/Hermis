# portfolio_sim/optimizer.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
try:
    import cvxpy as cp  # optional
except Exception:  # pragma: no cover
    cp = None
from numpy.linalg import solve, norm




# ---------------------------
# Helpers
# ---------------------------
# --- Compatibility wrapper for existing callers that expect mv_optimize ---
def _normalize_box(box):
    """
    Accept box as dict {'min':..., 'max':...} or tuple (min, max) or None.
    Return dict or None.
    """
    if box is None:
        return None
    if isinstance(box, dict):
        return {"min": box.get("min", None), "max": box.get("max", None)}
    if isinstance(box, (list, tuple)) and len(box) >= 2:
        return {"min": box[0], "max": box[1]}
    return None

def mv_optimize(expected_returns: pd.Series,
                cov: pd.DataFrame,
                target_return: Optional[float] = None,
                box: Optional[object] = None,
                long_only: bool = True,
                **kwargs) -> Dict:
    """
    Compatibility wrapper for older code that calls mv_optimize(...).
    Forwards to mv_reg_optimize using a default lambda sweep.
    - target_return is currently ignored (this wrapper optimizes regularized MV).
    - box accepted as dict or tuple.
    """
    # normalize inputs
    if expected_returns is None or cov is None:
        return {"weights": None, "status": "invalid_inputs"}
    box_norm = _normalize_box(box)
    lambdas = kwargs.get("lambdas", [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0])
    return mv_reg_optimize(expected_returns, cov, lambdas=lambdas, box=box_norm, long_only=long_only)

def _clip_box_and_long_only(w, box: Optional[dict], long_only: bool):
    """Apply long-only / box constraints by clipping and renormalizing.

    Accepts numpy array-like OR pd.Series. Returns the same *kind* (Series in, Series out;
    ndarray-like in, ndarray out).
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


# ---------------------------
# Online portfolio optimizers (stateful)
# ---------------------------

def project_to_k_set(w, k: int):
    """Project a weight vector onto a top-k simplex (keep only k largest entries, renormalize)."""
    w_arr = np.asarray(w, dtype=float).copy()
    n = w_arr.shape[0]
    if k is None or int(k) <= 0 or int(k) >= n:
        # nothing to do
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


def _price_relatives_from_prices_window(prices_window: pd.DataFrame) -> Optional[np.ndarray]:
    """Compute last-step price relatives r_t = p_t / p_{t-1} from a price window."""
    if prices_window is None or not isinstance(prices_window, pd.DataFrame) or len(prices_window) < 2:
        return None

    p_t = prices_window.iloc[-1].astype(float)
    p_prev = prices_window.iloc[-2].astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        r = p_t / p_prev

    # Clean anomalies: keep strictly positive
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

    # multiplicative update
    w_new = w_prev * np.exp(-float(eta) * grad)
    s = float(w_new.sum())
    w_new = (w_new / s) if s > 0 else np.ones_like(w_new) / len(w_new)

    # enforce constraints
    if k_cardinality is not None and int(k_cardinality) > 0:
        w_new = project_to_k_set(w_new, int(k_cardinality))

    w_new = _clip_box_and_long_only(w_new, box=box, long_only=long_only)

    # sparsify again after clipping if requested
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

    # bounds: default [0,1] for long-only; otherwise allow shorting within box if provided
    bounds = [(None, None)] * n
    if long_only:
        bounds = [(0.0, 1.0)] * n
    if box is not None:
        low = box.get('min', None)
        high = box.get('max', None)
        bounds = [(low if low is not None else b0, high if high is not None else b1) for (b0, b1) in bounds]

    x0 = w_prev.copy()
    # make feasible
    x0 = np.clip(x0, 1e-12, None)
    x0 = x0 / x0.sum()

    status = "ok"
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': int(max_iter), 'ftol': float(tol)})
        if res.success and res.x is not None:
            w_dense = np.asarray(res.x, dtype=float)
        else:
            w_dense = x0
            status = "solver_failed"
    except Exception:
        w_dense = x0
        status = "solver_exception"

    w_new = w_dense

    # enforce constraints (clip+renorm) and optional k-sparsity
    if k_cardinality is not None and int(k_cardinality) > 0:
        w_new = project_to_k_set(w_new, int(k_cardinality))

    w_new = _clip_box_and_long_only(w_new, box=box, long_only=long_only)

    if k_cardinality is not None and int(k_cardinality) > 0:
        w_new = project_to_k_set(w_new, int(k_cardinality))

    info = {"daily_return": daily, "loss": loss, "shortfall": shortfall}
    return np.asarray(w_new, dtype=float), B_t, v_t, info, status

# ---------------------------
# MV Regularized (sweep lambda)
# ---------------------------

def mv_reg_optimize(mu: pd.Series,
                    Sigma: pd.DataFrame,
                    lambdas: Optional[List[float]] = None,
                    lambda_reg: Optional[float] = None,
                    box: Optional[dict] = None,
                    long_only: bool = True,
                    solver: Optional[str] = None,
                    **kwargs) -> Dict:
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

    # support legacy single-value param `lambda_reg`
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

    # bounds
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
            # safety
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
                best_res = {"weights": pd.Series(w_np, index=assets), "status": "ok", "lambda": lam, "sharpe_insample": float(sharpe)}
        except Exception:
            continue

    if best_res is None:
        w0 = pd.Series(np.ones(n) / n, index=assets)
        if box or long_only:
            w0 = pd.Series(_clip_box_and_long_only(w0.values, box, long_only), index=assets)
        return {"weights": w0, "status": "fallback_equal", "lambda": None, "sharpe_insample": None}

    return best_res


# ---------------------------
# Minimum variance (convex)
# ---------------------------

def min_variance_optimize(Sigma: pd.DataFrame,
                          box: Optional[dict] = None,
                          long_only: bool = True,
                          solver: Optional[str] = None,
                          **kwargs) -> Dict:
    """Minimum variance portfolio without requiring CVXPY (SciPy SLSQP).

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


# ---------------------------
# Risk parity (numerical)
# ---------------------------

def risk_parity_optimize(Sigma: pd.DataFrame,
                         tol: float = 1e-6,
                         max_iter: int = 2000,
                         init_w: Optional[pd.Series] = None,
                         long_only: bool = True,
                         box: Optional[dict] = None,
                         **kwargs) -> Dict:
    """
    Numerical Risk Parity / Equal Risk Contribution solver (iterative).
    Accepts **kwargs to be tolerant of extra forwarded parameters.
    """
    assets = list(Sigma.index)
    n = len(assets)
    if n == 0:
        return {"weights": None, "status": "no_assets"}
    S = Sigma.reindex(index=assets, columns=assets).fillna(0.0).values.astype(float)
    # start with equal weights or provided
    if init_w is None:
        w = np.ones(n) / n
    else:
        w = init_w.reindex(assets).fillna(0.0).values
        if w.sum() <= 0:
            w = np.ones(n) / n
        else:
            w = w / w.sum()

    # iterative algorithm: Newton-like with scaling (see literature)
        # iterative algorithm: Newton-like with scaling (see literature)
    for it in range(max_iter):
        # compute marginal risk: M = Sigma @ w
        M = S.dot(w)  # expected to be >= 0, but numerical noise may produce tiny negatives

        # risk contributions RC_i = w_i * M_i
        RC = w * M
        RC_sum = RC.sum()
        if RC_sum == 0:
            break
        # target per-asset share
        target = RC_sum / n

        # --- NUMERICAL SAFEGUARDS ---
        # Avoid division by zero and negative values inside sqrt.
        eps = 1e-12

        # Ensure M is non-negative and finite before division
        M_safe = np.where(np.isfinite(M) & (M > 0), M, eps)

        # Ensure target is a positive finite scalar
        tgt_safe = float(target) if np.isfinite(target) and target > 0 else eps

        # Compute factors robustly
        factors = np.sqrt(M_safe / (tgt_safe + eps))

        # Prevent any non-finite or non-positive factors
        factors = np.where(np.isfinite(factors) & (factors > 0), factors, 1.0)

        # multiplicative update (keeps positivity)
        w_new = w / factors

        # apply box/long_only constraints by clipping and renormalizing
        if long_only:
            w_new = np.maximum(w_new, 0.0)
        if box is not None:
            low = box.get("min", None)
            high = box.get("max", None)
            if low is not None:
                w_new = np.maximum(w_new, float(low))
            if high is not None:
                w_new = np.minimum(w_new, float(high))
        if w_new.sum() <= 0:
            w_new = np.ones(n) / n
        w_new = w_new / w_new.sum()

        # convergence check: max relative change
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new


    M = S.dot(w)
    RC = w * M
    rc_mean = RC.mean() if RC.size > 0 else 0.0
    rc_err = float(np.linalg.norm(RC - rc_mean) / (np.linalg.norm(RC) + 1e-12))
    return {"weights": pd.Series(w, index=assets), "status": "ok", "erc_error": rc_err}


# ---------------------------
# Simple greedy k-cardinality helper
# ---------------------------

def greedy_k_cardinality(expected: pd.Series,
                         Sigma: pd.DataFrame,
                         k: int,
                         method: str = "mv_reg",
                         box: Optional[dict] = None,
                         long_only: bool = True,
                         **kwargs) -> Dict:
    """
    Simple greedy k-cardinality approach:
      - select top-k assets by expected return (or by risk-adjusted mu/sigma)
      - run chosen optimizer on that subset
    kwargs passed to underlying optimizer.
    """
    if expected is None or len(expected) == 0:
        return {"weights": None, "status": "no_expected"}

    # rank assets by score = mu / (sqrt(var) + eps)
    assets = expected.dropna().index.tolist()
    if len(assets) == 0:
        return {"weights": None, "status": "no_assets_after_dropna"}

    # compute per-asset vol from Sigma diag
    sigma_diag = pd.Series(np.sqrt(np.diag(Sigma.reindex(index=assets, columns=assets).fillna(0.0).values)), index=assets)
    score = expected.reindex(assets) / (sigma_diag + 1e-8)
    topk = list(score.sort_values(ascending=False).head(k).index) if k > 0 else assets

    mu_sub = expected.reindex(topk).dropna()
    Sigma_sub = Sigma.reindex(index=topk, columns=topk).fillna(0.0)

    if method == "mv_reg":
        return mv_reg_optimize(mu_sub, Sigma_sub, box=box, long_only=long_only, **kwargs)
    elif method == "minvar":
        return min_variance_optimize(Sigma_sub, box=box, long_only=long_only, **kwargs)
    elif method == "risk_parity":
        return risk_parity_optimize(Sigma_sub, box=box, long_only=long_only, **kwargs)
    elif method == "sharpe":
        return sharpe_optimize(mu_sub, Sigma_sub, box=box, long_only=long_only)
    else:
        # default fallback
        return mv_reg_optimize(mu_sub, Sigma_sub, box=box, long_only=long_only, **kwargs)


def sharpe_optimize(mu: pd.Series, Sigma: pd.DataFrame, box=None, long_only=True, solver=None) -> Dict:
    """Maximize Sharpe ratio using SciPy SLSQP (no CVXPY dependency).

    Maximize: (mu^T w) / sqrt(w^T Sigma w)
    Subject to: sum(w)=1 and optional bounds.

    Note: This is non-convex; we use a single-start local optimizer. For stability,
    prefer long-only + box constraints.
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

"""
Optimizers for Hermis Prism

This module implements a small collection of portfolio optimizers that are
pure-Python (numpy/pandas) friendly with an optional scipy fallback for
constrained (no-short) problems.

Functions included:
- min_variance_weights(returns, target_return=None, allow_short=True)
    Closed-form minimum-variance (or target-return) solution. If
    `allow_short` is False then a bounded quadratic program using
    `scipy.optimize.minimize` is used (if scipy is available).

- global_minimum_variance(returns, allow_short=True)
    Convenience wrapper for the GMV portfolio.

Usage example::

    import pandas as pd
    from portfolio_viz.optimizers import min_variance_weights

    # `rets` can be a DataFrame of periodic returns (rows = dates, cols = assets)
    w = min_variance_weights(rets, target_return=0.01, allow_short=False)

Notes:
- This implementation estimates expected returns as the mean of input returns
  and covariance with the sample covariance (ddof=1).
- When `allow_short=False` and scipy is not installed, a simple heuristic
  (clip negative weights to zero and renormalize) is used as a fallback.

"""
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
    """Return weights for the global minimum variance portfolio.

    Parameters
    ----------
    returns : DataFrame or ndarray
        Historical returns (rows = dates, columns = assets). If a 1-D series is
        provided it is treated as a single-asset series (trivial result).
    allow_short : bool
        If False then weights are constrained to be >= 0. In that case the
        function will attempt to use scipy.optimize; if scipy is unavailable a
        clipped-and-renormalized fallback is used.

    Returns
    -------
    pd.Series
        Asset weights (index preserved if `returns` is DataFrame).
    """
    return min_variance_weights(returns, target_return=None, allow_short=allow_short)


def min_variance_weights(returns, target_return: Optional[float] = None, allow_short: bool = True):
    """Compute minimum-variance (or target-return) portfolio weights.

    Closed-form solution when shorting is allowed. Constrained (no-short)
    solution uses scipy.optimize.minimize with bounds if available.

    Parameters
    ----------
    returns : pd.DataFrame or ndarray
        Historical periodic returns (rows = dates, columns = assets).
    target_return : float, optional
        If provided, solve the minimum-variance portfolio subject to the
        portfolio expected return equaling `target_return`. If None, return
        the global minimum variance portfolio (GMV).
    allow_short : bool
        If True, short positions are allowed and closed-form solution is used.
        If False, weights are constrained to be >= 0 and sum to 1.

    Returns
    -------
    pd.Series
        Weights indexed by asset names (if input is DataFrame) or integer
        positions if numpy array input.
    """
    mu, sigma = _to_mu_sigma(returns)
    n = len(mu)

    # Small regularization for numerical stability
    eps = 1e-8
    sigma = sigma + np.eye(n) * eps

    inv_sigma = np.linalg.inv(sigma)

    ones = np.ones(n)
    A = ones @ inv_sigma @ ones
    B = ones @ inv_sigma @ mu
    C = mu @ inv_sigma @ mu

    # Global minimum variance (no return constraint)
    if target_return is None and allow_short:
        w = (inv_sigma @ ones) / A
    elif target_return is not None and allow_short:
        # Solve for coefficients alpha, beta such that w = alpha * inv_sigma * mu + beta * inv_sigma * ones
        # Constraints: ones'w = 1, mu'w = target_return
        # Which gives: [B A; C B] [alpha; beta] = [1; target_return]
        M = np.array([[B, A], [C, B]])
        rhs = np.array([1.0, float(target_return)])
        alpha_beta = np.linalg.solve(M, rhs)
        alpha, beta = alpha_beta
        w = alpha * (inv_sigma @ mu) + beta * (inv_sigma @ ones)
    else:
        # Constrained problem (no shorting) or user forced no-short
        try:
            from scipy.optimize import minimize

            def port_var(x):
                return float(x @ sigma @ x)

            constraints = [{
                'type': 'eq',
                'fun': lambda x: float(np.sum(x) - 1.0)
            }]
            if target_return is not None:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: float(mu @ x - target_return)
                })

            bounds = [(0.0, 1.0) for _ in range(n)]
            x0 = np.ones(n) / n
            res = minimize(port_var, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if not res.success:
                # fallback to clipping heuristic
                w = np.clip(x0, 0, 1)
                w = w / w.sum()
            else:
                w = res.x
        except Exception:
            # SciPy not available -> heuristic
            # Take the unconstrained opt and clip negatives to zero then renormalize.
            try:
                if target_return is None:
                    w_un = (inv_sigma @ ones) / A
                else:
                    M = np.array([[B, A], [C, B]])
                    rhs = np.array([1.0, float(target_return)])
                    alpha_beta = np.linalg.solve(M, rhs)
                    alpha, beta = alpha_beta
                    w_un = alpha * (inv_sigma @ mu) + beta * (inv_sigma @ ones)
                w = np.clip(w_un, 0.0, None)
                s = w.sum()
                if s <= 0:
                    # fallback to equal weights
                    w = np.ones(n) / n
                else:
                    w = w / s
            except Exception:
                w = np.ones(n) / n

    # Return as pd.Series if input was DataFrame
    if isinstance(returns, pd.DataFrame):
        return pd.Series(w, index=returns.columns)
    else:
        return pd.Series(w)


# Additional helper: simple risk-parity heuristic (equal risk contributions)
# This is optional but handy for users who want a non-mean-based allocation.

def risk_parity_weights(returns, allow_short: bool = False, maxiter: int = 1000, tol: float = 1e-8):
    """Estimate equal risk contribution weights (a simple iterative algorithm).

    The algorithm solves for w s.t. w_i * (Sigma w)_i are (approximately) equal.
    We use a cyclical coordinate descent-like approach described in literature.

    Parameters
    ----------
    returns : DataFrame or ndarray
        Asset returns matrix.
    allow_short : bool
        Risk parity usually implies long-only weights; `allow_short` is
        accepted for API compatibility but currently ignored.

    Returns
    -------
    pd.Series
        Long-only weights summing to 1.
    """
    mu, sigma = _to_mu_sigma(returns)
    n = sigma.shape[0]
    x = np.ones(n) / n

    for it in range(maxiter):
        sigma_x = sigma @ x
        rc = x * sigma_x
        total_rc = rc.sum()
        # target risk contribution per asset
        target = total_rc / n
        # update rule: scale each weight by target / actual contribution
        x_new = x * np.sqrt(target / (rc + 1e-12))
        x_new = np.clip(x_new, 1e-12, None)
        x_new = x_new / x_new.sum()
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    if isinstance(returns, pd.DataFrame):
        return pd.Series(x, index=returns.columns)
    else:
        return pd.Series(x)


# --- Entropy-regularized damped-Newton projected optimizer ---
# Adapted from user-supplied implementation (damped Newton with entropy
# regularization and projection onto {w >= 0, sum(w) <= K}).

def _proj_simplex(v: np.ndarray, K: float = 1.0) -> np.ndarray:
    """Projection of vector v onto {x >= 0, sum(x) <= K}.

    If sum(max(v,0)) <= K returns max(v,0). Otherwise projects onto the
    simplex of radius K (i.e. nonnegative vector summing to exactly K).
    """
    x = np.maximum(v, 0.0)
    s = x.sum()
    if s <= K:
        return x
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - K))[0]
    if len(rho_idx) == 0:
        theta = 0.0
    else:
        rho = rho_idx[-1]
        theta = (cssv[rho] - K) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


def _objective_and_grad_hess(w: np.ndarray, R: np.ndarray, v_sum: np.ndarray, p_avg: float, lam: float, eps: float = 1e-12):
    """Objective, gradient and Hessian for the entropy-regularized problem.

    f(w) = 0.5 w^T R w - p_avg w^T v_sum + lam * sum_i w_i log w_i
    """
    w_safe = np.maximum(w, eps)
    f = 0.5 * float(w.dot(R.dot(w))) - float(p_avg * w.dot(v_sum)) + float(lam * np.sum(w_safe * np.log(w_safe)))
    grad = R.dot(w) - p_avg * v_sum + lam * (1.0 + np.log(w_safe))
    H = R.copy()
    H = H + np.diag(lam / w_safe)
    return f, grad, H


def _damped_newton_projected(w_init: np.ndarray, R: np.ndarray, v_sum: np.ndarray, p_avg: float, lam: float, K: float = 1.0,
                              max_iters: int = 50, tol: float = 1e-8, alpha0: float = 1.0, eps: float = 1e-12):
    """Minimize the entropy-regularized objective using damped Newton + projection.

    Parameters
    ----------
    w_init : ndarray
        Initial guess (length n).
    R : ndarray
        Accumulated quadratic term (n x n).
    v_sum : ndarray
        Accumulated linear term (length n).
    p_avg : float
        Scalar multiplier for the linear term.
    lam : float
        Entropy (log) regularization weight (>=0).
    K : float
        Simplex radius (sum(w) <= K).

    Returns
    -------
    ndarray
        Projected optimal weights (length n).
    """
    w = w_init.copy().astype(float)
    n = len(w)
    for it in range(max_iters):
        f, g, H = _objective_and_grad_hess(w, R, v_sum, p_avg, lam, eps=eps)
        reg = 1e-8
        try:
            d = solve(H + reg * np.eye(n), g)
        except np.linalg.LinAlgError:
            d = g
        alpha = alpha0
        found = False
        for _ in range(25):
            w_trial = w - alpha * d
            w_trial_proj = _proj_simplex(w_trial, K=K)
            f_trial, _, _ = _objective_and_grad_hess(w_trial_proj, R, v_sum, p_avg, lam, eps=eps)
            # Armijo-like condition (sufficient decrease)
            if f_trial <= f - 1e-4 * alpha * float(np.dot(g, d)):
                found = True
                break
            alpha *= 0.5
        if not found:
            step = -0.01 * g
            w_next = _proj_simplex(w + step, K=K)
        else:
            w_next = w_trial_proj
        if norm(w_next - w) < tol:
            w = w_next
            break
        w = w_next
    return w


def entropy_newton_weights(returns, p_avg: float = 0.0, lam: float = 1e-2, K: float = 1.0,
                            max_iters: int = 50, tol: float = 1e-8, warm_start: bool = True):
    """Compute weights by minimizing a quadratic objective with entropy regularization.

    Problem (variables w >= 0, sum(w) <= K):
        0.5 w^T R w - p_avg * w^T v_sum + lam * sum_i w_i log w_i

    Inputs are accepted as pandas DataFrame / Series / numpy arrays. When
    `returns` is a DataFrame it is treated as a (T x N) matrix of past returns
    and we use the accumulators R = sum_t r_t r_t^T, v_sum = sum_t r_t.

    Parameters
    ----------
    returns : DataFrame or ndarray
    p_avg : float
    lam : float
    K : float
    max_iters, tol : numeric
    warm_start : bool
        If True and `returns` is a DataFrame, initialize with uniform weights
        across columns. If a 1-D array is provided, the initial value will be
        a normalized positive vector.

    Returns
    -------
    pd.Series or np.ndarray
        Weights summing to <= K. If `returns` is a DataFrame, a pd.Series with
        the same column index is returned.
    """
    # Convert inputs
    if isinstance(returns, pd.DataFrame):
        # construct accumulators from historical returns: R = sum outer, v_sum = sum
        arr = returns.dropna(how='all').values
        if arr.size == 0:
            raise ValueError("Empty returns provided")
        R = arr.T.dot(arr)
        v_sum = arr.sum(axis=0)
        cols = returns.columns
    elif isinstance(returns, pd.Series):
        arr = returns.values.reshape(-1, 1)
        R = arr.T.dot(arr)
        v_sum = arr.sum(axis=0)
        cols = returns.index
    elif isinstance(returns, np.ndarray):
        if returns.ndim == 1:
            arr = returns.reshape(-1, 1)
        else:
            arr = returns
        R = arr.T.dot(arr)
        v_sum = arr.sum(axis=0)
        cols = None
    else:
        raise ValueError("Unsupported returns type")

    n = R.shape[0]
    if warm_start:
        w0 = np.ones(n) * (K / n)
    else:
        w0 = np.maximum(np.random.randn(n), 0.0)
        s = w0.sum()
        if s <= 0:
            w0 = np.ones(n) * (K / n)
        else:
            w0 = w0 / s * K

    w_opt = _damped_newton_projected(w0, R, v_sum, p_avg, lam, K=K, max_iters=max_iters, tol=tol)

    if cols is not None:
        return pd.Series(w_opt, index=cols)
    else:
        return pd.Series(w_opt)

