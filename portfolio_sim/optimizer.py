# portfolio_sim/optimizer.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import cvxpy as cp

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

def _clip_box_and_long_only(w: pd.Series, box: Optional[dict], long_only: bool) -> pd.Series:
    if box is None:
        low, high = None, None
    else:
        low, high = box.get("min", None), box.get("max", None)
    if long_only:
        if low is None:
            low = 0.0
        else:
            low = max(0.0, low)
    # apply clipping
    if low is not None:
        w = np.maximum(w, low)
    if high is not None:
        w = np.minimum(w, high)
    # renormalize to sum 1 if any positive mass
    s = np.nansum(w)
    if s > 0:
        w = w / s
    return pd.Series(w, index=w.index if isinstance(w, pd.Series) else None)

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
    """
    Regularized mean-variance: maximize mu^T w - lambda * w^T Sigma w
    Backwards-compatible: accepts either `lambdas` (list) or `lambda_reg` (single float)
    and normalizes to a list of lambda values to try.
    We try several lambda values and pick the best in-sample Sharpe.
    Returns dict {weights: pd.Series, status: "ok"/"failed", lambda: chosen}
    """
    # support legacy single-value param `lambda_reg`
    if lambdas is None and lambda_reg is not None:
        # if user passed a single float, turn it into a list for the sweep
        lambdas = [float(lambda_reg)]

    # if someone passed a single scalar in lambdas, coerce to list
    if lambdas is not None and not isinstance(lambdas, (list, tuple, set)):
        try:
            lambdas = [float(lambdas)]
        except Exception:
            lambdas = None

    if lambdas is None:
        lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0]

    # inputs to numpy
    assets = list(mu.index)
    n = len(assets)
    if n == 0:
        return {"weights": None, "status": "no_assets"}
    mu_np = mu.reindex(assets).values.astype(float)
    Sigma_np = Sigma.reindex(index=assets, columns=assets).fillna(0.0).values.astype(float)

    best_sharpe = -np.inf
    best_res = None

    for lam in lambdas:
        try:
            w = cp.Variable(n)
            # objective: maximize mu^T w - lam * w^T Sigma w
            obj = cp.Maximize(mu_np @ w - float(lam) * cp.quad_form(w, Sigma_np))
            constraints = [cp.sum(w) == 1]
            if long_only:
                constraints += [w >= 0]
            if box is not None:
                low = box.get("min", None)
                high = box.get("max", None)
                if low is not None:
                    constraints += [w >= float(low)]
                if high is not None:
                    constraints += [w <= float(high)]
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=solver, warm_start=True)
            if w.value is None:
                continue
            w_np = np.array(w.value).flatten()
            # postprocess small negatives
            w_np = np.where(w_np < 1e-8, 0.0, w_np)
            if w_np.sum() <= 0:
                continue
            w_np = w_np / w_np.sum()
            # compute in-sample performance: mean return and vol (annualized)
            port_mean = float(mu_np @ w_np)
            port_var = float(w_np @ Sigma_np @ w_np)
            port_vol = np.sqrt(port_var) if port_var > 0 else 1e-9
            sharpe = port_mean / port_vol if port_vol > 0 else -np.inf
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_res = {"weights": pd.Series(w_np, index=assets), "status": "ok", "lambda": lam, "sharpe_insample": float(sharpe)}
        except Exception:
            # skip failing lambda
            continue

    if best_res is None:
        # fallback: equal weight among assets, respecting long_only/box
        w0 = np.ones(n) / n
        w0 = pd.Series(w0, index=assets)
        if box or long_only:
            # clip and renormalize
            w0 = _clip_box_and_long_only(w0.values, box, long_only)
            w0 = pd.Series(w0, index=assets)
        return {"weights": w0, "status": "fallback_equal", "lambda": None, "sharpe_insample": None}
    return best_res


# ---------------------------
# Minimum variance (convex)
# ---------------------------

def min_variance_optimize(Sigma: pd.DataFrame,
                          box: Optional[dict] = None,
                          long_only: bool = True,
                          solver: Optional[str] = None) -> Dict:
    """
    Minimum variance under sum(w)=1 and box/long-only constraints.
    """
    assets = list(Sigma.index)
    n = len(assets)
    if n == 0:
        return {"weights": None, "status": "no_assets"}
    Sigma_np = Sigma.reindex(index=assets, columns=assets).fillna(0.0).values.astype(float)
    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, Sigma_np))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints += [w >= 0]
    if box is not None:
        low = box.get("min", None)
        high = box.get("max", None)
        if low is not None:
            constraints += [w >= float(low)]
        if high is not None:
            constraints += [w <= float(high)]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=solver, warm_start=True)
        if w.value is None:
            raise RuntimeError("solver_failed")
        w_np = np.array(w.value).flatten()
        w_np = np.where(w_np < 1e-8, 0.0, w_np)
        if w_np.sum() <= 0:
            # fallback equal
            w_np = np.ones(n) / n
        w_np = w_np / w_np.sum()
        return {"weights": pd.Series(w_np, index=assets), "status": "ok"}
    except Exception as e:
        # fallback
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
                         box: Optional[dict] = None) -> Dict:
    """
    Numerical Risk Parity / Equal Risk Contribution solver (iterative).
    Sigma: covariance DataFrame (index/columns assets)
    Returns equal risk contribution weights (sum to 1).
    Reference: Maillard et al. 2010 / iterative algorithms.
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
    for it in range(max_iter):
        # compute marginal risk: M = Sigma @ w
        M = S.dot(w)
        # risk contributions RC_i = w_i * M_i
        RC = w * M
        RC_sum = RC.sum()
        if RC_sum == 0:
            break
        # target per-asset share
        target = RC_sum / n
        # gradient-like update: w_i <- w_i * (M_i / target)^{-1/2} (heuristic)
        # use multiplicative update to enforce positivity
        factors = np.sqrt(M / (target + 1e-12))
        # prevent division by zero/infs
        factors = np.where(np.isfinite(factors) & (factors > 0), factors, 1.0)
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

    # final check: compute status and RC parity error
    M = S.dot(w)
    RC = w * M
    rc_mean = RC.mean() if RC.size>0 else 0.0
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
    else:
        # default fallback
        return mv_reg_optimize(mu_sub, Sigma_sub, box=box, long_only=long_only, **kwargs)
