# portfolio_sim/optimizer.py
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional, Tuple, Dict, Any

def mv_optimize(expected_returns: pd.Series,
                cov: pd.DataFrame,
                target_return: Optional[float] = None,
                box: Optional[Tuple[float,float]] = None,
                long_only: bool = True,
                solver: Optional[str] = None) -> Dict[str,Any]:
    """
    Solve classical mean-variance: minimize w^T cov w subject to expected return >= target_return
    """
    n = len(expected_returns)
    mu = expected_returns.values.astype(float)
    Sigma = cov.values.astype(float)
    w = cp.Variable(n)
    objective = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w) == 1]
    if target_return is not None:
        constraints.append(mu @ w >= target_return)
    if long_only:
        constraints.append(w >= 0)
    if box is not None:
        lb, ub = box
        constraints += [w >= lb, w <= ub]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    try:
        if solver:
            prob.solve(solver=solver, verbose=False)
        else:
            prob.solve(verbose=False)
    except Exception:
        # try different solver fallback
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            prob.solve(solver=cp.OSQP, verbose=False)
    if w.value is None:
        # fallback: equal weight
        weights = pd.Series(np.repeat(1.0/n, n), index=expected_returns.index)
        return {"weights": weights, "status": "failed", "objective": None}
    weights = pd.Series(w.value.flatten(), index=expected_returns.index)
    return {"weights": weights, "status": prob.status, "objective": prob.value}

def greedy_k_cardinality(expected_returns: pd.Series,
                         cov: pd.DataFrame,
                         k: int,
                         box: Optional[Tuple[float,float]] = None,
                         long_only: bool = True) -> Dict[str,Any]:
    """
    Greedy selection:
    1. Score assets by r_i / sqrt(Sigma_ii)
    2. Select top k, solve continuous MV restricted to selected assets
    """
    n = len(expected_returns)
    if k >= n or k is None:
        return mv_optimize(expected_returns, cov, target_return=None, box=box, long_only=long_only)
    sigma_diag = np.sqrt(np.diag(cov.values))
    score = expected_returns.values / (sigma_diag + 1e-12)
    idx_sorted = np.argsort(-score)
    selected_idx = idx_sorted[:k]
    selected = expected_returns.index[selected_idx]
    cov_r = cov.loc[selected, selected]
    mu_r = expected_returns.loc[selected]
    res = mv_optimize(mu_r, cov_r, target_return=None, box=box, long_only=long_only)
    w_r = res['weights']
    w_full = pd.Series(0.0, index=expected_returns.index)
    w_full.loc[selected] = w_r
    res['weights'] = w_full
    res['selected'] = list(selected)
    return res
