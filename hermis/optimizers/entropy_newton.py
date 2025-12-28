from __future__ import annotations

import numpy as np
import pandas as pd

from numpy.linalg import solve, norm


def _proj_simplex(v: np.ndarray, K: float = 1.0) -> np.ndarray:
    """Projection onto {x >= 0, sum(x) <= K}.

    If sum(max(v,0)) <= K returns max(v,0). Otherwise projects onto the
    simplex of radius K (nonnegative vector summing to exactly K).
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


def _objective_and_grad_hess(
    w: np.ndarray,
    R: np.ndarray,
    v_sum: np.ndarray,
    p_avg: float,
    lam: float,
    eps: float = 1e-12,
):
    """Objective, gradient and Hessian for entropy-regularized quadratic."""
    w_safe = np.maximum(w, eps)
    f = 0.5 * float(w.dot(R.dot(w))) - float(p_avg * w.dot(v_sum)) + float(lam * np.sum(w_safe * np.log(w_safe)))
    grad = R.dot(w) - p_avg * v_sum + lam * (1.0 + np.log(w_safe))
    H = R.copy()
    H = H + np.diag(lam / w_safe)
    return f, grad, H


def _damped_newton_projected(
    w_init: np.ndarray,
    R: np.ndarray,
    v_sum: np.ndarray,
    p_avg: float,
    lam: float,
    K: float = 1.0,
    max_iters: int = 50,
    tol: float = 1e-8,
    alpha0: float = 1.0,
    eps: float = 1e-12,
):
    """Minimize entropy-regularized objective using damped Newton + projection."""
    w = w_init.copy().astype(float)
    n = len(w)
    for _ in range(int(max_iters)):
        f, g, H = _objective_and_grad_hess(w, R, v_sum, p_avg, lam, eps=eps)
        reg = 1e-8
        try:
            d = solve(H + reg * np.eye(n), g)
        except np.linalg.LinAlgError:
            d = g
        alpha = float(alpha0)
        found = False
        for _ in range(25):
            w_trial = w - alpha * d
            w_trial_proj = _proj_simplex(w_trial, K=K)
            f_trial, _, _ = _objective_and_grad_hess(w_trial_proj, R, v_sum, p_avg, lam, eps=eps)
            if f_trial <= f - 1e-4 * alpha * float(np.dot(g, d)):
                found = True
                break
            alpha *= 0.5
        if not found:
            step = -0.01 * g
            w_next = _proj_simplex(w + step, K=K)
        else:
            w_next = w_trial_proj
        if norm(w_next - w) < float(tol):
            w = w_next
            break
        w = w_next
    return w


def entropy_newton_weights(
    returns,
    p_avg: float = 0.0,
    lam: float = 1e-2,
    K: float = 1.0,
    max_iters: int = 50,
    tol: float = 1e-8,
    warm_start: bool = True,
):
    """Compute weights by minimizing a quadratic objective with entropy regularization."""
    if isinstance(returns, pd.DataFrame):
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
        arr = returns.reshape(-1, 1) if returns.ndim == 1 else returns
        R = arr.T.dot(arr)
        v_sum = arr.sum(axis=0)
        cols = None
    else:
        raise ValueError("Unsupported returns type")

    n = R.shape[0]
    if warm_start:
        w0 = np.ones(n) * (float(K) / n)
    else:
        w0 = np.maximum(np.random.randn(n), 0.0)
        s = w0.sum()
        if s <= 0:
            w0 = np.ones(n) * (float(K) / n)
        else:
            w0 = w0 / s * float(K)

    w_opt = _damped_newton_projected(w0, R, v_sum, p_avg, lam, K=K, max_iters=max_iters, tol=tol)

    if cols is not None:
        return pd.Series(w_opt, index=cols)
    return pd.Series(w_opt)
