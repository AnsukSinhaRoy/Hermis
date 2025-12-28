from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from . import (
    greedy_k_cardinality,
    min_variance_optimize,
    mv_reg_optimize,
    risk_parity_optimize,
    sharpe_optimize,
)
from .registry import register

# Register built-ins

@register("mv_reg", "Regularized mean-variance (lambda sweep)")
def mv_reg(mu: pd.Series, Sigma: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    return mv_reg_optimize(mu, Sigma, **kwargs)

@register("minvar", "Minimum variance")
def minvar(mu: pd.Series, Sigma: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    _ = mu
    return min_variance_optimize(Sigma, **kwargs)

@register("risk_parity", "Equal risk contribution")
def risk_parity(mu: pd.Series, Sigma: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    _ = mu
    return risk_parity_optimize(Sigma, **kwargs)

@register("sharpe", "Max Sharpe ratio")
def sharpe(mu: pd.Series, Sigma: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    return sharpe_optimize(mu, Sigma, **kwargs)

@register("greedy_k", "Greedy top-k + chosen optimizer")
def greedy_k(mu: pd.Series, Sigma: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    k = int(kwargs.pop("k", 20))
    method = kwargs.pop("method", "mv_reg")
    return greedy_k_cardinality(mu, Sigma, k=k, method=method, **kwargs)
