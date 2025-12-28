"""Optimizer implementations and registry.

All optimizers follow the same high-level contract:

    optimizer(mu: pd.Series, Sigma: pd.DataFrame, **context) -> dict

Where the returned dict must include a `weights` entry (pd.Series).

The legacy code expects several named functions (mv_reg_optimize, risk_parity_optimize, ...).
Those remain available via both:
- `hermis.optimizers.*`
- `portfolio_sim.optimizer.*` (compatibility wrapper)
"""

from __future__ import annotations

from .common import (
    _normalize_box,
    _clip_box_and_long_only,
    project_to_k_set,
)
from .online import omd_step, ftrl_step, _price_relatives_from_prices_window
from .mean_variance import mv_optimize, mv_reg_optimize
from .min_variance import min_variance_optimize
from .risk_parity import risk_parity_optimize
from .greedy import greedy_k_cardinality
from .sharpe import sharpe_optimize
from .entropy_newton import (
    _proj_simplex,
    _objective_and_grad_hess,
    _damped_newton_projected,
    entropy_newton_weights,
)
from .legacy_prism import (
    _to_mu_sigma,
    global_minimum_variance,
    min_variance_weights,
    risk_parity_weights,
)

__all__ = [
    "_normalize_box",
    "_clip_box_and_long_only",
    "project_to_k_set",
    "omd_step",
    "ftrl_step",
    "_price_relatives_from_prices_window",
    "mv_optimize",
    "mv_reg_optimize",
    "min_variance_optimize",
    "risk_parity_optimize",
    "greedy_k_cardinality",
    "sharpe_optimize",
    "_proj_simplex",
    "_objective_and_grad_hess",
    "_damped_newton_projected",
    "entropy_newton_weights",
    "_to_mu_sigma",
    "global_minimum_variance",
    "min_variance_weights",
    "risk_parity_weights",
]
