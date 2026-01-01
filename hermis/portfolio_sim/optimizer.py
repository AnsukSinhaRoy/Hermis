"""Compatibility optimizer surface.

The original project grew a large single-file optimizer module.

To make the simulator easier to extend (online algos, Deep RL policies, etc.),
the implementations were split into `hermis.optimizers.*` modules.

Existing imports keep working:

    from portfolio_sim.optimizer import mv_reg_optimize

New code should prefer:

    from hermis.optimizers import mv_reg_optimize

The full original implementation is preserved at
`portfolio_sim/optimizer_legacy.py`.
"""

from __future__ import annotations

# Re-export the modularized implementations
from hermis.optimizers import (  # noqa: F401
    _normalize_box,
    _clip_box_and_long_only,
    project_to_k_set,
    omd_step,
    ftrl_step,
    _price_relatives_from_prices_window,
    mv_optimize,
    mv_reg_optimize,
    min_variance_optimize,
    risk_parity_optimize,
    greedy_k_cardinality,
    sharpe_optimize,
    _to_mu_sigma,
    global_minimum_variance,
    min_variance_weights,
    risk_parity_weights,
    _proj_simplex,
    _objective_and_grad_hess,
    _damped_newton_projected,
    entropy_newton_weights,
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
    "_to_mu_sigma",
    "global_minimum_variance",
    "min_variance_weights",
    "risk_parity_weights",
    "_proj_simplex",
    "_objective_and_grad_hess",
    "_damped_newton_projected",
    "entropy_newton_weights",
]
