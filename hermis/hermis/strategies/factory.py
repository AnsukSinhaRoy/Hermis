from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from hermis.optimizers.registry import get as get_optimizer


def make_optimizer_strategy(name: str, **default_kwargs):
    """Create a callable strategy from an optimizer name.

    Example:
        strat = make_optimizer_strategy("mv_reg", lambdas=[1e-2], long_only=True)
        result = strat(mu, Sigma)
    """

    spec = get_optimizer(name)

    def _strategy(mu: pd.Series, Sigma: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        params = dict(default_kwargs)
        params.update(kwargs)
        return spec.fn(mu, Sigma, **params)

    _strategy.__name__ = f"strategy_{name}"
    return _strategy
