from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import pandas as pd


@dataclass
class StrategyContext:
    """Context passed into strategies.

    You can extend this without touching the optimizer APIs.
    """

    date: Optional[pd.Timestamp] = None
    prices_window: Optional[pd.DataFrame] = None
    returns_window: Optional[pd.DataFrame] = None
    extra: Dict[str, Any] = None


class PortfolioStrategy(Protocol):
    """Portfolio-mode strategy.

    Typical usage: called at each rebalance step.

    Must return a dict containing:
    - weights: pd.Series (index = assets)
    - status: str (optional)
    """

    def __call__(self, mu: pd.Series, Sigma: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        ...
