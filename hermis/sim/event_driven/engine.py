from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from hermis.data.feeds import Bar, DataFeed


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, float]  # symbol -> units
    last_price: Dict[str, float]


class EventDrivenEngine:
    """Minimal event-driven loop skeleton.

    This is intentionally *not* wired into the current daily backtester.

    Extension points:
    - a `policy` callable: `policy(state, bar) -> desired_position_changes`
    - an `execution` model (slippage, fees, fills)

    For now, we only keep the API shape and a basic book-keeping loop.
    """

    def __init__(
        self,
        feed: DataFeed,
        policy: Callable[[PortfolioState, Bar], Dict[str, float]],
        initial_cash: float = 1_000_000.0,
        fee_bps: float = 0.0,
    ):
        self.feed = feed
        self.policy = policy
        self.fee_bps = float(fee_bps)
        self.state = PortfolioState(cash=float(initial_cash), positions={}, last_price={})

    def _mark_to_market(self) -> float:
        value = self.state.cash
        for sym, qty in self.state.positions.items():
            px = self.state.last_price.get(sym)
            if px is None:
                continue
            value += float(qty) * float(px)
        return float(value)

    def run(self) -> pd.DataFrame:
        rows = []
        for bar in self.feed:
            self.state.last_price[bar.symbol] = float(bar.close)

            # Policy proposes a delta in units (can be positive or negative)
            delta_units = self.policy(self.state, bar) or {}

            # Naive immediate execution at close
            for sym, du in delta_units.items():
                du = float(du)
                if abs(du) < 1e-12:
                    continue
                px = float(self.state.last_price.get(sym, bar.close))
                notional = du * px
                fee = abs(notional) * (self.fee_bps / 10_000.0)

                self.state.cash -= notional
                self.state.cash -= fee
                self.state.positions[sym] = float(self.state.positions.get(sym, 0.0) + du)

            rows.append(
                {
                    "ts": bar.ts,
                    "symbol": bar.symbol,
                    "equity": self._mark_to_market(),
                    "cash": self.state.cash,
                }
            )

        if not rows:
            return pd.DataFrame(columns=["ts", "symbol", "equity", "cash"]).set_index("ts")

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts").sort_index()
