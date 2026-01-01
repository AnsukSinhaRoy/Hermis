"""
hermis/replay/rebalance.py

Rebalance schedule utilities on a trading-day index.

User requirement:
- daily rebalance: every trading session
- weekly rebalance: "on Monday; if Monday is not a trading day, use the next trading day"
  => which is simply: first trading day of each ISO week (week starting Monday).
"""
from __future__ import annotations

import pandas as pd


def trading_days_from_daily_index(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(daily_index, errors="coerce")).dropna().sort_values().unique()
    return pd.DatetimeIndex(idx).normalize()


def get_rebalance_days(trading_days: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    td = trading_days_from_daily_index(trading_days)
    if len(td) == 0:
        return pd.DatetimeIndex([])

    freq = str(freq).strip().lower()
    if freq in {"d", "daily", "1d"}:
        return td

    if freq in {"w", "weekly", "1w"}:
        # group by week starting Monday
        week_start = td - pd.to_timedelta(td.weekday, unit="D")
        picks = (
            pd.Series(td)
            .groupby(week_start)
            .min()
            .sort_values()
            .values
        )
        return pd.DatetimeIndex(picks).normalize()

    if freq in {"m", "monthly", "1m"}:
        period = td.to_period("M")
        picks = pd.Series(td).groupby(period).min().sort_values().values
        return pd.DatetimeIndex(picks).normalize()

    raise ValueError(f"Unsupported rebalance freq: {freq!r}")
