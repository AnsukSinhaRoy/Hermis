"""Event-driven data feed interfaces.

This is a *skeleton* for the future 1s / derivatives engine.

It lets you progressively migrate without rewriting everything:
- Start with a DataFrame of bars and adapt it via `DataFrameBarFeed`.
- Later swap in a true streaming feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd


@dataclass(frozen=True)
class Bar:
    ts: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class DataFeed(Iterable[Bar]):
    """Abstract bar feed."""

    def __iter__(self) -> Iterator[Bar]:  # pragma: no cover
        raise NotImplementedError


class DataFrameBarFeed(DataFeed):
    """Adapter: wide OHLCV DataFrames → stream of `Bar`.

    Expected input format:
      data[symbol] = DataFrame indexed by datetime with columns in {open,high,low,close,volume?}

    This is intentionally simple and not optimized.
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self._data = {k: v.sort_index() for k, v in data.items()}

    def __iter__(self) -> Iterator[Bar]:
        # Merge all timestamps (simple but memory-heavy; OK for a skeleton)
        all_ts = pd.Index([])
        for df in self._data.values():
            all_ts = all_ts.union(df.index)
        all_ts = all_ts.sort_values()

        for ts in all_ts:
            for symbol, df in self._data.items():
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                yield Bar(
                    ts=pd.Timestamp(ts),
                    symbol=symbol,
                    open=float(row.get("open", row.get("Open", row.get("close", row.get("Close", 0.0))))),
                    high=float(row.get("high", row.get("High", row.get("close", row.get("Close", 0.0))))),
                    low=float(row.get("low", row.get("Low", row.get("close", row.get("Close", 0.0))))),
                    close=float(row.get("close", row.get("Close", 0.0))),
                    volume=(float(row.get("volume")) if "volume" in row.index else (float(row.get("Volume")) if "Volume" in row.index else None)),
                )
