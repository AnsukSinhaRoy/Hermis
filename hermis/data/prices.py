"""Price loading and return/covariance utilities.

Today the codebase mostly uses daily bars.

This module intentionally keeps a stable, easy-to-use surface area:
- `generate_synthetic_prices(n_assets, start, end, seed)`
- `load_prices_from_csv(path, start_date=None, end_date=None)`
- `load_prices_from_parquet(path, start_date=None, end_date=None, ...)`

These are compatible with the legacy `portfolio_sim.data` functions, and
delegate to them to avoid divergence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

# Delegate to the battle-tested legacy implementations.
from portfolio_sim.data import (  # type: ignore
    apply_date_range as _apply_date_range,
    generate_synthetic_prices as _generate_synthetic_prices,
    load_prices_from_csv as _load_prices_from_csv,
    load_prices_from_parquet as _load_prices_from_parquet,
    compute_returns as _compute_returns,
    cov_matrix as _cov_matrix,
)


def apply_date_range(
    prices: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    return _apply_date_range(prices, start_date=start_date, end_date=end_date)


def generate_synthetic_prices(
    n_assets: int = 100,
    start: str = "2018-01-01",
    end: str = "2022-12-31",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic GBM prices (daily business days).

    Signature kept stable for legacy configs.
    """
    return _generate_synthetic_prices(n_assets=n_assets, start=start, end=end, seed=seed)


def load_prices_from_csv(
    path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load prices from CSV.

    Supports both:
      - long format: date,ticker,price
      - wide format: date + one column per ticker

    Optional date slicing is applied after load.
    """
    df = _load_prices_from_csv(path)
    df, _, _ = apply_date_range(df, start_date=start_date, end_date=end_date)
    return df


def load_prices_from_parquet(
    path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "price",
) -> pd.DataFrame:
    """Load prices from parquet (supports directory, wide, or long formats).

    Optional date slicing is pushed down when pyarrow is available.
    """
    return _load_prices_from_parquet(
        path,
        start_date=start_date,
        end_date=end_date,
        date_col=date_col,
        ticker_col=ticker_col,
        price_col=price_col,
    )


def compute_returns(prices: pd.DataFrame, method: str = "log", min_valid_obs: int = 2) -> pd.DataFrame:
    return _compute_returns(prices, method=method, min_valid_obs=min_valid_obs)


def cov_matrix(prices: pd.DataFrame, method: str = "log", use_gpu: bool = False, min_obs: int = 5) -> pd.DataFrame:
    return _cov_matrix(prices, method=method, use_gpu=use_gpu, min_obs=min_obs)
