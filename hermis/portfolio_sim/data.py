# portfolio_sim/data.py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path
from .backend import get_backend, to_numpy


def _to_timestamp(x: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse an optional date/datetime string into pandas Timestamp."""
    if x is None:
        return None
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date/datetime: {x!r}")
    return pd.Timestamp(ts)


def apply_date_range(prices: pd.DataFrame,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return prices filtered to [start_date, end_date] (inclusive).

    - If start/end fall on non-trading days, the nearest available dates inside the interval are used.
    - Returns (filtered_prices, effective_start, effective_end)
    """
    if prices is None or prices.empty:
        return prices, None, None

    p = prices.copy()
    # Ensure datetime index
    if not isinstance(p.index, pd.DatetimeIndex):
        p.index = pd.to_datetime(p.index, errors="coerce")
    p = p.sort_index()

    s = _to_timestamp(start_date)
    e = _to_timestamp(end_date)
    if s is not None and e is not None and s > e:
        raise ValueError(f"start_date {start_date!r} is after end_date {end_date!r}")

    if s is not None:
        p = p.loc[p.index >= s]
    if e is not None:
        p = p.loc[p.index <= e]

    if p.empty:
        return p, None, None

    return p, p.index.min(), p.index.max()


def _infer_date_col_from_schema(colnames, preferred: str = "date"):
    """Pick a likely date column from a parquet schema (or return None)."""
    if preferred in colnames:
        return preferred
    for c in ["Date", "datetime", "timestamp", "ts", "time", "index", "__index_level_0__"]:
        if c in colnames:
            return c
    return None


def _pyarrow_read_parquet_filtered(
    path: Path,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_col: Optional[str] = None,
    columns: Optional[list] = None,
) -> Optional[pd.DataFrame]:
    """Read parquet using pyarrow.dataset with predicate pushdown on the date column.

    Returns None if:
      - pyarrow is not installed
      - no suitable date column exists in parquet schema
      - filtering/read fails (caller should fall back to pandas)
    """
    try:
        import pyarrow.dataset as ds
    except Exception:
        return None

    try:
        dataset = ds.dataset(str(path), format="parquet")
        colnames = list(dataset.schema.names)

        dcol = date_col or _infer_date_col_from_schema(colnames, preferred="date")
        if dcol is None:
            return None

        s = _to_timestamp(start_date)
        e = _to_timestamp(end_date)

        filt = None
        if s is not None:
            filt = (ds.field(dcol) >= s.to_datetime64())
        if e is not None:
            filt = (ds.field(dcol) <= e.to_datetime64()) if filt is None else (filt & (ds.field(dcol) <= e.to_datetime64()))

        cols = None
        if columns is not None:
            cols = list(dict.fromkeys(columns + ([dcol] if dcol not in columns else [])))

        table = dataset.to_table(filter=filt, columns=cols)
        return table.to_pandas()
    except Exception:
        return None


def load_prices_from_parquet(
    path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "price",
) -> pd.DataFrame:
    """Load prices from parquet efficiently (pyarrow predicate pushdown when possible).

    Supported formats:
      1) *Wide* parquet: index is dates, columns are tickers.
      2) *Long* parquet: columns include (date, ticker, price) -> pivoted to wide.
      3) Directory of per-ticker parquet files.

    Efficiency improvement:
      - If `pyarrow` is installed and the parquet contains a date column
        (commonly `date` or `__index_level_0__`), we read only the requested
        `[start_date, end_date]` window using predicate pushdown.
      - If not possible, we fall back to pandas and slice in-memory.
    """
    p = Path(path)

    def _postprocess_wide(df: pd.DataFrame) -> pd.DataFrame:
        # Ensure datetime index and slice (safety)
        if isinstance(df.index, pd.DatetimeIndex):
            wide = df.sort_index()
            wide, _, _ = apply_date_range(wide, start_date, end_date)
            return wide

        # If there's a date-like column, set it as index
        dcol = date_col if date_col in df.columns else _infer_date_col_from_schema(df.columns, preferred=date_col)
        if dcol and dcol in df.columns:
            tmp = df.copy()
            tmp[dcol] = pd.to_datetime(tmp[dcol], errors="coerce")
            tmp = tmp.dropna(subset=[dcol]).set_index(dcol).sort_index()
            tmp, _, _ = apply_date_range(tmp, start_date, end_date)
            return tmp

        # Last resort: assume first column is date
        if df.shape[1] >= 2:
            maybe_date = df.columns[0]
            tmp = df.copy()
            tmp[maybe_date] = pd.to_datetime(tmp[maybe_date], errors="coerce")
            tmp = tmp.dropna(subset=[maybe_date]).set_index(maybe_date).sort_index()
            tmp, _, _ = apply_date_range(tmp, start_date, end_date)
            return tmp

        raise ValueError("Could not infer date index/column from parquet data.")

    # ---- Directory of parquet files ----
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in directory: {path}")

        parts = []
        for f in files:
            df = _pyarrow_read_parquet_filtered(f, start_date=start_date, end_date=end_date, date_col=date_col)
            if df is None:
                df = pd.read_parquet(f)

            if {date_col, ticker_col, price_col}.issubset(df.columns):
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                wide = df.pivot(index=date_col, columns=ticker_col, values=price_col).sort_index()
                wide, _, _ = apply_date_range(wide, start_date, end_date)
                parts.append(wide)
                continue

            if date_col in df.columns and price_col in df.columns and ticker_col not in df.columns:
                tmp = df[[date_col, price_col]].copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp = tmp.dropna(subset=[date_col]).set_index(date_col).sort_index()
                tmp = tmp.rename(columns={price_col: f.stem})
                tmp, _, _ = apply_date_range(tmp, start_date, end_date)
                parts.append(tmp)
                continue

            if isinstance(df.index, pd.DatetimeIndex):
                tmp = df.copy()
                if tmp.shape[1] == 1:
                    tmp = tmp.rename(columns={tmp.columns[0]: f.stem})
                tmp, _, _ = apply_date_range(tmp.sort_index(), start_date, end_date)
                parts.append(tmp)
                continue

            tmp = _postprocess_wide(df)
            if tmp.shape[1] == 1:
                tmp = tmp.rename(columns={tmp.columns[0]: f.stem})
            parts.append(tmp)

        wide = pd.concat(parts, axis=1).sort_index()
        wide, _, _ = apply_date_range(wide, start_date, end_date)
        return wide

    # ---- Single parquet file ----
    df = _pyarrow_read_parquet_filtered(p, start_date=start_date, end_date=end_date, date_col=date_col)
    if df is None:
        df = pd.read_parquet(path)

    if {date_col, ticker_col, price_col}.issubset(df.columns):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        wide = df.pivot(index=date_col, columns=ticker_col, values=price_col).sort_index()
        wide, _, _ = apply_date_range(wide, start_date, end_date)
        return wide

    if isinstance(df.index, pd.DatetimeIndex):
        wide = df.sort_index()
        wide, _, _ = apply_date_range(wide, start_date, end_date)
        return wide

    wide = _postprocess_wide(df)
    wide, _, _ = apply_date_range(wide, start_date, end_date)
    return wide


def load_prices_from_partitioned_minute_store(
    store_dir: str,
    symbols: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    field: str = "close",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Load *minute* prices from a partitioned parquet store.

    Intended for large universes (e.g., Nifty 500) where a full wide minute matrix is
    impractical. The preprocessing step writes a hive-partitioned store like:

      1min_store/
        symbol=RELIANCE/year=2016/month=3/part-00000.parquet

    This loader uses pyarrow.dataset predicate pushdown on:
      - symbol partition column
      - datetime column inside files

    Returns
    -------
    DataFrame
      Index: datetime
      Columns: symbols
      Values: requested `field` (default: close)
    """
    if symbols is None or len(symbols) == 0:
        raise ValueError("symbols must be a non-empty list")

    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise ImportError(
            "pyarrow is required to load partitioned minute stores. Install pyarrow."
        ) from e

    s = _to_timestamp(start_date)
    e = _to_timestamp(end_date)

    dataset = ds.dataset(str(store_dir), format="parquet", partitioning="hive")

    filt = ds.field("symbol").isin([str(x) for x in symbols])
    if s is not None:
        filt = filt & (ds.field(datetime_col) >= s.to_datetime64())
    if e is not None:
        filt = filt & (ds.field(datetime_col) <= e.to_datetime64())

    cols = [datetime_col, "symbol", field]
    table = dataset.to_table(filter=filt, columns=cols)
    df = table.to_pandas()
    if df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=datetime_col))

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col])
    wide = df.pivot(index=datetime_col, columns="symbol", values=field).sort_index()
    wide, _, _ = apply_date_range(wide, start_date, end_date)
    return wide

def generate_synthetic_prices(n_assets: int = 100,
                              start: str = "2018-01-01",
                              end: str = "2022-12-31",
                              seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic geometric Brownian motion prices for n_assets.
    Returns DataFrame indexed by business dates with columns tick_0 ... tick_{n-1}
    """
    if seed is not None:
        np.random.seed(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    mu = 0.06  # annual drift
    sigma = 0.2  # annual vol
    dt = 1/252
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    S0 = 100 * np.exp(0.5 * np.random.randn(n_assets))
    noise = np.random.randn(n_days, n_assets)
    log_returns = drift + vol * noise
    log_price_paths = np.vstack([np.log(S0), np.cumsum(log_returns, axis=0) + np.log(S0)])
    prices = np.exp(log_price_paths[1:])  # shape (n_days, n_assets)
    cols = [f"tick_{i}" for i in range(n_assets)]
    df = pd.DataFrame(prices, index=dates, columns=cols)
    return df

def load_prices_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    if {'date','ticker','price'}.issubset(df.columns):
        piv = df.pivot(index='date', columns='ticker', values='price')
        piv = piv.sort_index()
        return piv
    else:
        if 'date' in df.columns:
            wide = df.set_index('date')
            wide.index = pd.to_datetime(wide.index)
            return wide.sort_index()
        else:
            raise ValueError("CSV must contain either (date,ticker,price) or date + price columns per asset")

def compute_returns(prices: pd.DataFrame, method: str = 'log', min_valid_obs:int=2) -> pd.DataFrame:
    """
    Compute returns robustly.
    - Treat non-positive prices as NaN (can't take log or pct)
    - Forward-fill small gaps (ffill), but avoid filling long missing tails
    - Returns a DataFrame of returns (index shorter by one row for log/simple)
    """
    # copy to avoid mutating original
    p = prices.copy()
    # replace non-positive values with NaN
    mask_nonpos = (p <= 0)
    if mask_nonpos.any().any():
        p = p.mask(mask_nonpos, other=pd.NA)

    # forward-fill short gaps (limit can be tuned); here we fill up to 3 consecutive missing minutes/days
    #p = p.fillna(method='ffill', limit=3)
    p = p.ffill(limit=3)

    if method == 'log':
        # drop remaining non-positive / NaN
        p = p.dropna(how='all')
        # if after ffill some columns still have NaN at first row, we will later drop them
        # compute log prices then diff
        with np.errstate(divide='ignore', invalid='ignore'):
            logp = np.log(p.astype(float))
        rets = logp.diff().dropna(how='all')
    elif method == 'simple':
        p = p.dropna(how='all')
        rets = p.pct_change().dropna(how='all')
    else:
        raise ValueError("method must be 'log' or 'simple'")

    # drop columns that are entirely NaN in returns
    rets = rets.dropna(axis=1, how='all')
    return rets

def cov_matrix(prices: pd.DataFrame, method: str='log', use_gpu: bool=False, min_obs:int=5) -> pd.DataFrame:
    """
    Compute sample covariance matrix robustly.
    - Uses compute_returns() which handles zeros/NaNs.
    - Requires at least `min_obs` return observations; otherwise returns tiny-diagonal.
    - Ensures the covariance is symmetric and finite before returning.
    """
    # compute returns using robust helper
    rets = compute_returns(prices, method=method)
    # If too few observations, return tiny diagonal
    if rets.shape[0] < min_obs or rets.shape[1] == 0:
        eps = 1e-8
        cols = prices.columns.tolist()
        cov_np = np.eye(len(cols)) * eps
        return pd.DataFrame(cov_np, index=cols, columns=cols)

    # Use numpy to compute covariance (always return numpy-backed DataFrame)
    arr = rets.values.astype(float)  # shape (T-1, N_valid)
    # align columns
    cols = rets.columns.tolist()
    # demean
    arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    # compute covariance (pairwise) with nan-safe handling
    # using np.nanmean and masked multiplication
    T = arr.shape[0]
    # if T <= 1 fallback
    if T < 2:
        eps = 1e-8
        cov_np = np.eye(len(cols)) * eps
        return pd.DataFrame(cov_np, index=cols, columns=cols)

    cov_np = (arr.T @ arr) / float(T - 1)

    # force symmetry numerically
    cov_np = 0.5 * (cov_np + cov_np.T)

    # clean non-finite values
    if not np.isfinite(cov_np).all():
        # replace non-finite with small diag eps
        eps = 1e-8
        # set diagonal to finite if possible
        diag = np.diag(cov_np)
        diag = np.where(np.isfinite(diag) & (diag > 0), diag, eps)
        cov_np = np.nan_to_num(cov_np, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(cov_np, diag)

    # final symmetry pass
    cov_np = 0.5 * (cov_np + cov_np.T)

    return pd.DataFrame(cov_np, index=cols, columns=cols)
