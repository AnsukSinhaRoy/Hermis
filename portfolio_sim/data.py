# portfolio_sim/data.py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from .backend import get_backend, to_numpy

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
