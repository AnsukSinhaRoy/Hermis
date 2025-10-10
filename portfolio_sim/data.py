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

def compute_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    if method == 'log':
        return np.log(prices).diff().dropna()
    elif method == 'simple':
        return prices.pct_change().dropna()
    else:
        raise ValueError("method must be 'log' or 'simple'")

def cov_matrix(prices: pd.DataFrame, method: str='log', use_gpu: bool=False) -> pd.DataFrame:
    """
    Compute sample covariance matrix using CPU (numpy) or GPU (cupy) backend.
    Returns a pandas DataFrame (numpy) to be compatible with other libs.

    If the available price window has fewer than 2 observations (so returns are empty),
    return a small-diagonal covariance matrix (eps*I) to keep optimizers stable.
    """
    import numpy as _np
    xp, info = get_backend(use_gpu)
    # convert to array on selected backend
    arr = xp.asarray(prices.values, dtype=float)  # shape (T, N)
    T = arr.shape[0]
    cols = prices.columns.tolist()
    if T < 2:
        # Not enough data to compute returns -> return small diagonal covariance
        eps = 1e-8
        cov_np = _np.eye(len(cols)) * eps
        return pd.DataFrame(cov_np, index=cols, columns=cols)

    if method == 'log':
        logp = xp.log(arr)
        rets = logp[1:] - logp[:-1]
    else:
        rets = (arr[1:] - arr[:-1]) / arr[:-1]

    # demean
    # convert to float64 on backend to avoid precision issues
    rets = rets.astype(float)
    mean = rets.mean(axis=0, keepdims=True)
    rets_centered = rets - mean
    n_obs = int(rets_centered.shape[0])
    if n_obs < 2:
        # fallback to tiny-diagonal cov if returns length is too small
        eps = 1e-8
        cov_np = _np.eye(len(cols)) * eps
        return pd.DataFrame(cov_np, index=cols, columns=cols)

    cov = (rets_centered.T @ rets_centered) / (n_obs - 1)
    cov_np = to_numpy(cov)
    return pd.DataFrame(cov_np, index=cols, columns=cols)
