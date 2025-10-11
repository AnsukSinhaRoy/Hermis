# portfolio_sim/backtest.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any, Callable, List
from datetime import timedelta

def get_rebalance_dates(prices: pd.DataFrame, freq: str='monthly') -> List[pd.Timestamp]:
    """
    Return rebalance dates (as a list) for the given frequency.
    - 'daily', 'weekly', 'monthly', 'quarterly', 'yearly' supported.
    - This returns actual index values from the prices DataFrame.
    """
    idx = prices.index
    if freq == 'daily':
        return list(idx)
    if freq == 'weekly':
        # pick last trading date of each week
        weeks = idx.to_period('W')
        picks = idx[weeks != weeks.shift(1)].tolist()
        return picks
    if freq == 'monthly':
        months = idx.to_period('M')
        picks = idx[months != months.shift(1)].tolist()
        return picks
    if freq == 'quarterly':
        q = idx.to_period('Q')
        picks = idx[q != q.shift(1)].tolist()
        return picks
    if freq == 'yearly':
        y = idx.to_period('Y')
        picks = idx[y != y.shift(1)].tolist()
        return picks
    # fallback: treat freq as pandas offset alias e.g., '10D' -> use resample endpoints
    try:
        # build a masked series using resample to get endpoints
        # create a temporary df with index=prices.index
        tmp = pd.DataFrame(index=idx)
        picks = tmp.resample(freq).asfreq().dropna().index.tolist()
        if picks:
            return picks
    except Exception:
        pass
    # default to monthly if unknown
    months = idx.to_period('M')
    return idx[months != months.shift(1)].tolist()

class BacktestResult:
    def __init__(self, nav: pd.Series, weights: pd.DataFrame, turnover: pd.Series, trades: pd.DataFrame):
        self.nav = nav
        self.weights = weights
        self.turnover = turnover
        self.trades = trades

def run_backtest(prices: pd.DataFrame,
                 expected_return_estimator: Callable[[pd.DataFrame], pd.Series],
                 cov_estimator: Callable[[pd.DataFrame], pd.DataFrame],
                 optimizer_func: Callable[[pd.Series, pd.DataFrame], Dict[str,Any]],
                 config: Dict[str,Any]) -> BacktestResult:
    """
    Robust backtest that:
      - Rebalances according to config['rebalance'] frequency.
      - Skips rebalances if expected returns or covariance are missing/empty.
      - Falls back to single-asset or equal-weight when necessary.
      - Records opt_status in trades for diagnostics.
    """
    rebalance = config.get('rebalance', 'monthly')
    tc = config.get('transaction_costs', {}).get('proportional', 0.0)
    dates = prices.index
    reb_dates = set(get_rebalance_dates(prices, rebalance))
    nav = pd.Series(index=dates, dtype=float)
    weights_df = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    turnover_series = pd.Series(0.0, index=dates)
    trades_list: List[Dict] = []
    current_weights = pd.Series(0.0, index=prices.columns)
    nav_val = 1.0

    for i, date in enumerate(dates):
        if i == 0:
            nav.iloc[0] = nav_val

        if date in reb_dates:
            past = prices.loc[:date]
            expected = expected_return_estimator(past)
            cov = cov_estimator(past)

            # Quick validation of estimators
            if (expected is None) or (cov is None):
                trades_list.append({
                    "date": date,
                    "turnover": 0.0,
                    "cost": 0.0,
                    "opt_status": "skipped_no_estimators",
                    "selected": []
                })
                weights_df.loc[date] = current_weights
                turnover_series.loc[date] = 0.0
                continue

            # Align expected & cov (common tickers)
            common = [t for t in expected.index if t in cov.index]
            if len(common) == 0:
                trades_list.append({
                    "date": date,
                    "turnover": 0.0,
                    "cost": 0.0,
                    "opt_status": "skipped_no_common_tickers",
                    "selected": []
                })
                weights_df.loc[date] = current_weights
                turnover_series.loc[date] = 0.0
                continue

            expected_r = expected.reindex(common).dropna()
            cov_r = cov.reindex(index=common, columns=common).fillna(0.0)

            if expected_r.shape[0] == 0 or cov_r.shape[0] == 0:
                trades_list.append({
                    "date": date,
                    "turnover": 0.0,
                    "cost": 0.0,
                    "opt_status": "skipped_insufficient_data",
                    "selected": []
                })
                weights_df.loc[date] = current_weights
                turnover_series.loc[date] = 0.0
                continue

            # Single-asset fallback
            if expected_r.shape[0] == 1:
                only = expected_r.index[0]
                new_w = pd.Series(0.0, index=prices.columns)
                new_w[only] = 1.0
                turnover = np.abs(new_w - current_weights).sum()
                cost = turnover * tc
                nav_val = nav_val * (1.0 - cost)
                nav.loc[date] = nav_val
                current_weights = new_w.copy()
                trades_list.append({
                    "date": date,
                    "turnover": float(turnover),
                    "cost": float(cost),
                    "opt_status": "single_asset_fallback",
                    "selected": [only]
                })
                weights_df.loc[date] = current_weights
                turnover_series.loc[date] = turnover
                continue

            # Try optimizer; fallback on failure
            try:
                opt = optimizer_func(expected_r, cov_r)
                new_w = opt.get('weights', None)
                opt_status = opt.get('status', 'ok')
                selected = opt.get('selected', list(new_w[new_w>0].index) if (new_w is not None) else [])
                if new_w is None:
                    # equal-weight fallback among common
                    k = len(common)
                    ew = pd.Series(0.0, index=prices.columns)
                    ew.loc[common] = 1.0 / k
                    new_w = ew
                    opt_status = "fallback_equal_no_weights"
            except Exception as e:
                # fallback equal-weight or keep previous
                try:
                    k = len(common)
                    ew = pd.Series(0.0, index=prices.columns)
                    ew.loc[common] = 1.0 / k
                    new_w = ew
                    opt_status = f"optimizer_failed:{repr(e)}"
                    selected = list(common)
                except Exception:
                    new_w = current_weights.copy()
                    opt_status = f"optimizer_failed_and_fallback_failed:{repr(e)}"
                    selected = list(current_weights[current_weights>0].index)

            # Ensure new_w is full-length and numeric
            new_w = new_w.reindex(prices.columns).fillna(0.0).astype(float)

            turnover = float(np.abs(new_w - current_weights).sum())
            cost = turnover * tc
            nav_val = nav_val * (1.0 - cost)
            nav.loc[date] = nav_val
            current_weights = new_w.copy()
            trades_list.append({
                "date": date,
                "turnover": float(turnover),
                "cost": float(cost),
                "opt_status": opt_status,
                "selected": selected
            })
            weights_df.loc[date] = current_weights
            turnover_series.loc[date] = turnover

        else:
            # update NAV using returns; handle NaNs gracefully
            if i > 0:
                prev = dates[i-1]
                # compute simple returns for this day across assets
                with np.errstate(invalid='ignore', divide='ignore'):
                    ret = (prices.loc[date] / prices.loc[prev]) - 1.0
                ret = ret.fillna(0.0)
                port_ret = (current_weights * ret).sum()
                nav_val = nav_val * (1.0 + port_ret)
                nav.iloc[i] = nav_val
            else:
                nav.iloc[i] = nav_val

    # forward fill weights for days between rebalances, fill nan as 0 after ffilling
    weights_df = weights_df.replace(0.0, np.nan).ffill().fillna(0.0)
    trades_df = pd.DataFrame(trades_list)
    return BacktestResult(nav=nav, weights=weights_df, turnover=turnover_series, trades=trades_df)
