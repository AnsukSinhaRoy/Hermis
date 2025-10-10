# portfolio_sim/backtest.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any, Callable, List
from datetime import timedelta

def get_rebalance_dates(prices: pd.DataFrame, freq: str='monthly'):
    idx = prices.index
    if freq == 'daily':
        return list(idx)
    if freq == 'weekly':
        # pick last business day of each week
        return list(idx[::5])
    if freq == 'monthly':
        # choose month starts (first available business day)
        s = pd.Series(idx)
        months = s.dt.to_period('M')
        picks = s[months != months.shift(1)].tolist()
        return picks
    raise ValueError("Unsupported rebalance frequency")

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
    prices: DataFrame of prices (dates x assets)
    expected_return_estimator, cov_estimator: functions that accept price-window and return estimates
    optimizer_func: function(expected_returns, cov) -> dict with 'weights'
    config: dict with 'rebalance' and 'transaction_costs'
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
        # if first day, initialize nav
        if i == 0:
            nav.iloc[0] = nav_val
        if date in reb_dates:
            past = prices.loc[:date]
            expected = expected_return_estimator(past)
            cov = cov_estimator(past)
            opt = optimizer_func(expected, cov)
            new_w = opt['weights'].reindex(prices.columns).fillna(0.0)
            turnover = np.abs(new_w - current_weights).sum()
            cost = turnover * tc
            nav_val = nav_val * (1.0 - cost)
            nav.iloc[i] = nav_val
            current_weights = new_w.copy()
            trades_list.append({
                "date": date,
                "turnover": float(turnover),
                "cost": float(cost),
                "opt_status": opt.get('status'),
                "selected": list(current_weights[current_weights>0].index)
            })
            weights_df.loc[date] = current_weights
            turnover_series.loc[date] = turnover
        else:
            # update NAV via price returns using last known weights
            if i > 0:
                prev = dates[i-1]
                ret = (prices.loc[date] / prices.loc[prev]) - 1.0
                port_ret = (current_weights * ret).sum()
                nav_val = nav_val * (1.0 + port_ret)
                nav.iloc[i] = nav_val
            else:
                nav.iloc[i] = nav_val
    # forward-fill weights
    weights_df = weights_df.replace(0.0, np.nan).ffill().fillna(0.0)
    trades_df = pd.DataFrame(trades_list)
    return BacktestResult(nav=nav, weights=weights_df, turnover=turnover_series, trades=trades_df)
