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
        # pick first trading date of each week
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
      - Correctly handles portfolio weight drift between rebalances.
      - Falls back to equal-weight when necessary and holds position if all else fails.
      - Records opt_status in trades for diagnostics.
    """
    rebalance_freq = config.get('rebalance', 'monthly')
    tc = config.get('transaction_costs', {}).get('proportional', 0.0)
    dates = prices.index
    rebalance_dates = set(get_rebalance_dates(prices, rebalance_freq))

    # --- Data structures to store results ---
    nav_series = pd.Series(index=dates, dtype=float)
    weights_df = pd.DataFrame(index=dates, columns=prices.columns, dtype=float)
    turnover_series = pd.Series(index=dates, dtype=float)
    trades_list: List[Dict] = []

    # --- State variables for the loop ---
    current_weights = pd.Series(0.0, index=prices.columns)
    nav = 1.0

    # --- Main backtest loop ---
    for i, date in enumerate(dates):
        prev_date = dates[i-1] if i > 0 else None

        # --- Daily Weight Drift Calculation ---
        # Before any rebalancing, update weights based on last day's returns.
        if i > 0:
            # Calculate daily returns
            with np.errstate(invalid='ignore', divide='ignore'):
                daily_returns = (prices.loc[date] / prices.loc[prev_date]) - 1.0
            daily_returns = daily_returns.fillna(0.0)

            # Calculate portfolio return based on *previous day's* weights
            portfolio_return = (current_weights * daily_returns).sum()
            nav *= (1.0 + portfolio_return)

            # Update weights due to market drift
            # Numerator: weight * (1 + asset_return)
            new_weights_drifted = current_weights * (1 + daily_returns)
            # Denominator: 1 + portfolio_return
            total_portfolio_value = new_weights_drifted.sum()
            
            # The new weights are the proportion of the new total value
            if total_portfolio_value > 1e-8:
                current_weights = new_weights_drifted / total_portfolio_value
            else:
                # If portfolio value collapsed, weights go to zero
                current_weights.values[:] = 0.0

        # --- Rebalancing Logic ---
        if date in rebalance_dates:
            past_prices = prices.loc[:date]
            expected_returns = expected_return_estimator(past_prices)
            covariance = cov_estimator(past_prices)

            target_weights = None
            opt_status = "ok"

            # Align inputs and run optimizer
            common_assets = list(set(expected_returns.index) & set(covariance.index))
            if not common_assets:
                opt_status = "skipped_no_common_tickers"
            else:
                try:
                    opt_result = optimizer_func(expected_returns[common_assets], covariance.loc[common_assets, common_assets])
                    target_weights = opt_result.get('weights')
                    opt_status = opt_result.get('status', 'ok')
                except Exception as e:
                    opt_status = f"optimizer_failed:{repr(e)}"

            # Fallback to equal weight if optimizer fails
            if target_weights is None:
                opt_status = opt_status if opt_status != "ok" else "fallback_ew"
                valid_assets = prices.loc[date].dropna().index.tolist()
                if valid_assets:
                    ew = 1.0 / len(valid_assets)
                    target_weights = pd.Series(ew, index=valid_assets)
                else: # If no valid assets, hold position
                    opt_status = "hold_no_valid_assets"
                    target_weights = current_weights

            # --- Apply Transaction Costs and Update Weights ---
            if target_weights is not None:
                # Align with main price columns before calculating turnover
                target_weights = target_weights.reindex(prices.columns).fillna(0.0)
                
                # Calculate turnover
                turnover = float(np.abs(target_weights - current_weights).sum())

                # FIX: Do not charge transaction costs on the very first day (i==0)
                cost = turnover * tc if i > 0 else 0.0
                
                nav *= (1.0 - cost) # Apply cost to NAV

                # Update to the new target weights
                current_weights = target_weights.copy()
                
                # Log the trade
                trades_list.append({
                    "date": date, "turnover": turnover, "cost": cost, "opt_status": opt_status,
                    "selected": list(current_weights[current_weights > 1e-6].index)
                })
                turnover_series.loc[date] = turnover

        # --- Store daily results ---
        nav_series.loc[date] = nav
        weights_df.loc[date] = current_weights

    # --- Finalize and return ---
    trades_df = pd.DataFrame(trades_list)
    return BacktestResult(nav=nav_series, weights=weights_df, turnover=turnover_series, trades=trades_df)