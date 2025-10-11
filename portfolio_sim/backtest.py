# portfolio_sim/backtest.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any, Callable, List
from datetime import timedelta

# replace get_rebalance_dates in portfolio_sim/backtest.py with this
def get_rebalance_dates(prices: pd.DataFrame, freq: str='monthly') -> pd.DatetimeIndex:
    """
    Return rebalance dates as a pandas.DatetimeIndex (sorted, unique).
    - supported freq: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    - fallback: treat freq as a pandas offset alias (e.g., '10D')
    """
    idx = prices.index
    if freq == 'daily':
        picks = idx
    elif freq == 'weekly':
        weeks = idx.to_period('W')
        picks = idx[weeks != weeks.shift(1)]
    elif freq == 'monthly':
        months = idx.to_period('M')
        picks = idx[months != months.shift(1)]
    elif freq == 'quarterly':
        q = idx.to_period('Q')
        picks = idx[q != q.shift(1)]
    elif freq == 'yearly':
        y = idx.to_period('Y')
        picks = idx[y != y.shift(1)]
    else:
        # fallback to pandas resample-asfreq approach for offsets like '10D'
        try:
            tmp = pd.DataFrame(index=idx)
            picks = tmp.resample(freq).asfreq().dropna().index
        except Exception:
            # final fallback: monthly boundaries
            months = idx.to_period('M')
            picks = idx[months != months.shift(1)]

    # Normalize and return clean DatetimeIndex
    picks = pd.DatetimeIndex(pd.Series(picks).drop_duplicates().sort_values().values)
    return picks


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

    # Get rebalance dates as a DatetimeIndex (do NOT convert to set)
    rebalance_dates = pd.DatetimeIndex(get_rebalance_dates(prices, rebalance_freq))
    # Normalize both sides to avoid tz / midnight mismatches
    rebalance_dates = rebalance_dates.normalize()

    # whether to record non-rebalance day entries (useful for debugging)
    record_non_rebalance = config.get('record_non_rebalance', False)

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
        if i > 0:
            with np.errstate(invalid='ignore', divide='ignore'):
                daily_returns = (prices.loc[date] / prices.loc[prev_date]) - 1.0
            daily_returns = daily_returns.fillna(0.0)

            # portfolio return using previous weights
            portfolio_return = (current_weights * daily_returns).sum()
            nav *= (1.0 + portfolio_return)

            # update weights by market drift
            new_weights_drifted = current_weights * (1 + daily_returns)
            total_portfolio_value = new_weights_drifted.sum()
            if total_portfolio_value > 1e-8:
                current_weights = new_weights_drifted / total_portfolio_value
            else:
                current_weights[:] = 0.0

        # normalize the loop date for membership testing
        date_norm = pd.Timestamp(date).normalize()

        # --- Rebalancing Logic ---
        if date_norm in rebalance_dates:
            past_prices = prices.loc[:date]
            expected_returns = expected_return_estimator(past_prices)
            covariance = cov_estimator(past_prices)

            target_weights = None
            opt_status = "ok"

            # prefer Index.intersection to keep ordering / dtype
            common_assets = expected_returns.index.intersection(covariance.index)
            if len(common_assets) == 0:
                opt_status = "skipped_no_common_tickers"
            else:
                try:
                    # pass aligned inputs to optimizer
                    er = expected_returns.reindex(common_assets)
                    cov = covariance.loc[common_assets, common_assets]
                    opt_result = optimizer_func(er, cov)

                    # robustly extract weights from optimizer result
                    target_weights = opt_result.get('weights') if isinstance(opt_result, dict) else None
                    opt_status = opt_result.get('status', 'ok') if isinstance(opt_result, dict) else "ok"

                    # coerce to pd.Series if needed
                    if target_weights is not None:
                        if isinstance(target_weights, dict):
                            target_weights = pd.Series(target_weights)
                        elif isinstance(target_weights, np.ndarray):
                            target_weights = pd.Series(target_weights, index=common_assets)
                        elif isinstance(target_weights, pd.Series):
                            # ensure index alignment
                            target_weights = target_weights.reindex(common_assets)
                        else:
                            # final try: attempt to build series from it
                            try:
                                target_weights = pd.Series(target_weights, index=common_assets)
                            except Exception:
                                target_weights = None
                                opt_status = "optimizer_return_unusable"
                except Exception as e:
                    target_weights = None
                    opt_status = f"optimizer_failed:{repr(e)}"

            # Fallback to equal weight if optimizer fails or returns None
            if target_weights is None:
                opt_status = opt_status if opt_status != "ok" else "fallback_ew"
                valid_assets = prices.loc[date].dropna().index.tolist()
                if valid_assets:
                    ew = 1.0 / len(valid_assets)
                    target_weights = pd.Series(ew, index=valid_assets)
                else:
                    opt_status = "hold_no_valid_assets"
                    target_weights = current_weights.copy()

            # --- Apply Transaction Costs and Update Weights ---
            if target_weights is not None:
                # Align with main price columns before calculating turnover
                target_weights = target_weights.reindex(prices.columns).fillna(0.0)

                # Calculate turnover
                turnover = float(np.abs(target_weights - current_weights).sum())

                # Do not charge transaction costs on the very first day (i==0)
                cost = turnover * tc if i > 0 else 0.0

                nav *= (1.0 - cost)  # Apply cost to NAV

                # Update to the new target weights
                current_weights = target_weights.copy()

                # Log the trade
                trades_list.append({
                    "date": date, "turnover": turnover, "cost": cost, "opt_status": opt_status,
                    "selected": list(current_weights[current_weights > 1e-6].index)
                })
                turnover_series.loc[date] = turnover
        else:
            # not a rebalance day
            turnover_series.loc[date] = 0.0
            if record_non_rebalance:
                trades_list.append({
                    "date": date, "turnover": 0.0, "cost": 0.0, "opt_status": "no_rebalance",
                    "selected": list(current_weights[current_weights > 1e-6].index)
                })

        # --- Store daily results ---
        nav_series.loc[date] = nav
        weights_df.loc[date] = current_weights

    # --- Finalize and return ---
    trades_df = pd.DataFrame(trades_list)
    return BacktestResult(nav=nav_series, weights=weights_df, turnover=turnover_series, trades=trades_df)
