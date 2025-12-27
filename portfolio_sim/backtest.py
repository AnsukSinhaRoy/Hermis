# portfolio_sim/backtest.py
import logging
from typing import Dict, Optional, Any, Callable, List

import numpy as np
import pandas as pd


def get_rebalance_dates(prices: pd.DataFrame, freq: str = "monthly") -> pd.DatetimeIndex:
    """
    Return rebalance dates as a pandas.DatetimeIndex (sorted, unique).

    Supported `freq`:
      - 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    """
    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
    idx = pd.DatetimeIndex(idx).dropna().sort_values()

    if len(idx) == 0:
        return pd.DatetimeIndex([])

    if freq == "daily":
        picks = idx

    elif freq in {"weekly", "monthly", "quarterly", "yearly"}:
        # Group by the period label and take the FIRST available timestamp in each period
        period_code = {"weekly": "W", "monthly": "M", "quarterly": "Q", "yearly": "Y"}[freq]
        period_labels = idx.to_period(period_code)
        picks = (
            pd.Series(idx)
            .groupby(period_labels)
            .min()
            .sort_values()
            .values
        )

    else:
        # Treat freq as a pandas offset alias (e.g., '10D')
        try:
            tmp = pd.Series(1.0, index=idx)
            picks = tmp.resample(freq).first().dropna().index
        except Exception:
            # fallback monthly
            period_labels = idx.to_period("M")
            picks = (
                pd.Series(idx)
                .groupby(period_labels)
                .min()
                .sort_values()
                .values
            )

    return pd.DatetimeIndex(pd.Series(picks).drop_duplicates().sort_values().values)



class BacktestResult:
    def __init__(self, nav: pd.Series, weights: pd.DataFrame, turnover: pd.Series, trades: pd.DataFrame):
        self.nav = nav
        self.weights = weights
        self.turnover = turnover
        self.trades = trades


def run_backtest(
    prices: pd.DataFrame,
    expected_return_estimator: Callable[[pd.DataFrame], pd.Series],
    cov_estimator: Callable[[pd.DataFrame], pd.DataFrame],
    optimizer_func: Callable[[pd.Series, pd.DataFrame], Dict[str, Any]],
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    step_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> BacktestResult:
    """
    Robust backtest that:
      - Rebalances according to config['rebalance'] frequency.
      - Correctly handles portfolio weight drift between rebalances.
      - Falls back to equal-weight when necessary and holds position if all else fails.
      - Records opt_status in trades for diagnostics.

    New (for observability):
      - `logger` for detailed diagnostics.
      - `step_callback(step_idx_1_based, total_steps, info_dict)` to report progress.
    """
    rebalance_freq = config.get('rebalance', 'monthly')
    lookback = config.get('lookback', None)

    tc = config.get('transaction_costs', {}).get('proportional', 0.0)
    record_non_rebalance = bool(config.get('record_non_rebalance', False))

    dates = prices.index
    total_steps = int(len(dates))
    if total_steps <= 0:
        raise ValueError("prices has no rows; cannot run backtest")

    # Get rebalance dates as a DatetimeIndex (do NOT convert to set)
    rebalance_dates = pd.DatetimeIndex(get_rebalance_dates(prices, rebalance_freq)).normalize()

    if logger:
        try:
            logger.info(
                "Backtest starting | steps=%s | rebalance=%s | tc(proportional)=%s | record_non_rebalance=%s",
                total_steps, rebalance_freq, tc, record_non_rebalance,
            )
            logger.info(
                "Date range: %s -> %s | assets=%s",
                str(dates.min()), str(dates.max()), prices.shape[1]
            )
            logger.info(
                "Rebalance dates computed: %s (first=%s last=%s)",
                len(rebalance_dates),
                str(rebalance_dates.min()) if len(rebalance_dates) else None,
                str(rebalance_dates.max()) if len(rebalance_dates) else None,
            )
        except Exception:
            # logging should never crash the backtest
            pass

    # --- Data structures to store results ---
    nav_series = pd.Series(index=dates, dtype=float)
    weights_df = pd.DataFrame(index=dates, columns=prices.columns, dtype=float)
    turnover_series = pd.Series(index=dates, dtype=float)
    trades_list: List[Dict[str, Any]] = []

    # --- State variables for the loop ---
    current_weights = pd.Series(0.0, index=prices.columns)
    nav = 1.0

    # --- Main backtest loop ---
    for i, date in enumerate(dates):
        prev_date = dates[i - 1] if i > 0 else None

        # Track day-level diagnostics for the step_callback
        day_opt_status = "no_rebalance"
        day_did_rebalance = False
        day_turnover = 0.0
        day_cost = 0.0

        # --- Daily Weight Drift Calculation ---
        if i > 0:
            with np.errstate(invalid='ignore', divide='ignore'):
                daily_returns = (prices.loc[date] / prices.loc[prev_date]) - 1.0
            daily_returns = daily_returns.fillna(0.0)

            # portfolio return using previous weights
            portfolio_return = float((current_weights * daily_returns).sum())
            nav *= (1.0 + portfolio_return)

            # update weights by market drift
            new_weights_drifted = current_weights * (1.0 + daily_returns)
            total_portfolio_value = float(new_weights_drifted.sum())
            if total_portfolio_value > 1e-8:
                current_weights = new_weights_drifted / total_portfolio_value
            else:
                current_weights[:] = 0.0

        # normalize the loop date for membership testing
        date_norm = pd.Timestamp(date).normalize()

        # --- Rebalancing Logic ---
        if date_norm in rebalance_dates:
            day_did_rebalance = True

            if logger:
                logger.debug("Rebalance day %s (i=%s/%s)", str(date), i + 1, total_steps)

            past_prices = prices.loc[:date]
            # optional rolling lookback window (shared by estimators + optimizer context)
            if lookback is not None:
                try:
                    lb = int(lookback)
                    if lb > 0:
                        # ensure we keep at least 2 rows so online optimizers can compute price relatives
                        past_prices = past_prices.tail(max(lb, 2))
                except Exception:
                    pass

            expected_returns = expected_return_estimator(past_prices)
            covariance = cov_estimator(past_prices)

            target_weights = None
            opt_status = "ok"

            # prefer Index.intersection to keep ordering / dtype
            common_assets = expected_returns.index.intersection(covariance.index)
            if len(common_assets) == 0:
                opt_status = "skipped_no_common_tickers"
                if logger:
                    logger.warning(
                        "No common tickers between expected_returns (%s) and covariance (%s) on %s",
                        len(expected_returns.index), len(covariance.index), str(date)
                    )
            else:
                try:
                    er = expected_returns.reindex(common_assets)
                    cov = covariance.loc[common_assets, common_assets]

                    opt_result = optimizer_func(er, cov, prices_window=past_prices, date=date, prev_date=prev_date)
                    target_weights = opt_result.get('weights') if isinstance(opt_result, dict) else None
                    opt_status = opt_result.get('status', 'ok') if isinstance(opt_result, dict) else "ok"

                    # coerce to pd.Series if needed
                    if target_weights is not None:
                        if isinstance(target_weights, dict):
                            target_weights = pd.Series(target_weights)
                        elif isinstance(target_weights, np.ndarray):
                            target_weights = pd.Series(target_weights, index=common_assets)
                        elif isinstance(target_weights, pd.Series):
                            target_weights = target_weights.reindex(common_assets)
                        else:
                            try:
                                target_weights = pd.Series(target_weights, index=common_assets)
                            except Exception:
                                target_weights = None
                                opt_status = "optimizer_return_unusable"
                except Exception as e:
                    target_weights = None
                    opt_status = f"optimizer_failed:{repr(e)}"
                    if logger:
                        logger.exception("Optimizer exception on %s", str(date))

            # Fallback to equal weight if optimizer fails or returns None
            if target_weights is None:
                opt_status = opt_status if opt_status != "ok" else "fallback_ew"
                valid_assets = prices.loc[date].dropna().index.tolist()
                if valid_assets:
                    ew = 1.0 / len(valid_assets)
                    target_weights = pd.Series(ew, index=valid_assets)
                    if logger and opt_status != "ok":
                        logger.warning(
                            "Using fallback equal-weight on %s | opt_status=%s | n_valid_assets=%s",
                            str(date), opt_status, len(valid_assets)
                        )
                else:
                    opt_status = "hold_no_valid_assets"
                    target_weights = current_weights.copy()
                    if logger:
                        logger.error(
                            "No valid assets available on %s; holding existing weights",
                            str(date)
                        )

            # --- Apply Transaction Costs and Update Weights ---
            if target_weights is not None:
                target_weights = target_weights.reindex(prices.columns).fillna(0.0)

                turnover = float(np.abs(target_weights - current_weights).sum())
                cost = float(turnover * tc) if i > 0 else 0.0

                nav *= (1.0 - cost)
                current_weights = target_weights.copy()

                # Log the trade
                selected_assets = list(current_weights[current_weights > 1e-6].index)
                trades_list.append({
                    "date": date,
                    "turnover": turnover,
                    "cost": cost,
                    "opt_status": opt_status,
                    "selected": selected_assets,
                })

                turnover_series.loc[date] = turnover

                # stash per-day info for step_callback
                day_opt_status = opt_status
                day_turnover = turnover
                day_cost = cost

                if logger:
                    logger.debug(
                        "Rebalance applied | date=%s | opt_status=%s | turnover=%.6f | cost=%.6f | selected=%s",
                        str(date), opt_status, turnover, cost, len(selected_assets)
                    )
        else:
            # not a rebalance day
            turnover_series.loc[date] = 0.0
            if record_non_rebalance:
                trades_list.append({
                    "date": date,
                    "turnover": 0.0,
                    "cost": 0.0,
                    "opt_status": "no_rebalance",
                    "selected": list(current_weights[current_weights > 1e-6].index),
                })

        # --- Store daily results ---
        nav_series.loc[date] = nav
        weights_df.loc[date] = current_weights

        # --- Progress callback (called every step; caller can choose to log only every 1%) ---
        if step_callback is not None:
            try:
                step_callback(
                    i + 1,
                    total_steps,
                    {
                        "date": date,
                        "nav": nav,
                        "did_rebalance": day_did_rebalance,
                        "opt_status": day_opt_status,
                        "turnover": day_turnover,
                        "cost": day_cost,
                    },
                )
            except Exception:
                # callbacks must never crash the backtest
                if logger:
                    logger.debug("step_callback failed (ignored)", exc_info=True)

    trades_df = pd.DataFrame(trades_list)
    return BacktestResult(nav=nav_series, weights=weights_df, turnover=turnover_series, trades=trades_df)
