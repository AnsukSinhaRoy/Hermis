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
    """Robust backtest with optional cash asset support.

    Core behaviors (same as before):
      - Rebalances according to config['rebalance'] frequency.
      - Correctly handles portfolio weight drift between rebalances.
      - Falls back to equal-weight when necessary and holds position if all else fails.
      - Records opt_status in trades for diagnostics.

    Cash behaviors (NEW, opt-in via config['cash']):
      - Adds a constant-price cash column (default ticker: 'CASH') if not present.
      - Treats any unallocated weight as cash (so sum(weights)=1 including cash).
      - Optional risky exposure cap via cash.max_gross_exposure (<= 1.0).
      - Optional drawdown brake via cash.drawdown_brake (see below).
    """
    rebalance_freq = config.get('rebalance', 'monthly')
    lookback = config.get('lookback', None)

    tc = config.get('transaction_costs', {}).get('proportional', 0.0)
    record_non_rebalance = bool(config.get('record_non_rebalance', False))

    # --- Cash config ---
    cash_cfg = config.get('cash', {}) if isinstance(config.get('cash', {}), dict) else {}
    cash_enabled = bool(cash_cfg.get('enabled', cash_cfg.get('use_cash', False)))
    cash_name = str(cash_cfg.get('asset_name', cash_cfg.get('cash_asset_name', cash_cfg.get('ticker', 'CASH')))).strip() or "CASH"

    # Ensure cash exists in prices so that downstream analytics (weights * returns) works.
    if cash_enabled and cash_name not in prices.columns:
        prices = prices.copy()
        prices[cash_name] = 1.0

    dates = prices.index
    total_steps = int(len(dates))
    if total_steps <= 0:
        raise ValueError("prices has no rows; cannot run backtest")

    # Get rebalance dates as a DatetimeIndex (do NOT convert to set)
    rebalance_dates = pd.DatetimeIndex(get_rebalance_dates(prices, rebalance_freq)).normalize()

    if logger:
        try:
            logger.info(
                "Backtest starting | steps=%s | rebalance=%s | tc(proportional)=%s | record_non_rebalance=%s | cash=%s(%s)",
                total_steps, rebalance_freq, tc, record_non_rebalance, cash_enabled, cash_name,
            )
            logger.info(
                "Date range: %s -> %s | assets=%s",
                str(dates.min()), str(dates.max()), prices.shape[1]
            )
        except Exception:
            pass

    # --- Data structures to store results ---
    nav_series = pd.Series(index=dates, dtype=float)
    weights_df = pd.DataFrame(index=dates, columns=prices.columns, dtype=float)
    turnover_series = pd.Series(index=dates, dtype=float)
    trades_list: List[Dict[str, Any]] = []

    # --- State variables for the loop ---
    current_weights = pd.Series(0.0, index=prices.columns, dtype=float)
    if cash_enabled and cash_name in current_weights.index:
        current_weights.loc[cash_name] = 1.0

    nav = 1.0
    peak_nav = nav

    def _apply_cash_overlay(w: pd.Series) -> pd.Series:
        """Sanitize weights, enforce cash residual, apply exposure caps."""
        w = w.reindex(prices.columns).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
        w = w.clip(lower=0.0)

        if not cash_enabled or cash_name not in w.index:
            # Keep legacy behavior (ensure weights sum > 0 then normalize).
            s = float(w.sum())
            if not np.isfinite(s) or s <= 0:
                return w * 0.0
            return w / s

        risky_cols = [c for c in w.index if c != cash_name]
        risky = w.reindex(risky_cols).fillna(0.0).astype(float)

        # If optimizer included an explicit cash weight, ignore it and treat cash as residual.
        risky_sum = float(risky.sum())

        # Static cap
        max_exp = float(cash_cfg.get('max_gross_exposure', cash_cfg.get('max_exposure', 1.0)))
        if not np.isfinite(max_exp):
            max_exp = 1.0
        max_exp = max(0.0, min(1.0, max_exp))

        # Drawdown brake (dynamic cap)
        dd_cfg = cash_cfg.get('drawdown_brake', {}) if isinstance(cash_cfg.get('drawdown_brake', {}), dict) else {}
        if bool(dd_cfg.get('enabled', False)):
            try:
                threshold = float(dd_cfg.get('threshold', 0.10))
                reduced_exp = float(dd_cfg.get('reduced_exposure', dd_cfg.get('exposure', 0.50)))
                threshold = max(0.0, min(1.0, threshold))
                reduced_exp = max(0.0, min(1.0, reduced_exp))

                dd = (float(nav) / float(peak_nav)) - 1.0 if peak_nav > 0 else 0.0
                if dd <= -threshold:
                    max_exp = min(max_exp, reduced_exp)
            except Exception:
                pass

        # If the optimizer wants more risky exposure than allowed, scale down proportionally.
        if risky_sum > max_exp + 1e-12 and risky_sum > 0:
            risky = risky * (max_exp / risky_sum)
            risky_sum = float(risky.sum())

        # Cash is the residual
        cash_w = 1.0 - risky_sum
        cash_w = float(max(0.0, min(1.0, cash_w)))

        out = pd.Series(0.0, index=prices.columns, dtype=float)
        out.loc[risky_cols] = risky
        out.loc[cash_name] = cash_w

        # Final sanity: ensure sum=1 by topping up cash / renormalizing if needed
        s_all = float(out.sum())
        if np.isfinite(s_all) and s_all < 1.0 - 1e-8:
            out.loc[cash_name] += (1.0 - s_all)
        elif np.isfinite(s_all) and s_all > 1.0 + 1e-8 and s_all > 0:
            out = out / s_all
        return out

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
            daily_returns = daily_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)

            # portfolio return using previous weights (cash return is 0 if cash price is constant)
            portfolio_return = float((current_weights * daily_returns).sum())
            nav *= (1.0 + portfolio_return)

            # update peak NAV for drawdown-aware cash overlays
            if nav > peak_nav:
                peak_nav = nav

            # update weights by market drift
            new_weights_drifted = current_weights * (1.0 + daily_returns)
            total_portfolio_value = float(new_weights_drifted.sum())
            if total_portfolio_value > 1e-12:
                current_weights = new_weights_drifted / total_portfolio_value
            else:
                current_weights[:] = 0.0
                if cash_enabled and cash_name in current_weights.index:
                    current_weights.loc[cash_name] = 1.0

        # normalize the loop date for membership testing
        date_norm = pd.Timestamp(date).normalize()

        # --- Rebalancing Logic ---
        if date_norm in rebalance_dates:
            day_did_rebalance = True

            past_prices = prices.loc[:date]
            # optional rolling lookback window (shared by estimators + optimizer context)
            if lookback is not None:
                try:
                    lb = int(lookback)
                    if lb > 0:
                        past_prices = past_prices.tail(max(lb, 2))
                except Exception:
                    pass

            # IMPORTANT: estimators/optimizer should not see cash
            past_prices_risky = past_prices.drop(columns=[cash_name], errors='ignore') if cash_enabled else past_prices

            expected_returns = expected_return_estimator(past_prices_risky)
            covariance = cov_estimator(past_prices_risky)

            target_weights = None
            opt_status = "ok"

            # prefer Index.intersection to keep ordering / dtype
            common_assets = expected_returns.index.intersection(covariance.index)
            if len(common_assets) == 0:
                opt_status = "skipped_no_common_tickers"
            else:
                try:
                    er = expected_returns.reindex(common_assets)
                    cov = covariance.loc[common_assets, common_assets]

                    opt_result = optimizer_func(er, cov, prices_window=past_prices_risky, date=date, prev_date=prev_date)
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
                valid_assets = past_prices_risky.loc[date].dropna().index.tolist() if date in past_prices_risky.index else []
                if valid_assets:
                    ew = 1.0 / len(valid_assets)
                    target_weights = pd.Series(ew, index=valid_assets)
                else:
                    opt_status = "hold_no_valid_assets"
                    target_weights = current_weights.drop(labels=[cash_name], errors='ignore').copy() if cash_enabled else current_weights.copy()

            # --- Apply Transaction Costs and Update Weights ---
            if target_weights is not None:
                # Bring weights onto full universe and apply cash overlay
                target_weights = _apply_cash_overlay(target_weights)

                if cash_enabled and cash_name in prices.columns:
                    turnover_assets = [c for c in prices.columns if c != cash_name]
                    turnover = float(np.abs(target_weights.reindex(turnover_assets) - current_weights.reindex(turnover_assets)).sum())
                else:
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
                    "cash_weight": float(current_weights.get(cash_name, 0.0)) if cash_enabled else 0.0,
                })

                turnover_series.loc[date] = turnover

                day_opt_status = opt_status
                day_turnover = turnover
                day_cost = cost
        else:
            turnover_series.loc[date] = 0.0
            if record_non_rebalance:
                trades_list.append({
                    "date": date,
                    "turnover": 0.0,
                    "cost": 0.0,
                    "opt_status": "no_rebalance",
                    "selected": list(current_weights[current_weights > 1e-6].index),
                    "cash_weight": float(current_weights.get(cash_name, 0.0)) if cash_enabled else 0.0,
                })

        # --- Store daily results ---
        nav_series.loc[date] = nav
        weights_df.loc[date] = current_weights

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
                if logger:
                    logger.debug("step_callback failed (ignored)", exc_info=True)

    trades_df = pd.DataFrame(trades_list)
    return BacktestResult(nav=nav_series, weights=weights_df, turnover=turnover_series, trades=trades_df)
