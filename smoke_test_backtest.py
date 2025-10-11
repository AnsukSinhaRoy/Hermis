# debug_backtest_step_by_step.py
import pandas as pd
import numpy as np
from pathlib import Path
import time

# --- Replace this 'prices' with your test prices (the same one you used) ---
dates = pd.date_range("2025-01-01", periods=20, freq="D")
prices = pd.DataFrame({
    "A": [100, 101, 102, 101, 103, 105, 106, 107, 109, 108, 110, 112, 113, 111, 114, 116, 117, 115, 118, 120],
    "B": [100,  99,  98,  99,  98,  99, 100, 101, 100,  99, 100, 101, 103, 102, 104, 103, 105, 106, 104, 107]
}, index=dates)
# ------------------------------

def expected_return_estimator(pr_window):
    r = (pr_window / pr_window.shift(1) - 1.0).dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return r.mean()

def cov_estimator(pr_window):
    r = (pr_window / pr_window.shift(1) - 1.0).dropna()
    if r.empty or r.shape[0] < 2:
        return pd.DataFrame(np.diag([1e-8]*pr_window.shape[1]), index=pr_window.columns, columns=pr_window.columns)
    return r.cov().fillna(0.0)

# Trivial optimizer: equal weight on available assets
def optimizer(expected, cov):
    if expected is None or len(expected)==0:
        return {"weights": None, "status": "no_expected"}
    w = pd.Series(1.0/len(expected), index=expected.index)
    return {"weights": w, "status": "equal_weight"}

# Choose rebalance freq: 'daily' for the smoke test
rebalance = 'daily'
tc = 0.0  # transaction cost for test

# helper to compute rebalance dates (simple daily)
reb_dates = list(prices.index) if rebalance == 'daily' else list(prices.index)

# state
dates = prices.index
current_weights = pd.Series(0.0, index=prices.columns)   # initial weights (all zeros)
nav_val = 1.0
nav = pd.Series(index=dates, dtype=float)

print("=== DEBUG BACKTEST STEP-BY-STEP ===")
print("dates:", list(dates))
print("prices:\n", prices)
print()

for i, date in enumerate(dates):
    print(f"--- Date {date.date()} (i={i}) ---")
    # compute returns from prev -> this date (if possible)
    if i == 0:
        prev = None
        ret = pd.Series(0.0, index=prices.columns)
        print("No prior date; initial NAV:", nav_val)
    else:
        prev = dates[i-1]
        ret = (prices.loc[date] / prices.loc[prev]) - 1.0
        # ensure align to columns
        ret = ret.reindex(current_weights.index).fillna(0.0)
        print("Price this date    :", prices.loc[date].to_dict())
        print("Price prev date    :", prices.loc[prev].to_dict())
        print("Asset returns (this/prev -1):", ret.to_dict())

    # portfolio return using current_weights (before rebalance)
    port_ret = float((current_weights * ret).sum())
    print("Current weights (before rebalance):", current_weights.to_dict())
    print(f"Portfolio return (before rebalance): {port_ret:.12f}")
    prev_nav = nav_val
    nav_val = nav_val * (1.0 + port_ret)
    nav.loc[date] = nav_val
    print(f"NAV before: {prev_nav:.6f} -> after applying port_ret: {nav_val:.6f}")

    # Is this a rebalance date?
    is_reb = date in reb_dates
    print("Is rebalance date?:", is_reb)

    if is_reb:
        # compute estimators on history up to this date
        past = prices.loc[:date]
        expected = expected_return_estimator(past)
        cov = cov_estimator(past)
        print("Expected returns estimator:", expected.to_dict() if not expected.empty else {})
        # call optimizer (only on tickers present in expected)
        common = [t for t in expected.index if t in cov.index]
        print("Common tickers between expected and cov:", common)
        if len(common) > 0:
            opt = optimizer(expected.reindex(common), cov.reindex(index=common, columns=common))
            new_w = opt.get('weights', None)
            opt_status = opt.get('status', 'ok')
            if new_w is None:
                # fallback equal
                new_w = pd.Series(0.0, index=prices.columns)
                new_w.loc[common] = 1.0 / len(common)
                opt_status = "fallback_equal"
            # reindex to full asset list
            new_w = new_w.reindex(prices.columns).fillna(0.0).astype(float)
        else:
            new_w = current_weights.copy()
            opt_status = "no_common"
        turnover = float(np.abs(new_w - current_weights).sum())
        cost = turnover * tc
        # apply transaction cost immediately (deduct)
        nav_val = nav_val * (1.0 - cost)
        nav.loc[date] = nav_val
        print(f"Rebalance -> opt_status: {opt_status}, turnover: {turnover:.6f}, cost: {cost:.6f}, NAV after cost: {nav_val:.6f}")
        print("New weights (after rebalance):", new_w.to_dict())
        current_weights = new_w.copy()
    print()

print("=== FINAL NAV SERIES ===")
print(nav)
print("=== FINAL WEIGHTS (last rows) ===")
# show weights as they would be on rebalance days (we printed them each step)
