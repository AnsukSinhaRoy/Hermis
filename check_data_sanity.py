import pandas as pd
import numpy as np

# Load the exact prices the run used (or load your parquet directly)
# Example if you load the Nifty500 daily parquet:
from portfolio_sim.data import load_prices_from_parquet
prices = load_prices_from_parquet("data/processed/nifty500/prices_1D_nifty500.parquet")

print("Date range:", prices.index.min(), "->", prices.index.max())
print("Bars:", len(prices), "Assets:", prices.shape[1])

# 1) Underlying single-stock daily return extremes
rets = prices.pct_change()
mx = rets.max().max()
mn = rets.min().min()
print("Max single-stock daily return:", mx)
print("Min single-stock daily return:", mn)

# Show the worst offenders
stacked = rets.stack().dropna()
print("\nTop 10 daily moves:")
print(stacked.sort_values(ascending=False).head(10))
print("\nBottom 10 daily moves:")
print(stacked.sort_values(ascending=True).head(10))

# 2) Missing-history profile (how many valid days per ticker)
valid_days = prices.notna().sum()
print("\nValid-day quantiles:")
print(valid_days.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
