# portfolio_sim/viz.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

def _ensure_series(x: pd.Series) -> pd.Series:
    """
    Ensure x is a pandas Series. If x is a DataFrame with one column, return that column.
    If it's a DataFrame with multiple columns, raise an informative error.
    """
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        else:
            # If DataFrame has multiple columns, attempt to find a common 'value' column
            if 'value' in x.columns and x.shape[1] == 1:
                return x['value']
            raise ValueError("Expected nav to be a Series or single-column DataFrame; got DataFrame with multiple columns.")
    raise TypeError("nav must be a pandas Series or DataFrame")

def plot_nav(nav: pd.Series, title: str = "Portfolio NAV", save_path: str = None):
    nav = _ensure_series(nav)
    plt.figure(figsize=(10,5))
    plt.plot(nav.index, nav.values, label='Portfolio NAV', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('NAV')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_allocation_heatmap(weights_df: pd.DataFrame, title: str="Allocation over time", top_n: int = 20, save_path: str = None):
    top_assets = weights_df.mean().sort_values(ascending=False).head(top_n).index
    sub = weights_df[top_assets]
    plt.figure(figsize=(12,6))
    plt.imshow(sub.T, aspect='auto', interpolation='nearest')
    plt.yticks(range(len(top_assets)), top_assets)
    plt.xticks([0, len(sub)//2, len(sub)-1],
               [sub.index[0].date(), sub.index[len(sub)//2].date(), sub.index[-1].date()])
    plt.title(title)
    plt.xlabel('Time')
    plt.colorbar(label='Weight')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_prices_and_nav(prices: pd.DataFrame,
                        nav,
                        normalize_prices: bool = True,
                        sample_max_assets: int = None,
                        alpha: float = 0.4,
                        figsize: tuple = (14,6),
                        save_path: str = None):
    """
    Plot price series (or sampled subset) normalized to 1 at start (if normalize_prices=True),
    and overlay the portfolio NAV on the same axis.

    - prices: DataFrame indexed by date, columns = tickers
    - nav: Series or single-column DataFrame indexed by date
    """
    nav = _ensure_series(nav)

    # Align indices: reindex nav to prices index if needed (forward/backward fill as reasonable)
    prices_idx = prices.index
    nav_aligned = nav.reindex(prices_idx).ffill().bfill()

    # Optionally sample a subset for readability/performance
    tickers = list(prices.columns)
    n_assets = len(tickers)
    if sample_max_assets is not None and sample_max_assets < n_assets:
        # deterministic sampling for reproducibility: take first N sorted tickers
        tickers = sorted(tickers)[:sample_max_assets]

    plot_prices = prices[tickers].copy()

    # Normalize prices to start at 1 so NAV and prices have comparable scales
    if normalize_prices:
        # first valid (non-NaN) value for each column
        def _first_non_na(col):
            non_na = col.dropna()
            return non_na.iloc[0] if non_na.shape[0] > 0 else np.nan
        first_vals = plot_prices.apply(_first_non_na)
        first_vals = first_vals.replace(0, np.nan)
        plot_prices = plot_prices.divide(first_vals, axis=1)

    plt.figure(figsize=figsize)
    # plot each asset as faint line
    for col in plot_prices.columns:
        plt.plot(plot_prices.index, plot_prices[col].values, linewidth=0.8, alpha=alpha, label="_nolegend_")

    # Plot median price path as a faint thicker line to give a sense of center
    try:
        median_path = plot_prices.median(axis=1)
        plt.plot(plot_prices.index, median_path.values, color='gray', linewidth=1.0, alpha=0.6, label='Median asset')
    except Exception:
        pass

    # Overlay NAV prominently
    nav_norm = nav_aligned.copy()
    if normalize_prices:
        if nav_norm.shape[0] == 0:
            # nothing to plot for NAV
            pass
        else:
            # make sure nav_norm is numeric series and has a scalar first element
            first_nav = nav_norm.iloc[0]
            # first_nav might be a scalar (float) or a 0-d array; coerce to float safely
            try:
                first_nav_val = float(first_nav)
            except Exception:
                # fallback: compute as nav_norm.values[0] then float
                first_nav_val = float(nav_norm.values[0])
            if first_nav_val != 0:
                nav_norm = nav_norm / first_nav_val

    plt.plot(nav_norm.index, nav_norm.values, color='red', linewidth=2.2, label='Portfolio NAV')

    plt.xlabel('Date')
    plt.ylabel('Normalized Price / NAV (start = 1)')
    plt.title('Asset Price Paths (normalized) and Portfolio NAV')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.2)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
