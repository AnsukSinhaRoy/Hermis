
# Hermis_Prism/engine/utils.py
from typing import Any, Optional, Dict
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def ensure_series(data: Any) -> Optional[pd.Series]:
    if data is None:
        return None
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    try:
        return pd.Series(data)
    except Exception:
        return None


def ann_stats(nav_s: pd.Series, rf_annual: float = 0.0) -> Dict[str, float]:
    """Basic annualized stats for UI fallback (when precomputed metrics aren't present)."""
    nav = ensure_series(nav_s)
    if nav is None or nav.empty or len(nav) < 2:
        return {"total_return": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    nav = nav.dropna().astype(float).sort_index()
    days = (nav.index[-1] - nav.index[0]).days or 1
    total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    ann_ret = float((1 + total_ret) ** (365.0 / days) - 1.0)

    dr = nav.pct_change().dropna()
    ann_vol = float(dr.std(ddof=0) * np.sqrt(TRADING_DAYS)) if not dr.empty else 0.0

    roll_max = nav.cummax()
    drawdown = (nav - roll_max) / roll_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    # Sharpe: annualized return / annualized vol (rough)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }


def create_calendar_heatmap(nav_s: pd.Series) -> pd.DataFrame:
    nav = ensure_series(nav_s)
    if nav is None or nav.empty:
        return pd.DataFrame()
    returns = nav.pct_change().dropna()
    returns.name = 'returns'
    res = returns.reset_index()
    res['year'] = res['index'].dt.year
    res['month'] = res['index'].dt.month
    monthly_returns = res.groupby(['year', 'month'])['returns'].apply(lambda x: (1 + x).prod() - 1)
    heatmap = monthly_returns.unstack(level='month')
    yearly_returns = res.groupby('year')['returns'].apply(lambda x: (1 + x).prod() - 1)
    heatmap['Year'] = yearly_returns
    month_names = {i: pd.to_datetime(i, format='%m').strftime('%b') for i in range(1, 13)}
    heatmap = heatmap.rename(columns=month_names)
    return heatmap
