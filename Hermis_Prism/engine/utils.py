# portfolio_viz/utils.py
from typing import Any, Optional, Dict
import numpy as np
import pandas as pd


def ensure_series(data: Any) -> Optional[pd.Series]:
    if data is None:
        return None
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return pd.Series(data)


def ann_stats(nav_s: pd.Series) -> Dict[str, float]:
    if nav_s is None or nav_s.empty:
        return {"total_return": 0, "ann_return": 0, "ann_vol": 0, "max_drawdown": 0}
    days = (nav_s.index[-1] - nav_s.index[0]).days or 1
    total_ret = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    ann_ret = (1 + total_ret) ** (365.0 / days) - 1
    dr = nav_s.pct_change().dropna()
    ann_vol = dr.std() * np.sqrt(252) if not dr.empty else 0.0
    roll_max = nav_s.cummax()
    drawdown = (nav_s - roll_max) / roll_max
    max_dd = drawdown.min()
    return {
        "total_return": total_ret, "ann_return": ann_ret,
        "ann_vol": ann_vol, "max_drawdown": max_dd,
    }


def create_calendar_heatmap(nav_s: pd.Series) -> pd.DataFrame:
    returns = nav_s.pct_change().dropna()
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