
# Hermis_Prism/engine/analytics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS = 252

def nav_returns(nav: pd.Series) -> pd.Series:
    nav = nav.dropna().astype(float).sort_index()
    return nav.pct_change().dropna()

def max_drawdown(nav: pd.Series) -> Tuple[float, pd.Series]:
    nav = nav.dropna().astype(float).sort_index()
    roll_max = nav.cummax()
    dd = (nav - roll_max) / roll_max
    return float(dd.min()) if not dd.empty else 0.0, dd

def drawdown_duration(dd: pd.Series) -> int:
    # longest consecutive period where drawdown < 0
    if dd is None or dd.empty:
        return 0
    underwater = (dd < 0).astype(int)
    # run-length encoding
    longest = 0
    cur = 0
    for v in underwater.values:
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return int(longest)

def ann_metrics(nav: pd.Series, rf_annual: float = 0.0) -> Dict[str, Any]:
    nav = nav.dropna().astype(float).sort_index()
    if nav.empty or len(nav) < 2:
        return {}

    # time span
    days = max(int((nav.index[-1] - nav.index[0]).days), 1)
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = float((1.0 + total_return) ** (365.0 / days) - 1.0)

    r = nav_returns(nav)
    if r.empty:
        return {"total_return": total_return, "cagr": cagr}

    ann_vol = float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))
    rf_daily = (1.0 + float(rf_annual)) ** (1.0 / TRADING_DAYS) - 1.0
    excess = r - rf_daily

    sharpe = float((excess.mean() * TRADING_DAYS) / (r.std(ddof=0) * np.sqrt(TRADING_DAYS) + 1e-12))
    downside = r[r < 0]
    sortino = float((excess.mean() * TRADING_DAYS) / ((downside.std(ddof=0) * np.sqrt(TRADING_DAYS)) + 1e-12)) if len(downside) else np.nan

    mdd, dd = max_drawdown(nav)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    dd_dur = drawdown_duration(dd)

    hit_rate = float((r > 0).mean())
    skew = float(r.skew()) if len(r) >= 3 else np.nan
    kurt = float(r.kurtosis()) if len(r) >= 4 else np.nan
    best_day = float(r.max())
    worst_day = float(r.min())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(mdd),
        "calmar": calmar,
        "max_dd_duration_days": dd_dur,
        "hit_rate": hit_rate,
        "skew": skew,
        "kurtosis": kurt,
        "best_day": best_day,
        "worst_day": worst_day,
    }

def concentration_stats(weights: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Per-date concentration measures:
    - n_holdings: count of weights > 0
    - hhi: sum(w^2)
    - eff_n: 1 / hhi
    - top1/top5: cumulative weight
    """
    if weights is None or weights.empty:
        return pd.DataFrame()
    w = weights.fillna(0.0).astype(float)
    n_hold = (w > eps).sum(axis=1)
    hhi = (w ** 2).sum(axis=1)
    eff_n = 1.0 / (hhi.replace(0.0, np.nan))
    top1 = w.apply(lambda row: float(row.nlargest(1).sum()), axis=1)
    top5 = w.apply(lambda row: float(row.nlargest(5).sum()), axis=1)
    return pd.DataFrame({"n_holdings": n_hold, "hhi": hhi, "eff_n": eff_n, "top1": top1, "top5": top5})

def pnl_contributions(weights: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute daily asset returns and contribution time series.
    Assumes weights[t-1] applied to returns from t-1 -> t.
    """
    if weights is None or prices is None:
        return {}
    w = weights.copy()
    p = prices.copy()
    if not isinstance(w.index, pd.DatetimeIndex):
        w.index = pd.to_datetime(w.index, errors="coerce")
    if not isinstance(p.index, pd.DatetimeIndex):
        p.index = pd.to_datetime(p.index, errors="coerce")

    w = w.sort_index()
    p = p.sort_index()

    # align on dates
    common_idx = w.index.intersection(p.index)
    w = w.loc[common_idx]
    p = p.loc[common_idx]

    # returns (t-1 -> t)
    r = p.pct_change().fillna(0.0)
    contrib = w.shift(1).fillna(0.0) * r
    port_ret = contrib.sum(axis=1)

    # cumulative contribution in NAV units (starting at 1)
    cum_port = (1.0 + port_ret).cumprod()
    cum_contrib = (1.0 + contrib).cumprod()

    return {
        "asset_returns": r,
        "contrib_returns": contrib,
        "portfolio_return": port_ret,
        "cum_portfolio": cum_port,
        "cum_contrib": cum_contrib,
    }

def parse_params_yaml(params_path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(params_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
