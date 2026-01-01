import numpy as np
import pandas as pd

from engine.analytics.ema import ema


def _ema_last_from_df(df: pd.DataFrame, span: int) -> pd.Series:
    """Vectorized EMA last value for each column."""
    # pandas ewm is vectorized across columns
    return df.ewm(span=int(span), adjust=False).mean().iloc[-1]


def ema_trend_score(
    price_window: pd.DataFrame,
    fast_span: int,
    slow_span: int,
) -> pd.Series | None:
    """Compute an EMA trend score for each asset.

    Data assumptions:
      - `price_window` contains close-only prices
      - Columns are tickers/assets
      - Index is datetime-like

    Lookahead safety:
      - Uses data up to *previous day* by slicing `iloc[:-1]`.
    """
    if price_window is None or not isinstance(price_window, pd.DataFrame):
        return None

    pw = price_window.iloc[:-1]
    if len(pw) < max(int(fast_span), int(slow_span)) + 1:
        return None

    prev_close = pw.iloc[-1].astype(float)

    # Fast/slow EMA last value per asset
    try:
        ema_fast = _ema_last_from_df(pw.astype(float), int(fast_span))
        ema_slow = _ema_last_from_df(pw.astype(float), int(slow_span))
    except Exception:
        # fallback (slower but robust) if needed
        ema_fast = pw.apply(lambda s: ema(s.astype(float), int(fast_span)).iloc[-1])
        ema_slow = pw.apply(lambda s: ema(s.astype(float), int(slow_span)).iloc[-1])

    with np.errstate(divide='ignore', invalid='ignore'):
        score = (ema_fast - ema_slow) / prev_close

    score = pd.Series(score).replace([np.inf, -np.inf], np.nan)
    return score
