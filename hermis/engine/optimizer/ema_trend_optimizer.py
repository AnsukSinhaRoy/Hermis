import pandas as pd

from engine.analytics.ema import ema
from engine.optimizer.ema_greedy import greedy_simplex_from_scores


class EMATrendOptimizer:
    """State-less EMA trend optimizer helper.

    NOTE: In the Hermis architecture, the backtester owns the rolling window and
    passes `prices_window` into the optimizer wrapper. This class is kept for
    convenience/testing, but the recommended integration is:
      - compute EMA trend score from the provided `prices_window` (up to prev day)
      - call `ema_trend_optimize` (which calls `greedy_simplex_from_scores`)
    """

    def __init__(self, cfg: dict):
        ema_cfg = cfg.get("ema", {}) if isinstance(cfg.get("ema", {}), dict) else {}
        self.fast_span = int(ema_cfg.get("fast_span", 12))
        self.slow_span = int(ema_cfg.get("slow_span", 26))
        self.fast_field = str(ema_cfg.get("fast_price_field", "close"))
        self.slow_field = str(ema_cfg.get("slow_price_field", "close"))
        self.fallback_k = int(ema_cfg.get("fallback_k", 5))
        self.weight_power = float(ema_cfg.get("weight_power", 1.0))
        self.epsilon = float(ema_cfg.get("epsilon", 1e-12))
        self.box = cfg.get("box", {"min": 0.0, "max": 1.0})
        self.long_only = bool(cfg.get("long_only", True))

    def compute_weights(self, price_window: pd.DataFrame):
        """Compute weights from a price window.

        Parameters
        ----------
        price_window : pd.DataFrame
            Indexed by date, columns are tickers. Contains close-only prices.

        Returns
        -------
        pd.Series or None
            Weights on selected tickers (sum to 1), or None on insufficient data.
        """
        if price_window is None or not isinstance(price_window, pd.DataFrame):
            return None

        # Use data up to the previous day to avoid lookahead.
        pw = price_window.iloc[:-1]
        if len(pw) < max(self.fast_span, self.slow_span) + 1:
            return None

        if (self.fast_field != "close") or (self.slow_field != "close"):
            # Current dataset is close-only; callers may still set these fields.
            # Fall back to close.
            pass

        prev_close = pw.iloc[-1]
        ema_fast = pw.apply(lambda s: ema(s, self.fast_span).iloc[-1])
        ema_slow = pw.apply(lambda s: ema(s, self.slow_span).iloc[-1])

        scores = (ema_fast - ema_slow) / prev_close

        return greedy_simplex_from_scores(
            scores=scores,
            fallback_k=self.fallback_k,
            weight_power=self.weight_power,
            epsilon=self.epsilon,
            box=self.box,
            long_only=self.long_only,
        )
