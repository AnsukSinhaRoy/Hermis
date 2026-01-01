import numpy as np
import pandas as pd


def _project_to_box_simplex(v: np.ndarray, low: float, high: float, target_sum: float = 1.0, iters: int = 80) -> np.ndarray:
    """Project vector v onto {w | sum w = target_sum, low <= w_i <= high}."""
    v = np.asarray(v, dtype=float)
    n = v.size
    if n == 0:
        return v

    # Feasibility guards
    low = float(low)
    high = float(high)
    if low > high:
        low, high = high, low

    if n * low > target_sum:
        low = target_sum / n
    if n * high < target_sum:
        high = target_sum / n
    if low > high:
        low = high = target_sum / n

    # Bisection on lambda for clipped affine shift
    # f(lam) = sum(clip(v - lam, low, high)) - target_sum is monotone decreasing in lam
    lo = np.min(v - high)
    hi = np.max(v - low)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        w = np.clip(v - mid, low, high)
        s = w.sum()
        if s > target_sum:
            lo = mid
        else:
            hi = mid

    w = np.clip(v - hi, low, high)
    # final tiny renormalization (keeps within bounds)
    s = w.sum()
    if s != 0:
        w *= (target_sum / s)
        w = np.clip(w, low, high)
        # correct any drift again with one more bisection step
        s2 = w.sum()
        if abs(s2 - target_sum) > 1e-10:
            # re-run quickly
            lo = np.min(v - high)
            hi = np.max(v - low)
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                w = np.clip(v - mid, low, high)
                s = w.sum()
                if s > target_sum:
                    lo = mid
                else:
                    hi = mid
            w = np.clip(v - hi, low, high)
    return w


def greedy_simplex_from_scores(
    scores: pd.Series,
    fallback_k: int,
    weight_power: float,
    epsilon: float,
    box: dict,
    long_only: bool = True,
):
    """Greedy EMA trend allocation from a per-asset trend score.

    Selection:
      - If any score > 0: select all bullish assets (score > 0)
      - Else: select top `fallback_k` assets with highest score (least bearish)

    Weighting:
      - Bullish set: w ∝ score^weight_power
      - All-bearish fallback: w ∝ (score - min(score) + epsilon)^weight_power

    Returns a pd.Series over the selected assets (sums to 1), or None if inputs invalid.
    """
    if scores is None:
        return None

    scores = pd.Series(scores).replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        return None

    try:
        fallback_k = int(fallback_k)
    except Exception:
        fallback_k = 5
    fallback_k = max(1, fallback_k)

    weight_power = float(weight_power) if weight_power is not None else 1.0
    weight_power = max(0.0, weight_power)

    epsilon = float(epsilon) if epsilon is not None else 1e-12
    epsilon = max(1e-18, epsilon)

    bullish = scores[scores > 0]

    if len(bullish) > 0:
        selected = bullish.sort_values(ascending=False)
        base = np.maximum(selected.values.astype(float), epsilon)
    else:
        selected = scores.sort_values(ascending=False).head(min(fallback_k, len(scores)))
        shifted = (selected - float(selected.min())) + epsilon
        base = np.maximum(shifted.values.astype(float), epsilon)

    raw = np.power(base, weight_power)
    if raw.size == 0 or not np.isfinite(raw).all():
        raw = np.ones(len(selected), dtype=float)

    s = float(raw.sum())
    if not np.isfinite(s) or s <= 0:
        raw = np.ones(len(selected), dtype=float)
        s = float(raw.sum())

    v = raw / s  # initial simplex point (long-only)

    # Constraints
    low = 0.0
    high = 1.0
    if box is not None and isinstance(box, dict):
        low = box.get("min", low)
        high = box.get("max", high)
    try:
        low = float(low) if low is not None else 0.0
    except Exception:
        low = 0.0
    try:
        high = float(high) if high is not None else 1.0
    except Exception:
        high = 1.0

    if long_only:
        low = max(0.0, low)

    v = _project_to_box_simplex(v, low=low, high=high, target_sum=1.0)

    w = pd.Series(v, index=selected.index)
    return w
