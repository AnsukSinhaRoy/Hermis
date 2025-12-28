import pandas as pd
from engine.optimizer.ema_greedy import greedy_simplex_from_scores

def ema_trend_optimize(
    scores: pd.Series,
    box: dict,
    long_only: bool,
    fallback_k: int,
    weight_power: float,
    epsilon: float,
):
    return greedy_simplex_from_scores(
        scores=scores,
        fallback_k=fallback_k,
        weight_power=weight_power,
        epsilon=epsilon,
        box=box,
        long_only=long_only,
    )
