from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .mean_variance import mv_reg_optimize
from .min_variance import min_variance_optimize
from .risk_parity import risk_parity_optimize
from .sharpe import sharpe_optimize


def greedy_k_cardinality(
    expected: pd.Series,
    Sigma: pd.DataFrame,
    k: int,
    method: str = "mv_reg",
    box: Optional[dict] = None,
    long_only: bool = True,
    **kwargs,
) -> Dict:
    """Simple greedy k-cardinality approach.

    - select top-k by score = mu / vol
    - run chosen optimizer on subset
    """
    if expected is None or len(expected) == 0:
        return {"weights": None, "status": "no_expected"}

    assets = expected.dropna().index.tolist()
    if len(assets) == 0:
        return {"weights": None, "status": "no_assets_after_dropna"}

    sigma_diag = pd.Series(
        np.sqrt(np.diag(Sigma.reindex(index=assets, columns=assets).fillna(0.0).values)),
        index=assets,
    )
    score = expected.reindex(assets) / (sigma_diag + 1e-8)
    topk = list(score.sort_values(ascending=False).head(int(k)).index) if int(k) > 0 else assets

    mu_sub = expected.reindex(topk).dropna()
    Sigma_sub = Sigma.reindex(index=topk, columns=topk).fillna(0.0)

    method = str(method).strip().lower()
    if method == "mv_reg":
        return mv_reg_optimize(mu_sub, Sigma_sub, box=box, long_only=long_only, **kwargs)
    if method == "minvar":
        return min_variance_optimize(Sigma_sub, box=box, long_only=long_only, **kwargs)
    if method == "risk_parity":
        return risk_parity_optimize(Sigma_sub, box=box, long_only=long_only, **kwargs)
    if method == "sharpe":
        return sharpe_optimize(mu_sub, Sigma_sub, box=box, long_only=long_only)

    return mv_reg_optimize(mu_sub, Sigma_sub, box=box, long_only=long_only, **kwargs)
