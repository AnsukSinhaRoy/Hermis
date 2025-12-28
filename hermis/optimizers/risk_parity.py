from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def risk_parity_optimize(
    Sigma: pd.DataFrame,
    tol: float = 1e-6,
    max_iter: int = 2000,
    init_w: Optional[pd.Series] = None,
    long_only: bool = True,
    box: Optional[dict] = None,
    **kwargs,
) -> Dict:
    """Numerical Risk Parity / Equal Risk Contribution solver (iterative)."""
    assets = list(Sigma.index)
    n = len(assets)
    if n == 0:
        return {"weights": None, "status": "no_assets"}

    S = Sigma.reindex(index=assets, columns=assets).fillna(0.0).values.astype(float)

    if init_w is None:
        w = np.ones(n) / n
    else:
        w = init_w.reindex(assets).fillna(0.0).values
        if w.sum() <= 0:
            w = np.ones(n) / n
        else:
            w = w / w.sum()

    for _ in range(int(max_iter)):
        M = S.dot(w)
        RC = w * M
        RC_sum = RC.sum()
        if RC_sum == 0:
            break
        target = RC_sum / n

        eps = 1e-12
        M_safe = np.where(np.isfinite(M) & (M > 0), M, eps)
        tgt_safe = float(target) if np.isfinite(target) and target > 0 else eps
        factors = np.sqrt(M_safe / (tgt_safe + eps))
        factors = np.where(np.isfinite(factors) & (factors > 0), factors, 1.0)

        w_new = w / factors

        if long_only:
            w_new = np.maximum(w_new, 0.0)
        if box is not None:
            low = box.get("min", None)
            high = box.get("max", None)
            if low is not None:
                w_new = np.maximum(w_new, float(low))
            if high is not None:
                w_new = np.minimum(w_new, float(high))

        if w_new.sum() <= 0:
            w_new = np.ones(n) / n
        w_new = w_new / w_new.sum()

        if np.max(np.abs(w_new - w)) < float(tol):
            w = w_new
            break
        w = w_new

    M = S.dot(w)
    RC = w * M
    rc_mean = RC.mean() if RC.size > 0 else 0.0
    rc_err = float(np.linalg.norm(RC - rc_mean) / (np.linalg.norm(RC) + 1e-12))

    return {"weights": pd.Series(w, index=assets), "status": "ok", "erc_error": rc_err}
