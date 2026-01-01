#!/usr/bin/env python3
"""
run_experiment.py

Run an experiment using portfolio_sim runner_func, ensure the exact prices
used are saved into the experiment folder, and precompute performance
artifacts to make the Streamlit app fast to load.

NEW (observability + usability):
- Writes logs into the experiment folder's built-in `logs/` directory (e.g. `<timestamp>__<name>/logs/`) with:
    - run.log (detailed structured logging)
    - stdout_stderr.log (only used for --detach runs)
    - failure.traceback.txt (written on failure)
    - run_metadata.json
    - params.yaml config snapshot (copied into the experiment folder)
- Logs progress every 1% of backtest steps.
- Supports `--detach` to free your terminal immediately.
- Accepts both `--config` and `--configs`.

Examples:
  python run_experiment.py --config configs/newconfig.yaml
  python run_experiment.py --config configs/newconfig.yaml --detach

NOTE:
  Convenience: `python run_experiment.py --configs/newconfig.yaml` is accepted and
  treated as `--config configs/newconfig.yaml` (missing space).
  Recommended: `python run_experiment.py --config configs/newconfig.yaml`
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from engine.optimizer.ema_trend import ema_trend_optimize
from engine.analytics.ema_signals import ema_trend_score
# portfolio_sim imports (local package)
from portfolio_sim.data import (
    generate_synthetic_prices,
    load_prices_from_csv,
    load_prices_from_parquet,
    load_prices_from_partitioned_minute_store,
    apply_date_range,
    compute_returns,
    cov_matrix,
)
from portfolio_sim.optimizer import (
    greedy_k_cardinality,
    mv_reg_optimize,
    min_variance_optimize,
    risk_parity_optimize,
    sharpe_optimize,
    omd_step,
    ftrl_step,
    _damped_newton_projected,
)
from portfolio_sim.backtest import run_backtest
from portfolio_sim.experiment import (
    run_experiment_from_config,
    load_experiment,
    make_experiment_folder,
    save_params_yaml,
    resolve_experiment_name,
)
from portfolio_sim import viz

from utils.run_logging import (
    RunPaths,
    setup_logging,
    write_metadata,
    PercentProgressLogger,
    log_failure,
)


def _infer_bars_per_day(idx: pd.DatetimeIndex) -> int:
    """Infer typical bars-per-day from an index.

    - For daily data, returns 1.
    - For intraday minute-ish data, returns the median number of rows per calendar day.
    """
    try:
        if idx is None or len(idx) < 3:
            return 1
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        idx = pd.DatetimeIndex(idx).dropna().sort_values()
        # Detect daily vs intraday by median spacing
        dt = np.median(np.diff(idx.view('i8')))
        if not np.isfinite(dt):
            return 1
        # 23h in ns
        if dt >= (23 * 3600 * 1_000_000_000):
            return 1
        counts = pd.Series(1, index=idx).groupby(idx.normalize()).sum()
        if len(counts) == 0:
            return 1
        bpd = int(max(1, round(float(counts.median()))))
        return bpd
    except Exception:
        return 1


def expected_return_estimator(prices_window: pd.DataFrame):
    """Legacy estimator (kept for backwards-compat): mean of log returns per bar."""
    rets = compute_returns(prices_window, method='log')
    if rets is None or rets.shape[0] == 0:
        return pd.Series(dtype=float)
    return rets.mean()


def cov_estimator_factory(use_gpu: bool = False):
    """Return a callable that computes covariance for a rolling window."""

    def cov_estimator(prices_window: pd.DataFrame):
        return cov_matrix(prices_window, method='log', use_gpu=use_gpu)

    return cov_estimator


def optimizer_wrapper_factory_from_cfg(cfg, logger=None):
    """Build an optimizer function from the YAML config.

    Key promise:
      - Changing `optimizer.type` in the config is sufficient to switch optimizers.

    The returned callable is tolerant of extra context kwargs (e.g. `prices_window`)
    so the backtester can pass useful information without breaking old optimizers.
    """

    opt_cfg = cfg.get('optimizer', {}) or {}

    raw_type = str(opt_cfg.get('type', 'mv_reg')).strip().lower()
    aliases = {
        # mean-variance
        'mv': 'mv_reg',
        'mean_variance': 'mv_reg',
        'mean-variance': 'mv_reg',
        'greedy': 'mv_reg',  # historical configs used "greedy" to mean "mv_reg + k_cardinality"

        # min-variance
        'minvar': 'minvar',
        'min_variance': 'minvar',
        'minimum_variance': 'minvar',
        'gmv': 'minvar',

        # others
        'erc': 'risk_parity',
        'riskparity': 'risk_parity',
        'max_sharpe': 'sharpe',
        'entropy': 'entropy_newton',
        'damped_newton': 'entropy_newton',
        'exp_grad': 'omd',
        'exponentiated_gradient': 'omd',

        # EMA trend
        'ema': 'ema_trend',
        'ema_trend': 'ema_trend',
        'ema_filter': 'ema_hybrid',
        'ema_hybrid': 'ema_hybrid',
        'ema_mv_reg': 'ema_hybrid',
        'ema_online': 'ema_online',
        'ema_multi': 'ema_online',
        'ema_multiwindow': 'ema_online',
        'ema_online_opt': 'ema_online',
        'eg': 'omd',
    }
    opt_type = aliases.get(raw_type, raw_type)

    # common knobs
    k = opt_cfg.get('k_cardinality', None)
    try:
        k = int(k) if k is not None else None
    except Exception:
        k = None

    box_cfg = opt_cfg.get('box', None)
    box = None
    if isinstance(box_cfg, dict):
        box = {"min": box_cfg.get("min", None), "max": box_cfg.get("max", None)}

    long_only = bool(opt_cfg.get('long_only', True))

    # mean-variance reg knobs
    lambda_reg = opt_cfg.get('lambda_reg', opt_cfg.get('lambda', 1.0))
    lambdas = opt_cfg.get('lambdas', None)
    solver = opt_cfg.get('solver', None)

    # risk parity knobs
    rp_cfg = opt_cfg.get('risk_parity', {}) if isinstance(opt_cfg.get('risk_parity', {}), dict) else {}
    rp_tol = float(rp_cfg.get('tol', opt_cfg.get('tol', 1e-6)))
    rp_max_iter = int(rp_cfg.get('max_iter', opt_cfg.get('max_iter', 2000)))

    # entropy-newton knobs (existing)
    p_avg = float(opt_cfg.get('p_avg', 0.0))
    lam = float(opt_cfg.get('lam', lambda_reg))
    K = float(opt_cfg.get('K', 1.0))

    # online optimizer knobs
    v_target = float(opt_cfg.get('v_target', opt_cfg.get('v_tar', 1.01)))

    ftrl_cfg = opt_cfg.get('ftrl', {}) if isinstance(opt_cfg.get('ftrl', {}), dict) else {}
    lambda_2 = float(ftrl_cfg.get('lambda_2', opt_cfg.get('lambda_2', 22.0)))
    gamma = float(ftrl_cfg.get('gamma', opt_cfg.get('gamma', 0.999)))
    ftrl_max_iter = int(ftrl_cfg.get('max_iter', opt_cfg.get('max_iter', 200)))
    ftrl_tol = float(ftrl_cfg.get('tol', opt_cfg.get('tol', 1e-9)))

    omd_cfg = opt_cfg.get('omd', {}) if isinstance(opt_cfg.get('omd', {}), dict) else {}
    eta = float(omd_cfg.get('eta', opt_cfg.get('eta', 0.1)))

    ema_cfg = opt_cfg.get('ema', {}) if isinstance(opt_cfg.get('ema', {}), dict) else {}
    ema_fast_span = int(ema_cfg.get('fast_span', opt_cfg.get('fast_span', 12)))
    ema_slow_span = int(ema_cfg.get('slow_span', opt_cfg.get('slow_span', 26)))
    ema_fallback_k = int(ema_cfg.get('fallback_k', opt_cfg.get('fallback_k', 5)))
    ema_weight_power = float(ema_cfg.get('weight_power', opt_cfg.get('weight_power', 1.0)))
    ema_epsilon = float(ema_cfg.get('epsilon', opt_cfg.get('epsilon', 1e-12)))
    ema_fast_field = str(ema_cfg.get('fast_price_field', opt_cfg.get('fast_price_field', 'close')))
    ema_slow_field = str(ema_cfg.get('slow_price_field', opt_cfg.get('slow_price_field', 'close')))

    ema_post_optimizer = str(ema_cfg.get('post_optimizer', ema_cfg.get('base_optimizer', opt_cfg.get('post_optimizer', 'mv_reg')))).strip().lower()
    ema_mu_source = str(ema_cfg.get('mu_source', opt_cfg.get('mu_source', 'ema_score'))).strip().lower()
    ema_mu_blend_alpha = float(ema_cfg.get('mu_blend_alpha', opt_cfg.get('mu_blend_alpha', 1.0)))
    ema_mu_scale = float(ema_cfg.get('mu_scale', opt_cfg.get('mu_scale', 1.0)))
    ema_bullish_threshold = float(ema_cfg.get('bullish_threshold', opt_cfg.get('bullish_threshold', 0.0)))
    ema_min_assets = ema_cfg.get('min_assets', opt_cfg.get('min_assets', None))
    ema_max_assets = ema_cfg.get('max_assets', opt_cfg.get('max_assets', None))
    try:
        ema_min_assets = int(ema_min_assets) if ema_min_assets is not None else None
    except Exception:
        ema_min_assets = None
    try:
        ema_max_assets = int(ema_max_assets) if ema_max_assets is not None else None
    except Exception:
        ema_max_assets = None

    # EMA-online (multi-window) configuration (separate key; falls back to ema.* if absent)
    emao_cfg = opt_cfg.get('ema_online', ema_cfg) if isinstance(opt_cfg.get('ema_online', ema_cfg), dict) else ema_cfg
    # windows: list of {fast_span, slow_span, weight}
    emao_windows = emao_cfg.get('windows', emao_cfg.get('window_pairs', None))
    if not isinstance(emao_windows, list) or len(emao_windows) == 0:
        # sensible defaults across time-scales
        emao_windows = [
            {"fast_span": 8, "slow_span": 21, "weight": 1.0},
            {"fast_span": 21, "slow_span": 55, "weight": 1.0},
            {"fast_span": 55, "slow_span": 144, "weight": 0.7},
        ]

    # Units: bars (default), minutes, hours, days
    emao_windows_unit = str(emao_cfg.get('windows_unit', emao_cfg.get('unit', 'bars'))).strip().lower()
    emao_score_smooth_span = emao_cfg.get('score_smooth_span', emao_cfg.get('score_smoothing', None))
    try:
        emao_score_smooth_span = int(emao_score_smooth_span) if emao_score_smooth_span is not None else None
    except Exception:
        emao_score_smooth_span = None

    emao_bullish_threshold = float(emao_cfg.get('bullish_threshold', ema_bullish_threshold))
    emao_min_assets = emao_cfg.get('min_assets', ema_min_assets)
    emao_max_assets = emao_cfg.get('max_assets', ema_max_assets)
    emao_fallback_k = int(emao_cfg.get('fallback_k', ema_fallback_k))

    emao_base_optimizer = str(emao_cfg.get('base_optimizer', emao_cfg.get('post_optimizer', ema_post_optimizer))).strip().lower()
    emao_mu_source = str(emao_cfg.get('mu_source', ema_mu_source)).strip().lower()
    emao_mu_blend_alpha = float(emao_cfg.get('mu_blend_alpha', ema_mu_blend_alpha))
    emao_mu_scale = float(emao_cfg.get('mu_scale', ema_mu_scale))
    emao_risk_blend = emao_cfg.get('risk_blend', emao_cfg.get('risk_parity_blend', None))

    # Ranking: optionally blend EMA score with recent return/volatility to bias selection toward high-return assets.
    emao_rank_mode = str(emao_cfg.get('rank_mode', emao_cfg.get('selection', 'score'))).strip().lower()
    emao_rank_unit = str(emao_cfg.get('rank_unit', 'bars')).strip().lower()
    emao_rank_ret_lb = emao_cfg.get('rank_return_lookback', emao_cfg.get('rank_ret_lookback', None))
    emao_rank_vol_lb = emao_cfg.get('rank_vol_lookback', emao_cfg.get('rank_volatility_lookback', None))
    emao_rank_dd_lb = emao_cfg.get('rank_dd_lookback', emao_cfg.get('rank_drawdown_lookback', None))
    try:
        emao_rank_ret_lb = int(emao_rank_ret_lb) if emao_rank_ret_lb is not None else None
    except Exception:
        emao_rank_ret_lb = None
    try:
        emao_rank_vol_lb = int(emao_rank_vol_lb) if emao_rank_vol_lb is not None else None
    except Exception:
        emao_rank_vol_lb = None
    try:
        emao_rank_dd_lb = int(emao_rank_dd_lb) if emao_rank_dd_lb is not None else None
    except Exception:
        emao_rank_dd_lb = None

    w_rank = emao_cfg.get('rank_weights', {}) if isinstance(emao_cfg.get('rank_weights', {}), dict) else {}
    emao_rank_w_ema = float(w_rank.get('ema', w_rank.get('score', 1.0)))
    emao_rank_w_ret = float(w_rank.get('ret', w_rank.get('return', 0.0)))
    emao_rank_w_vol = float(w_rank.get('vol', w_rank.get('volatility', 0.0)))
    emao_rank_w_dd = float(w_rank.get('dd', w_rank.get('drawdown', 0.0)))
    try:
        emao_risk_blend = float(emao_risk_blend) if emao_risk_blend is not None else None
    except Exception:
        emao_risk_blend = None

    try:
        emao_min_assets = int(emao_min_assets) if emao_min_assets is not None else None
    except Exception:
        emao_min_assets = None
    try:
        emao_max_assets = int(emao_max_assets) if emao_max_assets is not None else None
    except Exception:
        emao_max_assets = None

    if emao_risk_blend is not None:
        emao_risk_blend = max(0.0, min(1.0, float(emao_risk_blend)))

    if logger:
        try:
            logger.info(
                "Optimizer configured | raw_type=%s resolved=%s | k=%s | long_only=%s | box=%s",
                raw_type, opt_type, k, long_only, box,
            )
        except Exception:
            pass

    # state for online optimizers (persist across rebalance calls)
    online_state = {
        'assets': None,  # tuple[str]
        'w': None,       # np.ndarray
        'B': None,       # np.ndarray
        'v': None,       # np.ndarray
    }

    # Separate state for EMA-online when it uses online base optimizers (OMD/FTRL)
    emao_online_state = {
        'assets': None,
        'w': None,
        'B': None,
        'v': None,
    }

    def _select_topk_assets(mu: pd.Series, cov: pd.DataFrame, k_: int):
        assets = mu.dropna().index.tolist()
        if not assets:
            return []
        sigma_diag = pd.Series(np.sqrt(np.diag(cov.reindex(index=assets, columns=assets).fillna(0.0).values)), index=assets)
        score = mu.reindex(assets) / (sigma_diag + 1e-8)
        return list(score.sort_values(ascending=False).head(k_).index)

    def _prices_to_relatives(prices_window: pd.DataFrame, assets: list[str], period_window: pd.DataFrame | None = None):
        """Compute price relatives for online updates.

        - If `period_window` is provided (and has >=2 rows), return last/first (full holding-period).
        - Otherwise, return last/prev (single-step).
        """
        base = period_window if (period_window is not None and isinstance(period_window, pd.DataFrame) and len(period_window) >= 2) else prices_window
        if base is None or not isinstance(base, pd.DataFrame) or len(base) < 2:
            return None
        p_t = base.iloc[-1].astype(float).reindex(assets)
        p_prev = base.iloc[0].astype(float).reindex(assets) if base is period_window else base.iloc[-2].astype(float).reindex(assets)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = p_t / p_prev
        r = r.replace([np.inf, -np.inf], 1.0).fillna(1.0).clip(lower=1e-12)
        return r.values.astype(float)

    def optimizer_func(expected: pd.Series, cov: pd.DataFrame, **ctx):
        """Optimizer hook called by the backtester.

        expected: pd.Series indexed by tickers
        cov: pd.DataFrame indexed/columns by tickers
        ctx: optional context (prices_window/date/etc.)
        """
        if expected is None or cov is None:
            return {"weights": None, "status": "invalid_inputs"}

        assets = list(expected.index)
        if not assets:
            return {"weights": None, "status": "no_assets"}

        common_args = {"box": box, "long_only": long_only}

        # ------------------
        # Online optimizers
        # ------------------
        if opt_type in {"ftrl", "omd"}:
            prices_window = ctx.get('prices_window', None)
            period_window = ctx.get('period_prices_window', None)
            r_t = _prices_to_relatives(prices_window, assets, period_window=period_window)
            if r_t is None:
                # cannot update without at least 2 price rows
                w_ew = pd.Series(1.0 / len(assets), index=assets)
                return {"weights": w_ew, "status": "fallback_equal_insufficient_prices"}

            # init / reset state if universe changes
            if online_state['assets'] != tuple(assets) or online_state['w'] is None:
                n = len(assets)
                online_state['assets'] = tuple(assets)
                online_state['w'] = (np.ones(n) / n)
                online_state['B'] = np.zeros((n, n), dtype=float)
                online_state['v'] = np.zeros(n, dtype=float)

            if opt_type == "omd":
                w_new, info = omd_step(
                    online_state['w'],
                    r_t,
                    eta=eta,
                    v_target=v_target,
                    k_cardinality=k,
                    **common_args,
                )
                online_state['w'] = w_new
                return {"weights": pd.Series(w_new, index=assets), "status": "ok", **info}

            # ftrl
            w_new, B_t, v_t, info, st = ftrl_step(
                online_state['w'],
                r_t,
                online_state['B'],
                online_state['v'],
                lambda_2=lambda_2,
                gamma=gamma,
                v_target=v_target,
                k_cardinality=k,
                **common_args,
                max_iter=ftrl_max_iter,
                tol=ftrl_tol,
            )
            online_state['w'] = w_new
            online_state['B'] = B_t
            online_state['v'] = v_t
            return {"weights": pd.Series(w_new, index=assets), "status": st, **info}

        # ------------------
        # k-cardinality wrapper (subset selection)
        # ------------------
        if k is not None and int(k) > 0 and opt_type not in {'ema_trend','ema_hybrid','ema_online'}:
            if opt_type == "entropy_newton":
                # manual subset selection then entropy solve
                topk = _select_topk_assets(expected, cov, int(k))
                mu_sub = expected.reindex(topk).dropna()
                cov_sub = cov.reindex(index=topk, columns=topk).fillna(0.0)
                # fall through by calling entropy solver below on the subset
                expected_use, cov_use = mu_sub, cov_sub
            else:
                return greedy_k_cardinality(
                    expected,
                    cov,
                    k=int(k),
                    method=opt_type,
                    **common_args,
                    lambda_reg=lambda_reg,
                    lambdas=lambdas,
                    solver=solver,
                )
        else:
            expected_use, cov_use = expected, cov

        # ------------------
        # Classical optimizers
        # ------------------
        if opt_type == "mv_reg":
            return mv_reg_optimize(expected_use, cov_use, **common_args, lambda_reg=lambda_reg, lambdas=lambdas, solver=solver)

        if opt_type == "minvar":
            return min_variance_optimize(cov_use, **common_args, solver=solver)

        if opt_type == "risk_parity":
            return risk_parity_optimize(cov_use, **common_args, tol=rp_tol, max_iter=rp_max_iter)

        if opt_type == "sharpe":
            return sharpe_optimize(expected_use, cov_use, **common_args, solver=solver)

        if opt_type == "entropy_newton":
            try:
                Sigma = cov_use.values if isinstance(cov_use, pd.DataFrame) else np.asarray(cov_use, dtype=float)
                mu = np.asarray(expected_use).ravel().astype(float)
                n = Sigma.shape[0]
                if mu.shape[0] != n:
                    return {"weights": None, "status": "invalid_dims"}

                w0 = np.ones(n) * (K / n)
                w_opt = _damped_newton_projected(
                    w0,
                    Sigma,
                    mu,
                    p_avg,
                    lam,
                    K=K,
                    max_iters=50,
                    tol=1e-8,
                )

                idx = cov_use.columns if isinstance(cov_use, pd.DataFrame) else None
                return {"weights": pd.Series(w_opt, index=idx), "status": "success"}
            except Exception as e:
                return {"weights": None, "status": f"error_entropy_opt:{e}"}

        if opt_type in {"ema_trend", "ema_hybrid", "ema_online"}:
            prices_window = ctx.get("prices_window", None)

            if prices_window is None or not isinstance(prices_window, pd.DataFrame) or len(prices_window) < 3:
                return {"weights": None, "status": "ema_insufficient_window"}

            # Close-only assumption (we keep the field config for forward-compat).
            if (ema_fast_field != "close") or (ema_slow_field != "close"):
                if logger:
                    logger.warning(
                        "EMA strategy configured with non-close fields (fast=%s slow=%s) but prices are close-only; falling back to close.",
                        str(ema_fast_field), str(ema_slow_field)
                    )

            pw = prices_window.reindex(columns=assets)
            # avoid nonsense / log issues
            pw = pw.where(pw > 0)

            def _trend_score_single(df: pd.DataFrame, fast: int, slow: int, smooth_span: int | None = None) -> Optional[pd.Series]:
                if df is None or not isinstance(df, pd.DataFrame):
                    return None
                if len(df) < max(int(fast), int(slow)) + 2:
                    return None
                x = df.astype(float)
                ema_f = x.ewm(span=int(fast), adjust=False).mean()
                ema_s = x.ewm(span=int(slow), adjust=False).mean()
                with np.errstate(divide='ignore', invalid='ignore'):
                    score_ts = (ema_f - ema_s) / x
                if smooth_span is not None and int(smooth_span) > 1:
                    score_ts = score_ts.ewm(span=int(smooth_span), adjust=False).mean()
                s_last = score_ts.iloc[-1]
                return pd.Series(s_last).replace([np.inf, -np.inf], np.nan)

            def _trend_score_multi(df: pd.DataFrame, windows: list[dict], smooth_span: int | None = None) -> Optional[pd.Series]:
                total_w = 0.0
                out = None
                for wcfg in windows:
                    try:
                        f = int(wcfg.get("fast_span", wcfg.get("fast", wcfg.get("f", 12))))
                        s = int(wcfg.get("slow_span", wcfg.get("slow", wcfg.get("s", 26))))
                        w = float(wcfg.get("weight", 1.0))
                    except Exception:
                        continue
                    if f <= 0 or s <= 0:
                        continue
                    sc = _trend_score_single(df, f, s, smooth_span=smooth_span)
                    if sc is None:
                        continue
                    if out is None:
                        out = (w * sc)
                    else:
                        out = out.add(w * sc, fill_value=0.0)
                    total_w += abs(w)
                if out is None:
                    return None
                if total_w > 0:
                    out = out / total_w
                return out.replace([np.inf, -np.inf], np.nan)

            # choose score config
            if opt_type == "ema_online":
                # Convert configured window spans to bars based on the data frequency.
                # For 1-minute data: minutes == bars, hours == 60 bars, days == bars_per_day.
                bpd = _infer_bars_per_day(pw.index)

                def _to_bars(val: float, unit: str) -> int:
                    try:
                        v = float(val)
                    except Exception:
                        return 0
                    u = (unit or "bars").strip().lower()
                    if u in {"bar", "bars", "step", "steps"}:
                        return int(max(1, round(v)))
                    if u in {"min", "mins", "minute", "minutes"}:
                        return int(max(1, round(v)))
                    if u in {"h", "hr", "hrs", "hour", "hours"}:
                        return int(max(1, round(v * 60.0)))
                    if u in {"d", "day", "days", "trading_day", "trading_days"}:
                        return int(max(1, round(v * float(max(1, bpd)))))
                    # fallback: treat as bars
                    return int(max(1, round(v)))

                windows_bars: list[dict] = []
                for wcfg in emao_windows:
                    if not isinstance(wcfg, dict):
                        continue
                    f_raw = wcfg.get("fast_span", wcfg.get("fast", wcfg.get("f", 12)))
                    s_raw = wcfg.get("slow_span", wcfg.get("slow", wcfg.get("s", 26)))
                    wgt = float(wcfg.get("weight", 1.0))
                    f_b = _to_bars(float(f_raw), emao_windows_unit)
                    s_b = _to_bars(float(s_raw), emao_windows_unit)
                    if f_b <= 0 or s_b <= 0:
                        continue
                    windows_bars.append({"fast_span": f_b, "slow_span": s_b, "weight": wgt})

                if len(windows_bars) == 0:
                    windows_bars = [
                        {"fast_span": 15, "slow_span": 60, "weight": 1.0},
                        {"fast_span": 60, "slow_span": 240, "weight": 1.0},
                        {"fast_span": max(1, bpd), "slow_span": max(2, 5 * bpd), "weight": 0.7},
                    ]

                scores = _trend_score_multi(pw, windows_bars, smooth_span=emao_score_smooth_span)
                thr = float(emao_bullish_threshold)
                min_assets_cfg = emao_min_assets
                max_assets_cfg = emao_max_assets
                fb_k = int(emao_fallback_k)
                base_mode = str(emao_base_optimizer).strip().lower()
                mu_source = str(emao_mu_source).strip().lower()
                mu_alpha = float(emao_mu_blend_alpha)
                mu_scale = float(emao_mu_scale)
                risk_blend = emao_risk_blend
            else:
                # single window
                scores = _trend_score_multi(pw, [{"fast_span": ema_fast_span, "slow_span": ema_slow_span, "weight": 1.0}], smooth_span=None)
                thr = float(ema_bullish_threshold)
                min_assets_cfg = ema_min_assets
                max_assets_cfg = ema_max_assets
                fb_k = int(ema_fallback_k)
                base_mode = str(ema_post_optimizer).strip().lower()
                mu_source = str(ema_mu_source).strip().lower()
                mu_alpha = float(ema_mu_blend_alpha)
                mu_scale = float(ema_mu_scale)
                risk_blend = None

            if scores is None or len(scores) == 0:
                return {"weights": None, "status": "ema_insufficient_window"}

            scores = pd.Series(scores).replace([np.inf, -np.inf], np.nan).reindex(assets)
            scores_clean = scores.dropna()
            if scores_clean.empty:
                return {"weights": None, "status": "ema_no_scores"}

            # --- Selection / ranking ---
            # By default we filter on EMA trend (scores > thr). For "greedy returns", you can
            # set ema_online.rank_mode: composite and include return/vol/dd in rank_weights.
            eligible = scores_clean[scores_clean > thr]

            def _z(x: pd.Series) -> pd.Series:
                x = x.astype(float)
                m = float(x.mean())
                s = float(x.std())
                return (x - m) / (s + 1e-8)

            rank = None
            if opt_type == "ema_online" and emao_rank_mode in {"composite", "rank", "mix", "blend"}:
                # Convert rank lookbacks if configured in days/hours.
                bpd = _infer_bars_per_day(pw.index)
                def _lb_to_bars(lb: int | None, unit: str) -> int | None:
                    if lb is None:
                        return None
                    try:
                        v = float(lb)
                    except Exception:
                        return None
                    u = (unit or "bars").strip().lower()
                    if u in {"bars", "bar", "steps"}:
                        return int(max(1, round(v)))
                    if u in {"minute", "minutes", "min", "mins"}:
                        return int(max(1, round(v)))
                    if u in {"hour", "hours", "h", "hr", "hrs"}:
                        return int(max(1, round(v * 60.0)))
                    if u in {"day", "days", "d", "trading_days", "trading_day"}:
                        return int(max(1, round(v * float(max(1, bpd)))))
                    return int(max(1, round(v)))

                ret_lb = _lb_to_bars(emao_rank_ret_lb, emao_rank_unit) or int(max(1, bpd))
                vol_lb = _lb_to_bars(emao_rank_vol_lb, emao_rank_unit) or int(max(5, 2 * bpd))
                dd_lb = _lb_to_bars(emao_rank_dd_lb, emao_rank_unit) or 0

                # Recent return (log) over ret_lb
                if len(pw) >= ret_lb + 1:
                    p_last = pw.iloc[-1].astype(float)
                    p_prev = pw.iloc[-1 - ret_lb].astype(float)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ret = np.log(p_last / p_prev)
                    ret = pd.Series(ret, index=pw.columns).replace([np.inf, -np.inf], np.nan)
                else:
                    ret = pd.Series(index=pw.columns, dtype=float)

                # Recent vol (std of log returns) over vol_lb
                if len(pw) >= vol_lb + 2:
                    r_win = compute_returns(pw.tail(vol_lb + 1), method='log')
                    vol = r_win.std() if r_win is not None and not r_win.empty else pd.Series(index=pw.columns, dtype=float)
                else:
                    vol = pd.Series(index=pw.columns, dtype=float)

                # Recent drawdown magnitude over dd_lb (optional)
                dd_mag = pd.Series(index=pw.columns, dtype=float)
                if dd_lb and len(pw) >= dd_lb + 2:
                    x = pw.tail(dd_lb + 1).astype(float).values
                    # rolling max
                    roll_max = np.maximum.accumulate(x, axis=0)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dd = (x / roll_max) - 1.0
                    dd_min = np.nanmin(dd, axis=0)
                    dd_mag = pd.Series(np.abs(dd_min), index=pw.columns).replace([np.inf, -np.inf], np.nan)

                # Cross-sectional z-score blend
                rank = (
                    float(emao_rank_w_ema) * _z(scores_clean.reindex(pw.columns))
                    + float(emao_rank_w_ret) * _z(ret)
                    - float(emao_rank_w_vol) * _z(vol)
                    - float(emao_rank_w_dd) * _z(dd_mag)
                )
                rank = rank.replace([np.inf, -np.inf], np.nan)

            # Default rank is the EMA score itself
            if rank is None:
                rank = scores_clean

            # Filter on bullish trend but rank by composite
            selected = rank.reindex(eligible.index).dropna().sort_values(ascending=False)
            if max_assets_cfg is not None and len(selected) > 0:
                try:
                    selected = selected.head(min(int(max_assets_cfg), len(selected)))
                except Exception:
                    pass

            # minimum diversification fill
            min_assets_use = min_assets_cfg if (min_assets_cfg is not None) else max(1, min(fb_k, len(scores_clean)))
            try:
                min_assets_use = int(min_assets_use)
            except Exception:
                min_assets_use = max(1, min(fb_k, len(scores_clean)))

            if len(selected) < min_assets_use:
                remaining = rank.drop(index=selected.index, errors="ignore").dropna().sort_values(ascending=False)
                need = min_assets_use - len(selected)
                if need > 0:
                    selected = pd.concat([selected, remaining.head(need)])

            if selected.empty:
                selected = rank.dropna().sort_values(ascending=False).head(min(fb_k, len(scores_clean)))

            selected_assets = list(selected.index)

            # Greedy mode: score-proportional weights
            if base_mode in {"greedy", "trend_greedy"}:
                w = ema_trend_optimize(
                    scores=scores,
                    box=box,
                    long_only=long_only,
                    fallback_k=fb_k,
                    weight_power=ema_weight_power,
                    epsilon=ema_epsilon,
                )
                return {"weights": w, "status": f"ok_{opt_type}_greedy", "selected_n": int(len(selected_assets))}

            # Build mu for the base optimizer on the selected subset
            mu_returns = expected.reindex(selected_assets).astype(float)
            mu_score = (scores.reindex(selected_assets).astype(float) * float(mu_scale))
            if mu_source in {"ema", "ema_score", "score"}:
                mu_sub = mu_score
            elif mu_source in {"returns", "mean_returns", "er"}:
                mu_sub = mu_returns
            elif mu_source in {"blend", "mix", "hybrid"}:
                a = max(0.0, min(1.0, float(mu_alpha)))
                mu_sub = (a * mu_score) + ((1.0 - a) * mu_returns)
            else:
                mu_sub = mu_score

            cov_sub = cov.reindex(index=selected_assets, columns=selected_assets).fillna(0.0)

            # Run base optimizer
            res = None
            if base_mode in {"mv_reg", "mv", "mean_variance", "mean-variance"}:
                res = mv_reg_optimize(mu_sub, cov_sub, box=box, long_only=long_only, lambda_reg=lambda_reg, lambdas=lambdas, solver=solver)
            elif base_mode in {"minvar", "min_variance", "gmv"}:
                res = min_variance_optimize(cov_sub, box=box, long_only=long_only, solver=solver)
            elif base_mode in {"risk_parity", "erc"}:
                res = risk_parity_optimize(cov_sub, box=box, long_only=long_only, tol=rp_tol, max_iter=rp_max_iter)
            elif base_mode in {"sharpe", "max_sharpe"}:
                res = sharpe_optimize(mu_sub, cov_sub, box=box, long_only=long_only, solver=solver)
            elif base_mode in {"entropy_newton", "entropy"}:
                try:
                    Sigma = cov_sub.values.astype(float)
                    mu_np = mu_sub.reindex(selected_assets).fillna(0.0).values.astype(float)
                    n = Sigma.shape[0]
                    w0 = np.ones(n) * (K / n)
                    w_opt = _damped_newton_projected(
                        w0,
                        Sigma,
                        mu_np,
                        p_avg,
                        lam,
                        K=K,
                        max_iters=50,
                        tol=1e-8,
                    )
                    res = {"weights": pd.Series(w_opt, index=selected_assets), "status": "success"}
                except Exception as e:
                    res = {"weights": None, "status": f"error_entropy_opt:{e}"}
            elif base_mode in {"omd", "ftrl"}:
                r_t = _prices_to_relatives(prices_window, selected_assets, period_window=ctx.get('period_prices_window', None))
                if r_t is None:
                    w_ew = pd.Series(1.0 / len(selected_assets), index=selected_assets)
                    return {"weights": w_ew, "status": "ema_online_fallback_equal_insufficient_prices"}

                # reset state if universe changes
                if emao_online_state['assets'] != tuple(selected_assets) or emao_online_state['w'] is None:
                    n = len(selected_assets)
                    emao_online_state['assets'] = tuple(selected_assets)
                    emao_online_state['w'] = (np.ones(n) / n)
                    emao_online_state['B'] = np.zeros((n, n), dtype=float)
                    emao_online_state['v'] = np.zeros(n, dtype=float)

                if base_mode == "omd":
                    w_new, info = omd_step(
                        emao_online_state['w'],
                        r_t,
                        eta=eta,
                        v_target=v_target,
                        k_cardinality=None,
                        **common_args,
                    )
                    emao_online_state['w'] = w_new
                    res = {"weights": pd.Series(w_new, index=selected_assets), "status": "ok", **info}
                else:
                    w_new, B_t, v_t, info, st = ftrl_step(
                        emao_online_state['w'],
                        r_t,
                        emao_online_state['B'],
                        emao_online_state['v'],
                        lambda_2=lambda_2,
                        gamma=gamma,
                        v_target=v_target,
                        k_cardinality=None,
                        **common_args,
                        max_iter=ftrl_max_iter,
                        tol=ftrl_tol,
                    )
                    emao_online_state['w'] = w_new
                    emao_online_state['B'] = B_t
                    emao_online_state['v'] = v_t
                    res = {"weights": pd.Series(w_new, index=selected_assets), "status": st, **info}
            else:
                # default fallback
                res = mv_reg_optimize(mu_sub, cov_sub, box=box, long_only=long_only, lambda_reg=lambda_reg, lambdas=lambdas, solver=solver)

            w = res.get("weights") if isinstance(res, dict) else res
            st = res.get("status", "ok") if isinstance(res, dict) else "ok"
            if w is None:
                return {"weights": None, "status": f"{opt_type}_base_opt_failed:{base_mode}"}

            if isinstance(w, np.ndarray):
                w = pd.Series(w, index=selected_assets)
            elif isinstance(w, pd.Series):
                w = w.reindex(selected_assets)
            else:
                try:
                    w = pd.Series(w, index=selected_assets)
                except Exception:
                    return {"weights": None, "status": f"{opt_type}_bad_weights"}

            w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            # Optional risk parity blend (stabilize vol/DD while prioritizing return)
            if opt_type == "ema_online" and risk_blend is not None:
                try:
                    rp_res = risk_parity_optimize(cov_sub, box=box, long_only=long_only, tol=rp_tol, max_iter=rp_max_iter)
                    w_rp = rp_res.get("weights") if isinstance(rp_res, dict) else rp_res
                    if isinstance(w_rp, np.ndarray):
                        w_rp = pd.Series(w_rp, index=selected_assets)
                    elif isinstance(w_rp, pd.Series):
                        w_rp = w_rp.reindex(selected_assets)
                    else:
                        w_rp = pd.Series(w_rp, index=selected_assets)
                    w_rp = w_rp.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    a = float(risk_blend)
                    w = (a * w) + ((1.0 - a) * w_rp)
                except Exception:
                    pass

            if long_only:
                w = w.clip(lower=0.0)
            if box is not None:
                try:
                    if isinstance(box, dict):
                        lo = box.get("min", None)
                        hi = box.get("max", None)
                    else:
                        lo, hi = box  # type: ignore
                    if lo is not None:
                        w = w.clip(lower=float(lo))
                    if hi is not None:
                        w = w.clip(upper=float(hi))
                except Exception:
                    pass

            s = float(w.sum())
            if (not np.isfinite(s)) or s <= 0:
                w = pd.Series(1.0 / len(selected_assets), index=selected_assets)
            else:
                w = w / s

            return {"weights": w, "status": f"ok_{opt_type}_{base_mode}", "selected_n": int(len(selected_assets)), "base_status": st}
        return mv_reg_optimize(expected_use, cov_use, **common_args, lambda_reg=lambda_reg, lambdas=lambdas, solver=solver)

    return optimizer_func


def runner_func(cfg: dict, logger=None):
    """
    Runner used by run_experiment_from_config.

    Loads prices, builds estimators & optimizer, runs backtest,
    and RETURNS a dict containing results for the experiment saving routine.

    NEW:
      - If `logger` is provided, logs detailed diagnostics and progress.
    """
    dcfg = cfg.get('data', {})
    exp_cfg = cfg.get('experiment', {})
    use_gpu = bool(exp_cfg.get('use_gpu', False))

    if logger:
        logger.info("Runner starting | data.mode=%s | use_gpu=%s", dcfg.get('mode', 'synthetic'), use_gpu)
        try:
            logger.debug("Full config:\n%s", json.dumps(cfg, indent=2, default=str))
        except Exception:
            pass

    # 1) load prices
    mode = dcfg.get('mode', 'synthetic')
    prices = None

    if mode == 'synthetic':
        prices = generate_synthetic_prices(
            n_assets=dcfg.get('n_assets', 100),
            start=exp_cfg.get('start_date'),
            end=exp_cfg.get('end_date'),
            seed=exp_cfg.get('seed', None),
        )
    elif mode == 'processed':
        processed_path = dcfg.get('processed_path')
        # NEW: allow selecting dataset via data.processed.dataset without hardcoding a file path
        if not processed_path:
            pcfg = dcfg.get('processed', {}) or {}
            dataset = str(pcfg.get('dataset', pcfg.get('market', pcfg.get('region', 'both')))).strip().lower()
            aliases = {
                'in': 'india', 'ind': 'india', 'indian': 'india',
                'us': 'us', 'usa': 'us',
                'all': 'both', 'both': 'both',
            }
            dataset = aliases.get(dataset, dataset)
            freq = str(pcfg.get('freq', '1D'))
            base_dir = str(pcfg.get('base_dir', 'data/processed'))

            # Special dataset: Nifty 500 (daily matrix + partitioned minute store)
            ds_aliases = {"nifty_500": "nifty500", "nifty": "nifty500"}
            dataset = ds_aliases.get(dataset, dataset)

            freq_norm = str(freq).strip().lower()
            if freq_norm in {"1min", "1m", "minute", "min", "1t", "1"}:
                freq_norm = "1min"
            if freq_norm in {"1d", "d", "day", "daily"}:
                freq_norm = "1d"

            if dataset == "nifty500":
                if freq_norm == "1d":
                    processed_path = os.path.join(base_dir, "nifty500", "prices_1D_nifty500.parquet")
                else:
                    # minute store is loaded via load_prices_from_partitioned_minute_store
                    processed_path = None
                    minute_store_dir = os.path.join(base_dir, "nifty500", "1min_store")
                    symbols = pcfg.get("symbols", pcfg.get("tickers", pcfg.get("universe", None)))
                    if symbols is None:
                        # default: first N symbols from universe_symbols.json if present
                        n_default = int(pcfg.get("n_symbols", pcfg.get("n_assets", 50)))
                        try:
                            import json
                            u_path = os.path.join(base_dir, "nifty500", "universe_symbols.json")
                            with open(u_path, "r", encoding="utf-8") as f:
                                u = json.load(f)
                            symbols = list(u)[: max(1, n_default)]
                        except Exception:
                            raise ValueError(
                                "For Nifty500 minute data, set data.processed.symbols (list of tickers) "
                                "or ensure universe_symbols.json exists in the processed folder."
                            )
                    if isinstance(symbols, str):
                        # allow comma-separated
                        symbols = [s.strip() for s in symbols.split(",") if s.strip()]

                    prices = load_prices_from_partitioned_minute_store(
                        minute_store_dir,
                        symbols=list(symbols),
                        start_date=exp_cfg.get('start_date'),
                        end_date=exp_cfg.get('end_date'),
                        field=str(pcfg.get("field", "close")),
                    )
            elif dataset == 'both':
                processed_path = os.path.join(base_dir, f"prices_{freq}.parquet")
            elif dataset == 'india':
                processed_path = os.path.join(base_dir, f"prices_{freq}_india.parquet")
            elif dataset == 'us':
                processed_path = os.path.join(base_dir, f"prices_{freq}_us.parquet")
            else:
                raise ValueError(
                    "data.mode == 'processed' but neither data.processed_path is set nor "
                    "data.processed.dataset is one of: india|us|both|nifty500"
                )

        if prices is None:
            if not processed_path:
                raise ValueError("Processed mode requires data.processed_path or a resolved dataset path")
            prices = load_prices_from_parquet(
                processed_path,
                start_date=exp_cfg.get('start_date'),
                end_date=exp_cfg.get('end_date'),
            )
    elif mode == 'csv':
        csv_path = dcfg.get('csv_path')
        if not csv_path:
            raise ValueError("data.mode == 'csv' but data.csv_path is not set")
        prices = load_prices_from_csv(csv_path)
        prices, _, _ = apply_date_range(prices, exp_cfg.get('start_date'), exp_cfg.get('end_date'))
    else:
        raise ValueError(f"Unknown data.mode: {mode!r}")

    if prices is None or prices.empty:
        raise ValueError(
            "No price data available after applying the requested date range. "
            f"start_date={exp_cfg.get('start_date')!r} end_date={exp_cfg.get('end_date')!r}"
        )

    if logger:
        try:
            logger.info(
                "Loaded prices | rows=%s cols=%s | start=%s end=%s",
                prices.shape[0], prices.shape[1], str(prices.index.min()), str(prices.index.max())
            )
        except Exception:
            pass


# --- Optional cash asset (constant price series) ---
# Enables proper cash handling in the backtest + UI/analytics compatibility.
    cash_cfg = cfg.get('cash', {}) if isinstance(cfg.get('cash', {}), dict) else {}
    if bool(cash_cfg.get('enabled', cash_cfg.get('use_cash', False))):
        cash_name = str(cash_cfg.get('asset_name', cash_cfg.get('cash_asset_name', cash_cfg.get('ticker', 'CASH')))).strip() or "CASH"
        if cash_name not in prices.columns:
            prices = prices.copy()
            prices[cash_name] = 1.0

    # 2) estimators (frequency-aware)
    # For intraday data, it is often useful to scale per-bar log-return moments
    # to the rebalance horizon (e.g., daily rebalance => ~bars_per_day multiplier).
    risk_cfg = (cfg.get('risk_model', {}) or {}) if isinstance(cfg.get('risk_model', {}), dict) else {}
    bpd = _infer_bars_per_day(prices.index)
    is_intraday = bpd > 1
    rebalance_freq = str(exp_cfg.get('rebalance', 'monthly')).strip().lower()

    # Lookback: allow units so the same config works for 1D and 1min data.
    # Backtest expects lookback in BARS.
    lookback_bars = None
    lb_days = risk_cfg.get('lookback_days', None)
    lb_unit = str(risk_cfg.get('lookback_unit', '')).strip().lower()
    lb_raw = risk_cfg.get('lookback', None)
    if lb_days is not None:
        try:
            lookback_bars = int(max(2, round(float(lb_days) * float(max(1, bpd)))))
        except Exception:
            lookback_bars = None
    elif lb_raw is not None:
        if not lb_unit:
            lb_unit = 'trading_days' if is_intraday else 'bars'
        try:
            v = float(lb_raw)
            if lb_unit in {'bars', 'bar', 'steps'}:
                lookback_bars = int(max(2, round(v)))
            elif lb_unit in {'minutes', 'minute', 'min', 'mins'}:
                lookback_bars = int(max(2, round(v)))  # 1min bars
            elif lb_unit in {'hours', 'hour', 'h', 'hr', 'hrs'}:
                lookback_bars = int(max(2, round(v * 60.0)))
            elif lb_unit in {'days', 'day', 'd', 'trading_days', 'trading_day'}:
                lookback_bars = int(max(2, round(v * float(max(1, bpd)))))
            elif lb_unit in {'weeks', 'week', 'w'}:
                lookback_bars = int(max(2, round(v * 5.0 * float(max(1, bpd)))))
            else:
                lookback_bars = int(max(2, round(v)))
        except Exception:
            lookback_bars = None

    # Horizon scaling (log-return approximation: mean and cov scale linearly with time)
    scale_to_horizon = risk_cfg.get('scale_to_rebalance_horizon', None)
    if scale_to_horizon is None:
        scale_to_horizon = bool(is_intraday)
    horizon_mult = 1.0
    if bool(scale_to_horizon):
        days_map = {'daily': 1, 'weekly': 5, 'monthly': 21, 'quarterly': 63, 'yearly': 252}
        d_mult = float(days_map.get(rebalance_freq, 1))
        horizon_mult = d_mult * float(max(1, bpd)) if is_intraday else d_mult

    def expected_return_estimator_scaled(prices_window: pd.DataFrame) -> pd.Series:
        mu = expected_return_estimator(prices_window)
        try:
            return mu.astype(float) * float(horizon_mult)
        except Exception:
            return mu

    base_cov_est = cov_estimator_factory(use_gpu=use_gpu)

    def cov_estimator_scaled(prices_window: pd.DataFrame) -> pd.DataFrame:
        c = base_cov_est(prices_window)
        try:
            return c.astype(float) * float(horizon_mult)
        except Exception:
            return c

    expected_return_estimator_callable = expected_return_estimator_scaled
    cov_estimator_callable = cov_estimator_scaled

    # 3) optimizer wrapper from cfg
    optimizer_func = optimizer_wrapper_factory_from_cfg(cfg, logger=logger)

    # 4) backtest config
    backtest_cfg = {
        "rebalance": rebalance_freq,
        "transaction_costs": cfg.get("transaction_costs", {}),
        "lookback": lookback_bars,
        # optional debug setting:
        "record_non_rebalance": bool(exp_cfg.get("record_non_rebalance", False)),
        "cash": cash_cfg,
        "optimizer": cfg.get("optimizer", {}),
    }

    # 5) progress (log every 1%)
    progress = None
    step_cb = None
    if logger:
        total_steps = int(len(prices.index))
        progress = PercentProgressLogger(logger, total_steps=total_steps, percent_step=1)

        def step_cb(step_idx_1_based: int, total: int, info: dict):
            # Keep log line readable (avoid huge floats)
            try:
                progress.update(
                    step_idx_1_based,
                    date=str(info.get("date")),
                    nav=round(float(info.get("nav", 0.0)), 8),
                    opt_status=str(info.get("opt_status")),
                )
            except Exception:
                # never allow logging to break the run
                logger.debug("Progress logger failed (ignored)", exc_info=True)

    # 6) run backtest
    bt_result = run_backtest(
        prices,
        expected_return_estimator_callable,
        cov_estimator_callable,
        optimizer_func,
        backtest_cfg,
        logger=logger,
        step_callback=step_cb,
    )

    # 7) produce output dict (include prices snapshot and meta)
    out = {
        "nav": bt_result.nav,
        "weights": bt_result.weights,
        "turnover": bt_result.turnover,
        "trades": bt_result.trades,
        "prices": prices,
        "meta": {
            "cfg": cfg,
            "n_assets": prices.shape[1],
            "price_index_start": str(prices.index.min()),
            "price_index_end": str(prices.index.max()),
        },
    }
    return out


def _ann_stats(nav_s: pd.Series):
    """Compute a small set of annualized stats for JSON-friendly saving.

    Defensive: accepts Series or single-column DataFrame, handles very short series,
    non-finite values, and always returns numeric scalars (or None) so the saving step
    doesn't fail with UnboundLocalError.
    """
    # Normalize input to a Series
    try:
        if nav_s is None:
            return {"total_return": None, "ann_return": None, "ann_vol": None, "max_drawdown": None, "sharpe": None}
        if isinstance(nav_s, pd.DataFrame):
            if nav_s.shape[1] == 1:
                nav = nav_s.iloc[:, 0].astype(float)
            else:
                if 'value' in nav_s.columns:
                    nav = nav_s['value'].astype(float)
                else:
                    nav = nav_s.select_dtypes(include=['number']).iloc[:, 0].astype(float)
        elif isinstance(nav_s, pd.Series):
            nav = nav_s.astype(float)
        else:
            nav = pd.Series(nav_s).astype(float)
    except Exception:
        return {"total_return": None, "ann_return": None, "ann_vol": None, "max_drawdown": None, "sharpe": None}

    if nav is None or len(nav) < 2:
        return {"total_return": None, "ann_return": None, "ann_vol": None, "max_drawdown": None, "sharpe": None}

    try:
        nav = nav.sort_index()
    except Exception:
        pass

    try:
        days = (pd.to_datetime(nav.index[-1]) - pd.to_datetime(nav.index[0])).days
        days = max(int(days), 1)
    except Exception:
        days = max(len(nav) - 1, 1)

    try:
        total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    except Exception:
        total_ret = None

    try:
        ann_ret = float((1 + total_ret) ** (365.0 / float(days)) - 1.0) if total_ret is not None else None
    except Exception:
        ann_ret = None

    try:
        dr = nav.pct_change().dropna()
        ann_vol = float(dr.std() * np.sqrt(252)) if not dr.empty else None
    except Exception:
        ann_vol = None

    try:
        roll_max = nav.cummax()
        drawdown = (nav - roll_max) / roll_max
        min_drawdown = drawdown.min()
        max_dd = float(min_drawdown) if pd.notna(min_drawdown) else None
    except Exception:
        max_dd = None

    try:
        if ann_ret is None or ann_vol is None or ann_vol == 0:
            sharpe = None
        else:
            sharpe = float(ann_ret / ann_vol)
    except Exception:
        sharpe = None

    return {
        "total_return": None if total_ret is None else float(total_ret),
        "ann_return": None if ann_ret is None else float(ann_ret),
        "ann_vol": None if ann_vol is None else float(ann_vol),
        "max_drawdown": None if max_dd is None else float(max_dd),
        "sharpe": None if sharpe is None else float(sharpe),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    # Convenience: allow accidental `--configs/<file>.yaml` (missing space)
    # by interpreting it as `--config configs/<file>.yaml`.
    fixed: list[str] = []
    for a in argv:
        if a.startswith("--configs/"):
            fixed.extend(["--config", "configs/" + a[len("--configs/") :]])
        elif a.startswith("--config/"):
            fixed.extend(["--config", a[len("--config/") :]])
        else:
            fixed.append(a)
    argv = fixed

    p = argparse.ArgumentParser(description="Run a portfolio_sim experiment with detailed logging.")

    p.add_argument(
        "--config",
        "--configs",
        dest="config",
        default="configs/newconfig.yaml",
        help="Path to YAML config file.",
    )

    p.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Where experiment folders are written (used by Hermis_Prism).",
    )

    p.add_argument(
        "--runs-dir",
        default="runs",
        help="Where per-run logs/config snapshots are written.",
    )

    p.add_argument(
        "--detach",
        action="store_true",
        help="Start the simulation in the background and free the terminal immediately.",
    )

    # internal: used by the detached child process
    p.add_argument(
        "--experiment-folder",
        "--run-dir",
        dest="experiment_folder",
        default=None,
        help=argparse.SUPPRESS,
    )

    p.add_argument(
        "--name",
        default=None,
        help="Override experiment name (optional).",
    )

    p.add_argument(
        "--tag",
        default=None,
        help="Tag appended to experiment folder name (optional).",
    )

    return p.parse_args(argv)


def _runpaths_from_experiment_folder(exp_folder: Path, config_path: Path) -> RunPaths:
    """Create RunPaths pointing into the experiment folder's built-in logs/ directory."""
    exp_folder = Path(exp_folder)
    logs_dir = exp_folder / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=exp_folder,
        config_copy=config_path,
        log_file=logs_dir / "run.log",
        stdio_log_file=logs_dir / "stdout_stderr.log",
        failure_file=logs_dir / "failure.traceback.txt",
        metadata_file=logs_dir / "run_metadata.json",
    )


def _update_run_metadata(paths: RunPaths, updates: dict) -> None:
    try:
        if paths.metadata_file.exists():
            meta = json.loads(paths.metadata_file.read_text(encoding="utf-8"))
        else:
            meta = {}
        meta.update(updates)
        paths.metadata_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        # best-effort
        pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        return 2

    # Create or reuse the experiment folder (this also creates `<date_id>__<name>/logs/`).
    # We want ALL logs to live inside that built-in logs/ folder.
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    exp_name = resolve_experiment_name(cfg, args.name)

    # If a folder is provided (internal use by detached child), reuse it.
    if args.experiment_folder:
        exp_folder = Path(args.experiment_folder)
        exp_folder.mkdir(parents=True, exist_ok=True)
        (exp_folder / "logs").mkdir(parents=True, exist_ok=True)
        # Prefer the snapshot config inside the folder if present.
        if (exp_folder / "params.yaml").exists():
            config_path = exp_folder / "params.yaml"
        else:
            save_params_yaml(str(config_path), exp_folder)
            config_path = exp_folder / "params.yaml"
    else:
        exp_folder = make_experiment_folder(str(args.experiments_dir), exp_name, args.tag)
        save_params_yaml(str(config_path), exp_folder)
        config_path = exp_folder / "params.yaml"

    paths = _runpaths_from_experiment_folder(exp_folder, config_path)

    # --detach mode: create experiment folder + config snapshot NOW, then spawn child and exit.
    if args.detach:
        script_path = Path(__file__).resolve()

        # Child uses the COPIED config so the run is reproducible even if the original changes.
        cmd = [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--experiments-dir",
            str(args.experiments_dir),
            "--experiment-folder",
            str(exp_folder),
        ]

        # Capture ALL stdout/stderr to a file (helps debug even if logging fails early)
        out_f = open(paths.stdio_log_file, "a", encoding="utf-8")

        p = subprocess.Popen(
            cmd,
            stdout=out_f,
            stderr=out_f,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

        # Write a small launcher breadcrumb into the run log.
        launch_logger = setup_logging(paths.log_file, verbose=False)
        write_metadata(
            paths,
            extra={
                "config_path": str(config_path),
                "experiments_dir": str(args.experiments_dir),
                "experiment_folder": str(exp_folder),
                "launcher_pid": os.getpid(),
                "detached_pid": p.pid,
                "cwd": os.getcwd(),
            },
        )
        launch_logger.info("Launched detached simulation. PID=%s", p.pid)

        print(f"[DETACHED] PID={p.pid}")
        print(f"[DETACHED] Experiment folder: {exp_folder}")
        print(f"[DETACHED] Config snapshot: {config_path}")
        print(f"[DETACHED] Logs: {paths.log_file} (and {paths.stdio_log_file})")
        return 0

# Foreground run (also used by detached child)
    logger = setup_logging(paths.log_file, verbose=True)

    write_metadata(
        paths,
        extra={
            "config_path": str(config_path),
            "experiments_dir": str(args.experiments_dir),
            "experiment_folder": str(exp_folder),
            "name_override": args.name,
            "tag": args.tag,
            "cwd": os.getcwd(),
        },
    )

    logger.info("Run initialized")
    logger.info("Run folder: %s", str(paths.run_dir))
    logger.info("Config used: %s", str(config_path))
    logger.info("Experiments dir: %s", str(args.experiments_dir))

    try:
        logger.info("Running experiment from config")

        exp_folder_out = run_experiment_from_config(
            str(config_path),
            str(args.experiments_dir),
            runner_func=lambda cfg: runner_func(cfg, logger=logger),
            name_override=args.name,
            tag=args.tag,
            exp_folder=str(exp_folder),
        )

        logger.info("Experiment saved to: %s", str(exp_folder_out))
        _update_run_metadata(paths, {"experiment_folder": str(exp_folder_out)})

        # load outputs saved by run_experiment_from_config
        out = load_experiment(str(exp_folder))
        nav = out.get("nav")
        weights = out.get("weights")
        trades = out.get("trades")

        # Precompute and save performance artifacts
        perf_dir = Path(exp_folder) / "outputs" / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)

        try:
            if nav is not None and not nav.empty:
                metrics = _ann_stats(nav)
                with open(perf_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

                # rolling sharpe
                try:
                    returns = nav.pct_change().dropna()
                    if len(returns) > 63:
                        win = 63
                        r_mean = returns.rolling(win).mean()
                        r_std = returns.rolling(win).std()
                        rs = (r_mean / (r_std + 1e-8)) * np.sqrt(252)

                        if isinstance(rs, pd.DataFrame):
                            if rs.shape[1] == 1:
                                rs_series = rs.iloc[:, 0]
                            else:
                                rs_series = rs.mean(axis=1)
                        else:
                            rs_series = rs

                        rs_df = rs_series.to_frame(name="rolling_sharpe")
                        rs_df.to_parquet(perf_dir / "rolling_sharpe.parquet")
                except Exception as e:
                    logger.warning("Failed to write rolling sharpe artifact: %s", e)

                # drawdown
                try:
                    roll_max = nav.cummax()
                    drawdown = ((nav - roll_max) / roll_max)

                    if isinstance(drawdown, pd.DataFrame):
                        if drawdown.shape[1] == 1:
                            dd_series = drawdown.iloc[:, 0]
                        else:
                            dd_series = drawdown.min(axis=1)
                    else:
                        dd_series = drawdown

                    dd_df = dd_series.to_frame(name="drawdown")
                    dd_df.to_parquet(perf_dir / "drawdown.parquet")
                except Exception as e:
                    logger.warning("Failed to write drawdown artifact: %s", e)

                # cumulative returns
                try:
                    cum = (nav / nav.iloc[0])

                    if isinstance(cum, pd.DataFrame):
                        if cum.shape[1] == 1:
                            cum_series = cum.iloc[:, 0]
                        else:
                            try:
                                cum.to_parquet(perf_dir / "cum_returns_per_asset.parquet")
                            except Exception as e:
                                logger.warning("Failed to write cum_returns_per_asset: %s", e)
                            cum_series = cum.mean(axis=1)
                    else:
                        cum_series = cum

                    cum_df = cum_series.to_frame(name="cum_returns")
                    cum_df.to_parquet(perf_dir / "cum_returns.parquet")
                except Exception as e:
                    logger.warning("Failed to write cumulative returns artifact: %s", e)

            # selected assets artifact
            if trades is not None and hasattr(trades, 'columns') and (not trades.empty) and ('selected' in trades.columns):
                try:
                    trades[['selected']].to_parquet(perf_dir / "selected.parquet")
                except Exception as e:
                    logger.warning("Failed to write selected.parquet: %s", e)

            logger.info("Saved performance artifacts to %s", str(perf_dir))

        except Exception as e:
            logger.warning("Failed to write some performance artifacts: %s", e)
            logger.debug("Artifact error details", exc_info=True)

        # Load prices for plotting
        prices_path = Path(exp_folder) / "data" / "prices.parquet"
        if prices_path.exists():
            prices = pd.read_parquet(prices_path)
        else:
            logger.warning("Could not find prices snapshot for plotting: %s", str(prices_path))
            prices = None

        # Save figures
        if prices is not None:
            figures_dir = Path(exp_folder) / "artifacts" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            try:
                if nav is not None:
                    viz.plot_nav(nav, save_path=str(figures_dir / "nav.png"))
                    viz.plot_prices_and_nav(prices, nav, save_path=str(figures_dir / "prices_and_nav.png"))
                if weights is not None:
                    viz.plot_allocation_heatmap(weights, save_path=str(figures_dir / "alloc.png"))
                logger.info("Saved figures to: %s", str(figures_dir))
            except Exception as e:
                logger.warning("Failed to save figures: %s", e)
                logger.debug("Plot error details", exc_info=True)

        logger.info("Run complete")
        return 0

    except Exception as e:
        # Ensure we always leave a clear failure trail
        log_failure(logger, paths.failure_file, e)
        _update_run_metadata(paths, {"status": "failed", "error": repr(e)})
        logger.error("Run failed: %s", repr(e))
        # also print to console for immediate feedback
        print("ERROR: simulation failed. See logs:")
        print(f"  {paths.log_file}")
        print(f"  {paths.failure_file}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
