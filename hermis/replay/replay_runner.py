"""
hermis/replay/replay_runner.py

Minute-driven (but daily decision) historical replay engine.

Key behavior:
- Consume minute store -> derive daily close + execution prices at (market_open + latency).
- Make rebalance decisions on a *daily* schedule (or weekly first trading day).
- Execute at the configured execution price (1 minute delay by default).
- Mark NAV at daily close.
- Output artifacts compatible with Prism: nav (Series), weights (DataFrame), trades (DataFrame), prices (DataFrame).

Extra ergonomics:
- Month-by-month streaming with optional prefetch.
- Month-end NAV logging + optional partial output flushing (so you can open Prism while the sim is still running).
- Optional live NAV plotting window (matplotlib) for real-time monitoring.

This is intentionally designed so the same engine can later swap the data source
to a live API feed without changing portfolio logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import time
import gc

import numpy as np
import pandas as pd

from .minute_to_daily import iter_daily_months_from_partitioned_minute_store
from .rebalance import get_rebalance_days

# We reuse estimators + optimizer factory already in run_experiment.py to stay consistent with existing configs.
from run_experiment import expected_return_estimator, cov_estimator_factory, optimizer_wrapper_factory_from_cfg


@dataclass
class ReplayResult:
    nav: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    prices: pd.DataFrame


def _apply_cash_overlay(
    w_risky: pd.Series,
    all_assets: pd.Index,
    cash_name: str,
    max_gross_exposure: float = 1.0,
) -> pd.Series:
    """Long-only + cash residual + gross exposure cap."""
    w_risky = w_risky.copy()
    w_risky = w_risky.reindex([a for a in all_assets if a != cash_name]).fillna(0.0).astype(float)
    w_risky[w_risky < 0] = 0.0

    gross = float(w_risky.sum())
    cap = float(max_gross_exposure) if max_gross_exposure is not None else 1.0
    cap = max(0.0, min(1.0, cap))

    if gross > cap and gross > 0:
        w_risky = (w_risky / gross) * cap
        gross = cap

    cash_w = 1.0 - gross
    out = pd.Series(0.0, index=all_assets, dtype=float)
    out.loc[w_risky.index] = w_risky
    if cash_name in out.index:
        out.loc[cash_name] = cash_w
    else:
        # if cash column isn't present, renormalize risky to sum 1
        if gross > 0:
            out.loc[w_risky.index] = w_risky / gross
    # numerical cleanup
    s = float(out.sum())
    if np.isfinite(s) and s > 0 and abs(s - 1.0) > 1e-8:
        out = out / s
    return out


def _apply_drawdown_brake(
    w: pd.Series,
    nav: float,
    peak_nav: float,
    threshold: float,
    reduced_exposure: float,
    cash_name: str,
) -> pd.Series:
    if peak_nav <= 0:
        return w
    dd = 1.0 - float(nav / peak_nav)
    if dd <= threshold:
        return w

    risky_cols = [c for c in w.index if c != cash_name]
    risky_sum = float(w.reindex(risky_cols).sum())
    cap = float(reduced_exposure)
    cap = max(0.0, min(1.0, cap))
    if risky_sum <= cap:
        return w

    if risky_sum > 0:
        w2 = w.copy()
        w2.loc[risky_cols] = w2.loc[risky_cols] / risky_sum * cap
        w2.loc[cash_name] = 1.0 - cap
        return w2
    return w


def _weights_drift(w: pd.Series, rel: pd.Series) -> pd.Series:
    """Drift weights through a price-relative vector."""
    rel = rel.reindex(w.index).astype(float)
    rel = rel.replace([np.inf, -np.inf], np.nan)
    # Treat missing relatives as 1.0 (no move) to avoid blowing up.
    rel = rel.fillna(1.0)

    port_rel = float(np.dot(w.values, rel.values))
    if not np.isfinite(port_rel) or port_rel <= 0:
        return w

    return (w * rel) / port_rel


def run_minute_replay_from_cfg(cfg: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """Runner compatible with portfolio_sim.experiment.run_experiment_from_config."""

    exp_cfg = cfg.get("experiment", {}) or {}
    dcfg = cfg.get("data", {}) or {}

    # --- locate minute store ---
    replay_cfg = dcfg.get("replay", {}) if isinstance(dcfg.get("replay", {}), dict) else {}
    store_dir = replay_cfg.get("store_dir")

    if not store_dir:
        # mirror run_experiment processed path convention for nifty500
        pcfg = dcfg.get("processed", {}) if isinstance(dcfg.get("processed", {}), dict) else {}
        base_dir = pcfg.get("base_dir", "data/processed")
        dataset = pcfg.get("dataset", "nifty500")
        store_dir = str(Path(base_dir) / dataset / "1min_store")

    store_dir = str(Path(store_dir))

    # --- universe ---
    symbols = replay_cfg.get("symbols", replay_cfg.get("tickers", None))
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    if symbols is None:
        # try universe file if provided
        uni_file = replay_cfg.get("universe_file")
        if uni_file:
            try:
                import json
                with open(uni_file, "r", encoding="utf-8") as f:
                    symbols = json.load(f)
            except Exception:
                symbols = None

    if symbols is None:
        # fall back to data-processing artifact if present
        uni_guess = Path("data/processed/nifty500/universe_symbols.json")
        if uni_guess.exists():
            try:
                import json
                symbols = json.loads(uni_guess.read_text(encoding="utf-8"))
            except Exception:
                symbols = None

    if not symbols:
        raise ValueError("Replay: universe symbols not provided. Set data.replay.symbols or data.replay.universe_file.")

    # --- date range ---
    start_date = replay_cfg.get("start_date") or dcfg.get("start_date") or exp_cfg.get("start_date")
    end_date = replay_cfg.get("end_date") or dcfg.get("end_date") or exp_cfg.get("end_date")

    # --- market timing ---
    market_start = str(replay_cfg.get("market_start", "09:15:00"))
    latency_minutes = int(replay_cfg.get("latency_minutes", 1))

    # --- streaming / prefetch ---
    prefetch = int(replay_cfg.get("prefetch", 1))
    prefetch = max(0, min(4, prefetch))  # keep it small; month chunks can be large

    # --- schedule ---
    rebalance_freq = str(exp_cfg.get("rebalance", "weekly")).strip().lower()

    # --- costs / constraints ---
    tc = float(exp_cfg.get("transaction_cost", exp_cfg.get("transaction_cost_bps", 0.0)))
    if tc > 1.0:  # interpret as bps if user provided 5 => 5 bps
        tc = tc / 10000.0

    max_gross = float(exp_cfg.get("max_gross_exposure", 1.0))

    # cash + drawdown brake (compatible with existing run_experiment configs)
    cash_cfg = exp_cfg.get("cash", {}) if isinstance(exp_cfg.get("cash", {}), dict) else {}
    cash_enabled = bool(cash_cfg.get("enabled", True))
    cash_name = str(cash_cfg.get("asset_name", cash_cfg.get("ticker", "CASH"))).strip() or "CASH"

    dd_cfg = exp_cfg.get("drawdown_brake", {}) if isinstance(exp_cfg.get("drawdown_brake", {}), dict) else {}
    dd_enabled = bool(dd_cfg.get("enabled", False))
    dd_threshold = float(dd_cfg.get("threshold", 0.15))
    dd_reduced = float(dd_cfg.get("reduced_exposure", 0.3))

    # --- stoploss (intraday using minute data, but only active between rebalances by default) ---
    sl_cfg = exp_cfg.get("stoploss", {}) if isinstance(exp_cfg.get("stoploss", {}), dict) else {}
    sl_enabled = bool(sl_cfg.get("enabled", False))
    sl_active_on = str(sl_cfg.get("active_on", "non_rebalance")).strip().lower()  # always|non_rebalance
    sl_price_field = str(sl_cfg.get("trigger_price", "close")).strip().lower()   # close|low (future)
    sl_exec_field = str(sl_cfg.get("exec_price", "open")).strip().lower()       # open|close

    # Per-position stops (percent, e.g. 0.08 for 8%). 0 disables.
    sl_hard = float(sl_cfg.get("hard_stop_pct", sl_cfg.get("stop_pct", 0.0)) or 0.0)
    sl_trail = float(sl_cfg.get("trailing_stop_pct", sl_cfg.get("trail_pct", 0.0)) or 0.0)
    sl_min_weight = float(sl_cfg.get("min_weight", 0.001))
    sl_cooldown = str(sl_cfg.get("cooldown", "until_rebalance")).strip().lower()
    sl_latency = int(sl_cfg.get("latency_minutes", latency_minutes))
    sl_latency = max(0, sl_latency)

    # Sanity: trailing stop should be positive if enabled, otherwise disable to avoid surprises
    if sl_enabled and (sl_hard <= 0 and sl_trail <= 0):
        sl_enabled = False

    # --- estimators / optimizer ---
    use_gpu = bool(exp_cfg.get("use_gpu", False))
    cov_estimator = cov_estimator_factory(use_gpu=use_gpu)
    optimizer_func = optimizer_wrapper_factory_from_cfg(cfg)

    # replay lookback / warmup
    lookback = int(exp_cfg.get("lookback", exp_cfg.get("cov_lookback", 252)))
    min_lookback = int(exp_cfg.get("min_lookback", max(10, min(lookback, 60))))

    # --- live diagnostics: month-end NAV logging + optional flush ---
    log_month_nav = bool(replay_cfg.get("log_month_nav", True))
    flush_outputs_each_month = bool(replay_cfg.get("flush_outputs_each_month", False))
    flush_weights_each_month = bool(replay_cfg.get("flush_weights_each_month", False))
    flush_trades_each_month = bool(replay_cfg.get("flush_trades_each_month", True))
    outputs_folder = cfg.get("_exp_outputs_folder")

    # --- live NAV plot ---
    viz_cfg = cfg.get("visualization", {}) if isinstance(cfg.get("visualization", {}), dict) else {}
    live_plot = bool(replay_cfg.get("live_plot", viz_cfg.get("live_nav_plot", False)))
    live_plot_update_every = int(replay_cfg.get("live_plot_update_every", 1))
    live_plot_update_every = max(1, live_plot_update_every)
    live_plot_theme = str(replay_cfg.get("live_plot_theme", viz_cfg.get("live_plot_theme", "dark"))).strip().lower()
    live_plot_hold_on_finish = bool(replay_cfg.get("live_plot_hold_on_finish", True))

    plotter = None
    if live_plot:
        try:
            from .live_plot import NavLivePlotter
            plotter = NavLivePlotter(
                title="Hermis Live NAV (Replay)",
                subtitle=f"{rebalance_freq} rebalance | latency={latency_minutes}m | symbols={len(symbols)} | gpu={use_gpu}",
                theme=live_plot_theme,
            )
        except Exception as e:
            plotter = None
            if logger:
                logger.warning("Live plot disabled (matplotlib error): %s", repr(e))

    if logger is not None:
        logger.info(
            "Replay: minute stream | store_dir=%s | symbols=%s | prefetch=%s | use_gpu=%s | live_plot=%s",
            store_dir,
            len(symbols),
            prefetch,
            use_gpu,
            bool(plotter is not None),
        )

    def _week_key(ts: pd.Timestamp) -> Tuple[int, int]:
        iso = ts.isocalendar()
        return int(iso.year), int(iso.week)

    def _safe_rel(a: pd.Series, b: pd.Series) -> pd.Series:
        rel = (a / b).replace([np.inf, -np.inf], np.nan)
        # if missing, assume no move (keeps NAV finite)
        return rel.fillna(1.0)

    def _sanitize_target_for_tradability(w: pd.Series, p_exec: pd.Series) -> pd.Series:
        w = w.copy()
        tradable = p_exec.drop(labels=[cash_name], errors="ignore").dropna().index
        risky = [c for c in w.index if c != cash_name]
        non_tradable = [c for c in risky if c not in set(tradable)]
        if non_tradable:
            w.loc[non_tradable] = 0.0

        s = float(w.sum())
        if s <= 0:
            if cash_enabled and cash_name in w.index:
                w[:] = 0.0
                w[cash_name] = 1.0
            else:
                w[:] = 1.0 / max(1, len(risky))
            return w

        if abs(s - 1.0) > 1e-8:
            if cash_enabled and cash_name in w.index:
                risky_sum = float(w.drop(labels=[cash_name], errors="ignore").sum())
                w[cash_name] = max(0.0, 1.0 - risky_sum)
                if risky_sum > 1.0:
                    w_r = w.drop(labels=[cash_name], errors="ignore")
                    w_r = w_r / risky_sum
                    w.update(w_r)
                    w[cash_name] = 0.0
            else:
                w = w / s
        return w

    def _flush_partial_outputs(nav_idx: List[pd.Timestamp], nav_vals: List[float], weights_rows: List[pd.Series], trade_rows: List[Dict[str, Any]]):
        if not outputs_folder:
            return
        try:
            from portfolio_sim.experiment import save_series_as_parquet, save_dataframe_as_parquet
            out = Path(outputs_folder)
            out.mkdir(parents=True, exist_ok=True)

            nav_series = pd.Series(nav_vals, index=pd.DatetimeIndex(nav_idx, name="date"), name="nav")

            # atomic-ish writes: write to temp then replace (parquet path).
            tmp_nav = out / "nav.parquet.tmp"
            save_series_as_parquet(nav_series, tmp_nav)
            if tmp_nav.exists():
                tmp_nav.replace(out / "nav.parquet")

            # We keep weights/trades optional because writing large matrices repeatedly is expensive.
            if flush_weights_each_month:
                weights_df = pd.DataFrame(weights_rows, index=nav_series.index)
                tmp_w = out / "weights.parquet.tmp"
                save_dataframe_as_parquet(weights_df, tmp_w)
                if tmp_w.exists():
                    tmp_w.replace(out / "weights.parquet")

            if flush_trades_each_month:
                trades_df = pd.DataFrame(trade_rows)
                if not trades_df.empty:
                    tmp_t = out / "trades.parquet.tmp"
                    save_dataframe_as_parquet(trades_df, tmp_t)
                    if tmp_t.exists():
                        tmp_t.replace(out / "trades.parquet")
        except Exception as e:
            if logger:
                logger.warning("Replay: failed flushing partial outputs: %s", repr(e))

    # -------- streaming producer (month-by-month) --------
    import queue
    import threading

    SENTINEL = object()
    q: "queue.Queue[Any]" = queue.Queue(maxsize=prefetch if prefetch > 0 else 1)
    err_holder: List[BaseException] = []

    def _producer():
        try:
            for item in iter_daily_months_from_partitioned_minute_store(
                store_dir=store_dir,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                market_start=market_start,
                latency_minutes=latency_minutes,
                include_ohlcv=False,
                include_minutes=sl_enabled,
                logger=logger,
                log_every_month=True,
            ):
                q.put(item)
        except BaseException as e:
            err_holder.append(e)
        finally:
            q.put(SENTINEL)

    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    # -------- replay state --------
    all_assets = pd.Index(list(symbols) + ([cash_name] if cash_enabled else []))
    close_hist = pd.DataFrame(columns=symbols, dtype=float)  # close history used for estimators (risky only)

    nav_close = 1.0
    peak_nav = 1.0

    # start fully in cash (or equal-weight if cash disabled)
    if cash_enabled:
        w_close_prev = pd.Series(0.0, index=all_assets, dtype=float)
        w_close_prev[cash_name] = 1.0
    else:
        w_close_prev = pd.Series(1.0 / len(symbols), index=all_assets, dtype=float)

    p_close_prev: Optional[pd.Series] = None

    last_rebalance_exec_prices: Optional[pd.Series] = None
    last_rebalance_day: Optional[pd.Timestamp] = None

    last_week_key: Optional[Tuple[int, int]] = None

    # stoploss state (risky only)
    sl_entry = pd.Series(np.nan, index=pd.Index(symbols, dtype=str), dtype=float)
    sl_peak = pd.Series(np.nan, index=pd.Index(symbols, dtype=str), dtype=float)
    sl_stopped: set[str] = set()

    # outputs
    nav_idx: List[pd.Timestamp] = []
    nav_vals: List[float] = []
    weights_rows: List[pd.Series] = []
    trade_rows: List[Dict[str, Any]] = []
    close_parts: List[pd.DataFrame] = []

    plot_updates = 0

    # -------- main consume loop --------
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        if err_holder:
            raise err_holder[0]

        # iterator yields: (yy,mm), close_w, exec_w, ohlcv_w, minutes_long
        (yy, mm), close_w, exec_w, _, minutes_long = item
        if close_w is None or close_w.empty:
            continue

        month_t0 = time.time()
        month_start_nav_pos = len(nav_vals)
        month_trade_pos = len(trade_rows)

        # align columns to full universe
        close_w = close_w.reindex(columns=symbols).sort_index()
        exec_w = exec_w.reindex(columns=symbols)
        exec_w = exec_w.reindex(index=close_w.index).sort_index()

        close_parts.append(close_w)

        for day in close_w.index:
            day = pd.Timestamp(day).normalize()
            p_close = close_w.loc[day].astype(float)
            p_exec = exec_w.loc[day].astype(float) if day in exec_w.index else p_close.copy()

            if cash_enabled:
                p_close = p_close.copy(); p_exec = p_exec.copy()
                p_close[cash_name] = 1.0
                p_exec[cash_name] = 1.0

            # initialize previous close on first day
            if p_close_prev is None:
                p_close_prev = p_close.copy()
                close_hist.loc[day, :] = p_close.reindex(symbols).values
                nav_idx.append(day)
                nav_vals.append(nav_close)
                weights_rows.append(w_close_prev.copy())

                if plotter is not None:
                    try:
                        plotter.update(day, nav_close)
                        plotter.pump()
                    except Exception:
                        plotter = None
                continue

            # Drift: prev close -> today's exec
            rel_overnight = _safe_rel(p_exec.reindex(all_assets), p_close_prev.reindex(all_assets))
            nav_exec = nav_close * float((w_close_prev * rel_overnight).sum())
            w_exec_pre = _weights_drift(w_close_prev, rel_overnight)

            # Decide rebalance
            do_rebalance = False
            if rebalance_freq in ("daily", "1d", "d"):
                do_rebalance = True
            elif rebalance_freq in ("weekly", "1w", "w"):
                wk = _week_key(day)
                if wk != last_week_key:
                    do_rebalance = True
                    last_week_key = wk
            else:
                wk = _week_key(day)
                if wk != last_week_key:
                    do_rebalance = True
                    last_week_key = wk

            have_history = len(close_hist) >= min_lookback

            # Cooldown reset: once per rebalance window.
            if sl_enabled and do_rebalance and sl_cooldown in ("until_rebalance", "rebalance"):
                sl_stopped.clear()

            w_target = w_exec_pre.copy()
            opt_status = "hold"

            if do_rebalance and have_history:
                window = close_hist.tail(lookback).copy()
                expected = expected_return_estimator(window)
                cov = cov_estimator(window)

                period_prices_window = None
                if last_rebalance_exec_prices is not None:
                    idx = pd.DatetimeIndex([
                        last_rebalance_day if last_rebalance_day is not None else day,
                        day
                    ], name="date")
                    period_prices_window = pd.DataFrame([
                        last_rebalance_exec_prices,
                        p_exec.reindex(all_assets)
                    ], index=idx)

                try:
                    opt_res = optimizer_func(
                        expected,
                        cov,
                        prices_window=window,
                        period_prices_window=period_prices_window,
                        date=day,
                    ) or {}
                    w_risky = opt_res.get("weights", None)
                    opt_status = str(opt_res.get("status", "ok")).strip()
                except Exception as e:
                    w_risky = None
                    opt_status = f"optimizer_failed:{repr(e)}"
                    if logger:
                        logger.exception("Replay optimizer failed on %s", str(day))

                if w_risky is None or (isinstance(w_risky, pd.Series) and w_risky.empty):
                    valid = p_exec.drop(labels=[cash_name], errors="ignore").dropna().index.tolist()
                    if valid:
                        w_risky = pd.Series(1.0 / len(valid), index=valid, dtype=float)
                    else:
                        w_risky = pd.Series(dtype=float)

                w_target = _apply_cash_overlay(w_risky, all_assets, cash_name=cash_name, max_gross_exposure=max_gross)
                w_target = _sanitize_target_for_tradability(w_target, p_exec.reindex(all_assets))

                if dd_enabled and cash_enabled and cash_name in w_target.index:
                    w_target = _apply_drawdown_brake(
                        w_target,
                        nav=nav_exec,
                        peak_nav=peak_nav,
                        threshold=dd_threshold,
                        reduced_exposure=dd_reduced,
                        cash_name=cash_name,
                    )

                turnover = float(np.abs(w_target - w_exec_pre).sum())
                cost = float(turnover * tc) if len(nav_idx) > 0 else 0.0
                nav_exec = nav_exec * (1.0 - cost)

                trade_rows.append({
                    "date": day,
                    "type": "rebalance",
                    "turnover": turnover,
                    "tc": tc,
                    "cost": cost,
                    "status": opt_status,
                })

                last_rebalance_exec_prices = p_exec.reindex(all_assets).copy()
                last_rebalance_day = day
                w_exec_post = w_target

                # Reset stoploss references at rebalance execution (entry price and peak start here)
                if sl_enabled:
                    if sl_cooldown in ("until_rebalance", "rebalance", "on_rebalance"):
                        sl_stopped.clear()
                    held = w_exec_post.drop(labels=[cash_name], errors="ignore")
                    held = held[held > sl_min_weight]
                    sl_entry[:] = np.nan
                    sl_peak[:] = np.nan
                    if len(held) > 0:
                        px = p_exec.reindex(held.index)
                        sl_entry.loc[held.index] = px.values
                        sl_peak.loc[held.index] = px.values
            else:
                w_exec_post = w_exec_pre

            # Drift: exec -> close (with optional intraday stoploss overrides)
            rel_intraday = _safe_rel(p_close.reindex(all_assets), p_exec.reindex(all_assets))

            # --- intraday stoploss ---
            stop_syms: List[str] = []
            stop_exec_px: Dict[str, float] = {}
            stop_exec_ts: Dict[str, pd.Timestamp] = {}

            should_check_sl = sl_enabled and (sl_active_on == "always" or (sl_active_on in ("non_rebalance", "between_rebalances") and not do_rebalance))
            if should_check_sl and minutes_long is not None and not minutes_long.empty:
                # Only evaluate for held positions above threshold and not already stopped.
                held_w = w_exec_post.drop(labels=[cash_name], errors="ignore")
                held_w = held_w[held_w > sl_min_weight]
                if len(held_w) > 0:
                    held_syms = [s for s in held_w.index.astype(str).tolist() if s not in sl_stopped]
                else:
                    held_syms = []

                if held_syms:
                    dfd = minutes_long
                    # filter to this day
                    dfd = dfd.loc[dfd["date"] == day]
                    if not dfd.empty:
                        dfd = dfd.loc[dfd["symbol"].isin(held_syms)]

                    if not dfd.empty:
                        # Ensure the stop refs exist (if you start mid-week without rebalance, seed from today's exec)
                        seed = sl_entry.loc[held_syms].isna()
                        if bool(seed.any()):
                            seed_syms = sl_entry.loc[held_syms].index[seed].tolist()
                            px0 = p_exec.reindex(seed_syms).astype(float)
                            sl_entry.loc[seed_syms] = px0.values
                            sl_peak.loc[seed_syms] = px0.values

                        # group by symbol and find first trigger minute
                        for sym, g in dfd.groupby("symbol", sort=False):
                            # pull refs
                            entry = float(sl_entry.get(sym, np.nan))
                            peak0 = float(sl_peak.get(sym, np.nan))
                            if not np.isfinite(entry) or entry <= 0:
                                continue
                            if not np.isfinite(peak0) or peak0 <= 0:
                                peak0 = entry

                            # choose columns
                            closes = pd.to_numeric(g.get("close"), errors="coerce").to_numpy(dtype=float)
                            opens = pd.to_numeric(g.get("open"), errors="coerce").to_numpy(dtype=float) if "open" in g.columns else closes.copy()
                            dts = pd.to_datetime(g.get("datetime"), errors="coerce").to_numpy()
                            if closes.size == 0:
                                continue

                            # update peak path (for trailing)
                            peak_path = np.maximum.accumulate(np.where(np.isfinite(closes), closes, -np.inf))
                            peak_path = np.maximum(peak_path, peak0)

                            # effective stop level per minute
                            lvl = np.full_like(closes, -np.inf, dtype=float)
                            if sl_trail > 0:
                                lvl = np.maximum(lvl, peak_path * (1.0 - sl_trail))
                            if sl_hard > 0:
                                lvl = np.maximum(lvl, entry * (1.0 - sl_hard))

                            # trigger on close
                            trig = np.where(np.isfinite(closes) & (closes <= lvl))[0]
                            if trig.size == 0:
                                # update peak state with max close
                                mx = float(np.nanmax(closes)) if np.isfinite(closes).any() else np.nan
                                if np.isfinite(mx):
                                    sl_peak.loc[sym] = max(peak0, mx)
                                continue

                            i0 = int(trig[0])
                            iexec = min(i0 + sl_latency, closes.size - 1)
                            # execute at next minute open (preferred) else close
                            px_exec = opens[iexec] if (sl_exec_field == "open" and np.isfinite(opens[iexec]) and opens[iexec] > 0) else closes[iexec]
                            if not np.isfinite(px_exec) or px_exec <= 0:
                                px_exec = closes[i0]
                            ts_exec = pd.Timestamp(dts[iexec]) if iexec < dts.size else day

                            stop_syms.append(str(sym))
                            stop_exec_px[str(sym)] = float(px_exec)
                            stop_exec_ts[str(sym)] = ts_exec

                            # mark stopped; reset refs
                            sl_stopped.add(str(sym))
                            sl_entry.loc[str(sym)] = np.nan
                            sl_peak.loc[str(sym)] = np.nan

                        # Deduplicate in case categories/strings mismatch
                        stop_syms = sorted(list(set(stop_syms)))

            # Apply stoploss by overriding intraday relatives for stopped symbols.
            if stop_syms:
                # Build a rel vector that uses the stop execution price for sold names.
                rel_eff = rel_intraday.copy()
                for sym in stop_syms:
                    px0 = float(p_exec.get(sym, np.nan))
                    pxs = float(stop_exec_px.get(sym, np.nan))
                    if np.isfinite(px0) and px0 > 0 and np.isfinite(pxs) and pxs > 0:
                        rel_eff.loc[sym] = pxs / px0

                # Transaction costs on stop trades (approx): sell notional ~= weight at exec.
                # Turnover includes cash leg, so we use 2*sum(weights_sold).
                sold_w = float(w_exec_post.reindex(stop_syms).sum())
                turnover_sl = 2.0 * max(0.0, sold_w)
                cost_sl = float(turnover_sl * tc) if tc > 0 else 0.0
                cost_sl = min(max(cost_sl, 0.0), 0.99)

                nav_close = nav_exec * float((w_exec_post * rel_eff).sum())
                nav_close = nav_close * (1.0 - cost_sl)

                # Close weights: move stopped value into cash.
                value = (w_exec_post * rel_eff).copy()
                if cash_enabled and cash_name in value.index:
                    cash_val = float(value.get(cash_name, 0.0))
                    for sym in stop_syms:
                        cash_val += float(value.get(sym, 0.0))
                        value.loc[sym] = 0.0
                    value.loc[cash_name] = cash_val
                # normalize
                tot = float(value.sum())
                if tot > 0 and np.isfinite(tot):
                    w_close_prev = value / tot
                else:
                    w_close_prev = w_exec_post.copy()

                # Record stop events
                for sym in stop_syms:
                    trade_rows.append({
                        "date": day,
                        "type": "stoploss",
                        "symbol": sym,
                        "exec_time": stop_exec_ts.get(sym),
                        "exec_price": stop_exec_px.get(sym),
                        "hard_stop_pct": sl_hard,
                        "trailing_stop_pct": sl_trail,
                    })
                trade_rows.append({
                    "date": day,
                    "type": "stoploss_cost",
                    "turnover": turnover_sl,
                    "tc": tc,
                    "cost": cost_sl,
                    "n_exits": int(len(stop_syms)),
                })
            else:
                nav_close = nav_exec * float((w_exec_post * rel_intraday).sum())
                w_close_prev = _weights_drift(w_exec_post, rel_intraday)

            peak_nav = max(peak_nav, nav_close)

            nav_idx.append(day)
            nav_vals.append(float(nav_close))
            weights_rows.append(w_close_prev.copy())

            close_hist.loc[day, :] = p_close.reindex(symbols).values
            p_close_prev = p_close.copy()

            if plotter is not None:
                plot_updates += 1
                if (plot_updates % live_plot_update_every) == 0:
                    try:
                        plotter.update(day, float(nav_close))
                    except Exception as e:
                        if logger:
                            logger.warning("Live plot disabled (update error): %s", repr(e))
                        plotter = None

                # Keep GUI responsive even if we don't add a point every step.
                if plotter is not None:
                    try:
                        plotter.pump()
                    except Exception:
                        pass

        # ---- end-of-month logging / flushing ----
        if log_month_nav and len(nav_vals) > month_start_nav_pos:
            seg = np.array(nav_vals[month_start_nav_pos:], dtype=float)
            nav_end = float(seg[-1])
            nav_start = float(seg[0])
            mtd = (nav_end / nav_start - 1.0) if nav_start > 0 else np.nan
            # month max drawdown
            peak = np.maximum.accumulate(seg)
            dd = np.min(seg / peak - 1.0) if len(seg) else 0.0
            n_trades = len(trade_rows) - month_trade_pos
            elapsed = time.time() - month_t0

            if logger:
                logger.info(
                    "Replay month done | %04d-%02d | days=%d | NAV=%.4f | MTD=%+.2f%% | DD=%+.2f%% | trades=%d | %.1fs",
                    int(yy), int(mm), len(seg), nav_end, 100*mtd, 100*dd, n_trades, elapsed,
                )

        if flush_outputs_each_month:
            _flush_partial_outputs(nav_idx, nav_vals, weights_rows, trade_rows)

        # keep memory stable for long runs
        gc.collect()

    if err_holder:
        raise err_holder[0]

    if plotter is not None:
        try:
            plotter.finalize()
        except Exception:
            pass

    nav_series = pd.Series(nav_vals, index=pd.DatetimeIndex(nav_idx, name="date"), name="nav")
    weights_df = pd.DataFrame(weights_rows, index=nav_series.index).reindex(columns=all_assets)

    trades_df = pd.DataFrame(trade_rows)
    if not trades_df.empty and "date" in trades_df.columns:
        trades_df = trades_df.sort_values("date")

    prices_snapshot = pd.concat(close_parts, axis=0).sort_index()
    if cash_enabled:
        prices_snapshot[cash_name] = 1.0


    # Optionally keep the live plot window open after the run finishes.
    # We also flush final outputs to disk first so you can inspect them in Prism.
    if plotter is not None and live_plot_hold_on_finish:
        try:
            _flush_partial_outputs(nav_idx, nav_vals, weights_rows, trade_rows)
        except Exception:
            pass
        try:
            logger.info("Replay finished. Close the live NAV window to exit.")
        except Exception:
            pass
        try:
            plotter.wait_until_closed()
        except Exception:
            pass

    return {
        "nav": nav_series,
        "weights": weights_df,
        "trades": trades_df,
        "prices": prices_snapshot,
    }
