#!/usr/bin/env python3
"""
examples/run_experiment.py

Run an experiment using portfolio_sim runner_func, ensure the exact prices
used are saved into the experiment folder, and precompute performance
artifacts to make the Streamlit app fast to load.
"""
from pathlib import Path
import time
import json

import pandas as pd
import numpy as np

# portfolio_sim imports (local package)
from portfolio_sim.config import Config
from portfolio_sim.data import generate_synthetic_prices, load_prices_from_csv, compute_returns, cov_matrix
from portfolio_sim.optimizer import greedy_k_cardinality, mv_reg_optimize, min_variance_optimize, risk_parity_optimize
from portfolio_sim.backtest import run_backtest
from portfolio_sim.experiment import run_experiment_from_config, load_experiment
from portfolio_sim import viz

# path to your config
cfg_path = "configs/newconfig.yaml"


def expected_return_estimator(prices_window: pd.DataFrame):
    """Simple estimator: mean of log returns (in-sample)."""
    rets = compute_returns(prices_window, method='log')
    if rets is None or rets.shape[0] == 0:
        return pd.Series(dtype=float)
    return rets.mean()


def cov_estimator_factory(use_gpu: bool = False):
    """Return a callable that computes covariance for a rolling window."""
    def cov_estimator(prices_window: pd.DataFrame):
        return cov_matrix(prices_window, method='log', use_gpu=use_gpu)
    return cov_estimator


def optimizer_wrapper_factory_from_cfg(cfg):
    """
    Return optimizer_func(expected: pd.Series, cov: pd.DataFrame) -> dict
    that our backtester expects. Reads optimizer settings from cfg.
    """
    opt_cfg = cfg.get('optimizer', {})
    opt_type = opt_cfg.get('type', 'mv_reg')
    k = opt_cfg.get('k_cardinality', None)
    box_cfg = opt_cfg.get('box', None)
    box = None
    if box_cfg is not None:
        box = {"min": box_cfg.get("min", None), "max": box_cfg.get("max", None)}
    long_only = bool(opt_cfg.get('long_only', True))
    lambdas = opt_cfg.get('lambdas', None)

    def optimizer_func(expected: pd.Series, cov: pd.DataFrame):
        if expected is None or cov is None:
            return {"weights": None, "status": "invalid_inputs"}
        if k is not None and int(k) > 0:
            return greedy_k_cardinality(expected, cov, k=int(k), method=opt_type, box=box, long_only=long_only, lambdas=lambdas)
        if opt_type == "mv_reg":
            return mv_reg_optimize(expected, cov, lambdas=lambdas, box=box, long_only=long_only)
        elif opt_type == "minvar":
            return min_variance_optimize(cov, box=box, long_only=long_only)
        elif opt_type == "risk_parity":
            return risk_parity_optimize(cov, long_only=long_only, box=box)
        else:
            return mv_reg_optimize(expected, cov, lambdas=lambdas, box=box, long_only=long_only)
    return optimizer_func


def runner_func(cfg: dict):
    """
    Runner used by run_experiment_from_config.
    Loads prices (synthetic/processed/csv), builds estimators & optimizer, runs backtest,
    and RETURNS a dict containing nav/weights/trades/turnover/prices/meta so the experiment
    saving routine can snapshot the exact prices and outputs.
    """
    dcfg = cfg.get('data', {})
    exp_cfg = cfg.get('experiment', {})
    use_gpu = bool(exp_cfg.get('use_gpu', False))

    # 1) load prices
    mode = dcfg.get('mode', 'synthetic')
    prices = None
    if mode == 'synthetic':
        prices = generate_synthetic_prices(
            n_assets=dcfg.get('n_assets', 100),
            start=exp_cfg.get('start_date'),
            end=exp_cfg.get('end_date'),
            seed=exp_cfg.get('seed', None)
        )
    elif mode == 'processed':
        processed_path = dcfg.get('processed_path')
        if processed_path is None:
            raise ValueError("data.mode == 'processed' but data.processed_path is not set in config")
        prices = pd.read_parquet(processed_path)
    elif mode == 'csv':
        csv_path = dcfg.get('csv_path')
        if csv_path is None:
            raise ValueError("data.mode == 'csv' but data.csv_path is not set in config")
        prices = load_prices_from_csv(csv_path)
    else:
        raise ValueError(f"Unknown data.mode: {mode!r}")

    # 2) estimators
    expected_return_estimator_callable = expected_return_estimator
    cov_estimator_callable = cov_estimator_factory(use_gpu=use_gpu)

    # 3) optimizer wrapper from cfg
    optimizer_func = optimizer_wrapper_factory_from_cfg(cfg)

    # 4) backtest config
    import time
    from pathlib import Path

    # base experiment directory that will be created by run_experiment_from_config
    # If run_experiment_from_config doesn't create the folder before calling runner_func,
    # we keep logs in ./runs/ for visibility and the file is printed so you can move it later.
    runs_dir = Path.cwd() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    live_log_filename = f"live_{ts}.log"
    log_path = str(runs_dir / live_log_filename)

    print(f"[runner] Writing live backtest log to: {log_path}")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)        

    backtest_cfg = {
    "rebalance": exp_cfg.get("rebalance", "monthly"),
    "transaction_costs": cfg.get("transaction_costs", {}),
    # NEW: enable verbose per-step logging and write updates to log_path
    "verbose": True,
    "log_path": log_path,
    }

    # 5) run backtest
    bt_result = run_backtest(prices,
                             expected_return_estimator_callable,
                             cov_estimator_callable,
                             optimizer_func,
                             backtest_cfg)

    # 6) produce output dict (include prices snapshot and meta)
    out = {
        "nav": getattr(bt_result, "nav", None),
        "weights": getattr(bt_result, "weights", None),
        "turnover": getattr(bt_result, "turnover", None),
        "trades": getattr(bt_result, "trades", None),
        "prices": prices,
        "meta": {
            "cfg": cfg,
            "n_assets": None if prices is None else prices.shape[1],
            "price_index_start": None if prices is None else str(prices.index.min()),
            "price_index_end": None if prices is None else str(prices.index.max()),
        }
    }
    return out


def _ann_stats(nav_s: pd.Series):
    """Compute a small set of annualized stats for JSON-friendly saving."""
    if nav_s is None or len(nav_s) < 2:
        return {"total_return": None, "ann_return": None, "ann_vol": None, "max_drawdown": None}
    days = (nav_s.index[-1] - nav_s.index[0]).days or 1
    total_ret = float(nav_s.iloc[-1] / nav_s.iloc[0] - 1.0)
    ann_ret = (1 + total_ret) ** (365.0 / days) - 1
    dr = nav_s.pct_change().dropna()
    ann_vol = float(dr.std() * np.sqrt(252)) if not dr.empty else 0.0
    roll_max = nav_s.cummax()
    drawdown = (nav_s - roll_max) / roll_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    return {"total_return": total_ret, "ann_return": ann_ret, "ann_vol": ann_vol, "max_drawdown": max_dd}


def main():
    experiments_base = "experiments"
    Path(experiments_base).mkdir(exist_ok=True)

    print("Running experiment from config:", cfg_path)
    exp_folder = run_experiment_from_config(cfg_path, experiments_base, runner_func)
    print("Experiment saved to:", exp_folder)

    # load outputs saved by run_experiment_from_config
    out = load_experiment(str(exp_folder))
    nav = out.get("nav")
    weights = out.get("weights")
    trades = out.get("trades")
    meta = out.get("meta", {})

    # Determine prices used in the experiment (prefer exact saved snapshot)
    prices = None
    prices_path = Path(exp_folder) / "data" / "prices.parquet"
    if prices_path.exists():
        try:
            prices = pd.read_parquet(prices_path)
            print(f"Loaded experiment snapshot prices from {prices_path}")
        except Exception as e:
            print(f"Warning: failed to read saved prices snapshot {prices_path}: {e}")

    # fallback: if snapshot not present, check config or synthetic generator
    if prices is None:
        try:
            cfg = Config.load(cfg_path).raw
            dcfg = cfg.get("data", {})
            exp_cfg = cfg.get("experiment", {})
            processed_path = dcfg.get("processed_path", None)
            csv_path = dcfg.get("csv_path", None)
            mode = dcfg.get("mode", "synthetic")
            if processed_path and Path(processed_path).exists():
                prices = pd.read_parquet(processed_path)
                print(f"Loaded processed prices from {processed_path}")
            elif mode == "synthetic":
                prices = generate_synthetic_prices(
                    n_assets=dcfg.get("n_assets", 100),
                    start=exp_cfg.get("start_date"),
                    end=exp_cfg.get("end_date"),
                    seed=exp_cfg.get("seed", None)
                )
                print("Generated synthetic prices (mode=synthetic).")
            elif mode == "csv" and csv_path:
                prices = load_prices_from_csv(csv_path)
                print(f"Loaded prices from CSV at {csv_path}")
            else:
                print("No prices snapshot found and no valid fallback available.")
        except Exception as e:
            print("Warning while attempting fallback price load:", e)

    if prices is None:
        raise RuntimeError("Failed to obtain prices from snapshot, processed_path, synthetic generator, or csv.")

    # Save / show figures into experiment artifacts
    figures_dir = Path(exp_folder) / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        if nav is not None:
            viz.plot_nav(nav, save_path=str(figures_dir / "nav.png"))
        if weights is not None:
            viz.plot_allocation_heatmap(weights, save_path=str(figures_dir / "alloc.png"))
        # plot prices and nav (saves to file)
        viz.plot_prices_and_nav(
            prices,
            nav,
            normalize_prices=True,
            sample_max_assets=None,
            alpha=0.25,
            figsize=(14, 6),
            save_path=str(figures_dir / "prices_and_nav.png")
        )
        print("Saved figures to:", str(figures_dir))
    except Exception as e:
        print("Warning while saving figures:", e)

    # Precompute and save performance artifacts (into outputs/performance/)
    out_dir = Path(exp_folder) / "outputs"
    perf_dir = out_dir / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)

    try:
        if nav is not None:
            # metrics.json
            metrics = _ann_stats(nav)
            with open(perf_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # rolling_sharpe (63 day)
            returns = nav.pct_change().dropna()
            if not returns.empty:
                win = 63
                r_mean = returns.rolling(win).mean()
                r_std = returns.rolling(win).std()
                rs = (r_mean / r_std) * np.sqrt(252)
                rs = rs.dropna()
                if not rs.empty:
                    rs.to_frame(name="rolling_sharpe").to_parquet(perf_dir / "rolling_sharpe.parquet")

            # drawdown series
            roll_max = nav.cummax()
            drawdown = ((nav - roll_max) / roll_max).to_frame(name="drawdown")
            drawdown.to_parquet(perf_dir / "drawdown.parquet")

            # cumulative returns
            cum = (nav / float(nav.iloc[0])).to_frame(name="cum_returns")
            cum.to_parquet(perf_dir / "cum_returns.parquet")

        # save selected mapping if trades exist and include 'selected'
        if trades is not None:
            try:
                if 'selected' in trades.columns:
                    # trades['date'] might be a column; ensure index is date
                    sel = trades.set_index('date')['selected'] if 'date' in trades.columns else trades['selected']
                    sel.to_frame(name="selected").to_parquet(perf_dir / "selected.parquet")
            except Exception:
                pass

        print(f"Saved performance artifacts to {perf_dir}")
    except Exception as e:
        print("Warning: failed to write performance artifacts:", e)


if __name__ == "__main__":
    main()


