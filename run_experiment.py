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
from portfolio_sim.optimizer import greedy_k_cardinality, mv_reg_optimize, risk_parity_optimize
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
    lambda_reg = opt_cfg.get('lambda_reg', 1.0) # Use lambda_reg

    def optimizer_func(expected: pd.Series, cov: pd.DataFrame):
        if expected is None or cov is None:
            return {"weights": None, "status": "invalid_inputs"}
        
        # Consolidate common arguments
        common_args = {"box": box, "long_only": long_only}

        if k is not None and int(k) > 0:
            return greedy_k_cardinality(expected, cov, k=int(k), method=opt_type, **common_args, lambda_reg=lambda_reg)
        
        if opt_type == "mv_reg":
            return mv_reg_optimize(expected, cov, **common_args, lambda_reg=lambda_reg)
        elif opt_type == "risk_parity":
            return risk_parity_optimize(cov, **common_args)
        elif opt_type == "sharpe":
            from portfolio_sim.optimizer import sharpe_optimize
            return sharpe_optimize(expected, cov, **common_args)
        else: # Default to mv_reg
            return mv_reg_optimize(expected, cov, **common_args, lambda_reg=lambda_reg)
            
    return optimizer_func


def runner_func(cfg: dict):
    """
    Runner used by run_experiment_from_config.
    Loads prices, builds estimators & optimizer, runs backtest,
    and RETURNS a dict containing results for the experiment saving routine.
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
        if not processed_path:
            raise ValueError("data.mode == 'processed' but data.processed_path is not set")
        prices = pd.read_parquet(processed_path)
    elif mode == 'csv':
        csv_path = dcfg.get('csv_path')
        if not csv_path:
            raise ValueError("data.mode == 'csv' but data.csv_path is not set")
        prices = load_prices_from_csv(csv_path)
    else:
        raise ValueError(f"Unknown data.mode: {mode!r}")

    # 2) estimators
    expected_return_estimator_callable = expected_return_estimator
    cov_estimator_callable = cov_estimator_factory(use_gpu=use_gpu)

    # 3) optimizer wrapper from cfg
    optimizer_func = optimizer_wrapper_factory_from_cfg(cfg)

    # 4) backtest config
    backtest_cfg = {
        "rebalance": exp_cfg.get("rebalance", "monthly"),
        "transaction_costs": cfg.get("transaction_costs", {}),
    }

    # 5) run backtest
    bt_result = run_backtest(prices,
                             expected_return_estimator_callable,
                             cov_estimator_callable,
                             optimizer_func,
                             backtest_cfg)

    # 6) produce output dict (include prices snapshot and meta)
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
        }
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
                # prefer column named 'value' or first numeric column
                if 'value' in nav_s.columns:
                    nav = nav_s['value'].astype(float)
                else:
                    # pick first numeric column
                    nav = nav_s.select_dtypes(include=['number']).iloc[:, 0].astype(float)
        elif isinstance(nav_s, pd.Series):
            nav = nav_s.astype(float)
        else:
            # attempt to build a Series
            nav = pd.Series(nav_s).astype(float)
    except Exception:
        # If conversion fails, return Nones
        return {"total_return": None, "ann_return": None, "ann_vol": None, "max_drawdown": None, "sharpe": None}

    if nav is None or len(nav) < 2:
        return {"total_return": None, "ann_return": None, "ann_vol": None, "max_drawdown": None, "sharpe": None}

    # Ensure index sorted and has datetime-like index for days difference
    try:
        nav = nav.sort_index()
    except Exception:
        pass

    # days between first and last (guard against zero)
    try:
        days = (pd.to_datetime(nav.index[-1]) - pd.to_datetime(nav.index[0])).days
        days = max(int(days), 1)
    except Exception:
        days = max(len(nav) - 1, 1)

    # total return (coerce to scalar)
    try:
        total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    except Exception:
        total_ret = None

    # annualized return (if total_ret computed)
    try:
        ann_ret = float((1 + total_ret) ** (365.0 / float(days)) - 1.0) if total_ret is not None else None
    except Exception:
        ann_ret = None

    # daily return series
    try:
        dr = nav.pct_change().dropna()
        ann_vol = float(dr.std() * np.sqrt(252)) if not dr.empty else None
    except Exception:
        ann_vol = None

    # max drawdown
    try:
        roll_max = nav.cummax()
        drawdown = (nav - roll_max) / roll_max
        # .min() returns scalar; coerce to float or None
        min_drawdown = drawdown.min()
        max_dd = float(min_drawdown) if pd.notna(min_drawdown) else None
    except Exception:
        max_dd = None

    # Sharpe: use annualized return divided by ann_vol (if both available and ann_vol>0)
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
        "sharpe": None if sharpe is None else float(sharpe)
    }



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
    
    # Precompute and save performance artifacts
    perf_dir = Path(exp_folder) / "outputs" / "performance"
    perf_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if nav is not None and not nav.empty:
            # Save metrics
            metrics = _ann_stats(nav)
            with open(perf_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Save rolling sharpe
            try:
                returns = nav.pct_change().dropna()
                if len(returns) > 63:
                    win = 63
                    r_mean = returns.rolling(win).mean()
                    r_std = returns.rolling(win).std()
                    rs = (r_mean / (r_std + 1e-8)) * np.sqrt(252)

                    # Convert rs to a single-column DataFrame for saving.
                    # If rs is DataFrame with a single column, take that column.
                    # If multiple columns, summarize by taking the row-wise mean (gives a single Sharpe series).
                    if isinstance(rs, pd.DataFrame):
                        if rs.shape[1] == 1:
                            rs_series = rs.iloc[:, 0]
                        else:
                            # Multiple columns: produce a single time series summary (mean across columns).
                            # This is better than failing; if you prefer a different aggregation, change this line.
                            rs_series = rs.mean(axis=1)
                    else:
                        # already a Series
                        rs_series = rs

                    # Ensure Series -> DataFrame with a stable column name, then save
                    rs_df = rs_series.to_frame(name="rolling_sharpe")
                    rs_df.to_parquet(perf_dir / "rolling_sharpe.parquet")
            except Exception as e:
                print(f"Warning: failed to write rolling sharpe artifact: {e}")

            # Save drawdown series
            try:
                roll_max = nav.cummax()
                drawdown = ((nav - roll_max) / roll_max)

                # If drawdown is a DataFrame:
                #  - if single column, take that column
                #  - if multiple columns, produce a single summary series:
                #      use the row-wise minimum (most negative drawdown across columns) as the portfolio drawdown summary.
                if isinstance(drawdown, pd.DataFrame):
                    if drawdown.shape[1] == 1:
                        dd_series = drawdown.iloc[:, 0]
                    else:
                        # multiple columns -> summarize to a single series for the UI
                        # We choose the per-row minimum so the drawdown plot shows the worst drawdown across assets.
                        dd_series = drawdown.min(axis=1)
                        # OPTIONAL: if you want to keep per-asset drawdowns as well, uncomment the next line:
                        # drawdown.to_parquet(perf_dir / "drawdown_per_asset.parquet")
                else:
                    # already a Series
                    dd_series = drawdown

                # Ensure a DataFrame with a stable column name is saved
                dd_df = dd_series.to_frame(name="drawdown")
                dd_df.to_parquet(perf_dir / "drawdown.parquet")

            except Exception as e:
                print(f"Warning: failed to write drawdown artifact: {e}")

            # Save cumulative returns
            cum = (nav / nav.iloc[0])
            cum.to_frame(name="cum_returns").to_parquet(perf_dir / "cum_returns.parquet")

        # --- FIX for artifact saving error ---
        # The backtester now returns trades with the date as the index.
        # We can simplify this logic to directly select the column as a DataFrame.
        if trades is not None and 'selected' in trades.columns and not trades.empty:
            # Select as a DataFrame and save
            trades[['selected']].to_parquet(perf_dir / "selected.parquet")

        print(f"Saved performance artifacts to {perf_dir}")

    except Exception as e:
        print(f"Warning: failed to write some performance artifacts: {e}")
        import traceback
        traceback.print_exc()

    # Load prices for plotting
    prices_path = Path(exp_folder) / "data" / "prices.parquet"
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
    else:
        print("Warning: Could not find prices snapshot for plotting.")
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
            print("Saved figures to:", str(figures_dir))
        except Exception as e:
            print(f"Warning: failed to save figures: {e}")


if __name__ == "__main__":
    main()