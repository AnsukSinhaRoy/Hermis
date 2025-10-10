# examples/run_experiment.py
import pandas as pd
from pathlib import Path
from portfolio_sim.config import Config
from portfolio_sim.data import generate_synthetic_prices, load_prices_from_csv, compute_returns, cov_matrix
from portfolio_sim.optimizer import greedy_k_cardinality, mv_optimize
from portfolio_sim.backtest import run_backtest
from portfolio_sim.experiment import run_experiment_from_config
from portfolio_sim import viz

def expected_return_estimator(prices_window: pd.DataFrame):
    rets = compute_returns(prices_window, method='log')
    return rets.mean()

def cov_estimator_factory(use_gpu: bool = False):
    def cov_estimator(prices_window: pd.DataFrame):
        return cov_matrix(prices_window, method='log', use_gpu=use_gpu)
    return cov_estimator

def optimizer_wrapper_factory(cfg):
    k = cfg['optimizer'].get('k_cardinality')
    box = None
    if cfg['optimizer'].get('box') is not None:
        b = cfg['optimizer']['box']
        box = (b['min'], b['max'])
    long_only = cfg['optimizer'].get('long_only', True)
    def optimizer_func(expected, cov):
        if k is None:
            return mv_optimize(expected, cov, target_return=None, box=box, long_only=long_only)
        else:
            return greedy_k_cardinality(expected, cov, k=k, box=box, long_only=long_only)
    return optimizer_func

def runner_func(cfg: dict):
    dcfg = cfg['data']
    exp_cfg = cfg['experiment']
    use_gpu = bool(exp_cfg.get('use_gpu', False))
    if dcfg['mode'] == 'synthetic':
        prices = generate_synthetic_prices(n_assets=100,
                                           start=exp_cfg.get('start_date'),
                                           end=exp_cfg.get('end_date'),
                                           seed=exp_cfg.get('seed', None))
    else:
        prices = load_prices_from_csv(dcfg['csv_path'])
    cov_estimator = cov_estimator_factory(use_gpu=use_gpu)
    optimizer = optimizer_wrapper_factory(cfg)
    backtest_cfg = {'rebalance': exp_cfg.get('rebalance', 'monthly'),
                    'transaction_costs': cfg.get('transaction_costs', {})}
    result = run_backtest(prices, expected_return_estimator, cov_estimator, optimizer, backtest_cfg)
    meta = {
        "seed": exp_cfg.get('seed'),
        "use_gpu": use_gpu,
        "optimizer": cfg['optimizer'],
        "risk_model": cfg['risk_model']
    }
    return {"nav": result.nav, "weights": result.weights, "trades": result.trades, "meta": meta}

def main():
    # local imports so the function is drop-in
    from pathlib import Path
    import pandas as pd

    from portfolio_sim.config import Config
    from portfolio_sim.data import generate_synthetic_prices, load_prices_from_csv
    from portfolio_sim.experiment import run_experiment_from_config, load_experiment
    from portfolio_sim import viz
    from portfolio_sim import data as data_module  # to access names if needed

    cfg_path = "configs/example_experiment.yaml"
    experiments_base = "experiments"
    Path(experiments_base).mkdir(exist_ok=True)

    # Run the experiment (this will create an experiment folder and save outputs)
    exp_folder = run_experiment_from_config(cfg_path, experiments_base, runner_func)
    print("Experiment saved to:", exp_folder)

    # Load experiment outputs (nav, weights, trades, meta)
    out = load_experiment(str(exp_folder))
    nav = out["nav"]
    weights = out["weights"]
    meta = out.get("meta", {})

    # -----------------------------
    # Determine prices used in the experiment
    # -----------------------------
    # Prefer exact saved snapshot if present
    prices_path = Path(exp_folder) / "data" / "prices.parquet"
    if prices_path.exists():
        try:
            prices = pd.read_parquet(prices_path)
        except Exception:
            prices = None
    else:
        prices = None

    # If no saved prices, reconstruct based on YAML config
    if prices is None:
        cfg = Config.load(cfg_path).raw
        dcfg = cfg.get("data", {})
        exp_cfg = cfg.get("experiment", {})

        if dcfg.get("mode", "synthetic") == "synthetic":
            # read n_assets if user added it to YAML, else default to 100
            n_assets = dcfg.get("n_assets", 100)
            prices = generate_synthetic_prices(
                n_assets=n_assets,
                start=exp_cfg.get("start_date"),
                end=exp_cfg.get("end_date"),
                seed=exp_cfg.get("seed", None)
            )
        else:
            # csv mode: load file specified in csv_path
            csv_path = dcfg.get("csv_path", "")
            if not csv_path:
                raise ValueError("data.mode == 'csv' but csv_path is empty in config.")
            prices = load_prices_from_csv(csv_path)

    # -----------------------------
    # Save / show figures into experiment artifacts
    # -----------------------------
    figures_dir = Path(exp_folder) / "artifacts" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) NAV
    viz.plot_nav(nav, save_path=str(figures_dir / "nav.png"))

    # 2) Allocation heatmap (top 20 by avg weight)
    viz.plot_allocation_heatmap(weights, save_path=str(figures_dir / "alloc.png"))

    # 3) All prices (normalized) + NAV overlay
    # Set sample_max_assets: None to plot all, or e.g., 200 to limit clutter
    viz.plot_prices_and_nav(
        prices,
        nav,
        normalize_prices=True,
        sample_max_assets=None,   # adjust if your universe is large
        alpha=0.25,
        figsize=(14, 6),
        save_path=str(figures_dir / "prices_and_nav.png")
    )

    print("Saved figures to:", str(figures_dir))

if __name__ == "__main__":
    main()
