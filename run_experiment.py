# examples/run_experiment.py
import pandas as pd
from pathlib import Path
from portfolio_sim.config import Config
from portfolio_sim.data import generate_synthetic_prices, load_prices_from_csv, compute_returns, cov_matrix
from portfolio_sim.optimizer import greedy_k_cardinality, mv_optimize
from portfolio_sim.backtest import run_backtest
from portfolio_sim.experiment import run_experiment_from_config
from portfolio_sim import viz
from portfolio_sim.optimizer import mv_reg_optimize, min_variance_optimize, risk_parity_optimize, greedy_k_cardinality

cfg_path = "configs/newconfig.yaml"

def expected_return_estimator(prices_window: pd.DataFrame):
    rets = compute_returns(prices_window, method='log')
    return rets.mean()

def cov_estimator_factory(use_gpu: bool = False):
    def cov_estimator(prices_window: pd.DataFrame):
        return cov_matrix(prices_window, method='log', use_gpu=use_gpu)
    return cov_estimator

def optimizer_wrapper_factory_from_cfg(cfg):
    """
    Build an optimizer function that matches run_backtest's expectation:
      optimizer_func(expected_series, cov_df) -> dict with 'weights' and 'status'
    This wrapper will read optimizer settings from cfg and dispatch to the right method.
    """
    opt_cfg = cfg['optimizer']
    opt_type = opt_cfg.get('type', 'mv_reg')
    k = opt_cfg.get('k_cardinality', None)  # could be None or int
    box_cfg = opt_cfg.get('box', None)
    box = None
    if box_cfg is not None:
        # keep as dict for our optimizers which expect dicts not tuples
        box = {"min": box_cfg.get("min", None), "max": box_cfg.get("max", None)}
    long_only = bool(opt_cfg.get('long_only', True))

    # pass through lambda list or other optimizer params
    lambdas = opt_cfg.get('lambdas', None)

    def optimizer_func(expected: pd.Series, cov: pd.DataFrame):
        # ensure expected and cov are aligned and non-empty
        if expected is None or cov is None:
            return {"weights": None, "status": "invalid_inputs"}
        # if k specified and >0, use greedy_k_cardinality selection first
        if k is not None and int(k) > 0:
            # greedy_k_cardinality will call requested method on subset
            return greedy_k_cardinality(expected, cov, k=int(k), method=opt_type, box=box, long_only=long_only, lambdas=lambdas)
        # otherwise call optimizer on full universe
        if opt_type == "mv_reg":
            return mv_reg_optimize(expected, cov, lambdas=lambdas, box=box, long_only=long_only)
        elif opt_type == "minvar":
            return min_variance_optimize(cov, box=box, long_only=long_only)
        elif opt_type == "risk_parity":
            return risk_parity_optimize(cov, long_only=long_only, box=box)
        else:
            # fallback to mv_reg
            return mv_reg_optimize(expected, cov, lambdas=lambdas, box=box, long_only=long_only)

    return optimizer_func


def runner_func(cfg: dict):
    """
    Runner that:
      - loads prices
      - builds estimator callables
      - builds optimizer wrapper
      - runs run_backtest and returns a dict with expected keys
    """
    dcfg = cfg['data']
    exp_cfg = cfg['experiment']
    use_gpu = bool(exp_cfg.get('use_gpu', False))

    # 1) load prices according to data.mode
    if dcfg['mode'] == 'synthetic':
        prices = generate_synthetic_prices(
            n_assets=dcfg.get('n_assets', 100),
            start=exp_cfg.get('start_date'),
            end=exp_cfg.get('end_date'),
            seed=exp_cfg.get('seed', None)
        )
    elif dcfg['mode'] == 'processed':
        prices = pd.read_parquet(dcfg['processed_path'])
    else:
        prices = load_prices_from_csv(dcfg['csv_path'])

    # 2) estimator callables
    expected_return_estimator_callable = expected_return_estimator  # function, DO NOT CALL
    cov_estimator_callable = cov_estimator_factory(use_gpu=use_gpu)

    # 3) optimizer wrapper from config
    optimizer_func = optimizer_wrapper_factory_from_cfg(cfg)

    # 4) build backtest config (what run_backtest expects)
    backtest_cfg = {
        "rebalance": exp_cfg.get("rebalance", "monthly"),
        "transaction_costs": cfg.get("transaction_costs", {}),
    }

    # 5) run backtest (returns BacktestResult)
    bt_result = run_backtest(prices,
                             expected_return_estimator_callable,
                             cov_estimator_callable,
                             optimizer_func,
                             backtest_cfg)

    # 6) convert BacktestResult -> plain dict expected by run_experiment_from_config
    # bt_result has attributes: nav (pd.Series), weights (pd.DataFrame), turnover (pd.Series), trades (pd.DataFrame)
    out = {
        "nav": getattr(bt_result, "nav", None),
        "weights": getattr(bt_result, "weights", None),
        "turnover": getattr(bt_result, "turnover", None),
        "trades": getattr(bt_result, "trades", None),
        "meta": {
            "cfg": cfg,
            "n_assets": None if prices is None else prices.shape[1],
            "price_index_start": None if prices is None else str(prices.index.min()),
            "price_index_end": None if prices is None else str(prices.index.max()),
        }
    }

    return out



def main():
    # local imports so the function is drop-in
    from pathlib import Path
    import pandas as pd

    from portfolio_sim.config import Config
    from portfolio_sim.data import generate_synthetic_prices, load_prices_from_csv
    from portfolio_sim.experiment import run_experiment_from_config, load_experiment
    from portfolio_sim import viz
    from portfolio_sim import data as data_module  # to access names if needed

    
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
    # 1) Prefer exact saved snapshot if present in the experiment folder
    prices = None
    prices_path = Path(exp_folder) / "data" / "prices.parquet"
    if prices_path.exists():
        try:
            prices = pd.read_parquet(prices_path)
            print(f"Loaded experiment snapshot prices from {prices_path}")
        except Exception as e:
            print(f"Warning: failed to read saved prices snapshot {prices_path}: {e}")

    # 2) If no saved snapshot, consult config. Prefer processed_path (if present on disk).
    if prices is None:
        cfg = Config.load(cfg_path).raw
        dcfg = cfg.get("data", {})
        exp_cfg = cfg.get("experiment", {})

        processed_path = dcfg.get("processed_path", None)
        csv_path = dcfg.get("csv_path", None)
        mode = dcfg.get("mode", "synthetic")

        # If a processed parquet path is provided and exists, use it regardless of mode
        if processed_path is not None and Path(processed_path).exists():
            try:
                prices = pd.read_parquet(processed_path)
                print(f"Loaded processed prices from {processed_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to read processed_path '{processed_path}': {e}") from e
        else:
            # fallback by declared mode
            if mode == "synthetic":
                n_assets = dcfg.get("n_assets", 100)
                prices = generate_synthetic_prices(
                    n_assets=n_assets,
                    start=exp_cfg.get("start_date"),
                    end=exp_cfg.get("end_date"),
                    seed=exp_cfg.get("seed", None)
                )
                print("Generated synthetic prices (mode=synthetic).")
            elif mode == "processed":
                # user requested processed but no valid file found
                raise FileNotFoundError(
                    f"data.mode == 'processed' but processed_path is not provided or does not exist: {processed_path}"
                )
            elif mode == "csv":
                # CSV mode: require csv_path or fallback to processed_path if exists
                if csv_path:
                    prices = load_prices_from_csv(csv_path)
                    print(f"Loaded prices from CSV at {csv_path}")
                else:
                    # final fallback: if processed_path was provided but missing, error
                    raise ValueError("data.mode == 'csv' but csv_path is empty in config and no processed_path found.")
            else:
                raise ValueError(f"Unknown data.mode: {mode!r} in config {cfg_path}")

    # at this point `prices` must be set
    if prices is None:
        raise RuntimeError("Failed to obtain prices from snapshot, processed_path, synthetic generator, or csv.")

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
