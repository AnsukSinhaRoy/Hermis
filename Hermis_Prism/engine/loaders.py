import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import pandas as pd
import streamlit as st

from .utils import ensure_series, ann_stats

@st.cache_data
def load_experiment(path: str) -> Dict[str, Any]:
    """Load experiment outputs from `path`. Tries to import `portfolio_sim.experiment` first."""
    try:
        from portfolio_sim.experiment import load_experiment as load_sim_experiment
        return load_sim_experiment(path)
    except (ImportError, ModuleNotFoundError):
        out = Path(path) / "outputs"
        nav = pd.read_parquet(out / "nav.parquet") if (out / "nav.parquet").exists() else None
        weights = pd.read_parquet(out / "weights.parquet") if (out / "weights.parquet").exists() else None
        trades = pd.read_parquet(out / "trades.parquet") if (out / "trades.parquet").exists() else None
        meta = json.load(open(out / "metadata.json")) if (out / "metadata.json").exists() else {}
        return {"nav": nav, "weights": weights, "trades": trades, "meta": meta}


def load_experiment_list(experiments_root: Path) -> List[Path]:
    """Return list of experiment directories containing an `outputs` folder."""
    if not experiments_root.exists():
        return []
    dirs = [p for p in experiments_root.iterdir() if p.is_dir() and (p / 'outputs').exists()]
    return sorted(dirs, reverse=True)

@st.cache_data
def safe_read_prices(exp_folder: Path) -> Optional[pd.DataFrame]:
    p = exp_folder / 'data' / 'prices.parquet'
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as e:
            st.error(f"Failed to read prices file: {e}")
            return None
    return None

@st.cache_data
def load_precomputed_perf(exp_folder: Path) -> Dict[str, Any]:
    perf = {}
    perf_dir = Path(exp_folder) / 'outputs' / 'performance'
    if not perf_dir.exists():
        return perf
    try:
        if (perf_dir / 'metrics.json').exists():
            with open(perf_dir / 'metrics.json') as f:
                perf['metrics'] = json.load(f)
        for fname in ['rolling_sharpe', 'drawdown', 'cum_returns']:
            fpath = perf_dir / f'{fname}.parquet'
            if fpath.exists():
                perf[fname] = pd.read_parquet(fpath).iloc[:, 0]
    except Exception as e:
        st.warning(f"Could not load some precomputed performance files: {e}")
    return perf

@st.cache_data
def load_all_experiment_parameters_and_metrics(root_path: Path) -> pd.DataFrame:
    all_experiments_data = []
    exp_paths = load_experiment_list(root_path)
    for path in exp_paths:
        params_path = path / 'params.yaml'
        metrics_path = path / 'outputs' / 'performance' / 'metrics.json'
        if params_path.exists() and metrics_path.exists():
            try:
                with open(params_path, 'r') as f:
                    params = yaml.safe_load(f)
                params_flat = pd.json_normalize(params, sep='.')
                params_dict = params_flat.to_dict(orient='records')[0]
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                combined_data = {**params_dict, **metrics}
                combined_data['experiment_name'] = path.name
                all_experiments_data.append(combined_data)
            except Exception as e:
                st.warning(f"Could not process experiment {path.name}: {e}")
    if not all_experiments_data:
        return pd.DataFrame()
    return pd.DataFrame(all_experiments_data)