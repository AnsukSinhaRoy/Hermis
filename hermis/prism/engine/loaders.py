
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import pandas as pd
import streamlit as st

from .utils import ensure_series


@st.cache_data(show_spinner=False)
def load_experiment(path: str) -> Dict[str, Any]:
    """Load experiment outputs from `path` (experiment folder).

    Returns dict with keys:
      - nav: pd.Series or None
      - weights: pd.DataFrame or None
      - trades: pd.DataFrame or None
      - meta: dict (from <exp>/metadata.json if present)
      - params: dict (from <exp>/params.yaml if present)
      - params_path: str | None
    """
    p = Path(path)
    out = p / "outputs"

    data: Dict[str, Any] = {}
    # Prefer portfolio_sim loader (handles parquet/csv fallback consistently)
    try:
        from portfolio_sim.experiment import load_experiment as load_sim_experiment  # type: ignore
        data = load_sim_experiment(str(p)) or {}
    except Exception:
        # Fallback (parquet preferred; csv fallback)
        def _read_df(stem: str):
            pq = out / f"{stem}.parquet"
            csv = out / f"{stem}.csv"
            if pq.exists():
                try:
                    return pd.read_parquet(pq)
                except Exception:
                    if csv.exists():
                        return pd.read_csv(csv, index_col=0, parse_dates=True)
            if csv.exists():
                return pd.read_csv(csv, index_col=0, parse_dates=True)
            return None

        nav_df = _read_df("nav")
        nav = None
        if nav_df is not None:
            if isinstance(nav_df, pd.DataFrame) and nav_df.shape[1] >= 1:
                nav = nav_df.iloc[:, 0]
            elif isinstance(nav_df, pd.Series):
                nav = nav_df

        data = {
            "nav": nav,
            "weights": _read_df("weights"),
            "trades": _read_df("trades"),
        }

    # Normalize nav to Series
    data["nav"] = ensure_series(data.get("nav"))

    # Metadata lives at experiment root (written by portfolio_sim.experiment)
    meta_path = p / "metadata.json"
    if meta_path.exists():
        try:
            data["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            data["meta"] = {}
    else:
        data["meta"] = data.get("meta", {}) or {}

    # Params snapshot lives at experiment root
    params_path = p / "params.yaml"
    data["params_path"] = str(params_path) if params_path.exists() else None
    if params_path.exists():
        try:
            data["params"] = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
        except Exception:
            data["params"] = {}
    else:
        data["params"] = {}

    return data


def load_experiment_list(experiments_root: Path) -> List[Path]:
    """Return list of experiment directories containing an `outputs` folder."""
    if not experiments_root.exists():
        return []
    dirs = [p for p in experiments_root.iterdir() if p.is_dir() and (p / 'outputs').exists()]
    return sorted(dirs, reverse=True)



@st.cache_data(show_spinner=False)
def load_nav_only(path: str) -> Dict[str, Any]:
    """Fast path: load only NAV + meta/params (skip weights/trades)."""
    data = load_experiment(path)
    return {
        "nav": data.get("nav"),
        "meta": data.get("meta", {}),
        "params": data.get("params", {}),
        "params_path": data.get("params_path"),
    }


@st.cache_data(show_spinner=False)
def load_weights_only(path: str) -> Optional[pd.DataFrame]:
    p = Path(path) / "outputs"
    pq = p / "weights.parquet"
    csv = p / "weights.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if csv.exists():
        try:
            return pd.read_csv(csv, index_col=0, parse_dates=True)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_trades_only(path: str) -> Optional[pd.DataFrame]:
    p = Path(path) / "outputs"
    pq = p / "trades.parquet"
    csv = p / "trades.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if csv.exists():
        try:
            return pd.read_csv(csv, index_col=0, parse_dates=True)
        except Exception:
            return None
    return None
@st.cache_data(show_spinner=False)
def safe_read_prices(exp_folder: Path) -> Optional[pd.DataFrame]:
    """Read the exact prices snapshot saved for an experiment."""
    pq = exp_folder / 'data' / 'prices.parquet'
    csv = exp_folder / 'data' / 'prices.csv'
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            st.warning(f"Failed to read prices.parquet ({e}); trying CSV if present.")
    if csv.exists():
        try:
            return pd.read_csv(csv, index_col=0, parse_dates=True)
        except Exception as e:
            st.error(f"Failed to read prices.csv: {e}")
            return None
    return None


@st.cache_data(show_spinner=False)
def load_precomputed_perf(exp_folder: Path) -> Dict[str, Any]:
    """Load performance artifacts written by run_experiment.py."""
    perf: Dict[str, Any] = {}
    perf_dir = Path(exp_folder) / 'outputs' / 'performance'
    if not perf_dir.exists():
        return perf
    try:
        if (perf_dir / 'metrics.json').exists():
            perf['metrics'] = json.loads((perf_dir / 'metrics.json').read_text(encoding='utf-8'))
        for fname in ['rolling_sharpe', 'drawdown', 'cum_returns']:
            pq = perf_dir / f'{fname}.parquet'
            csv = perf_dir / f'{fname}.csv'
            if pq.exists():
                try:
                    perf[fname] = pd.read_parquet(pq).iloc[:, 0]
                    continue
                except Exception:
                    pass
            if csv.exists():
                try:
                    perf[fname] = pd.read_csv(csv, index_col=0, parse_dates=True).iloc[:, 0]
                except Exception:
                    pass
    except Exception as e:
        st.warning(f"Could not load some precomputed performance files: {e}")
    return perf


@st.cache_data(show_spinner=False)
def load_all_experiment_parameters_and_metrics(root_path: Path) -> pd.DataFrame:
    all_experiments_data = []
    exp_paths = load_experiment_list(root_path)
    for path in exp_paths:
        params_path = path / 'params.yaml'
        metrics_path = path / 'outputs' / 'performance' / 'metrics.json'
        if params_path.exists() and metrics_path.exists():
            try:
                params = yaml.safe_load(params_path.read_text(encoding='utf-8')) or {}
                params_flat = pd.json_normalize(params, sep='.')
                params_dict = params_flat.to_dict(orient='records')[0] if not params_flat.empty else {}
                metrics = json.loads(metrics_path.read_text(encoding='utf-8')) or {}
                combined_data = {**params_dict, **metrics, 'experiment_name': path.name}
                all_experiments_data.append(combined_data)
            except Exception as e:
                st.warning(f"Could not process experiment {path.name}: {e}")
    if not all_experiments_data:
        return pd.DataFrame()
    return pd.DataFrame(all_experiments_data)