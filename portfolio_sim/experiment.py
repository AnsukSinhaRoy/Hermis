# portfolio_sim/experiment.py
import os
import shutil
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import pandas as pd

from typing import Dict, Any, Optional

def resolve_experiment_name(cfg: Dict[str, Any], name_override: Optional[str] = None) -> str:
    """
    Resolve an experiment/simulation name from config.

    Precedence:
      1) name_override (CLI --name)
      2) cfg['experiment']['name']
      3) cfg['experiment_name']
      4) cfg['sim_name']
      5) cfg['name']
      6) 'simulation'
    """
    if name_override:
        return str(name_override)

    exp = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
    if isinstance(exp, dict) and exp.get("name"):
        return str(exp["name"])

    for key in ("experiment_name", "sim_name", "name"):
        if isinstance(cfg, dict) and cfg.get(key):
            return str(cfg[key])

    return "simulation"


def make_experiment_folder(base_dir: str, name: str, tag: Optional[str]=None) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_tag = f"__{tag}" if tag else ""
    folder_name = f"{ts}__{name}{safe_tag}"
    exp_path = Path(base_dir) / folder_name
    exp_path.mkdir(parents=True, exist_ok=False)
    (exp_path / "outputs").mkdir()
    (exp_path / "data").mkdir()
    (exp_path / "logs").mkdir()
    (exp_path / "artifacts").mkdir()
    (exp_path / "artifacts" / "figures").mkdir(parents=True, exist_ok=True)
    return exp_path

def save_params_yaml(src_yaml_path: str, dest_folder: Path):
    dest = dest_folder / "params.yaml"
    src = Path(src_yaml_path)
    try:
        if src.resolve() == dest.resolve():
            return
    except Exception:
        pass

    try:
        shutil.copy2(src_yaml_path, dest)
    except shutil.SameFileError:
        return


def save_dataframe_as_parquet(df: pd.DataFrame, path: Path):
    """Write a DataFrame to parquet, with a CSV fallback if parquet engine is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=True)
    except ImportError:
        # No pyarrow/fastparquet installed.
        csv_path = path.with_suffix('.csv')
        df.to_csv(csv_path, index=True)
    except Exception:
        # last resort fallback
        csv_path = path.with_suffix('.csv')
        df.to_csv(csv_path, index=True)


def save_series_as_parquet(s: pd.Series, path: Path):
    """Write a Series to parquet (as a 1-col frame), with CSV fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = s.to_frame(name="value")
    try:
        df.to_parquet(path, index=True)
    except ImportError:
        csv_path = path.with_suffix('.csv')
        df.to_csv(csv_path, index=True)
    except Exception:
        csv_path = path.with_suffix('.csv')
        df.to_csv(csv_path, index=True)


def save_metadata(meta: Dict[str,Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

# Replace run_experiment_from_config in portfolio_sim/experiment.py with this full function
def run_experiment_from_config(params_path: str,
                               base_experiments_dir: str,
                               runner_func: Callable[[Dict[str,Any]], Dict[str,Any]],
                               name_override: Optional[str]=None,
                               tag: Optional[str]=None,
                               exp_folder: Optional[str]=None) -> Path:
    """Run a full experiment.

    If `exp_folder` is provided, outputs are written into that folder (it may already exist).
    Otherwise, a new folder is created under `base_experiments_dir` using make_experiment_folder().
    """
    import yaml

    with open(params_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    exp_name = resolve_experiment_name(cfg, name_override)

    if exp_folder is None:
        exp_folder_path = make_experiment_folder(base_experiments_dir, exp_name, tag)
    else:
        exp_folder_path = Path(exp_folder)
        exp_folder_path.mkdir(parents=True, exist_ok=True)
        # Ensure expected subfolders exist (make_experiment_folder creates these too).
        (exp_folder_path / "outputs").mkdir(parents=True, exist_ok=True)
        (exp_folder_path / "data").mkdir(parents=True, exist_ok=True)
        (exp_folder_path / "logs").mkdir(parents=True, exist_ok=True)
        (exp_folder_path / "artifacts" / "figures").mkdir(parents=True, exist_ok=True)

    # Always snapshot params into the experiment folder for reproducibility.
    save_params_yaml(params_path, exp_folder_path)

    start = time.time()
    outputs = runner_func(cfg)
    end = time.time()

    out_folder = exp_folder_path / "outputs"
    out_folder.mkdir(parents=True, exist_ok=True)

    nav = outputs.get("nav", None)
    weights = outputs.get("weights", None)
    trades = outputs.get("trades", None)
    prices = outputs.get("prices", None)  # exact data used, if provided

    if nav is not None:
        save_series_as_parquet(nav, out_folder / "nav.parquet")
    if weights is not None:
        save_dataframe_as_parquet(weights, out_folder / "weights.parquet")
    if trades is not None:
        save_dataframe_as_parquet(trades, out_folder / "trades.parquet")

    # Save prices snapshot (exact data used in experiment) for reproducible viz
    if prices is not None:
        try:
            save_dataframe_as_parquet(prices, exp_folder_path / "data" / "prices.parquet")
        except Exception:
            pass

    # metadata
    meta = {
        "name": exp_name,
        "tag": tag,
        "started_utc": datetime.utcnow().isoformat(),
        "elapsed_seconds": end - start,
    }

    # best-effort git commit
    meta_update = {}
    try:
        import subprocess
        meta_update["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        meta_update["git_commit"] = None

    meta.update(meta_update)
    save_metadata(meta, exp_folder_path / "metadata.json")

    return exp_folder_path

def load_experiment(path: str) -> Dict[str,Any]:
    """Load a saved experiment (parquet preferred, CSV fallback)."""
    p = Path(path)
    out = p / "outputs"

    def _read_df(stem: str):
        pq = out / f"{stem}.parquet"
        csv = out / f"{stem}.csv"
        if pq.exists():
            try:
                return pd.read_parquet(pq)
            except ImportError:
                # parquet engine missing, fall back to CSV if present
                if csv.exists():
                    return pd.read_csv(csv, index_col=0, parse_dates=True)
                return None
            except Exception:
                if csv.exists():
                    return pd.read_csv(csv, index_col=0, parse_dates=True)
                return None
        if csv.exists():
            return pd.read_csv(csv, index_col=0, parse_dates=True)
        return None

    nav_df = _read_df('nav')
    nav = None
    if nav_df is not None:
        # nav is stored as 1-col frame named 'value'
        try:
            nav = nav_df['value']
        except Exception:
            if nav_df.shape[1] == 1:
                nav = nav_df.iloc[:, 0]

    return {
        "params_path": str(p / "params.yaml"),
        "nav": nav,
        "weights": _read_df('weights'),
        "trades": _read_df('trades'),
    }
