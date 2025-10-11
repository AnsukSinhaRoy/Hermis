# portfolio_sim/experiment.py
import os
import shutil
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import pandas as pd
import pkg_resources

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
    shutil.copy(src_yaml_path, dest)

def save_dataframe_as_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)

def save_series_as_parquet(s: pd.Series, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame(name="value").to_parquet(path, index=True)

def save_metadata(meta: Dict[str,Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

# Replace run_experiment_from_config in portfolio_sim/experiment.py with this full function
def run_experiment_from_config(params_path: str,
                               base_experiments_dir: str,
                               runner_func: Callable[[Dict[str,Any]], Dict[str,Any]],
                               name_override: Optional[str]=None,
                               tag: Optional[str]=None) -> Path:
    import yaml
    with open(params_path, "r") as f:
        cfg = yaml.safe_load(f)
        # Resolve experiment name from multiple possible locations in the config.
    # Priority (highest -> lowest):
    #   1) name_override
    #   2) result.name
    #   3) experiment.name
    #   4) top-level name
    #   5) fallback 'unnamed'
    exp_name = name_override or \
               cfg.get('result', {}).get('name') or \
               cfg.get('experiment', {}).get('name') or \
               cfg.get('name') or \
               'unnamed'

    # sanitize folder-friendly name (replace whitespace/newlines)
    exp_name = str(exp_name).strip().replace(' ', '_').replace('\n', '_')

    exp_folder = make_experiment_folder(base_experiments_dir, exp_name, tag)
    save_params_yaml(params_path, exp_folder)
    start = time.time()
    outputs = runner_func(cfg)
    end = time.time()
    out_folder = exp_folder / "outputs"
    nav = outputs.get("nav")
    weights = outputs.get("weights")
    trades = outputs.get("trades")
    meta = outputs.get("meta", {})
    prices = outputs.get("prices", None)  # <-- accept prices if runner returned them

    if nav is not None:
        save_series_as_parquet(nav, out_folder / "nav.parquet")
    if weights is not None:
        save_dataframe_as_parquet(weights, out_folder / "weights.parquet")
    if trades is not None:
        save_dataframe_as_parquet(trades, out_folder / "trades.parquet")

    # Save prices snapshot (exact data used in experiment) for reproducible viz
    if prices is not None:
        try:
            save_dataframe_as_parquet(prices, exp_folder / "data" / "prices.parquet")
        except Exception:
            # best-effort: ignore if saving prices fails
            pass

    # metadata augmentation (unchanged)
    meta_update = {
        "runtime_seconds": end - start,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "python_version": f"{os.sys.version}",
        "packages": {}
    }

    try:
        import importlib.metadata as importlib_metadata
    except Exception:
        import importlib_metadata

    for pkg in ["numpy","pandas","cvxpy","matplotlib","pyyaml","pyarrow"]:
        try:
            meta_update["packages"][pkg] = importlib_metadata.version(pkg)
        except Exception:
            meta_update["packages"][pkg] = None

    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )
        commit = res.stdout.strip()
        meta_update["git_commit"] = commit if commit else None
    except Exception:
        meta_update["git_commit"] = None

    meta.update(meta_update)
    save_metadata(meta, out_folder / "metadata.json")
    return exp_folder


def load_experiment(path: str) -> Dict[str,Any]:
    p = Path(path)
    out = p / "outputs"
    return {
        "params_path": str(p / "params.yaml"),
        "nav": pd.read_parquet(out / "nav.parquet") if (out / "nav.parquet").exists() else None,
        "weights": pd.read_parquet(out / "weights.parquet") if (out / "weights.parquet").exists() else None,
        "trades": pd.read_parquet(out / "trades.parquet") if (out / "trades.parquet").exists() else None,
        "meta": json.load(open(out / "metadata.json")) if (out / "metadata.json").exists() else {}
    }
