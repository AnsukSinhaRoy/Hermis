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

def run_experiment_from_config(params_path: str,
                               base_experiments_dir: str,
                               runner_func: Callable[[Dict[str,Any]], Dict[str,Any]],
                               name_override: Optional[str]=None,
                               tag: Optional[str]=None) -> Path:
    import yaml
    with open(params_path, "r") as f:
        cfg = yaml.safe_load(f)
    exp_name = name_override or cfg.get("experiment", {}).get("name", "unnamed")
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
    if nav is not None:
        save_series_as_parquet(nav, out_folder / "nav.parquet")
    if weights is not None:
        save_dataframe_as_parquet(weights, out_folder / "weights.parquet")
    if trades is not None:
        save_dataframe_as_parquet(trades, out_folder / "trades.parquet")
    # metadata augmentation
    meta_update = {
        "runtime_seconds": end - start,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "python_version": f"{os.sys.version}",
        "packages": {}
    }
    for pkg in ["numpy","pandas","cvxpy","matplotlib","pyyaml","pyarrow"]:
        try:
            meta_update["packages"][pkg] = pkg_resources.get_distribution(pkg).version
        except Exception:
            meta_update["packages"][pkg] = None
    # git commit
    try:
        import subprocess
        commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=os.getcwd()).decode().strip()
        meta_update["git_commit"] = commit
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
