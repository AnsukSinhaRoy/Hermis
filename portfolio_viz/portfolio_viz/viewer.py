# viewer.py
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_outputs(exp_folder: str):
    out = Path(exp_folder) / "outputs"
    nav = pd.read_parquet(out / "nav.parquet")
    weights = pd.read_parquet(out / "weights.parquet")
    trades = pd.read_parquet(out / "trades.parquet")
    meta = json.load(open(out / "metadata.json"))
    return {"nav": nav, "weights": weights, "trades": trades, "meta": meta}

def plot_nav(nav, save_path=None):
    plt.figure(figsize=(10,5))
    plt.plot(nav.index, nav.values)
    plt.title("Portfolio NAV")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
