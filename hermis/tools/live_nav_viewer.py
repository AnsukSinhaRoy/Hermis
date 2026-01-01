"""hermis/tools/live_nav_viewer.py

Real-time NAV viewer for a running experiment.

Run in a separate terminal while an experiment is running:

    python -m hermis.tools.live_nav_viewer --experiment experiments/<run_folder>

or:

    python -m hermis.tools.live_nav_viewer --nav experiments/<run_folder>/outputs/nav.parquet

The viewer will refresh periodically and update a live matplotlib window.

Notes:
- If parquet reading fails (no pyarrow), it will try CSV.
- The simulation must be periodically writing NAV to disk. In the replay
  engine we support monthly flushing via `data.replay.flush_outputs_each_month`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def _load_nav(nav_path: Path) -> pd.Series | None:
    if not nav_path.exists():
        return None

    # Prefer parquet written by portfolio_sim.experiment.save_series_as_parquet
    if nav_path.suffix.lower() == ".parquet" and nav_path.exists():
        try:
            df = pd.read_parquet(nav_path)
            if isinstance(df, pd.Series):
                return df
            if "value" in df.columns:
                s = df["value"]
            else:
                s = df.iloc[:, 0]
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            pass

    # Fallback: look for CSV next to it
    csv = nav_path.with_suffix(".csv")
    if csv.exists():
        try:
            df = pd.read_csv(csv, index_col=0)
            s = df.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
        except Exception:
            pass

    return None


def main():
    ap = argparse.ArgumentParser(description="Live NAV viewer (matplotlib)")
    ap.add_argument("--experiment", type=str, default=None, help="Experiment folder (contains outputs/nav.parquet)")
    ap.add_argument("--nav", type=str, default=None, help="Path to nav.parquet (or nav.csv)")
    ap.add_argument("--refresh", type=float, default=2.0, help="Refresh interval (seconds)")
    ap.add_argument("--theme", type=str, default="dark", help="Plot theme: dark/light or a matplotlib style name")
    args = ap.parse_args()

    nav_path = None
    if args.nav:
        nav_path = Path(args.nav)
    elif args.experiment:
        nav_path = Path(args.experiment) / "outputs" / "nav.parquet"
    else:
        raise SystemExit("Provide --experiment or --nav")

    from hermis.replay.live_plot import NavLivePlotter

    plotter = NavLivePlotter(title="Hermis Live NAV (Viewer)", subtitle=str(nav_path), theme=args.theme)

    last_len = 0
    while True:
        s = _load_nav(nav_path)
        if s is not None and len(s) > 0:
            if len(s) != last_len:
                # only push new points
                for dt, v in s.iloc[last_len:].items():
                    plotter.update_point(dt, float(v))
                last_len = len(s)
        time.sleep(max(0.2, float(args.refresh)))


if __name__ == "__main__":
    main()
