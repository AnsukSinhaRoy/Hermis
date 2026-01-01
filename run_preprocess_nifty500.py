"""run_preprocess_nifty500.py

Entry point to preprocess Nifty 500 (or any large universe) 1-minute CSV files.

Typical usage (from repo root):

  python run_preprocess_nifty500.py \
    --raw-dir data/raw \
    --processed-root data/processed \
    --start-date 2010-12-01 \
    --end-date 2025-12-31

This writes:
  data/processed/nifty500/1min_store/...
  data/processed/nifty500/prices_1D_nifty500.parquet
  data/processed/nifty500/universe_symbols.json
"""

from __future__ import annotations

import argparse

from portfolio_sim.preprocess_nifty500 import Nifty500PreprocessConfig, preprocess_nifty500_minute_csvs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Nifty 500 1-minute CSVs into partitioned parquet + daily matrix")
    p.add_argument("--raw-dir", default="data/raw", help="Folder containing per-symbol minute CSVs")
    p.add_argument("--processed-root", default="data/processed", help="Where to write processed artifacts")
    p.add_argument("--dataset-name", default="nifty500", help="Output dataset subfolder name")
    p.add_argument("--no-recursive", action="store_true", help="Do not search recursively for CSVs")
    p.add_argument("--glob", dest="file_glob", default="*.csv", help="Glob pattern for CSVs")
    p.add_argument("--chunksize", type=int, default=750_000, help="Rows per CSV chunk")
    p.add_argument("--market-start", default="09:15:00")
    p.add_argument("--market-end", default="15:30:00")
    p.add_argument("--start-date", default=None, help="Inclusive start date/datetime")
    p.add_argument("--end-date", default=None, help="Inclusive end date/datetime")
    p.add_argument("--overwrite", action="store_true", help="Delete existing minute store before writing")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    cfg = Nifty500PreprocessConfig(
        raw_dir=a.raw_dir,
        processed_root=a.processed_root,
        dataset_name=a.dataset_name,
        recursive=not a.no_recursive,
        file_glob=a.file_glob,
        chunksize=int(a.chunksize),
        market_start=a.market_start,
        market_end=a.market_end,
        start_date=a.start_date,
        end_date=a.end_date,
        overwrite=bool(a.overwrite),
    )
    summary = preprocess_nifty500_minute_csvs(cfg)
    print("\nDone.")
    print("Minute store:", summary["minute_store_dir"])
    print("Daily matrix:", summary["daily_prices_path"])
    print("Symbols:", len(summary["symbols"]))
