"""portfolio_sim/preprocess_nifty500.py

Nifty 500 (or any large universe) 1-minute CSV -> reusable Parquet store.

Design goals
------------
1) Stable + resumable preprocessing.
2) No giant "wide" 1-minute matrices written to disk.
3) Fast repeated research runs:
   - Daily close matrix for the full universe (manageable)
   - Minute store that can be queried for a *subset* of symbols and a date range.

Outputs
-------
processed_root/
  nifty500/
    1min_store/                     # partitioned parquet store (hive partitions)
      symbol=RELIANCE/year=2016/month=3/part-00000.parquet
      ...
    prices_1D_nifty500.parquet       # wide daily close matrix (Date x Symbol)
    universe_symbols.json
    preprocess_nifty500_metadata.json
    preprocess_nifty500_report.txt

CSV assumptions
---------------
Each raw file contains a single symbol, with at least:
  - a datetime-like column ("date", "datetime", "timestamp" or first column)
  - OHLCV columns (case-insensitive): open, high, low, close, volume

If your columns differ, tweak `_standardize_columns()`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DATETIME_CANDIDATES = ("datetime", "timestamp", "date", "time", "ts")


def _infer_datetime_col(cols: Iterable[str]) -> str:
    cols_l = {c.lower(): c for c in cols}
    for c in DATETIME_CANDIDATES:
        if c in cols_l:
            return cols_l[c]
    # fallback to first column
    return list(cols)[0]


def _infer_symbol_from_filename(p: Path) -> str:
    """Best-effort symbol inference from filename."""
    stem = p.stem
    # common patterns: SYMBOL.csv, SYMBOL_1min.csv, NSE_SYMBOL.csv
    parts = [x for x in stem.replace("-", "_").split("_") if x]
    if len(parts) == 0:
        return stem
    # if starts with exchange prefix
    if parts[0].lower() in {"nse", "bse", "ns", "india"} and len(parts) >= 2:
        return parts[1].upper()
    return parts[0].upper()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map columns to canonical names: open/high/low/close/volume."""
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in {"o", "open"}:
            rename[c] = "open"
        elif cl in {"h", "high"}:
            rename[c] = "high"
        elif cl in {"l", "low"}:
            rename[c] = "low"
        elif cl in {"c", "close", "last"}:
            rename[c] = "close"
        elif cl in {"v", "vol", "volume"}:
            rename[c] = "volume"
    return df.rename(columns=rename)


def _market_hours_mask(idx: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    st = pd.to_datetime(start).time()
    et = pd.to_datetime(end).time()
    t = idx.time
    return np.array([(x >= st) and (x < et) for x in t], dtype=bool)


@dataclass
class Nifty500PreprocessConfig:
    raw_dir: str = "data/raw"
    processed_root: str = "data/processed"
    dataset_name: str = "nifty500"
    recursive: bool = True
    file_glob: str = "*.csv"
    chunksize: int = 750_000
    market_start: str = "09:15:00"
    market_end: str = "15:30:00"
    tz: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    overwrite: bool = False


def preprocess_nifty500_minute_csvs(cfg: Nifty500PreprocessConfig) -> Dict[str, object]:
    raw_dir = Path(cfg.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {cfg.raw_dir}")

    processed_root = Path(cfg.processed_root)
    out_base = processed_root / cfg.dataset_name
    store_dir = out_base / "1min_store"
    out_base.mkdir(parents=True, exist_ok=True)

    if cfg.overwrite and store_dir.exists():
        # dangerous but convenient
        import shutil

        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    # file discovery
    files = list(raw_dir.rglob(cfg.file_glob) if cfg.recursive else raw_dir.glob(cfg.file_glob))
    files = [p for p in files if p.is_file()]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No CSVs found in {cfg.raw_dir} matching {cfg.file_glob}")

    # collect daily closes (per symbol -> Series)
    daily_closes: Dict[str, pd.Series] = {}
    failures: Dict[str, str] = {}

    # per-partition counters (symbol, year, month) -> int
    part_counter: Dict[Tuple[str, int, int], int] = {}

    s_ts = pd.to_datetime(cfg.start_date, errors="coerce") if cfg.start_date else None
    e_ts = pd.to_datetime(cfg.end_date, errors="coerce") if cfg.end_date else None

    for f in files:
        symbol = _infer_symbol_from_filename(f)
        # hold per-day last close with timestamp to ensure last-in-day even across chunks
        day_last: Dict[pd.Timestamp, Tuple[pd.Timestamp, float]] = {}

        try:
            reader = pd.read_csv(f, chunksize=int(cfg.chunksize))
        except Exception as e:
            failures[str(f)] = f"read_failed: {repr(e)}"
            continue

        # process chunks
        any_rows = False
        for chunk in reader:
            if chunk is None or len(chunk) == 0:
                continue
            any_rows = True

            dtcol = _infer_datetime_col(chunk.columns)
            chunk = _standardize_columns(chunk)

            # parse datetime
            chunk[dtcol] = pd.to_datetime(chunk[dtcol], errors="coerce")
            chunk = chunk.dropna(subset=[dtcol])
            if chunk.empty:
                continue
            if cfg.tz:
                try:
                    chunk[dtcol] = chunk[dtcol].dt.tz_localize(cfg.tz, nonexistent="shift_forward", ambiguous="NaT")
                except Exception:
                    # if already tz-aware or localization fails, ignore
                    pass

            # filter date range early
            if s_ts is not None:
                chunk = chunk.loc[chunk[dtcol] >= s_ts]
            if e_ts is not None:
                chunk = chunk.loc[chunk[dtcol] <= e_ts]
            if chunk.empty:
                continue

            # keep canonical cols only
            keep_cols = [dtcol] + [c for c in ["open", "high", "low", "close", "volume"] if c in chunk.columns]
            chunk = chunk[keep_cols].copy()

            # coerce numeric
            for c in ["open", "high", "low", "close", "volume"]:
                if c in chunk.columns:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            # drop bad close values
            if "close" in chunk.columns:
                chunk = chunk.loc[chunk["close"].notna() & (chunk["close"] > 0)]
            if chunk.empty:
                continue

            chunk = chunk.sort_values(dtcol)

            idx = pd.DatetimeIndex(chunk[dtcol])
            mh_mask = _market_hours_mask(idx, cfg.market_start, cfg.market_end)
            chunk = chunk.loc[mh_mask].copy()
            if chunk.empty:
                continue

            # de-duplicate timestamps (keep last)
            chunk = chunk.drop_duplicates(subset=[dtcol], keep="last")

            # --- update daily closes ---
            # group by normalized day, keep last timestamp/close
            by_day = chunk[[dtcol, "close"]].copy()
            by_day["day"] = by_day[dtcol].dt.normalize()
            g = by_day.sort_values(dtcol).groupby("day", sort=False).tail(1)
            for _, row in g.iterrows():
                day = pd.Timestamp(row["day"])  # normalized
                ts = pd.Timestamp(row[dtcol])
                close = float(row["close"]) if pd.notna(row["close"]) else np.nan
                if not np.isfinite(close):
                    continue
                prev = day_last.get(day)
                if prev is None or ts > prev[0]:
                    day_last[day] = (ts, close)

            # --- write minute parquet partitions ---
            out = chunk.rename(columns={dtcol: "datetime"})
            out["symbol"] = symbol
            out["year"] = pd.DatetimeIndex(out["datetime"]).year.astype(int)
            out["month"] = pd.DatetimeIndex(out["datetime"]).month.astype(int)

            # write by (year, month) to keep partitions stable
            for (yy, mm), sub in out.groupby(["year", "month"], sort=False):
                key = (symbol, int(yy), int(mm))
                n = part_counter.get(key, 0)
                part_counter[key] = n + 1

                part_dir = store_dir / f"symbol={symbol}" / f"year={int(yy)}" / f"month={int(mm)}"
                part_dir.mkdir(parents=True, exist_ok=True)
                part_path = part_dir / f"part-{n:05d}.parquet"
                # avoid writing partition cols in file (they exist in directory structure)
                sub = sub.drop(columns=["symbol", "year", "month"], errors="ignore")
                sub.to_parquet(part_path, index=False)

        if not any_rows:
            failures[str(f)] = "empty_file"
            continue

        if not day_last:
            failures[str(f)] = "no_valid_rows_after_cleaning"
            continue

        # finalize daily series
        days = sorted(day_last.keys())
        closes = [day_last[d][1] for d in days]
        s = pd.Series(closes, index=pd.DatetimeIndex(days), name=symbol).sort_index()
        daily_closes[symbol] = s

    if not daily_closes:
        raise RuntimeError("No symbols were successfully processed. Check raw_dir and CSV schema.")

    # --- build daily wide matrix ---
    daily_df = pd.concat(daily_closes.values(), axis=1).sort_index()
    # optional slice
    if s_ts is not None:
        daily_df = daily_df.loc[daily_df.index >= s_ts.normalize()]
    if e_ts is not None:
        daily_df = daily_df.loc[daily_df.index <= e_ts.normalize()]

    out_daily = out_base / "prices_1D_nifty500.parquet"
    daily_df.to_parquet(out_daily)

    universe = sorted(list(daily_df.columns))
    (out_base / "universe_symbols.json").write_text(json.dumps(universe, indent=2))

    meta = {
        "raw_dir": str(raw_dir),
        "processed_root": str(processed_root),
        "dataset_name": cfg.dataset_name,
        "files_discovered": int(len(files)),
        "symbols_processed": int(len(universe)),
        "minute_store_dir": str(store_dir),
        "daily_prices_path": str(out_daily),
        "date_range_daily": {
            "start": str(daily_df.index.min()),
            "end": str(daily_df.index.max()),
        },
        "config": {
            "market_start": cfg.market_start,
            "market_end": cfg.market_end,
            "chunksize": int(cfg.chunksize),
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
        },
        "failures": failures,
    }
    (out_base / "preprocess_nifty500_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    report_lines = [
        "Nifty500 preprocessing report",
        f"raw_dir={raw_dir}",
        f"processed_root={processed_root}",
        f"files_discovered={len(files)}",
        f"symbols_processed={len(universe)}",
        f"minute_store_dir={store_dir}",
        f"daily_prices_path={out_daily}",
        f"daily_date_range={meta['date_range_daily']}",
        f"failures={len(failures)}",
        "",
        "Failure details:",
    ]
    for k, v in list(failures.items())[:200]:
        report_lines.append(f"- {k}: {v}")
    if len(failures) > 200:
        report_lines.append(f"... ({len(failures) - 200} more)")
    (out_base / "preprocess_nifty500_report.txt").write_text("\n".join(report_lines))

    return {
        "minute_store_dir": str(store_dir),
        "daily_prices_path": str(out_daily),
        "symbols": universe,
        "failures": failures,
        "metadata": meta,
    }
