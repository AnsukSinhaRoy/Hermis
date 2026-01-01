"""
hermis/replay/minute_to_daily.py

Convert hive-partitioned minute parquet store (symbol/year/month partitions)
into daily aggregates and "execution prices" at a configured time (e.g. market open + 1 minute).

Design goals:
- Stream by month to keep memory bounded.
- Keep output as *daily* wide matrices (date index, symbol columns) for fast backtesting.
- Later, this module can be extended to emit intraday features / signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict, List, Any
from pathlib import Path
import gc

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class DailyMinuteDerived:
    """Daily data derived from minute bars."""
    close: pd.DataFrame            # daily close, wide
    ohlcv: Optional[pd.DataFrame]  # optional MultiIndex columns (field, symbol)
    exec_price: pd.DataFrame       # wide execution price at (market_start + latency)


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[int, int]]:
    start_p = start.to_period("M")
    end_p = end.to_period("M")
    months = pd.period_range(start_p, end_p, freq="M")
    return [(int(p.year), int(p.month)) for p in months]

def _infer_date_bounds_from_store(store_dir: str, symbols: Optional[Sequence[str]] = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Infer (min_date, max_date) from the partitioned minute store layout.

    We avoid scanning the full tree (which can be huge). Strategy:
    - pick one representative symbol partition (prefer first symbol in `symbols`, else first symbol=* folder)
    - scan year=*/month=* folders under it to infer the available month range
    """
    base = Path(store_dir)
    sym_dir = None
    if symbols:
        cand = base / f"symbol={symbols[0]}"
        if cand.exists():
            sym_dir = cand
    if sym_dir is None:
        for p in base.glob("symbol=*"):
            if p.is_dir():
                sym_dir = p
                break
    if sym_dir is None:
        # fallback: wide bounds
        return pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")

    years = []
    months = []
    for ydir in sym_dir.glob("year=*"):
        if not ydir.is_dir():
            continue
        try:
            yy = int(ydir.name.split("=", 1)[1])
        except Exception:
            continue
        for mdir in ydir.glob("month=*"):
            if not mdir.is_dir():
                continue
            try:
                mm = int(mdir.name.split("=", 1)[1])
            except Exception:
                continue
            years.append(yy)
            months.append((yy, mm))

    if not months:
        return pd.Timestamp("1900-01-01"), pd.Timestamp("2100-01-01")

    months_sorted = sorted(months)
    y0, m0 = months_sorted[0]
    y1, m1 = months_sorted[-1]
    start = pd.Timestamp(year=y0, month=m0, day=1)
    # end as last day of last month
    end = (pd.Timestamp(year=y1, month=m1, day=1) + pd.offsets.MonthEnd(0))
    return start, end


def iter_daily_months_from_partitioned_minute_store(
    store_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    market_start: str = "09:15:00",
    latency_minutes: int = 1,
    include_ohlcv: bool = False,
    logger: Optional[Any] = None,
    log_every_month: bool = True,
):
    """Yield (month_id, close_wide, exec_wide, ohlcv_wide_or_none) month-by-month.

    This is the streaming primitive used by the replay engine. It lets the strategy start
    as soon as the first month is processed (instead of waiting for the entire date range).
    """
    s = pd.to_datetime(start_date) if start_date else None
    e = pd.to_datetime(end_date) if end_date else None
    if s is None or e is None:
        s_inf, e_inf = _infer_date_bounds_from_store(store_dir, symbols=symbols)
        if s is None:
            s = s_inf
        if e is None:
            e = e_inf

    months = _month_range(s, e)

    # Only load what we need: open for execution price, close for daily close.
    cols = ["open", "close"] if not include_ohlcv else ["open", "high", "low", "close", "volume"]

    for yy, mm in months:
        if logger is not None and log_every_month:
            logger.info(f"Replay: processing month {yy}-{mm:02d}")

        raw = _load_month_pyarrow(store_dir, yy, mm, symbols=symbols, columns=cols)
        raw = _normalize_and_sort(raw)
        if raw.empty:
            continue

        # date filtering inside month (important near boundaries)
        if s is not None:
            raw = raw.loc[raw["date"] >= s.normalize()]
        if e is not None:
            raw = raw.loc[raw["date"] <= e.normalize()]
        if raw.empty:
            continue

        # daily aggregates
        daily = _daily_ohlcv_from_minute(raw)
        exec_s = _execution_price_from_minute(raw, market_start=market_start, latency_minutes=latency_minutes)

        # wide matrices
        close_w = daily["close"].unstack("symbol").sort_index() if "close" in daily.columns else pd.DataFrame()
        exec_w = exec_s.unstack("symbol").sort_index()

        ohlcv_w = None
        if include_ohlcv:
            wide_fields = []
            for field in ["open", "high", "low", "close", "volume"]:
                if field in daily.columns:
                    wide = daily[field].unstack("symbol").sort_index()
                    wide.columns = pd.MultiIndex.from_product([[field], wide.columns])
                    wide_fields.append(wide)
            if wide_fields:
                ohlcv_w = pd.concat(wide_fields, axis=1)

        # basic progress log
        if logger is not None and log_every_month:
            try:
                logger.info(
                    f"Replay: {yy}-{mm:02d} ready | days={close_w.shape[0]} | symbols={close_w.shape[1]}"
                )
            except Exception:
                pass

        yield (yy, mm), close_w, exec_w, ohlcv_w


def _load_month_pyarrow(
    store_dir: str,
    year: int,
    month: int,
    symbols: Optional[Sequence[str]],
    columns: Sequence[str],
):
    """
    Load a month chunk from a hive-partitioned parquet dataset using pyarrow.dataset.

    Returns a pandas DataFrame in *long* format with at least:
      - datetime (timestamp)
      - symbol (str)
      - open/high/low/close/volume (as available)
    """
    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise ImportError(
            "pyarrow is required to read the partitioned minute store. "
            "Install it with: pip install pyarrow"
        ) from e

    dataset = ds.dataset(store_dir, format="parquet", partitioning="hive")

    filt = (ds.field("year") == int(year)) & (ds.field("month") == int(month))
    if symbols is not None and len(symbols) > 0:
        filt = filt & ds.field("symbol").isin(list(symbols))

    # Ensure we always request datetime + symbol even if caller didn't include them
    cols = list(dict.fromkeys(["datetime", "symbol", *list(columns)]))
    # Try fast pyarrow.dataset scan first; fall back to manual file-by-file reads
    try:
        table = dataset.to_table(filter=filt, columns=cols)
        df = table.to_pandas()
        return df
    except Exception as e:
        # Some parquet stores contain fragments with schema/type oddities that cause
        # pyarrow.dataset to attempt unsafe casts (e.g., float -> int64) during unification.
        # Reading files individually avoids that global unification step.
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception:
            raise

        base = Path(store_dir)

        # Determine which symbol partitions to read
        if symbols is not None and len(symbols) > 0:
            sym_list = list(symbols)
        else:
            sym_list = []
            for p in base.glob("symbol=*"):
                if p.is_dir():
                    sym_list.append(p.name.split("=", 1)[1])

        tables = []
        yy = int(year)
        mm = int(month)

        for sym in sym_list:
            part_dir = base / f"symbol={sym}" / f"year={yy}" / f"month={mm}"
            if not part_dir.exists():
                continue
            for fp in sorted(part_dir.glob("*.parquet")):
                t = pq.read_table(fp, columns=cols)
                # Normalize volume type to float to avoid schema merge failures across fragments
                if "volume" in cols and "volume" in t.column_names:
                    try:
                        import pyarrow.compute as pc
                        vol = pc.cast(t["volume"], pa.float64(), safe=False)
                        t = t.set_column(t.schema.get_field_index("volume"), "volume", vol)
                    except Exception:
                        pass
                tables.append(t)

        if not tables:
            return pd.DataFrame(columns=cols)

        # Ensure consistent types for problematic columns (e.g., volume) before concatenation
        if "volume" in cols:
            import pyarrow.compute as pc
            casted = []
            for t in tables:
                if "volume" in t.column_names:
                    try:
                        vol = pc.cast(t["volume"], pa.float64(), safe=False)
                        t = t.set_column(t.schema.get_field_index("volume"), "volume", vol)
                    except Exception:
                        pass
                casted.append(t)
            tables = casted

        # Concatenate permissively if supported by this pyarrow version
        try:
            table = pa.concat_tables(tables, promote_options="permissive")
        except TypeError:
            table = pa.concat_tables(tables, promote=True)
        df = table.to_pandas()

        # if symbol wasn't stored as a column in files (some stores keep it only in partition),
        # reconstruct it from directory name
        if "symbol" not in df.columns:
            # best-effort: parse from path by re-reading per symbol
            # (shouldn't happen with this repo's preprocess script)
            pass

        return df


def _normalize_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # normalize types
    if "datetime" not in df.columns:
        raise ValueError("Minute store chunk is missing required column: 'datetime'")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    if "symbol" not in df.columns:
        raise ValueError("Minute store chunk is missing required column: 'symbol'")

    # Ensure sorted for groupby(first/last)
    df = df.sort_values(["symbol", "datetime"], kind="mergesort")
    df["date"] = df["datetime"].dt.normalize()

    return df


def _daily_ohlcv_from_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily OHLCV in long index (date, symbol) with columns open/high/low/close/volume."""
    agg_map = {}
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            if c == "open":
                agg_map[c] = "first"
            elif c == "close":
                agg_map[c] = "last"
            elif c == "high":
                agg_map[c] = "max"
            elif c == "low":
                agg_map[c] = "min"
            elif c == "volume":
                agg_map[c] = "sum"

    if not agg_map:
        raise ValueError("Minute store chunk has none of the expected price columns: open/high/low/close/volume")

    daily = df.groupby(["date", "symbol"], sort=False).agg(agg_map)
    return daily


def _execution_price_from_minute(
    df: pd.DataFrame,
    market_start: str = "09:15:00",
    latency_minutes: int = 1,
) -> pd.Series:
    """
    For each (date, symbol), pick the first bar whose datetime >= date + market_start + latency.
    Uses that bar's 'open' as the execution price. If missing, falls back to the first 'open' of the day.
    Returns a Series indexed by (date, symbol).
    """
    if "open" not in df.columns:
        # fallback to close
        if "close" not in df.columns:
            raise ValueError("Need at least 'open' or 'close' to compute execution price")
        price_col = "close"
    else:
        price_col = "open"

    start_td = pd.to_timedelta(market_start)
    lat_td = pd.to_timedelta(f"{int(latency_minutes)}min")
    target_dt = df["date"] + start_td + lat_td

    # first bar at/after target time
    mask = df["datetime"] >= target_dt
    exec_s = df.loc[mask].groupby(["date", "symbol"], sort=False)[price_col].first()

    # fallback: first bar of day
    first_s = df.groupby(["date", "symbol"], sort=False)[price_col].first()

    exec_s = exec_s.reindex(first_s.index)
    exec_s = exec_s.fillna(first_s)
    return exec_s


def build_daily_from_partitioned_minute_store(
    store_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    market_start: str = "09:15:00",
    latency_minutes: int = 1,
    include_ohlcv: bool = False,
    logger: Optional[Any] = None,
    log_every_month: bool = True,
) -> DailyMinuteDerived:
    """
    Stream the minute store month-by-month and produce:
      - daily close (wide)
      - execution price at open+latency (wide)
      - optional daily OHLCV with MultiIndex columns (field, symbol)

    Notes:
    - start_date/end_date are inclusive and interpreted as calendar dates (YYYY-MM-DD).
    """
    s = pd.to_datetime(start_date) if start_date else None
    e = pd.to_datetime(end_date) if end_date else None
    if s is None or e is None:
        s_inf, e_inf = _infer_date_bounds_from_store(store_dir, symbols=symbols)
        if s is None:
            s = s_inf
        if e is None:
            e = e_inf

    months = _month_range(s, e)

    close_parts: List[pd.DataFrame] = []
    exec_parts: List[pd.DataFrame] = []
    ohlcv_parts: List[pd.DataFrame] = []
    # Only load what we need: open for execution price, close for daily close.
    # High/low/volume are only needed when include_ohlcv=True.
    cols = ["open", "close"] if not include_ohlcv else ["open", "high", "low", "close", "volume"]

    for (yy, mm) in months:
        if logger is not None and log_every_month:
            logger.info(f"Replay: processing month {yy}-{mm:02d}")
        raw = _load_month_pyarrow(store_dir, yy, mm, symbols=symbols, columns=cols)
        raw = _normalize_and_sort(raw)
        if raw.empty:
            continue
        if logger is not None and log_every_month:
            try:
                logger.info(
                    f"Replay: {yy}-{mm:02d} loaded | rows={len(raw):,} | days={raw['date'].nunique()} | symbols={raw['symbol'].nunique()}"
                )
            except Exception:
                pass

        # date filtering inside month (important near boundaries)
        raw = raw[(raw["date"] >= s.normalize()) & (raw["date"] <= e.normalize())]
        if raw.empty:
            continue

        daily = _daily_ohlcv_from_minute(raw)
        close_wide = daily["close"].unstack("symbol").sort_index()
        close_parts.append(close_wide)

        exec_s = _execution_price_from_minute(raw, market_start=market_start, latency_minutes=latency_minutes)
        exec_wide = exec_s.unstack("symbol").sort_index()
        exec_parts.append(exec_wide)

        if include_ohlcv:
            # MultiIndex columns: (field, symbol)
            wide_fields = []
            for field in ["open", "high", "low", "close", "volume"]:
                if field in daily.columns:
                    wide_fields.append(daily[field].unstack("symbol").sort_index())
                    wide_fields[-1].columns = pd.MultiIndex.from_product([[field], wide_fields[-1].columns])
            if wide_fields:
                ohlcv_parts.append(pd.concat(wide_fields, axis=1))
        gc.collect()

    if not close_parts:
        empty_idx = pd.DatetimeIndex([], name="date")
        return DailyMinuteDerived(
            close=pd.DataFrame(index=empty_idx),
            ohlcv=pd.DataFrame(index=empty_idx) if include_ohlcv else None,
            exec_price=pd.DataFrame(index=empty_idx),
        )

    close = pd.concat(close_parts, axis=0).sort_index()
    exec_price = pd.concat(exec_parts, axis=0).sort_index()
    # align indices/columns
    common_idx = close.index.union(exec_price.index)
    common_cols = close.columns.union(exec_price.columns)

    close = close.reindex(index=common_idx, columns=common_cols)
    exec_price = exec_price.reindex(index=common_idx, columns=common_cols)

    ohlcv = None
    if include_ohlcv and ohlcv_parts:
        ohlcv = pd.concat(ohlcv_parts, axis=0).sort_index()
        # ensure same index and symbol universe
        ohlcv = ohlcv.reindex(index=common_idx)

    return DailyMinuteDerived(close=close, ohlcv=ohlcv, exec_price=exec_price)