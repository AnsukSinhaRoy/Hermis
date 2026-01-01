# portfolio_sim/preprocess.py
from pathlib import Path
import pandas as pd
import numpy as np
import json, traceback, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Optional, Dict, List, Any

# ---------------------------
# Helper utils
# ---------------------------

def _make_json_serializable(obj):
    """Recursively convert obj to JSON-serializable types."""
    # primitives
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    # numpy scalars
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_ ,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    # pandas Timestamp / datetime
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp):
            return str(obj)
        if isinstance(obj, _pd.Timedelta):
            return str(obj)
    except Exception:
        pass
    # pandas Series / DataFrame -> provide small summary (avoid dumping huge tables)
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Series):
            # turn into dict of up to N items + metadata
            N = 10
            head = obj.head(N).to_dict()
            return {"_type": "Series", "length": int(obj.shape[0]), "head": _make_json_serializable(head)}
        if isinstance(obj, _pd.DataFrame):
            # provide shape and up to first N rows as records
            N = 5
            try:
                head = obj.head(N).to_dict(orient="records")
            except Exception:
                head = str(obj.head(N))
            return {"_type": "DataFrame", "shape": list(obj.shape), "head": _make_json_serializable(head)}
    except Exception:
        pass
    # dict
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    # list / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(v) for v in obj]
    # objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _make_json_serializable(vars(obj))
        except Exception:
            return repr(obj)
    # fallback: repr
    return repr(obj)


def _ensure_cols_present(df: pd.DataFrame, required: List[str]) -> bool:
    cols_lower = [c.lower() for c in df.columns]
    return all(r in cols_lower for r in required)

def _map_columns_to_expected(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    # return df with standardized lower-case column names for expected fields
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if low in expected:
            col_map[col] = low
    return df.rename(columns=col_map)

def _parse_datetime_series(series: pd.Series, dayfirst: bool = True) -> pd.DatetimeIndex:
    # pandas robust parsing; coerce errors
    return pd.to_datetime(series, dayfirst=dayfirst, errors='coerce')

def _market_hours_filter(df_index: pd.DatetimeIndex, start_time: str = "09:15:00", end_time: str = "15:30:00") -> np.ndarray:
    # returns boolean mask of index within given hours (inclusive start, exclusive end)
    st = pd.to_datetime(start_time).time()
    et = pd.to_datetime(end_time).time()
    times = df_index.time
    mask = [(t >= st) and (t < et) for t in times]
    return np.array(mask, dtype=bool)

def _detect_outliers_by_returns(series: pd.Series, z_thresh: float = 6.0) -> pd.Series:
    # compute log returns and detect large z-score values
    s = series.dropna()
    if s.shape[0] < 5:
        return pd.Series(False, index=series.index)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(s).diff().dropna()
    mean = r.mean()
    std = r.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return pd.Series(False, index=series.index)
    z = (r - mean) / std
    out_idx = z.abs() > z_thresh
    # convert back to original index with False padding
    mask = pd.Series(False, index=series.index)
    mask.loc[r.index] = out_idx
    return mask

# ---------------------------
# Read + validate single file
# ---------------------------

def read_validate_ohlcv(path: Path,
                        datetime_col: str = "date",
                        dayfirst: bool = True,
                        tz: Optional[str] = None,
                        enforce_columns: bool = True) -> Dict[str, Any]:
    """
    Read a raw OHLCV CSV and validate it. Returns dict with:
      - 'ok': bool
      - 'df': DataFrame (if ok)
      - 'errors': list of error strings
      - diagnostic counts (dup_timestamps, nonpos_count, nan_ts, rows)
    """
    res = {"ok": False, "errors": [], "df": None, "dup_timestamps": 0, "nonpos_count": 0, "nan_timestamps": 0, "rows": 0}
    try:
        df = pd.read_csv(path)
        res["rows"] = df.shape[0]
        # find datetime column
        if datetime_col in df.columns:
            dtcol = datetime_col
        else:
            # fallback first column
            dtcol = df.columns[0]
        # parse datetime
        # auto-detect format: if ISO-like pattern, force dayfirst=False
        sample_val = str(df.iloc[0][dtcol]) if not df.empty else ""
        if sample_val and sample_val[:4].isdigit() and sample_val[4] == "-":
        # looks like YYYY-MM-DD -> safe to set dayfirst=False
            df[dtcol] = pd.to_datetime(df[dtcol], errors="coerce", dayfirst=False)
        else:
            df[dtcol] = pd.to_datetime(df[dtcol], errors="coerce", dayfirst=dayfirst)

        n_na_ts = df[dtcol].isna().sum()
        res["nan_timestamps"] = int(n_na_ts)
        df = df.dropna(subset=[dtcol]).copy()
        # lower-case mapping for expected columns
        expected = ["open", "high", "low", "close", "volume"]
        df = _map_columns_to_expected(df, expected)
        if enforce_columns and not _ensure_cols_present(df, expected):
            missing = [e for e in expected if e not in [c.lower() for c in df.columns]]
            res["errors"].append(f"Missing expected columns: {missing}")
            return res
        # set index and sort
        df = df.set_index(dtcol).sort_index()
        # duplicates
        dup = df.index.duplicated().sum()
        res["dup_timestamps"] = int(dup)
        if dup > 0:
            df = df[~df.index.duplicated(keep='first')].copy()
        # coerce numeric
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # non-positive prices count (close <= 0)
        if "close" in df.columns:
            nonpos = (df["close"] <= 0) | (df["close"].isna())
            res["nonpos_count"] = int(nonpos.sum())
        res["ok"] = True
        res["df"] = df
    except Exception as e:
        res["errors"].append(repr(e))
        res["errors"].append(traceback.format_exc())
    return res

# ---------------------------
# Single-file processing worker
# ---------------------------

def _process_file_worker(path: Path,
                         freq: str = "1D",
                         dayfirst: bool = True,
                         tz: Optional[str] = None,
                         market_start: Optional[str] = "09:15:00",
                         market_end: Optional[str] = "15:30:00",
                         z_outlier_threshold: float = 6.0,
                         ffill_limit: int = 5,
                         start_datetime: Optional[str] = None) -> Dict[str, Any]:
    """
    Process single CSV:
      - read+validate
      - **NEW**: if start_datetime provided:
          - if file's first valid timestamp > start_datetime -> skip file (return error reason 'starts_after_start_datetime')
          - else crop series to >= start_datetime before resampling/cleaning
      - filter trading hours (if configured)
      - resample to OHLCV at freq
      - detect outliers, compute stats, return series

    Returns dict with: 'ticker','series','diag' or 'error'
    """
    out = {"ticker": None, "series": None, "diag": {}, "error": None}
    try:
        info = read_validate_ohlcv(path, dayfirst=dayfirst, tz=tz)
        if not info["ok"]:
            out["error"] = {"reason": "validation_failed", "details": info["errors"]}
            out["diag"] = info
            return out
        df = info["df"]
        ticker = path.stem.split("_")[0]
        out["ticker"] = ticker

        # --- NEW: apply start_datetime logic ---
        if start_datetime is not None:
            # parse the requested start_datetime robustly
            start_dt = pd.to_datetime(start_datetime, errors='coerce')
            if pd.isna(start_dt):
                out["error"] = {"reason": "invalid_start_datetime", "details": str(start_datetime)}
                out["diag"] = info
                return out
            # get first valid timestamp in the file (after parse, before filtering)
            if df.shape[0] == 0:
                out["error"] = {"reason": "empty_after_parse"}
                out["diag"] = info
                return out
            first_ts = df.index.min()
            # If the file's first timestamp is strictly after requested start, skip file
            if first_ts > start_dt:
                out["error"] = {"reason": "starts_after_start_datetime", "file_first_ts": str(first_ts), "requested_start": str(start_dt)}
                out["diag"] = info
                return out
            # Otherwise crop the dataframe to start from start_dt
            df = df.loc[df.index >= start_dt]
            if df.empty:
                out["error"] = {"reason": "empty_after_crop_to_start_datetime"}
                out["diag"] = info
                return out
        # --- end start_datetime logic ---

        # market hours filter
        if market_start and market_end:
            mask = _market_hours_filter(df.index, start_time=market_start, end_time=market_end)
            df = df.loc[mask]
        if df.empty:
            out["error"] = {"reason": "empty_after_market_hours"}
            out["diag"] = info
            return out

        # Resample OHLCV
        agg_map = {}
        if "open" in df.columns:
            agg_map["open"] = "first"
        if "high" in df.columns:
            agg_map["high"] = "max"
        if "low" in df.columns:
            agg_map["low"] = "min"
        if "close" in df.columns:
            agg_map["close"] = "last"
        if "volume" in df.columns:
            agg_map["volume"] = "sum"
        res = df.resample(freq).agg(agg_map)

        # drop periods without close
        if "close" in res.columns:
            res = res.dropna(subset=["close"])
        else:
            numeric_cols = res.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                res = res.dropna(subset=[numeric_cols[0]])
            else:
                out["error"] = {"reason": "no_numeric_after_resample"}
                out["diag"] = info
                return out

        # Remove non-positive closes (set to NaN)
        if "close" in res.columns:
            nonpos_mask = (res["close"] <= 0) | (res["close"].isna())
            nonpos_count = int(nonpos_mask.sum())
            out["diag"]["nonpos_count_resampled"] = nonpos_count
            res.loc[nonpos_mask, "close"] = np.nan

        # Forward fill small gaps, limit ffill_limit
        res["close"] = res["close"].ffill(limit=ffill_limit)

        # Detect outliers in returns
        outlier_mask = _detect_outliers_by_returns(res["close"], z_thresh=z_outlier_threshold)
        out["diag"]["outlier_count"] = int(outlier_mask.sum())

        # If outliers detected, null them out then ffill small window
        if out["diag"]["outlier_count"] > 0:
            res.loc[outlier_mask, "close"] = np.nan
            res["close"] = res["close"].ffill(limit=ffill_limit)

        # Final cleaning and checks
        series = res["close"].dropna()
        if series.empty:
            out["error"] = {"reason": "empty_after_cleaning"}
            out["diag"] = info
            return out

        # small backfill/ffill around start if needed
        series = series.bfill(limit=ffill_limit).ffill(limit=ffill_limit)
        if series.empty:
            out["error"] = {"reason": "empty_after_fill"}
            out["diag"] = info
            return out

        out["series"] = series
        out["diag"].update({
            "rows_original": info["rows"],
            "nan_timestamps_original": info["nan_timestamps"],
            "dup_timestamps_original": info["dup_timestamps"],
            "nonpos_count_original": info["nonpos_count"],
            "resampled_rows": int(res.shape[0]),
            "final_valid_points": int(series.shape[0]),
        })
        return out

    except Exception as e:
        out["error"] = {"reason": "exception", "details": repr(e), "trace": traceback.format_exc()}
        return out


# ---------------------------
# Parallel pipeline
# ---------------------------

def preprocess_pipeline(raw_dir: str = "data/raw",
                        processed_dir: str = "data/processed",
                        file_pattern: str = "*_minute.csv",
                        freq: str = "1D",
                        tz: Optional[str] = None,
                        dayfirst: bool = True,
                        market_start: Optional[str] = "09:15:00",
                        market_end: Optional[str] = "15:30:00",
                        z_outlier_threshold: float = 6.0,
                        ffill_limit: int = 5,
                        min_coverage_frac: float = 0.5,
                        drop_tickers_with_low_coverage: bool = True,
                        save_ohlcv_per_ticker: bool = False,
                        max_workers: Optional[int] = None,
                        # NEW parameters:
                        restrict_to_common_intersection: bool = False,
                        min_length_days: Optional[int] = None,
                        start_datetime: Optional[str] = None) -> Dict[str, Any]:
    """
    Robust preprocessing pipeline with option to restrict final dataset to the common intersection
    of all tickers (i.e., date range where every ticker has at least one valid point).

    New args:
      - restrict_to_common_intersection: if True, compute common_start = max(starts) and
        common_end = min(ends) and restrict all series to that window BEFORE align/coverage.
      - min_length_days: optional minimum number of days required for the intersection;
        if the computed intersection is shorter than this, pipeline will NOT restrict and will warn.

    Returns summary dict (same as before).
    """
    rawp = Path(raw_dir)
    assert rawp.exists(), f"raw dir not found: {raw_dir}"
    processedp = Path(processed_dir)
    processedp.mkdir(parents=True, exist_ok=True)

    files = sorted(rawp.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {raw_dir} matching {file_pattern}")

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 2)

    ticker_series: Dict[str, pd.Series] = {}
    diagnostics = {}
    failed = {}

        # run in threads (same worker as before)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # create futures mapping -> future : path
        futures = {
            ex.submit(
                _process_file_worker,
                f,                 # path
                freq,              # freq
                dayfirst,          # dayfirst
                tz,                # tz
                market_start,      # market_start
                market_end,        # market_end
                z_outlier_threshold,
                ffill_limit,
                start_datetime     # pass start_datetime from outer scope
            ): f for f in files
        }

        # iterate over completed futures with a tqdm progress bar
        for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocessing", unit="file"):
            path = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                # record the failure but continue processing others
                failed[str(path)] = {"reason": "exception_in_worker", "detail": repr(e)}
                continue

            ticker = r.get("ticker") or path.stem
            if r.get("error"):
                # worker signalled an error (including 'starts_after_start_datetime', etc.)
                failed[str(path)] = r["error"]
                diagnostics[ticker] = r.get("diag", {})
                continue

            # successful result: collect series and diagnostics
            series = r["series"]
            diag = r.get("diag", {})
            ticker_series[ticker] = series
            diagnostics[ticker] = diag

            # optionally save per-ticker OHLCV resampled
            if save_ohlcv_per_ticker:
                try:
                    orig = Path(path)
                    orig_df = pd.read_csv(orig)
                    dtcol = "date" if "date" in orig_df.columns else orig_df.columns[0]
                    orig_df[dtcol] = pd.to_datetime(orig_df[dtcol], dayfirst=dayfirst, errors='coerce')
                    orig_df = orig_df.dropna(subset=[dtcol]).set_index(dtcol).sort_index()
                    orig_df = _map_columns_to_expected(orig_df, ["open","high","low","close","volume"])
                    agg = {}
                    if "open" in orig_df.columns: agg["open"] = "first"
                    if "high" in orig_df.columns: agg["high"] = "max"
                    if "low" in orig_df.columns: agg["low"] = "min"
                    if "close" in orig_df.columns: agg["close"] = "last"
                    if "volume" in orig_df.columns: agg["volume"] = "sum"
                    if agg:
                        res_ohlcv = orig_df.resample(freq).agg(agg)
                        res_dir = processedp / f"ohlcv_{freq}"
                        res_dir.mkdir(parents=True, exist_ok=True)
                        res_ohlcv.to_parquet(res_dir / f"{ticker}.parquet")
                except Exception as e:
                    diagnostics[ticker] = diagnostics.get(ticker, {})
                    diagnostics[ticker]["ohlcv_save_error"] = repr(e)

    if not ticker_series:
        raise RuntimeError("No valid ticker series produced by preprocessing.")

    # --- NEW: compute common intersection window if requested ---
    if restrict_to_common_intersection:
        # collect per-ticker start/end times (after cleaning)
        starts = {t: s.index.min() for t, s in ticker_series.items()}
        ends   = {t: s.index.max() for t, s in ticker_series.items()}
        common_start = max(starts.values())
        common_end = min(ends.values())

        # check for valid window
        if common_start >= common_end:
            # intersection empty -> warn and do not restrict
            # write warning to metadata after pipeline completes
            restrict_used = False
            restrict_msg = f"Common intersection empty: computed start {common_start} >= end {common_end}. Not restricting."
        else:
            # check min_length_days guard
            if min_length_days is not None:
                length_days = (common_end - common_start).days
                if length_days < min_length_days:
                    restrict_used = False
                    restrict_msg = f"Common intersection length {length_days}d < min_length_days ({min_length_days}). Not restricting."
                else:
                    restrict_used = True
                    restrict_msg = f"Restricting to common intersection {common_start} -> {common_end} (length {length_days} days)."
            else:
                restrict_used = True
                restrict_msg = f"Restricting to common intersection {common_start} -> {common_end}."

            if restrict_used:
                # restrict each series to that window (and drop if it becomes empty)
                new_series = {}
                for t, s in ticker_series.items():
                    s2 = s.loc[common_start:common_end]
                    if s2.empty:
                        diagnostics[t] = diagnostics.get(t, {})
                        diagnostics[t]["dropped_after_intersection"] = True
                        # we won't include this ticker
                    else:
                        new_series[t] = s2
                ticker_series = new_series
    else:
        restrict_used = False
        restrict_msg = "No restriction to common intersection applied."

    # --- continue with earlier flow: build coverage, union index, align, drop low coverage etc. ---
    coverage_rows = []
    for t, s in ticker_series.items():
        total = len(s.index.unique())
        coverage_rows.append({
            "ticker": t,
            "n_points": int(total),
            "start": str(s.index.min()),
            "end": str(s.index.max()),
            "diag": diagnostics.get(t, {})
        })
    cov_df = pd.DataFrame(coverage_rows).set_index("ticker").sort_values(by="n_points", ascending=False)

    # union index (after optional restriction)
    all_index = sorted({ts for s in ticker_series.values() for ts in s.index})
    if not all_index:
        raise RuntimeError("No valid series left after optional intersection restriction.")

    prices_wide = pd.DataFrame(index=all_index)
    for t, s in ticker_series.items():
        prices_wide[t] = s.reindex(all_index)

    total_len = len(prices_wide)
    non_na_frac = prices_wide.notna().sum() / float(total_len)
    coverage_table = pd.DataFrame({
        "n_non_na": prices_wide.notna().sum().astype(int),
        "frac_non_na": non_na_frac,
        "n_points_total": total_len
    })
    diag_map = {t: diagnostics.get(t,{}) for t in ticker_series.keys()}
    coverage_table["diag"] = coverage_table.index.map(lambda t: diag_map.get(t, {}))

    # drop tickers below coverage threshold if requested
    to_drop = coverage_table[coverage_table["frac_non_na"] < min_coverage_frac].index.tolist()
    if drop_tickers_with_low_coverage and to_drop:
        for t in to_drop:
            prices_wide.drop(columns=t, inplace=True, errors='ignore')
            diagnostics[t] = diagnostics.get(t, {})
            diagnostics[t]["dropped_low_coverage"] = True

    # final cleanup
    prices_wide = prices_wide.sort_index()
    prices_wide = prices_wide.ffill(limit=ffill_limit).bfill(limit=ffill_limit)
    prices_wide = prices_wide.dropna(how='all')

    out_prices = processedp / f"prices_{freq}.parquet"
    prices_wide.to_parquet(out_prices)

    coverage_out = processedp / f"coverage_{freq}.csv"
    coverage_export = coverage_table.copy()
    coverage_export["diag"] = coverage_export["diag"].apply(lambda x: json.dumps(x))
    coverage_export.to_csv(coverage_out)

    meta = {
        "n_tickers_processed": len(files),
        "n_tickers_saved": prices_wide.shape[1],
        "saved_prices": str(out_prices),
        "freq": freq,
        "range": {"start": str(prices_wide.index.min()), "end": str(prices_wide.index.max())},
        "failed_files": failed,
        "diagnostics": {t: diagnostics[t] for t in diagnostics},
        "restrict_to_common_intersection": restrict_used,
        "restrict_msg": restrict_msg
    }
    # sanitize meta for JSON serialization
    safe_meta = _make_json_serializable(meta)
    with open(processedp / "metadata.json", "w") as f:
        json.dump(safe_meta, f, indent=2)


    report_lines = []
    report_lines.append(f"Preprocessing report\nfreq={freq}\nprocessed_files={len(files)}\n")
    report_lines.append(f"tickers_processed={len(files)}\n")
    report_lines.append(f"tickers_saved={prices_wide.shape[1]}\n")
    report_lines.append(f"index_range={meta['range']}\n")
    report_lines.append(f"drop_tickers_count={len(to_drop)}\n")
    report_lines.append(f"restrict_to_common_intersection={restrict_used}\n")
    report_lines.append(f"restrict_msg={restrict_msg}\n")
    report_lines.append("\nFailed files summary:\n")
    for k, v in failed.items():
        report_lines.append(f"{k} -> {v}\n")
    open(processedp / "preprocess_report.txt", "w").write("\n".join(report_lines))

    summary = {
        "saved_prices": str(out_prices),
        "n_tickers_saved": prices_wide.shape[1],
        "n_tickers_attempted": len(files),
        "coverage_csv": str(coverage_out),
        "metadata": meta
    }
    return summary
