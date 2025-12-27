"""portfolio_sim/preprocess_xlsx.py

Preprocessing pipeline for raw Excel stock files.

Expected raw layout:

  raw_stock_data/
    India/
      <something>.xlsx
      <matching>.json
    US/
      <something>.xlsx
      <matching>.json

For each .xlsx we:
  1) read it robustly (date column can be "Date", "date", "Unnamed: 0", etc.)
  2) write a parquet copy next to the xlsx (same stem)
  3) extract a close-price series and build a wide prices matrix
  4) aggregate sidecar JSON metadata into a tidy table.

Outputs (in processed_dir):
  - prices_<freq>_india.parquet
  - prices_<freq>_us.parquet
  - prices_<freq>.parquet (only when dataset="both")
  - asset_metadata.parquet + asset_metadata.json
  - preprocess_xlsx_metadata.json + preprocess_xlsx_report.txt

This module is intentionally separate from portfolio_sim/preprocess.py, which
targets minute CSV inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import os
import traceback

import numpy as np
import pandas as pd


def _make_json_serializable(obj: Any) -> Any:
    """Best-effort conversion to JSON-friendly types."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(v) for v in obj]
    return repr(obj)


def _pick_date_col(cols: List[str]) -> Optional[str]:
    """Heuristic date column detection for typical Excel exports."""
    lowered = {c.lower(): c for c in cols}
    for k in ("date", "datetime", "timestamp", "time"):
        if k in lowered:
            return lowered[k]
    # common when saving a DF with index to Excel
    for c in cols:
        if str(c).lower().startswith("unnamed"):
            return c
    # last resort: first column
    return cols[0] if cols else None


def _pick_close_col(cols: List[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in cols}
    for k in ("close", "adj close", "adj_close", "adjusted close", "price"):
        if k in lowered:
            return lowered[k]
    return None


def _find_sidecar_json(xlsx_path: Path) -> Optional[Path]:
    """Find the JSON file corresponding to an xlsx.

    We try common patterns:
      - <stem>.json
      - <stem>_meta.json
      - <stem>-meta.json
      - <stem>.meta.json
    """
    folder = xlsx_path.parent
    stem = xlsx_path.stem
    candidates = [
        folder / f"{stem}.json",
        folder / f"{stem}_meta.json",
        folder / f"{stem}-meta.json",
        folder / f"{stem}.meta.json",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _read_excel_robust(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df is None or df.empty:
        return pd.DataFrame()

    date_col = _pick_date_col(list(df.columns))
    if date_col is None:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    tmp = tmp.set_index(date_col).sort_index()
    # remove duplicate timestamps (keep first)
    tmp = tmp.loc[~tmp.index.duplicated(keep="first")]

    # If timezone-aware, make naive
    try:
        if getattr(tmp.index, "tz", None) is not None:
            tmp.index = tmp.index.tz_convert(None)
    except Exception:
        try:
            tmp.index = tmp.index.tz_localize(None)
        except Exception:
            pass

    return tmp


def _xlsx_to_parquet(xlsx_path: Path, parquet_path: Path, overwrite: bool = True) -> Dict[str, Any]:
    """Convert a single xlsx to parquet (index preserved)."""
    out: Dict[str, Any] = {"ok": False, "rows": 0, "parquet": str(parquet_path), "error": None}
    try:
        if parquet_path.exists() and not overwrite:
            out["ok"] = True
            return out

        df = _read_excel_robust(xlsx_path)
        out["rows"] = int(df.shape[0])
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=True)
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = {"reason": "exception", "details": repr(e), "trace": traceback.format_exc()}
        return out


def _extract_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    close_col = _pick_close_col(list(df.columns))
    if close_col is None:
        return None

    s = pd.to_numeric(df[close_col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.where(s > 0)
    s = s.dropna()
    if s.empty:
        return None
    return s


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass
class SingleAssetResult:
    ok: bool
    ticker: Optional[str]
    market: str
    series: Optional[pd.Series]
    excel_path: str
    parquet_path: str
    json_path: Optional[str]
    meta: Dict[str, Any]
    diag: Dict[str, Any]
    error: Optional[Dict[str, Any]]


def _process_one_xlsx(xlsx_path: Path, market: str, overwrite_parquet: bool = True) -> SingleAssetResult:
    json_path = _find_sidecar_json(xlsx_path)
    side_meta = _read_json(json_path)

    # Use JSON 'ticker' if present (preferred), else fall back to filename stem.
    ticker = side_meta.get("ticker") or xlsx_path.stem

    parquet_path = xlsx_path.with_suffix(".parquet")
    conv = _xlsx_to_parquet(xlsx_path, parquet_path, overwrite=overwrite_parquet)

    diag: Dict[str, Any] = {
        "xlsx_rows": conv.get("rows", 0),
        "converted_to_parquet": bool(conv.get("ok")),
    }
    if not conv.get("ok"):
        return SingleAssetResult(
            ok=False,
            ticker=ticker,
            market=market,
            series=None,
            excel_path=str(xlsx_path),
            parquet_path=str(parquet_path),
            json_path=str(json_path) if json_path else None,
            meta=side_meta,
            diag=diag,
            error=conv.get("error") or {"reason": "xlsx_to_parquet_failed"},
        )

    try:
        df = pd.read_parquet(parquet_path)
        # parquet may load index as a column depending on engine; handle both
        if not isinstance(df.index, pd.DatetimeIndex):
            # try common index columns
            for c in ("__index_level_0__", "index", "date", "Date"):
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    df = df.dropna(subset=[c]).set_index(c).sort_index()
                    break
        if not isinstance(df.index, pd.DatetimeIndex):
            # last resort: try the first column
            if df.shape[1] >= 2:
                c0 = df.columns[0]
                df[c0] = pd.to_datetime(df[c0], errors="coerce")
                df = df.dropna(subset=[c0]).set_index(c0).sort_index()

        series = _extract_close_series(df)
        if series is None or series.empty:
            return SingleAssetResult(
                ok=False,
                ticker=ticker,
                market=market,
                series=None,
                excel_path=str(xlsx_path),
                parquet_path=str(parquet_path),
                json_path=str(json_path) if json_path else None,
                meta=side_meta,
                diag=diag,
                error={"reason": "missing_close_column_or_empty"},
            )

        diag.update({
            "series_points": int(series.shape[0]),
            "series_start": str(series.index.min()),
            "series_end": str(series.index.max()),
        })

        return SingleAssetResult(
            ok=True,
            ticker=str(ticker),
            market=market,
            series=series,
            excel_path=str(xlsx_path),
            parquet_path=str(parquet_path),
            json_path=str(json_path) if json_path else None,
            meta=side_meta,
            diag=diag,
            error=None,
        )
    except Exception as e:
        return SingleAssetResult(
            ok=False,
            ticker=str(ticker),
            market=market,
            series=None,
            excel_path=str(xlsx_path),
            parquet_path=str(parquet_path),
            json_path=str(json_path) if json_path else None,
            meta=side_meta,
            diag=diag,
            error={"reason": "exception", "details": repr(e), "trace": traceback.format_exc()},
        )


def _normalize_dataset_choice(dataset: str) -> str:
    d = (dataset or "both").strip().lower()
    aliases = {
        "in": "india",
        "ind": "india",
        "indian": "india",
        "us": "us",
        "usa": "us",
        "united_states": "us",
        "both": "both",
        "all": "both",
        "india+us": "both",
    }
    return aliases.get(d, d)


def _write_prices_bundle(
    prices: pd.DataFrame,
    processed_dir: Path,
    freq: str,
    suffix: str,
    ffill_limit: int = 5,
) -> Dict[str, str]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    prices = prices.sort_index()
    prices = prices.ffill(limit=ffill_limit).bfill(limit=ffill_limit)
    prices = prices.dropna(how="all")
    out = processed_dir / f"prices_{freq}{suffix}.parquet"
    prices.to_parquet(out)
    return {"path": str(out), "rows": int(prices.shape[0]), "cols": int(prices.shape[1])}


def preprocess_xlsx_pipeline(
    raw_root: str = "raw_stock_data",
    processed_dir: str = "data/processed",
    dataset: str = "both",  # india | us | both
    freq: str = "1D",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_coverage_frac: float = 0.5,
    drop_tickers_with_low_coverage: bool = True,
    restrict_to_common_intersection: bool = False,
    overwrite_parquet_cache: bool = True,
    max_workers: Optional[int] = None,
    ffill_limit: int = 5,
) -> Dict[str, Any]:
    """Build processed parquet(s) from raw Excel files.

    Notes:
      - freq is currently used only for naming outputs; Excel data is assumed
        to already be at the desired frequency (typically daily).
      - For `dataset='both'` we write both per-market price files and a combined
        `prices_<freq>.parquet` convenience file.
    """
    raw_root_p = Path(raw_root)
    processed_p = Path(processed_dir)

    dataset_norm = _normalize_dataset_choice(dataset)
    if dataset_norm not in {"india", "us", "both"}:
        raise ValueError(f"Unknown dataset={dataset!r} (expected india|us|both)")

    market_dirs: List[Tuple[str, Path]] = []
    if dataset_norm in {"india", "both"}:
        market_dirs.append(("India", raw_root_p / "India"))
    if dataset_norm in {"us", "both"}:
        market_dirs.append(("US", raw_root_p / "US"))

    for m, p in market_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Missing raw folder for {m}: {p}")

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 2)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    all_asset_meta_rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    per_market_prices: Dict[str, pd.DataFrame] = {}
    per_market_written: Dict[str, Any] = {}

    def _date_slice(s: pd.Series) -> pd.Series:
        out = s.copy()
        if start_date is not None:
            out = out.loc[out.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            out = out.loc[out.index <= pd.to_datetime(end_date)]
        return out

    # --- process each market folder ---
    for market_name, market_dir in market_dirs:
        xlsx_files = sorted(market_dir.glob("*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(f"No .xlsx files found in {market_dir}")

        results: List[SingleAssetResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_process_one_xlsx, p, market_name, overwrite_parquet_cache) for p in xlsx_files]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"xlsx:{market_name}", unit="file"):
                results.append(fut.result())

        # build prices
        series_map: Dict[str, pd.Series] = {}
        for r in results:
            meta_row = {
                "ticker": r.ticker,
                "market": r.market,
                "excel_path": r.excel_path,
                "parquet_path": r.parquet_path,
                "json_path": r.json_path,
                **(r.meta or {}),
                "diag": r.diag,
                "ok": bool(r.ok),
            }
            all_asset_meta_rows.append(meta_row)

            if not r.ok or r.series is None:
                failed.append({
                    "ticker": r.ticker,
                    "market": r.market,
                    "excel_path": r.excel_path,
                    "error": r.error,
                })
                continue

            s = _date_slice(r.series)
            if s.empty:
                failed.append({
                    "ticker": r.ticker,
                    "market": r.market,
                    "excel_path": r.excel_path,
                    "error": {"reason": "empty_after_date_slice"},
                })
                continue
            # ensure unique ticker key
            key = str(r.ticker)
            if key in series_map:
                # avoid silent overwrite; suffix with market
                key = f"{key}__{market_name}"
            series_map[key] = s

        if not series_map:
            raise RuntimeError(f"No valid series produced for market={market_name}")

        # optional common intersection restriction within this market
        if restrict_to_common_intersection:
            starts = [s.index.min() for s in series_map.values()]
            ends = [s.index.max() for s in series_map.values()]
            cs = max(starts)
            ce = min(ends)
            if cs < ce:
                series_map = {k: v.loc[cs:ce] for k, v in series_map.items() if not v.loc[cs:ce].empty}

        union_index = sorted({ts for s in series_map.values() for ts in s.index})
        prices = pd.DataFrame(index=pd.DatetimeIndex(union_index))
        for k, s in series_map.items():
            prices[k] = s.reindex(prices.index)

        # coverage
        total_len = len(prices.index)
        frac_non_na = prices.notna().sum() / float(total_len) if total_len else 0.0
        to_drop = frac_non_na[frac_non_na < float(min_coverage_frac)].index.tolist()
        if drop_tickers_with_low_coverage and to_drop:
            prices = prices.drop(columns=to_drop, errors="ignore")

        per_market_prices[market_name] = prices

        suffix = "_india" if market_name.lower() == "india" else "_us"
        per_market_written[market_name] = _write_prices_bundle(prices, processed_p, freq=freq, suffix=suffix, ffill_limit=ffill_limit)

    # --- combined bundle ---
    combined_written = None
    if dataset_norm == "both":
        # union across markets
        combined = None
        for _, df in per_market_prices.items():
            combined = df if combined is None else combined.join(df, how="outer")

        if combined is None or combined.empty:
            raise RuntimeError("Combined prices empty.")

        if restrict_to_common_intersection:
            starts = [combined[c].dropna().index.min() for c in combined.columns if combined[c].notna().any()]
            ends = [combined[c].dropna().index.max() for c in combined.columns if combined[c].notna().any()]
            if starts and ends:
                cs = max(starts)
                ce = min(ends)
                if cs < ce:
                    combined = combined.loc[cs:ce]

        combined_written = _write_prices_bundle(combined, processed_p, freq=freq, suffix="", ffill_limit=ffill_limit)
    # If only one market is requested, we intentionally DO NOT overwrite the
    # default `prices_<freq>.parquet` (which is reserved for the combined dataset).
    # Users should point to the per-market output (e.g. prices_1D_india.parquet).

    # --- write asset metadata table ---
    asset_meta_df = pd.DataFrame(all_asset_meta_rows)
    # keep diag as JSON string for parquet friendliness
    if "diag" in asset_meta_df.columns:
        asset_meta_df["diag"] = asset_meta_df["diag"].apply(lambda x: json.dumps(_make_json_serializable(x)) if isinstance(x, (dict, list)) else ("" if x is None else str(x)))
    asset_meta_path = processed_p / "asset_metadata.parquet"
    asset_meta_df.to_parquet(asset_meta_path, index=False)
    (processed_p / "asset_metadata.json").write_text(json.dumps(_make_json_serializable(all_asset_meta_rows), indent=2), encoding="utf-8")

    meta = {
        "raw_root": str(raw_root_p),
        "dataset": dataset_norm,
        "freq": freq,
        "start_date": start_date,
        "end_date": end_date,
        "min_coverage_frac": min_coverage_frac,
        "drop_tickers_with_low_coverage": drop_tickers_with_low_coverage,
        "restrict_to_common_intersection": restrict_to_common_intersection,
        "written": {
            "per_market": per_market_written,
            "combined": combined_written,
            "asset_metadata": str(asset_meta_path),
        },
        "failed": failed,
    }
    safe_meta = _make_json_serializable(meta)
    (processed_p / "preprocess_xlsx_metadata.json").write_text(json.dumps(safe_meta, indent=2), encoding="utf-8")

    # lightweight report
    lines = []
    lines.append("Preprocess XLSX report")
    lines.append(f"dataset={dataset_norm} freq={freq}")
    lines.append(f"start_date={start_date} end_date={end_date}")
    lines.append(f"asset_metadata={asset_meta_path}")
    lines.append(f"per_market_written={per_market_written}")
    lines.append(f"combined_written={combined_written}")
    lines.append(f"failed_count={len(failed)}")
    (processed_p / "preprocess_xlsx_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "prices_path": (combined_written or {}).get("path"),
        "per_market_prices": {k: v.get("path") for k, v in per_market_written.items()},
        "asset_metadata": str(asset_meta_path),
        "failed_count": len(failed),
        "metadata": meta,
    }
