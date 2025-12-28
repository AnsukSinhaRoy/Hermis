"""Benchmark helpers for Hermis Prism.

Benchmarks are best-effort and should never break the UI.

Providers:
- Yahoo Finance (via yfinance)
- NSE India (unofficial JSON endpoint; works in many cases where Yahoo is blocked)

Notes:
- Both sources may fail depending on network conditions / rate limits.
- We cache downloaded series locally to avoid repeated requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    providers: Tuple[str, ...] = ("yahoo",)
    yahoo_ticker: Optional[str] = None
    nse_index: Optional[str] = None
    column: str = "Adj Close"  # prefer total-return-ish if available


# Friendly names shown in the UI.
BENCHMARKS: Dict[str, BenchmarkSpec] = {
    # Prefer Yahoo (simpler), but fall back to NSE if Yahoo is blocked.
    "NIFTY 50 (Auto: NSE / Yahoo)": BenchmarkSpec(
        name="NIFTY 50",
        providers=("yahoo", "nse"),
        yahoo_ticker="^NSEI",
        nse_index="NIFTY 50",
        column="Adj Close",
    ),
    # Backward compatible label from earlier versions.
    "NIFTY 50 (Yahoo: ^NSEI)": BenchmarkSpec(
        name="NIFTY 50",
        providers=("yahoo", "nse"),
        yahoo_ticker="^NSEI",
        nse_index="NIFTY 50",
        column="Adj Close",
    ),
}


def _bench_cache_dir() -> Path:
    p = Path(".cache") / "benchmarks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(provider: str, ident: str) -> str:
    safe = (
        (ident or "")
        .replace("^", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    return f"{provider}__{safe}".strip("_")


def _load_cached(provider: str, ident: str) -> Optional[pd.Series]:
    fp = _bench_cache_dir() / f"{_cache_key(provider, ident)}.parquet"
    if not fp.exists():
        return None
    try:
        df = pd.read_parquet(fp)
        if "price" in df.columns:
            s = df["price"]
        else:
            # Backwards/unknown format
            s = df.iloc[:, 0]
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.sort_index()
        return s
    except Exception:
        return None


def _save_cached(provider: str, ident: str, s: pd.Series) -> None:
    fp = _bench_cache_dir() / f"{_cache_key(provider, ident)}.parquet"
    try:
        out = pd.DataFrame({"price": s}).sort_index()
        out.to_parquet(fp)
    except Exception:
        # cache failures shouldn't crash the app
        return



def _download_yahoo_chart_api(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Download daily close prices using Yahoo's public chart endpoint (no yfinance dependency)."""
    # Local import so the module doesn't hard-fail if requests isn't installed in some setups.
    import requests

    # Yahoo end is exclusive-ish; pad a bit on both sides
    start_epoch = int((start - pd.Timedelta(days=7)).timestamp())
    end_epoch = int((end + pd.Timedelta(days=3)).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": start_epoch,
        "period2": end_epoch,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }

    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        txt = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(f"Yahoo chart API failed with HTTP {r.status_code}: {txt}")

    payload = r.json()
    chart = payload.get("chart", {}) if isinstance(payload, dict) else {}
    if chart.get("error"):
        err = chart.get("error") or {}
        raise RuntimeError(f"Yahoo chart API returned error: {err.get('code')} {err.get('description')}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo chart API returned no results.")
    result = results[0]
    ts = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0] or {}
    adj = (indicators.get("adjclose") or [{}])[0] or {}

    close = quote.get("close") or []
    adjclose = adj.get("adjclose") or []

    # Prefer adjusted close if available and not all missing
    values = adjclose if (adjclose and any(v is not None for v in adjclose)) else close
    if not values or not ts:
        raise RuntimeError("Yahoo chart API returned empty price series.")

    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None)
    s = pd.Series(values, index=idx, dtype="float64").dropna()
    s.name = ticker

    # Filter to requested window (inclusive)
    s = s.loc[(s.index >= start) & (s.index <= end)]
    if s.empty:
        raise RuntimeError(f"Yahoo chart API returned no data in range for {ticker!r}.")
    return s


def _download_yahoo(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Try yfinance first (if available), else fall back to Yahoo chart API."""
    # Attempt yfinance if installed; if anything goes wrong, try the lightweight fallback.
    try:
        import yfinance as yf  # type: ignore

        # Yahoo end is exclusive-ish; pad a bit
        start_s = (start - pd.Timedelta(days=7)).date().isoformat()
        end_s = (end + pd.Timedelta(days=3)).date().isoformat()

        df = yf.download(ticker, start=start_s, end=end_s, progress=False, auto_adjust=False)
        if df is None or df.empty:
            raise RuntimeError(f"No data returned from Yahoo for ticker {ticker!r}.")

        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            raise RuntimeError(f"Yahoo data for {ticker!r} has no Close/Adj Close column.")

        s = df[col].copy()
        s.index = pd.to_datetime(s.index)
        s = s.loc[(s.index >= start) & (s.index <= end)].dropna()
        s.name = ticker
        if s.empty:
            raise RuntimeError(f"Yahoo returned no data in range for ticker {ticker!r}.")
        return s
    except ImportError:
        return _download_yahoo_chart_api(ticker, start, end)
    except Exception as e:
        # yfinance installed but failed (network / schema changes / etc.)
        try:
            return _download_yahoo_chart_api(ticker, start, end)
        except Exception as e2:
            raise RuntimeError(f"Yahoo download failed via yfinance ({type(e).__name__}: {e}) and chart API ({type(e2).__name__}: {e2}).") from e2


def _download_nse_index(index_name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Download historical index levels from NSE India.

    NSE runs anti-bot protections, so we:
    1) establish cookies via a homepage hit
    2) call the historical endpoint with browser-like headers

    This is *best-effort* and may still fail (403/401) depending on environment.
    """
    import json
    import requests

    if not index_name:
        raise RuntimeError("NSE index name is empty.")

    # NSE expects DD-MM-YYYY for these query params.
    # Pad slightly so we don't miss the first/last trading day.
    from_dt = (start - pd.Timedelta(days=7)).strftime("%d-%m-%Y")
    to_dt = (end + pd.Timedelta(days=3)).strftime("%d-%m-%Y")

    url = (
        "https://www.nseindia.com/api/historical/indicesHistory"
        f"?indexType={requests.utils.quote(index_name)}&from={from_dt}&to={to_dt}"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
        "Connection": "keep-alive",
    }

    s = requests.Session()
    # Cookie bootstrap
    s.get("https://www.nseindia.com", headers=headers, timeout=20)
    r = s.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"NSE request failed with HTTP {r.status_code}: {r.text[:200]}")

    try:
        payload = r.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"NSE response was not valid JSON: {r.text[:200]}") from e

    # Different shapes are seen across NSE endpoints/wrappers.
    rows = None
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            rows = payload["data"]
        elif "price" in payload and isinstance(payload["price"], list):
            rows = payload["price"]
        elif "result" in payload and isinstance(payload["result"], list):
            rows = payload["result"]

    if not rows:
        raise RuntimeError("NSE returned no rows for the requested window.")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("NSE returned empty dataframe.")

    # Date column candidates.
    date_col = None
    for c in [
        "HISTORICAL_DATE",
        "EOD_TIMESTAMP",
        "TIMESTAMP",
        "DATE",
        "Date",
        "date",
    ]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise RuntimeError(f"Could not find a date column in NSE response. Columns: {list(df.columns)[:20]}")

    # Close column candidates.
    px_col = None
    for c in [
        "CLOSE",
        "Close",
        "close",
        "CH_CLOSING_INDEX",
        "CLOSE_INDEX_VALUE",
        "CLOSE_PRICE",
    ]:
        if c in df.columns:
            px_col = c
            break
    if px_col is None:
        raise RuntimeError(f"Could not find a close column in NSE response. Columns: {list(df.columns)[:20]}")

    out = df[[date_col, px_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)
    out[px_col] = (
        out[px_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    out[px_col] = pd.to_numeric(out[px_col], errors="coerce")
    out = out.dropna(subset=[date_col, px_col]).sort_values(date_col)
    if out.empty:
        raise RuntimeError("NSE returned no valid (date, close) rows after parsing.")

    ser = out.set_index(date_col)[px_col]
    ser.index = pd.to_datetime(ser.index, errors="coerce")
    ser = ser.sort_index()
    ser.name = f"NSE:{index_name}"
    return ser


@st.cache_data(show_spinner=False)
def get_benchmark_price_series(
    spec: BenchmarkSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Return a price series for the given benchmark spec, using cache + download."""

    errors: List[str] = []

    for provider in (spec.providers or ("yahoo",)):
        if provider == "yahoo":
            ident = spec.yahoo_ticker or ""
        elif provider == "nse":
            ident = spec.nse_index or ""
        else:
            ident = ""

        cached = _load_cached(provider, ident)
        if cached is not None and not cached.empty:
            if cached.index.min() <= start and cached.index.max() >= end:
                return cached

        try:
            if provider == "yahoo":
                if not spec.yahoo_ticker:
                    raise RuntimeError("No Yahoo ticker configured.")
                s = _download_yahoo(spec.yahoo_ticker, start=start, end=end)
            elif provider == "nse":
                if not spec.nse_index:
                    raise RuntimeError("No NSE index name configured.")
                s = _download_nse_index(spec.nse_index, start=start, end=end)
            else:
                raise RuntimeError(f"Unknown provider: {provider}")
            _save_cached(provider, ident, s)
            return s
        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {e}")
            continue

    raise RuntimeError(" | ".join(errors) if errors else "Benchmark download failed.")


def try_get_benchmark_nav(
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Fetch benchmark NAV normalized to 1.

    Returns (nav, error). If nav is None, error contains a human-readable reason.
    """
    spec = BENCHMARKS.get(label)
    if spec is None:
        return None, f"Unknown benchmark label: {label!r}"

    try:
        px = get_benchmark_price_series(spec, start=start, end=end)
    except Exception as e:
        return None, str(e)

    px = px.loc[(px.index >= start) & (px.index <= end)].copy()
    px = px.ffill().dropna()
    if px.empty:
        return None, "No benchmark prices overlap the requested date window."

    nav = px / float(px.iloc[0])
    nav.name = f"Benchmark: {spec.name}"
    return nav, None


def get_benchmark_nav(
    label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.Series]:
    """Backward-compatible wrapper for existing UI callers."""
    nav, _err = try_get_benchmark_nav(label, start=start, end=end)
    return nav


def get_custom_yahoo_nav(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    label: Optional[str] = None,
) -> Optional[pd.Series]:
    """Custom ticker via Yahoo Finance."""
    ticker = (ticker or "").strip()
    if not ticker:
        return None
    try:
        px = _download_yahoo(ticker, start=start, end=end)
    except Exception:
        return None

    px = px.loc[(px.index >= start) & (px.index <= end)].copy()
    px = px.ffill().dropna()
    if px.empty:
        return None
    nav = px / float(px.iloc[0])
    nav.name = f"Benchmark: {label or ticker}"
    return nav
