# Hermis

Hermis is a small research/backtesting sandbox for **portfolio-level strategies**.

What you do with it:

1) Run an experiment (backtest) using a YAML config
2) Get an output folder under `experiments/` (NAV, weights, trades, logs)
3) Open **Prism** (Streamlit) to view and compare runs

What’s included out of the box:

- **EMA Trend / EMA Hybrid strategy** (EMA fast/slow crossover used as a universe filter)
- Several optimizers under `hermis/optimizers/` (mv_reg, minvar, risk_parity, sharpe, online OMD/FTRL, etc.)
  - `portfolio_sim/optimizer.py` is now a thin compatibility wrapper.
- Benchmark overlay in the UI (e.g., **NIFTY 50** via Yahoo ticker `^NSEI`)

---

## 1) Quickstart

### Install

From the `Hermis/` folder:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
```

### Run an experiment

```bash
python run_experiment.py --config configs/ema.yaml
```

Or using the convenience launcher:

```bash
./levitate --ema.yaml
```

This creates a new folder under `experiments/` containing:

- `nav.parquet` (NAV series)
- `weights.parquet` (portfolio weights by date)
- `trades.parquet` (optional; if enabled)
- `params.yaml` (snapshot of the config you ran)
- `logs/` (run logs + metadata)

### Open the UI (Prism)

```bash
streamlit run prism/app.py
```

---

## 2) Data: where prices come from

Hermis expects **daily prices** as a *wide* DataFrame:

- index = dates (`DatetimeIndex`)
- columns = tickers
- values = price (close)

The loader also supports:

1) **Long** parquet: columns like `(date, ticker, price)` (it pivots internally)
2) **Directory of per-ticker parquet files**

### Close-only parquet files (your case)

If your per-ticker parquet files contain **close only**, that’s fine.

You have two easy options:

**Option A (recommended): directory of per-ticker parquets**

Put files like:

```
data/my_parquets_1D/
  RELIANCE.parquet
  TCS.parquet
  INFY.parquet
  ...
```

Each file can be any of these shapes:

- a single column with a DatetimeIndex (we’ll treat that column as the price)
- columns like `date` + `close`

Then point your config at the directory:

```yaml
data:
  mode: processed
  processed_path: data/my_parquets_1D
```

**Option B: one wide parquet**

Create a single parquet like `data/processed/prices_1D_india.parquet` where:

- rows = dates
- columns = tickers
- values = close prices

Then use:

```yaml
data:
  mode: processed
  processed:
    dataset: india
    base_dir: data/processed
    freq: 1D
```

---

## 3) Config-driven experiments

Experiments are configured via YAML in `configs/`.

Common knobs:

- `data.*` : where the price data is
- `backtest.*` : start/end dates, rebalance frequency, transaction costs
- `strategy.*` : signal/selection logic
- `optimizer.*` : how weights are chosen

### EMA strategy (configs/ema.yaml)

The EMA strategy does two things at each rebalance date:

1) **Shortlist assets** using an EMA crossover signal (fast EMA vs slow EMA)
2) **Allocate weights** using an optimizer (default: `mv_reg`)

This structure is intentional: an EMA crossover by itself is usually weak. Using it as a **shortlist / universe filter** and then letting an optimizer choose weights is usually more robust.

Key knobs in `configs/ema.yaml`:

```yaml
optimizer:
  type: ema_hybrid
  ema:
    fast_span: 12
    slow_span: 26

    bullish_threshold: 0.0
    min_assets: 8
    max_assets: 30

    # how to turn scores into expected returns for the optimizer
    mu_source: blend      # ema_score | returns | blend
    mu_blend_alpha: 0.7   # blend weight toward ema_score

    # which optimizer to run after filtering
    post_optimizer: mv_reg
```

Notes:

- **Lookahead safe**: EMA signals are computed using prices up to the *previous trading day*.
- Your datasets are currently **close-only**. The config supports `fast_price_field`/`slow_price_field` for forward-compatibility, but the implementation falls back to close if anything else is set.

---

## 4) Benchmarks (e.g., NIFTY 50)

Hermis Prism can overlay a benchmark NAV alongside your strategy NAV.

### How NIFTY 50 is loaded

We load NIFTY 50 using **Yahoo Finance** (ticker: `^NSEI`).

Implementation: `prism/engine/benchmarks.py`

Order of attempts:

1) `yfinance` (if available)
2) Yahoo “chart” API endpoint (no yfinance dependency)

There is also a best-effort NSE fallback in the code, but NSE endpoints often break due to bot protection and site changes.

Benchmark series are cached locally to:

```
.cache/benchmarks/
```

### Common benchmark issues

**1) "yfinance is not installed" even though you installed it**

This usually means **Streamlit is running in a different Python environment** than the one you installed packages into.

Fix by:

- activating the correct venv/conda env **before** launching Streamlit
- or reinstalling from `requirements.txt` in the environment you actually run Streamlit in

**2) NSE endpoint errors**

NSE blocks many automated requests and frequently changes endpoints.

In practice: use Yahoo (`^NSEI`) as the benchmark source and treat NSE as best-effort.

### Adding more benchmarks

Open `prism/engine/benchmarks.py` and add an entry to `BENCHMARKS`.
Example (S&amp;P 500):

```python
"S&amp;P 500 (Yahoo: ^GSPC)": BenchmarkSpec(
    name="S&amp;P 500",
    providers=("yahoo",),
    yahoo_ticker="^GSPC",
)
```

---

## 5) Repo layout (where to look)

```
Hermis/
  run_experiment.py              # main experiment runner
  levitate / levitate.bat        # convenience launcher
  configs/                       # YAML configs
  experiments/                   # output folders for runs
  hermis/                        # new modular API (optimizers, data, event-driven skeleton)
  portfolio_sim/                 # legacy simulation code (still used) + compat wrappers
  engine/                        # strategy/analytics helpers
  prism/app.py                   # Streamlit entrypoint
  prism/engine/ui.py             # Streamlit UI implementation
  prism/engine/benchmarks.py
```

---

## 6) Adding your own strategy

The clean path is:

1) Add a new strategy function (signal + selection + optimizer wiring)
2) Register it in the experiment runner (so YAML `strategy.type` can pick it)
3) Make sure your output artifacts (nav/weights/trades) are written under the experiment folder

If you keep the interface similar to the existing strategies, Hermis Prism will “just work”.

---

## 7) Troubleshooting checklist

If something doesn’t load / runs slow / errors out, this is the fastest checklist:

1) **Are you running Streamlit in the same environment you installed requirements into?**
2) Does your price parquet load in Python?

```python
import pandas as pd
df = pd.read_parquet("data/processed/prices_1D_india.parquet")
print(df.shape, df.index.min(), df.index.max())
```

3) If benchmarks fail: try again (rate limits happen), or change the date range to a smaller window.
