# portfolio_viz/app.py
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List

try:
    from portfolio_sim.experiment import load_experiment
except Exception:
    def load_experiment(path: str) -> Dict[str, Any]:
        out = Path(path) / "outputs"
        nav = pd.read_parquet(out / "nav.parquet") if (out / "nav.parquet").exists() else None
        weights = pd.read_parquet(out / "weights.parquet") if (out / "weights.parquet").exists() else None
        trades = pd.read_parquet(out / "trades.parquet") if (out / "trades.parquet").exists() else None
        meta = json.load(open(out / "metadata.json")) if (out / "metadata.json").exists() else {}
        return {"nav": nav, "weights": weights, "trades": trades, "meta": meta}

st.set_page_config(page_title="Portfolio Viz", layout="wide")

# Utilities
def ensure_series(nav) -> Optional[pd.Series]:
    if nav is None:
        return None
    if isinstance(nav, pd.Series):
        return nav
    if isinstance(nav, pd.DataFrame):
        if nav.shape[1] == 1:
            return nav.iloc[:, 0]
        if "value" in nav.columns and nav.shape[1] == 1:
            return nav["value"]
        return nav.iloc[:, 0]
    return pd.Series(nav)

def load_experiment_list(experiments_root: Path) -> List[Path]:
    if not experiments_root.exists():
        return []
    dirs = [p for p in sorted(experiments_root.iterdir(), reverse=True) if p.is_dir() and (p / "outputs").exists()]
    return dirs

def safe_read_prices(exp_folder: Path) -> Optional[pd.DataFrame]:
    p = exp_folder / "data" / "prices.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    return None

# Attempt to load precomputed performance artifacts
def load_precomputed_perf(exp_folder: Path) -> Dict[str, Any]:
    perf = {}
    perf_dir = Path(exp_folder) / 'outputs' / 'performance'
    if not perf_dir.exists():
        return perf
    try:
        if (perf_dir / 'metrics.json').exists():
            perf['metrics'] = json.load(open(perf_dir / 'metrics.json'))
        if (perf_dir / 'rolling_sharpe.parquet').exists():
            perf['rolling_sharpe'] = pd.read_parquet(perf_dir / 'rolling_sharpe.parquet').iloc[:,0]
        if (perf_dir / 'drawdown.parquet').exists():
            perf['drawdown'] = pd.read_parquet(perf_dir / 'drawdown.parquet').iloc[:,0]
        if (perf_dir / 'cum_returns.parquet').exists():
            perf['cum_returns'] = pd.read_parquet(perf_dir / 'cum_returns.parquet').iloc[:,0]
        if (perf_dir / 'selected.parquet').exists():
            perf['selected'] = pd.read_parquet(perf_dir / 'selected.parquet')
    except Exception:
        perf = {}
    return perf

# Performance helpers (fallback)
def ann_stats(nav_s: pd.Series) -> Dict[str, float]:
    days = (nav_s.index[-1] - nav_s.index[0]).days or 1
    total_ret = float(nav_s.iloc[-1] / nav_s.iloc[0] - 1.0)
    ann_ret = (1 + total_ret) ** (365.0 / days) - 1
    dr = nav_s.pct_change().dropna()
    ann_vol = dr.std() * np.sqrt(252) if not dr.empty else 0.0
    roll_max = nav_s.cummax()
    drawdown = (nav_s - roll_max) / roll_max
    max_dd = float(drawdown.min())
    return {"total_return": total_ret, "ann_return": ann_ret, "ann_vol": ann_vol, "max_drawdown": max_dd}


# Sidebar
st.sidebar.title("Experiment Explorer")
root = st.sidebar.text_input("Experiments root folder", value="experiments")
rootp = Path(root)
exp_dirs = load_experiment_list(rootp)

if not exp_dirs:
    st.sidebar.warning(f"No experiments found under {root}. Run an experiment first.")
    st.stop()

exp_labels = [f"{p.name}" for p in exp_dirs]
sel_label = st.sidebar.selectbox("Choose an experiment", exp_labels, index=0)
selected_path = exp_dirs[exp_labels.index(sel_label)]

cmp_paths = []
if st.sidebar.checkbox("Enable comparison mode"):
    cmp_choices = st.sidebar.multiselect("Pick experiments to compare (NAV)", exp_labels, default=[exp_labels[0]])
    cmp_paths = [exp_dirs[exp_labels.index(x)] for x in cmp_choices]

st.sidebar.markdown("---")
st.sidebar.write("Plot options")
sample_max = st.sidebar.number_input("Max assets to draw (prices tab)", min_value=10, max_value=2000, value=100, step=10)
normalize_default = st.sidebar.checkbox("Normalize prices & NAV to 1 at start", value=True)

# Load experiment
with st.spinner('Loading experiment...'):
    exp = load_experiment(str(selected_path))
    nav_raw = exp.get("nav")
    nav = ensure_series(nav_raw)
    weights = exp.get("weights")
    trades = exp.get("trades")
    meta = exp.get("meta", {})
    prices = safe_read_prices(selected_path)
    preperf = load_precomputed_perf(selected_path)

tabs = st.tabs(["Overview", "Allocation", "Prices", "Trades", "Performance", "Asset Inspector", "Compare"])

# Overview
with tabs[0]:
    col1, col2 = st.columns([3,1])
    with col1:
        st.header("Experiment Overview")
        st.write(f"**Folder:** `{selected_path}`")
        st.write("**Metadata (quick)**")
        # show a compact subset of metadata for speed
        try:
            compact_meta = {k: meta.get(k) for k in ['python_version','runtime_seconds','packages'] if k in meta}
        except Exception:
            compact_meta = meta
        st.json(compact_meta)
        st.markdown("---")
        if nav is not None:
            nav_df = nav.reset_index()
            nav_df.columns = ["date", "nav"]
            fig = px.line(nav_df, x="date", y="nav", title="Portfolio NAV")
            st.plotly_chart(fig, use_container_width=True)

            # prefer precomputed metrics
            metrics = preperf.get('metrics') if preperf.get('metrics') else ann_stats(nav)
            st.metric("Total return", f"{metrics['total_return']*100:.2f} %")
            st.metric("Annualized return", f"{metrics['ann_return']*100:.2f} %")
            st.metric("Annualized vol", f"{metrics['ann_vol']*100:.2f} %")
            st.metric("Max drawdown", f"{metrics['max_drawdown']*100:.2f} %")
        else:
            st.warning("NAV not found for this experiment.")
    with col2:
        st.header("Quick actions")
        if st.button("Open outputs folder (print path)"):
            st.write(str(selected_path / "outputs"))
        if prices is None:
            st.info("No saved prices snapshot found in experiment/data. Prices tab may be unavailable.")
        # allow downloads of precomputed perf if present
        perf_dir = selected_path / 'outputs' / 'performance'
        if perf_dir.exists():
            if (perf_dir / 'metrics.json').exists():
                st.download_button("Download precomputed metrics.json", data=open(perf_dir / 'metrics.json').read(), file_name='metrics.json')
            if (perf_dir / 'rolling_sharpe.parquet').exists():
                st.download_button("Download rolling_sharpe.parquet", data=(perf_dir / 'rolling_sharpe.parquet').read_bytes(), file_name='rolling_sharpe.parquet')
        st.download_button("Download metadata.json", data=json.dumps(meta, indent=2), file_name="metadata.json")

# Allocation
with tabs[1]:
    st.header("Allocation Over Time")
    if weights is None:
        st.warning("No weights file found for this experiment.")
    else:
        top_n = st.slider("Show top N assets by average weight", min_value=5, max_value=min(200, weights.shape[1]), value=20)
        avg_weights = weights.mean().sort_values(ascending=False)
        top_assets = avg_weights.head(top_n).index.tolist()
        sub = weights[top_assets]
        fig = go.Figure(data=go.Heatmap(
            z=sub.T.values,
            x=[d.strftime("%Y-%m-%d") for d in sub.index],
            y=list(sub.columns),
            colorscale="Viridis"
        ))
        fig.update_layout(height=600, title=f"Allocation heatmap (top {top_n})", xaxis_nticks=10)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Selected assets each rebalance")
        selected_each = weights.apply(lambda row: row[row>0].index.tolist(), axis=1)
        show_n = st.number_input("Show last N rebalances", min_value=1, max_value=len(selected_each), value=min(10, len(selected_each)))
        rows_to_show = selected_each.dropna().tail(show_n)
        iterator = rows_to_show.items() if hasattr(rows_to_show, "items") else rows_to_show.iteritems()
        for d, sel in iterator:
            st.write(f"**{d.date()}** — {len(sel)} assets — {', '.join(sel[:10])}{'...' if len(sel)>10 else ''}")

# Prices
with tabs[2]:
    st.header("Price Paths & NAV")
    if prices is None:
        st.info("Saved price snapshot not found. Upload CSV or regenerate if synthetic.")
        uploaded = st.file_uploader("Upload price CSV (long or wide)", type=["csv"])
        if uploaded:
            try:
                df_csv = pd.read_csv(uploaded, parse_dates=['date'])
                if {'date','ticker','price'}.issubset(df_csv.columns):
                    pivot = df_csv.pivot(index='date', columns='ticker', values='price').sort_index()
                else:
                    pivot = df_csv.set_index('date').sort_index()
                prices = pivot
                st.success("Loaded CSV into prices.")
            except Exception as e:
                st.error(f"Failed to parse uploaded CSV: {e}")
    if prices is not None:
        st.write(f"Universe size: {prices.shape[1]} assets — {prices.shape[0]} dates")
        normalize = st.checkbox("Normalize to 1 at start (prices and NAV)", value=normalize_default)
        max_assets = st.number_input("Max assets to draw", min_value=10, max_value=1000, value=int(sample_max))
        tickers = sorted(prices.columns)[:max_assets] if max_assets and max_assets < prices.shape[1] else prices.columns.tolist()
        plot_prices = prices[tickers].copy()
        if normalize:
            first_vals = plot_prices.apply(lambda col: col.dropna().iloc[0] if col.dropna().shape[0]>0 else np.nan)
            plot_prices = plot_prices.divide(first_vals, axis=1)
        fig = go.Figure()
        for col in plot_prices.columns:
            fig.add_trace(go.Scatter(x=plot_prices.index, y=plot_prices[col], mode='lines', line=dict(width=1), opacity=0.25, name=col, showlegend=False))
        try:
            median_series = plot_prices.median(axis=1)
            fig.add_trace(go.Scatter(x=plot_prices.index, y=median_series, mode='lines', line=dict(color='gray', width=1.5), name='Median asset'))
        except Exception:
            pass
        if nav is not None:
            nav_s = ensure_series(nav)
            nav_aligned = nav_s.reindex(plot_prices.index).ffill().bfill()
            if normalize:
                nav_aligned = nav_aligned / float(nav_aligned.iloc[0])
            fig.add_trace(go.Scatter(x=nav_aligned.index, y=nav_aligned.values, mode='lines', line=dict(color='red', width=3), name='Portfolio NAV'))
        fig.update_layout(title="Price Paths and NAV", height=600, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

# Trades
with tabs[3]:
    st.header("Trades & Turnover")
    if trades is None:
        st.info("No trades data found.")
    else:
        if 'date' not in trades.columns:
            trades_display = trades.reset_index()
        else:
            trades_display = trades.copy()
        st.dataframe(trades_display.sort_values(by='date', ascending=False).head(500))
        if 'turnover' in trades.columns:
            t = trades.set_index('date')['turnover'].sort_index()
            fig = px.bar(t.reset_index(), x='date', y='turnover', title='Turnover per rebalance')
            st.plotly_chart(fig, use_container_width=True)

# Performance
with tabs[4]:
    st.header("Performance Analytics")
    if nav is None:
        st.info("No NAV available to compute performance.")
    else:
        nav_s = ensure_series(nav)
        # prefer precomputed cum_returns if available
        if 'cum_returns' in preperf:
            cum = preperf['cum_returns']
        else:
            cum = nav_s / float(nav_s.iloc[0])

        dr = nav_s.pct_change().dropna()

        # cumulative returns plot
        fig = px.line(x=cum.index, y=cum.values, title="Cumulative NAV (start=1)", labels={"x":"date","y":"cum_return"})
        st.plotly_chart(fig, use_container_width=True)

        # drawdown
        if 'drawdown' in preperf:
            dd = preperf['drawdown']
        else:
            roll_max = nav_s.cummax()
            dd = ((nav_s - roll_max) / roll_max)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dd.index, y=dd.values, fill='tozeroy', name='Drawdown'))
        fig2.update_layout(title="Drawdown", yaxis_tickformat="%", height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # rolling sharpe
        if 'rolling_sharpe' in preperf:
            rs = preperf['rolling_sharpe']
        else:
            win = st.slider("Sharpe rolling window (days)", min_value=21, max_value=252, value=63)
            rs = dr.rolling(win).mean() / dr.rolling(win).std() * np.sqrt(252)
            rs = rs.dropna()
        fig3 = px.line(x=rs.index, y=rs.values, title=f"Rolling Sharpe")
        st.plotly_chart(fig3, use_container_width=True)

        # table metrics
        metrics = preperf.get('metrics') if preperf.get('metrics') else ann_stats(nav_s)
        st.subheader("Summary metrics")
        st.write({
            "Total return": f"{metrics['total_return']*100:.2f} %",
            "Annualized return": f"{metrics['ann_return']*100:.2f} %",
            "Annualized volatility": f"{metrics['ann_vol']*100:.2f} %",
            "Max drawdown": f"{metrics['max_drawdown']*100:.2f} %"
        })

# Asset Inspector
with tabs[5]:
    st.header("Asset Inspector")
    if prices is None:
        st.info("No prices snapshot to inspect assets.")
    else:
        ticker_search = st.text_input("Filter tickers (substring, comma-separated list)", value="")
        tickers_all = sorted(list(prices.columns))
        if ticker_search.strip():
            tokens = [t.strip().lower() for t in ticker_search.split(",") if t.strip()]
            filtered = [t for t in tickers_all if any(tok in t.lower() for tok in tokens)]
        else:
            filtered = tickers_all
        max_show = st.number_input("Max tickers to show", min_value=1, max_value=min(500, len(filtered)), value=min(20, len(filtered)))
        chosen = st.multiselect("Choose tickers to highlight", options=filtered, default=filtered[:max_show])
        if chosen:
            plot_prices = prices[chosen].copy()
            normalize = st.checkbox("Normalize selected to 1 at start", value=True)
            if normalize:
                first_vals = plot_prices.apply(lambda col: col.dropna().iloc[0] if col.dropna().shape[0]>0 else np.nan)
                plot_prices = plot_prices.divide(first_vals, axis=1)
            fig = go.Figure()
            for col in plot_prices.columns:
                fig.add_trace(go.Scatter(x=plot_prices.index, y=plot_prices[col], mode='lines', name=col))
            # overlay weights (if present) as markers at rebalances
            if weights is not None:
                wsub = weights[chosen].copy()
                for col in wsub.columns:
                    fig.add_trace(go.Scatter(x=wsub.index, y=wsub[col], mode='markers', name=f"w:{col}", marker=dict(size=4), yaxis="y2", showlegend=True))
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Weight'), height=600)
            fig.update_layout(title="Selected asset prices (and weights)", height=600, xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)
            csv = plot_prices.to_csv(index=True)
            st.download_button("Download selected prices CSV", data=csv, file_name="selected_prices.csv")

# ---------- Compare (enhanced) ----------
with tabs[6]:
    st.header("Compare Experiments (detailed metrics)")

    if not cmp_paths:
        st.info("Enable comparison mode in the sidebar to pick experiments.")
    else:
        def compute_returns_from_nav(nav_s: pd.Series) -> pd.Series:
            return nav_s.pct_change().dropna()

        def sharpe_ratio(nav_s: pd.Series, rf=0.0):
            r = compute_returns_from_nav(nav_s)
            if r.empty:
                return np.nan
            mean_excess = r.mean() - rf/252.0
            return float((mean_excess / r.std()) * np.sqrt(252))

        def sortino_ratio(nav_s: pd.Series, rf=0.0):
            r = compute_returns_from_nav(nav_s)
            if r.empty:
                return np.nan
            downside = r[r < 0]
            dd = downside.std() * np.sqrt(252) if not downside.empty else 1e-9
            mean_excess = r.mean() - rf/252.0
            return float((mean_excess * np.sqrt(252)) / dd) if dd != 0 else np.nan

        def calmar_ratio(nav_s: pd.Series):
            metrics = ann_stats(nav_s)
            ann_ret = metrics['ann_return']
            max_dd = abs(metrics['max_drawdown'])
            return float(ann_ret / max_dd) if max_dd > 0 else np.nan

        def win_rate(nav_s: pd.Series):
            r = compute_returns_from_nav(nav_s)
            if r.empty:
                return np.nan
            return float((r > 0).sum() / len(r))

        metric_rows = []
        nav_series_dict = {}
        for p in cmp_paths:
            e = load_experiment(str(p))
            n = e.get("nav")
            if n is None:
                continue
            n_s = ensure_series(n).dropna()
            nav_series_dict[p.name] = n_s
            basic = ann_stats(n_s)  # contains total_return, ann_return, ann_vol, max_drawdown
            row = {
                "experiment": p.name,
                "total_return": basic["total_return"],
                "cagr": basic["ann_return"],
                "ann_vol": basic["ann_vol"],
                "max_drawdown": basic["max_drawdown"],
                "sharpe": sharpe_ratio(n_s),
                "sortino": sortino_ratio(n_s),
                "calmar": calmar_ratio(n_s),
                "win_rate": win_rate(n_s)
            }
            metric_rows.append(row)

        if not metric_rows:
            st.warning("No NAV data found for the selected experiments.")
        else:
            metrics_df = pd.DataFrame(metric_rows).set_index("experiment")
            # Formatting for display
            display_df = metrics_df.copy()
            fmt_pct = ["total_return", "cagr", "ann_vol", "max_drawdown", "sharpe", "sortino", "calmar", "win_rate"]
            # Show sortable table
            st.subheader("Metrics table (click column to sort in UI)")
            st.dataframe(display_df.style.format({
                "total_return": "{:.2%}",
                "cagr": "{:.2%}",
                "ann_vol": "{:.2%}",
                "max_drawdown": "{:.2%}",
                "sharpe": "{:.2f}",
                "sortino": "{:.2f}",
                "calmar": "{:.2f}",
                "win_rate": "{:.2%}"
            }))

            # Allow CSV download
            csv_buf = metrics_df.to_csv(index=True)
            st.download_button("Download metrics CSV", data=csv_buf, file_name="compare_metrics.csv")

            # Choose metric(s) to plot
            all_metrics = list(metrics_df.columns)
            sel_metrics = st.multiselect("Select metric(s) to plot", options=all_metrics, default=["cagr", "sharpe", "max_drawdown"])

            if sel_metrics:
                # Bar chart for each selected metric
                for m in sel_metrics:
                    fig = px.bar(metrics_df.reset_index(), x="experiment", y=m, title=f"Comparison: {m}", text=metrics_df.reset_index()[m])
                    fig.update_layout(xaxis_tickangle=-45, height=450)
                    # For percentage-like metrics label formatting on hover
                    if m in ["total_return", "cagr", "ann_vol", "max_drawdown", "win_rate"]:
                        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
                    else:
                        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

                # Optional radar chart when multiple metrics selected and experiments <= 6 (for readability)
                if len(sel_metrics) > 1 and len(metrics_df) <= 6:
                    radar_df = metrics_df[sel_metrics].copy()
                    # normalize metrics to 0..1 per metric for radar (min-max)
                    radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-12)
                    fig = go.Figure()
                    for idx, row in radar_norm.iterrows():
                        fig.add_trace(go.Scatterpolar(r=row.values.tolist(), theta=radar_norm.columns.tolist(), fill='toself', name=idx))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Radar (normalized) comparison")
                    st.plotly_chart(fig, use_container_width=True)

            # Keep NAV time-series comparison (normalized)
            st.markdown("---")
            st.subheader("NAV time-series (normalized)")
            fig_ts = go.Figure()
            for name, s in nav_series_dict.items():
                s_aligned = s / float(s.iloc[0])
                fig_ts.add_trace(go.Scatter(x=s_aligned.index, y=s_aligned.values, mode='lines', name=name))
            fig_ts.update_layout(height=500, xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_ts, use_container_width=True)
