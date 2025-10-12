"""
A Streamlit web application for visualizing and analyzing portfolio backtest results.

This application allows users to:
- Select and load different experiment outputs.
- View performance overview, including NAV charts and key metrics.
- Analyze portfolio allocation over time with heatmaps.
- Inspect underlying asset prices.
- Review trade logs and turnover.
- Dive deep into performance analytics with drawdown and rolling Sharpe plots.
- Compare multiple experiments side-by-side with detailed metrics tables and charts.
- Analyze hyperparameter sensitivity across all experiments.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Portfolio Viz",
    page_icon="🧪",
    layout="wide"
)

# --- CORRECTED: CSS for Tabs and NEW CSS for Sidebar ---
st.markdown("""
<style>
    /* --- Tab Styling (Theme-Aware) --- */
    /* Tab container */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; /* Spacing between tabs */
        border-bottom: 1px solid var(--border-color);
    }

    /* Inactive tab buttons */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent; /* Start with no background */
        border: none;
        border-radius: 8px 8px 0 0;
        padding: 12px 16px;
        color: var(--text-color);
        opacity: 0.7; /* Make inactive tabs slightly faded */
        transition: all 0.2s ease-in-out; /* Smooth transitions */
        border-bottom: 2px solid transparent; /* Placeholder for the active indicator */
    }
    
    /* Tab button on hover */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--secondary-background-color);
        opacity: 1;
        transform: translateY(-2px); /* Subtle lift effect */
    }

    /* Active tab button */
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-background-color);
        font-weight: 600; /* Make active tab text bold */
        opacity: 1;
        color: var(--primary-color); /* Use accent color for active tab text */
        border-bottom: 2px solid var(--primary-color); /* Use accent color for the indicator line */
    }

    /* --- NEW: Sidebar Styling --- */
    /* The main sidebar container */
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border-color);
    }

    /* Styling for the headers within the sidebar */
    [data-testid="stSidebar"] h2 {
        font-size: 1.1rem;
        text-transform: uppercase;
        color: var(--text-color);
        opacity: 0.8;
        letter-spacing: 1.5px;
        padding-bottom: 8px;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading & Utilities ---

@st.cache_data
def load_experiment(path: str) -> Dict[str, Any]:
    """Loads experiment artifacts from a directory."""
    try:
        from portfolio_sim.experiment import load_experiment as load_sim_experiment
        return load_sim_experiment(path)
    except (ImportError, ModuleNotFoundError):
        out = Path(path) / "outputs"
        nav = pd.read_parquet(out / "nav.parquet") if (out / "nav.parquet").exists() else None
        weights = pd.read_parquet(out / "weights.parquet") if (out / "weights.parquet").exists() else None
        trades = pd.read_parquet(out / "trades.parquet") if (out / "trades.parquet").exists() else None
        meta = json.load(open(out / "metadata.json")) if (out / "metadata.json").exists() else {}
        return {"nav": nav, "weights": weights, "trades": trades, "meta": meta}

def ensure_series(data: Any) -> Optional[pd.Series]:
    """Ensures the input data is a Pandas Series."""
    if data is None:
        return None
    if isinstance(data, pd.Series):
        return data
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return pd.Series(data)

def load_experiment_list(experiments_root: Path) -> List[Path]:
    """Scans a root directory and returns a sorted list of valid experiment paths."""
    if not experiments_root.exists():
        return []
    dirs = [
        p for p in experiments_root.iterdir()
        if p.is_dir() and (p / "outputs").exists()
    ]
    return sorted(dirs, reverse=True)

@st.cache_data
def safe_read_prices(exp_folder: Path) -> Optional[pd.DataFrame]:
    """Safely reads the prices.parquet file."""
    p = exp_folder / "data" / "prices.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as e:
            st.error(f"Failed to read prices file: {e}")
            return None
    return None

@st.cache_data
def load_precomputed_perf(exp_folder: Path) -> Dict[str, Any]:
    """Loads pre-computed performance artifacts."""
    perf = {}
    perf_dir = Path(exp_folder) / 'outputs' / 'performance'
    if not perf_dir.exists():
        return perf
    try:
        if (perf_dir / 'metrics.json').exists():
            with open(perf_dir / 'metrics.json') as f:
                perf['metrics'] = json.load(f)
        for fname in ['rolling_sharpe', 'drawdown', 'cum_returns']:
            fpath = perf_dir / f'{fname}.parquet'
            if fpath.exists():
                perf[fname] = pd.read_parquet(fpath).iloc[:, 0]
    except Exception as e:
        st.warning(f"Could not load some precomputed performance files: {e}")
    return perf

# --- Helper for Parameter Analysis ---
@st.cache_data
def load_all_experiment_parameters_and_metrics(root_path: Path) -> pd.DataFrame:
    """Scans all experiments, loads their parameters and metrics, and returns a unified DataFrame."""
    all_experiments_data = []
    exp_paths = load_experiment_list(root_path)

    for path in exp_paths:
        params_path = path / "params.yaml"
        metrics_path = path / "outputs" / "performance" / "metrics.json"

        if params_path.exists() and metrics_path.exists():
            try:
                with open(params_path, 'r') as f:
                    params = yaml.safe_load(f)
                params_flat = pd.json_normalize(params, sep='.')
                params_dict = params_flat.to_dict(orient='records')[0]
                
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                combined_data = {**params_dict, **metrics}
                combined_data['experiment_name'] = path.name
                all_experiments_data.append(combined_data)
            except Exception as e:
                st.warning(f"Could not process experiment {path.name}: {e}")
    
    if not all_experiments_data:
        return pd.DataFrame()
    return pd.DataFrame(all_experiments_data)

# --- Performance Calculation Helpers ---

def ann_stats(nav_s: pd.Series) -> Dict[str, float]:
    """Calculates annualized statistics for a given NAV series."""
    if nav_s.empty:
        return {"total_return": 0, "ann_return": 0, "ann_vol": 0, "max_drawdown": 0}
    days = (nav_s.index[-1] - nav_s.index[0]).days or 1
    total_ret = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    ann_ret = (1 + total_ret) ** (365.0 / days) - 1
    dr = nav_s.pct_change().dropna()
    ann_vol = dr.std() * np.sqrt(252) if not dr.empty else 0.0
    roll_max = nav_s.cummax()
    drawdown = (nav_s - roll_max) / roll_max
    max_dd = drawdown.min()
    return {
        "total_return": total_ret, "ann_return": ann_ret,
        "ann_vol": ann_vol, "max_drawdown": max_dd,
    }

def create_calendar_heatmap(nav_s: pd.Series) -> pd.DataFrame:
    """Creates a pivot table of monthly returns for a calendar heatmap."""
    returns = nav_s.pct_change().dropna()
    returns.name = "returns"
    res = returns.reset_index()
    res['year'] = res['index'].dt.year
    res['month'] = res['index'].dt.month
    monthly_returns = res.groupby(['year', 'month'])['returns'].apply(lambda x: (1 + x).prod() - 1)
    heatmap = monthly_returns.unstack(level='month')
    yearly_returns = res.groupby('year')['returns'].apply(lambda x: (1 + x).prod() - 1)
    heatmap['Year'] = yearly_returns
    month_names = {i: pd.to_datetime(i, format='%m').strftime('%b') for i in range(1, 13)}
    heatmap = heatmap.rename(columns=month_names)
    return heatmap

# --- Main App UI ---

st.title("🧪 Portfolio Analysis Workbench")
st.markdown("Analyze and compare backtesting experiments with interactive charts and metrics.")

# --- Sidebar ---

with st.sidebar:
    st.header("⚙️ Controls")
    root = st.text_input("Experiments Root Folder", value="experiments")
    rootp = Path(root)
    exp_dirs = load_experiment_list(rootp)

    if not exp_dirs:
        st.warning(f"No valid experiments found in `{root}`. Please run a simulation first.")
        st.stop()

    exp_labels = [p.name for p in exp_dirs]
    sel_label = st.selectbox("Choose an Experiment", exp_labels, index=0)
    selected_path = exp_dirs[exp_labels.index(sel_label)]

    st.header("📊 Comparison Mode")
    if st.checkbox("Enable Comparison"):
        default_choices = [sel_label] if sel_label in exp_labels else []
        cmp_choices = st.multiselect("Select Experiments to Compare", exp_labels, default=default_choices)
        cmp_paths = [exp_dirs[exp_labels.index(x)] for x in cmp_choices]
    else:
        cmp_paths = []

    st.header("🎨 Plot Options")
    sample_max = st.number_input("Max Assets to Plot", min_value=10, max_value=2000, value=100, step=10)
    normalize_default = st.checkbox("Normalize Prices & NAV to 1", value=True)

# --- Load Selected Experiment Data ---
with st.spinner(f"Loading experiment `{selected_path.name}`..."):
    exp = load_experiment(str(selected_path))
    nav = ensure_series(exp.get("nav"))
    weights = exp.get("weights")
    trades = exp.get("trades")
    meta = exp.get("meta", {})
    prices = safe_read_prices(selected_path)
    preperf = load_precomputed_perf(selected_path)

# --- Tabs for Different Views ---
tabs = st.tabs([
    "🌟 Overview", "📊 Allocation", "📈 Prices", "🔄 Trades",
    "🏆 Performance", "📅 Deep Dive", "🔬 Parameter Analysis",
    "🔍 Asset Inspector", "🆚 Compare"
])

# Overview Tab
with tabs[0]:
    st.header(f"Experiment Overview: `{selected_path.name}`")
    col1, col2 = st.columns([2, 1])
    with col1:
        if nav is not None:
            metrics = preperf.get('metrics', ann_stats(nav))
            st.subheader("Key Performance Indicators")
            m_cols = st.columns(4)
            m_cols[0].metric("Total Return", f"{metrics.get('total_return', 0)*100:.2f}%")
            m_cols[1].metric("Annualized Return", f"{metrics.get('ann_return', 0)*100:.2f}%")
            m_cols[2].metric("Annualized Volatility", f"{metrics.get('ann_vol', 0)*100:.2f}%")
            m_cols[3].metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%", delta_color="inverse")
            st.markdown("---")
            fig = px.area(nav, title="Portfolio Net Asset Value (NAV)", labels={"index": "Date", "value": "NAV"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("NAV data not found for this experiment.")
    with col2:
        st.subheader("Details & Actions")
        st.info(f"**Folder:** `{selected_path}`")
        if prices is None:
            st.warning("Prices snapshot not found. Some tabs may be limited.")
        with st.expander("Show Experiment Metadata", expanded=False):
            st.json(meta)
        st.download_button("Download Metadata (JSON)", data=json.dumps(meta, indent=2), file_name="metadata.json")
        perf_dir = selected_path / 'outputs' / 'performance'
        if (perf_dir / 'metrics.json').exists():
            st.download_button("Download Precomputed Metrics", data=(perf_dir / 'metrics.json').read_bytes(), file_name='metrics.json')

# Allocation Tab
with tabs[1]:
    with st.container():
        st.header("Portfolio Allocation")
        if weights is None:
            st.warning("No weights file found for this experiment.")
        else:
            st.subheader("Allocation Heatmap")
            top_n = st.slider("Show Top N Assets (by average weight)", 5, min(200, weights.shape[1]), 20)
            avg_weights = weights.mean().sort_values(ascending=False)
            top_assets = avg_weights.head(top_n).index.tolist()
            sub = weights[top_assets]
            fig = go.Figure(data=go.Heatmap(
                z=sub.T.values, x=sub.index, y=sub.columns, colorscale="Viridis",
                hovertemplate='Date: %{x}<br>Asset: %{y}<br>Weight: %{z:.2%}<extra></extra>'
            ))
            fig.update_layout(height=600, title=f"Top {top_n} Asset Weights Over Time", yaxis_nticks=top_n)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Holdings at Rebalance Dates")
            with st.expander("View detailed holdings per rebalance date", expanded=False):
                selected_each = weights.apply(lambda row: row[row > 1e-6].index.tolist(), axis=1)
                show_n = st.number_input("Show Last N Rebalances", 1, len(selected_each), min(10, len(selected_each)))
                for d, holdings in selected_each.dropna().tail(show_n).items():
                    assets_str = ', '.join(holdings[:10]) + ('...' if len(holdings) > 10 else '')
                    st.markdown(f"**{d.date()}** ({len(holdings)} assets): `{assets_str}`")

# Prices Tab
with tabs[2]:
    st.header("Asset Prices & NAV Overlay")
    if prices is None:
        st.info("Saved price snapshot not found. You can upload a CSV to proceed.")
    else:
        st.write(f"Universe size: {prices.shape[1]} assets over {prices.shape[0]} dates.")
        normalize = st.checkbox("Normalize to 1 at start", value=normalize_default, key="prices_normalize")
        max_assets_to_plot = min(sample_max, prices.shape[1])
        tickers = sorted(prices.columns)[:max_assets_to_plot]
        if sample_max > prices.shape[1]:
            st.info(f"Plotting all {prices.shape[1]} available assets (sidebar setting is {sample_max}).")
        plot_prices = prices[tickers].copy()
        if normalize:
            first_vals = plot_prices.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else np.nan)
            plot_prices = plot_prices.divide(first_vals, axis=1)
        fig = go.Figure()
        for col in plot_prices.columns:
            fig.add_trace(go.Scatter(x=plot_prices.index, y=plot_prices[col], mode='lines', line=dict(width=0.5, color='grey'), opacity=0.5, name=col, showlegend=False))
        median_series = plot_prices.median(axis=1)
        fig.add_trace(go.Scatter(x=median_series.index, y=median_series, mode='lines', line=dict(color='blue', width=2, dash='dash'), name='Median Asset'))
        if nav is not None:
            nav_aligned = nav.reindex(plot_prices.index, method='ffill')
            if normalize:
                nav_aligned /= nav_aligned.dropna().iloc[0]
            fig.add_trace(go.Scatter(x=nav_aligned.index, y=nav_aligned, mode='lines', line=dict(color='red', width=3), name='Portfolio NAV'))
        fig.update_layout(title="Asset Price Paths and Portfolio NAV", height=600, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

# Trades Tab
with tabs[3]:
    st.header("Trades and Turnover")
    if trades is None:
        st.info("No trades data found for this experiment.")
    else:
        trades_display = trades.reset_index() if 'date' not in trades.columns else trades.copy()
        st.subheader("Recent Trades")
        st.dataframe(trades_display.sort_values(by='date', ascending=False).head(500), use_container_width=True)
        if 'turnover' in trades.columns:
            st.subheader("Portfolio Turnover")
            t = trades.set_index('date')['turnover'].sort_index()
            fig = px.bar(t.reset_index(), x='date', y='turnover', title='Turnover per Rebalance')
            fig.update_layout(yaxis_tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)

# Performance Tab
with tabs[4]:
    st.header("Deep-Dive Performance Analytics")
    if nav is None:
        st.info("No NAV data available to compute performance metrics.")
    else:
        cum_returns = preperf.get('cum_returns', nav / nav.iloc[0])
        fig_cum = px.line(cum_returns, title="Cumulative Returns (Normalized to 1)", labels={"index": "Date", "value": "Cumulative Return"})
        st.plotly_chart(fig_cum, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            drawdown = preperf.get('drawdown', (nav - nav.cummax()) / nav.cummax())
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name='Drawdown', line=dict(color='indianred')))
            fig_dd.update_layout(title="Portfolio Drawdown", yaxis_tickformat=".2%", height=400)
            st.plotly_chart(fig_dd, use_container_width=True)
        with col2:
            if 'rolling_sharpe' in preperf:
                rs = preperf['rolling_sharpe']
                title = "Rolling Sharpe (Precomputed)"
            else:
                win = st.slider("Sharpe Rolling Window (days)", 21, 252, 63)
                dr = nav.pct_change()
                rs = dr.rolling(win).mean() / dr.rolling(win).std() * np.sqrt(252)
                title = f"{win}-Day Rolling Sharpe Ratio"
            fig_rs = px.line(rs, title=title)
            st.plotly_chart(fig_rs, use_container_width=True)

# Deep Dive Tab
with tabs[5]:
    with st.container():
        st.header("Deep Dive Analytics")
        if nav is None:
            st.info("No NAV data available for deep dive analysis.")
        else:
            st.subheader("📅 Calendar Returns Heatmap")
            heatmap_df = create_calendar_heatmap(nav)
            styled_heatmap = heatmap_df.style.background_gradient(cmap='RdYlGn', axis=None, low=0.4, high=0.4).format("{:.2%}")
            st.dataframe(styled_heatmap, use_container_width=True)
            st.caption("Heatmap shows compounded monthly returns.")

# Parameter Analysis Tab
with tabs[6]:
    st.header("🔬 Hyperparameter Sensitivity Analysis")
    st.markdown("Visualize how changing algorithm parameters affects performance metrics across all experiments.")
    
    param_df = load_all_experiment_parameters_and_metrics(rootp)

    if param_df.empty:
        st.warning("No experiment data found for parameter analysis. Ensure experiments have `params.yaml` and `outputs/performance/metrics.json` files.")
    else:
        metric_cols = [
            'total_return', 'ann_return', 'ann_vol', 'max_drawdown', 'sharpe',
            'sortino', 'calmar', 'win_rate', 'avg_turnover'
        ]
        available_metrics = sorted([m for m in metric_cols if m in param_df.columns])
        parameter_cols = sorted([c for c in param_df.columns if c not in available_metrics and c != 'experiment_name'])

        if not parameter_cols:
            st.error("No parameter columns found. Check the structure of your `params.yaml` files.")
            st.stop()

        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis_val = st.selectbox("X-Axis (Parameter)", options=parameter_cols, index=0)
        with col2:
            y_axis_val = st.selectbox("Y-Axis (Metric)", options=available_metrics, index=len(available_metrics)-1 if 'sharpe' in available_metrics else 0)
        with col3:
            color_val = st.selectbox("Color By (Parameter)", options=[None] + parameter_cols, index=0)
        
        fig = px.scatter(
            param_df,
            x=x_axis_val,
            y=y_axis_val,
            color=color_val,
            title=f"{y_axis_val} vs. {x_axis_val}",
            hover_name='experiment_name',
            hover_data=parameter_cols + available_metrics
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show Raw Parameter & Metric Data"):
            st.dataframe(param_df[['experiment_name'] + parameter_cols + available_metrics], use_container_width=True)

# Asset Inspector Tab
with tabs[7]:
    st.header("Asset Inspector")
    if prices is None:
        st.info("No prices snapshot available to inspect assets.")
    else:
        all_tickers = sorted(prices.columns.tolist())
        search_query = st.text_input("Filter tickers by name (e.g., 'AAPL, MSFT')", "")
        if search_query:
            tokens = [t.strip().lower() for t in search_query.split(",")]
            filtered_tickers = [t for t in all_tickers if any(tok in t.lower() for tok in tokens)]
        else:
            filtered_tickers = all_tickers
        selected_tickers = st.multiselect("Select assets to plot", options=filtered_tickers, default=filtered_tickers[:5])
        if selected_tickers:
            plot_prices = prices[selected_tickers]
            normalize = st.checkbox("Normalize selected prices", value=True, key="inspector_normalize")
            if normalize:
                plot_prices = plot_prices / plot_prices.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else np.nan)
            fig = px.line(plot_prices, title="Selected Asset Prices")
            fig.update_layout(height=600, xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)
            csv = plot_prices.to_csv().encode('utf-8')
            st.download_button("Download Selected Prices (CSV)", data=csv, file_name="selected_prices.csv")

# Compare Tab
with tabs[8]:
    st.header("Compare Experiments")
    if not cmp_paths:
        st.info("To compare, enable 'Comparison Mode' in the sidebar and select multiple experiments.")
    else:
        @st.cache_data
        def get_comparison_data(paths: List[Path]) -> pd.DataFrame:
            metric_rows = []
            for p in paths:
                exp_data = load_experiment(str(p))
                nav_s = ensure_series(exp_data.get("nav"))
                if nav_s is None or nav_s.empty:
                    continue
                perf_data = load_precomputed_perf(p)
                metrics = perf_data.get('metrics', ann_stats(nav_s))
                row = {
                    "Experiment": p.name, "Total Return": metrics.get('total_return'),
                    "CAGR": metrics.get('ann_return'), "Volatility": metrics.get('ann_vol'),
                    "Max Drawdown": metrics.get('max_drawdown'),
                }
                metric_rows.append(row)
            return pd.DataFrame(metric_rows).set_index("Experiment")

        metrics_df = get_comparison_data(cmp_paths)

        if metrics_df.empty:
            st.warning("No valid NAV data found for the selected experiments to compare.")
        else:
            st.subheader("Comparative Performance Metrics")
            format_mapping = {
                "Total Return": "{:.2%}", "CAGR": "{:.2%}",
                "Volatility": "{:.2%}", "Max Drawdown": "{:.2%}"
            }
            styled_df = metrics_df.style.format(format_mapping).highlight_max(
                subset=["Total Return", "CAGR"], color='lightgreen'
            ).highlight_min(
                subset=["Volatility", "Max Drawdown"], color='lightcoral'
            )
            st.dataframe(styled_df, use_container_width=True)
            st.subheader("Metric Comparison Charts")
            metric_to_plot = st.selectbox("Select metric to visualize", options=metrics_df.columns)
            if metric_to_plot:
                fig = px.bar(
                    metrics_df[metric_to_plot].sort_values(ascending=False),
                    orientation='h', title=f"Comparison: {metric_to_plot}", text_auto=True
                )
                fig.update_layout(yaxis_title="Experiment", xaxis_title=metric_to_plot, showlegend=False)
                if metric_to_plot in format_mapping:
                    fig.update_traces(texttemplate='%{x:.2%}')
                else:
                    fig.update_traces(texttemplate='%{x:.2f}')
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Normalized NAV Comparison")
            fig_ts = go.Figure()
            for path in cmp_paths:
                exp_data = load_experiment(str(path))
                nav_s = ensure_series(exp_data.get("nav"))
                if nav_s is not None and not nav_s.empty:
                    s_norm = nav_s / nav_s.dropna().iloc[0]
                    fig_ts.add_trace(go.Scatter(x=s_norm.index, y=s_norm, mode='lines', name=path.name))
            fig_ts.update_layout(height=500, title="Normalized NAV Over Time", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_ts, use_container_width=True)