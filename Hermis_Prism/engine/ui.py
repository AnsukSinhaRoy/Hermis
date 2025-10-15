# portfolio_viz/ui.py
from pathlib import Path
from typing import List
from .viz import plot_calendar_heatmap
import streamlit as st
import pandas as pd
import plotly.express as px

from .loaders import (
    load_experiment_list, load_experiment, safe_read_prices, load_precomputed_perf, load_all_experiment_parameters_and_metrics
)
from .utils import ensure_series, create_calendar_heatmap, ann_stats
from .viz import (
    plot_nav, plot_all_prices, plot_heatmap, plot_drawdown, plot_rolling_sharpe, plot_compare_navs
)

CSS = """
<style>
/* Light styling for tabs and sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--border-color); }
</style>
"""


def run_app():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("💎 Hermis Prism")
    st.markdown("An understandable spectrum of insights")

    with st.sidebar:
        st.header("⚙️ Controls")
        root = st.text_input("Experiments Root Folder", value='experiments')
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

    # Load data
    with st.spinner(f"Loading experiment `{selected_path.name}`..."):
        exp = load_experiment(str(selected_path))
        nav = ensure_series(exp.get('nav'))
        weights = exp.get('weights')
        trades = exp.get('trades')
        meta = exp.get('meta', {})
        prices = safe_read_prices(selected_path)
        preperf = load_precomputed_perf(selected_path)

    tabs = st.tabs([
        "🌟 Overview", "📊 Allocation", "📈 Prices", "🔄 Trades",
        "🏆 Performance", "📅 Deep Dive", "🔬 Parameter Analysis",
        "🔍 Asset Inspector", "🆚 Compare"
    ])

    # Overview
    with tabs[0]:
        st.header(f"Experiment Overview: `{selected_path.name}`")
        col1, col2 = st.columns([2, 1])
        with col1:
            if nav is not None:
                metrics = preperf.get('metrics', ann_stats(nav))
                st.subheader("Key Performance Indicators")
                m_cols = st.columns(5)
                m_cols[0].metric("Total Return", f"{metrics.get('total_return',0)*100:.2f}%")
                m_cols[1].metric("Annualized Return", f"{metrics.get('ann_return',0)*100:.2f}%")
                m_cols[2].metric("Annualized Volatility", f"{metrics.get('ann_vol',0)*100:.2f}%")
                m_cols[3].metric("Max Drawdown", f"{metrics.get('max_drawdown',0)*100:.2f}%", delta_color='inverse')
                m_cols[4].metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")

                st.markdown("---")
                st.plotly_chart(plot_nav(nav), use_container_width=True)
            else:
                st.warning("NAV data not found for this experiment.")
        with col2:
            st.subheader("Details & Actions")
            st.info(f"**Folder:** `{selected_path}`")
            if prices is None:
                st.warning("Prices snapshot not found. Some tabs may be limited.")
            with st.expander("Show Experiment Metadata", expanded=False):
                st.json(meta)
            import json  # make sure this is imported at the top

            st.download_button(
                "Download Metadata (JSON)",
                data=json.dumps(meta, indent=2),
                file_name="metadata.json",
                mime="application/json"
            )

    # Allocation
    with tabs[1]:
        st.header("Portfolio Allocation")
        if weights is None:
            st.warning("No weights file found for this experiment.")
        else:
            st.subheader("Allocation Heatmap")
            top_n = st.slider("Show Top N Assets (by average weight)", 5, min(200, weights.shape[1]), 20)
            avg_weights = weights.mean().sort_values(ascending=False)
            top_assets = avg_weights.head(top_n).index.tolist()
            st.plotly_chart(plot_heatmap(weights, top_assets), use_container_width=True)

            st.subheader("Holdings at Rebalance Dates")
            with st.expander("View detailed holdings per rebalance date", expanded=False):
                selected_each = weights.apply(lambda row: row[row > 1e-6].index.tolist(), axis=1)
                show_n = st.number_input("Show Last N Rebalances", 1, len(selected_each), min(10, len(selected_each)))
                for d, holdings in selected_each.dropna().tail(show_n).items():
                    assets_str = ', '.join(holdings[:10]) + ('...' if len(holdings) > 10 else '')
                    st.markdown(f"**{d.date()}** ({len(holdings)} assets): `{assets_str}`")

    # Prices
    with tabs[2]:
        st.header("Asset Prices & NAV Overlay")
        if prices is None:
            st.info("Saved price snapshot not found. You can upload a CSV to proceed.")
        else:
            st.write(f"Universe size: {prices.shape[1]} assets over {prices.shape[0]} dates.")
            normalize = st.checkbox("Normalize to 1 at start", value=normalize_default, key='prices_normalize')
            max_assets_to_plot = min(sample_max, prices.shape[1])
            tickers = sorted(prices.columns)[:max_assets_to_plot]
            plot_prices = prices[tickers].copy()
            st.plotly_chart(plot_all_prices(plot_prices, nav, normalize), use_container_width=True)

    # Trades
    with tabs[3]:
        st.header("Trades and Turnover")
        if trades is None:
            st.info("No trades data found for this experiment.")
        else:
            trades_display = trades.reset_index() if 'date' not in trades.columns else trades.copy()
            st.subheader("Recent Trades")
            st.dataframe(trades_display.sort_values(by='date', ascending=False).head(500), use_container_width=True)
            if 'turnover' in trades.columns:
                t = trades.set_index('date')['turnover'].sort_index()
                st.plotly_chart(px.bar(t.reset_index(), x='date', y='turnover', title='Turnover per Rebalance'), use_container_width=True)

    # Performance
    with tabs[4]:
        st.header("Deep-Dive Performance Analytics")
        if nav is None:
            st.info("No NAV data available to compute performance metrics.")
        else:
            cum_returns = preperf.get('cum_returns', nav / nav.iloc[0])
            st.plotly_chart(px.line(cum_returns, title='Cumulative Returns (Normalized to 1)'), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                drawdown = preperf.get('drawdown', (nav - nav.cummax()) / nav.cummax())
                st.plotly_chart(plot_drawdown(drawdown), use_container_width=True)
            with col2:
                if 'rolling_sharpe' in preperf:
                    rs = preperf['rolling_sharpe']
                    title = 'Rolling Sharpe (Precomputed)'
                else:
                    win = st.slider('Sharpe Rolling Window (days)', 21, 252, 63)
                    dr = nav.pct_change()
                    rs = dr.rolling(win).mean() / dr.rolling(win).std() * (252 ** 0.5)
                    title = f'{win}-Day Rolling Sharpe Ratio'
                st.plotly_chart(plot_rolling_sharpe(rs, title), use_container_width=True)

    # Deep Dive
    with tabs[5]:
        st.header('Deep Dive Analytics')
        if nav is None:
            st.info('No NAV data available for deep dive analysis.')
        else:
            st.subheader('📅 Calendar Returns Heatmap')
            heatmap_df = create_calendar_heatmap(nav)
            st.plotly_chart(plot_calendar_heatmap(heatmap_df), use_container_width=True)

    # Parameter Analysis
    with tabs[6]:
        st.header('🔬 Hyperparameter Sensitivity Analysis')
        param_df = load_all_experiment_parameters_and_metrics(rootp)

        if param_df.empty:
            st.warning('No experiment data found for parameter analysis.')
        else:
            # --- IMPROVED CODE STARTS HERE ---

            # 1. Prepare the list of options for the X-axis
            x_options = sorted(param_df.columns.drop('experiment_name'))

            # 2. Find the index of your desired default value. Default to 0 if not found.
            default_x_index = 0
            if 'optimizer.k_cardinality' in x_options:
                default_x_index = x_options.index('optimizer.k_cardinality')
            
            # 3. Create the selectbox with the default index
            x_axis_val = st.selectbox(
                'X-Axis (Parameter)', 
                options=x_options, 
                index=default_x_index
            )
            
            # 4. Repeat the process for the Y-axis
            y_options = [c for c in param_df.columns if c not in ['experiment_name', x_axis_val]]
            
            default_y_index = 0
            if 'ann_return' in y_options:
                default_y_index = y_options.index('ann_return')

            y_axis_val = st.selectbox(
                'Y-Axis (Metric)', 
                options=y_options, 
                index=default_y_index
            )

            # --- END OF IMPROVED CODE ---

            fig = px.scatter(param_df, x=x_axis_val, y=y_axis_val, hover_name='experiment_name')
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander('Show Raw Parameter & Metric Data'):
                st.dataframe(param_df, use_container_width=True)

    # Asset Inspector
    with tabs[7]:
        st.header('Asset Inspector')
        if prices is None:
            st.info('No prices snapshot available to inspect assets.')
        else:
            all_tickers = sorted(prices.columns.tolist())
            search_query = st.text_input('Filter tickers by name (e.g., "AAPL, MSFT")', '')
            if search_query:
                tokens = [t.strip().lower() for t in search_query.split(',')]
                filtered_tickers = [t for t in all_tickers if any(tok in t.lower() for tok in tokens)]
            else:
                filtered_tickers = all_tickers
            selected_tickers = st.multiselect('Select assets to plot', options=filtered_tickers, default=filtered_tickers[:5])
            if selected_tickers:
                plot_prices = prices[selected_tickers]
                normalize = st.checkbox('Normalize selected prices', value=True, key='inspector_normalize')
                if normalize:
                    plot_prices = plot_prices / plot_prices.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else None)
                st.plotly_chart(px.line(plot_prices, title='Selected Asset Prices'), use_container_width=True)
                csv = plot_prices.to_csv().encode('utf-8')
                st.download_button('Download Selected Prices (CSV)', data=csv, file_name='selected_prices.csv')

    # Compare
    with tabs[8]:
        st.header('Compare Experiments')
        if not cmp_paths:
            st.info('To compare, enable Comparison Mode in the sidebar and select multiple experiments.')
        else:
            @st.cache_data
            def get_comparison_data(paths: List[Path]):
                rows = []
                for p in paths:
                    exp_data = load_experiment(str(p))
                    nav_s = ensure_series(exp_data.get('nav'))
                    if nav_s is None or nav_s.empty:
                        continue
                    perf_data = load_precomputed_perf(p)
                    metrics = perf_data.get('metrics', ann_stats(nav_s))
                    rows.append({
                        'Experiment': p.name,
                        'Total Return': metrics.get('total_return'),
                        'CAGR': metrics.get('ann_return'),
                        'Volatility': metrics.get('ann_vol'),
                        'Max Drawdown': metrics.get('max_drawdown'),
                    })
                return pd.DataFrame(rows).set_index('Experiment')

            metrics_df = get_comparison_data(cmp_paths)
            if metrics_df.empty:
                st.warning('No valid NAV data found for the selected experiments to compare.')
            else:
                st.dataframe(metrics_df.style.format({col:'{:.2%}' for col in ['Total Return','CAGR','Volatility','Max Drawdown']}), use_container_width=True)
                metric_to_plot = st.selectbox('Select metric to visualize', options=metrics_df.columns)
                st.plotly_chart(px.bar(metrics_df[metric_to_plot].sort_values(ascending=False), orientation='h', title=f'Comparison: {metric_to_plot}'), use_container_width=True)

                # Normalized NAV comparison
                nav_series = []
                names = []
                for path in cmp_paths:
                    exp_data = load_experiment(str(path))
                    nav_s = ensure_series(exp_data.get('nav'))
                    if nav_s is not None and not nav_s.empty:
                        nav_series.append(nav_s)
                        names.append(path.name)
                st.plotly_chart(plot_compare_navs(nav_series, names), use_container_width=True)