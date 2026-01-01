
# Hermis_Prism/engine/ui.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .loaders import (
    load_experiment_list,
    load_nav_only,
    load_experiment,
    load_weights_only,
    load_trades_only,
    safe_read_prices,
    load_precomputed_perf,
    load_all_experiment_parameters_and_metrics,
)
from .utils import ensure_series, create_calendar_heatmap, ann_stats
from .analytics import ann_metrics, concentration_stats, pnl_contributions
from .viz import (
    plot_nav,
    plot_all_prices,
    plot_heatmap,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_compare_navs,
    plot_calendar_heatmap,
)

from .benchmarks import BENCHMARKS, try_get_benchmark_nav, get_custom_yahoo_nav


CSS = """
<style>
/* --- Hermis Prism: clean, Google-ish theme --- */
:root{
  --hp-blue:#1a73e8;
  --hp-bg:#ffffff;
  --hp-sub:#f8f9fa;
  --hp-border:#e0e0e0;
  --hp-text:#202124;
  --hp-muted:#5f6368;
}

html, body, [class*="css"]{
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Noto Sans", "Apple Color Emoji","Segoe UI Emoji";
  color: var(--hp-text);
}

/* Main page container */
.block-container{
  padding-top: 3.25rem;     /* keeps content below Streamlit top bar */
  padding-bottom: 2.5rem;
  max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: var(--hp-sub);
  border-right: 1px solid var(--hp-border);
}
section[data-testid="stSidebar"] .block-container{
  padding-top: 3.25rem;
}

/* Sidebar brand */
.hp-side-brand{
  display:flex;
  align-items:center;
  gap:10px;
  margin: 6px 0 2px 0;
}
.hp-side-dot{
  width:10px;
  height:10px;
  border-radius:50%;
  background: var(--hp-blue);
  flex: 0 0 auto;
}
.hp-side-title{
  font-size: 18px;
  font-weight: 760;
  letter-spacing: 0.1px;
  line-height: 1.1;
  color: var(--hp-text);
}
.hp-side-sub{
  font-size: 13px;
  color: var(--hp-muted);
  margin: 0 0 12px 20px;
}

/* Sidebar expanders as a clean nav list (only target Streamlit expanders) */
section[data-testid="stSidebar"] [data-testid="stExpander"]{
  margin: 0 !important;
}
/* Prevent any layout shift when Streamlit toggles focus/active states */
section[data-testid="stSidebar"] [data-testid="stExpander"] *{
  box-sizing: border-box !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details{
  background: transparent !important;
  border: none !important;
  border-radius: 12px !important;
  margin: 2px 0 !important;
  overflow: hidden !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary{
  list-style: none !important;
  display:flex !important;
  align-items:center !important;
  gap: 10px !important;
  /* keep left padding stable to prevent layout shift on expand */
  padding: 10px 10px 10px 7px !important;
  border-radius: 10px !important;
  font-size: 16px !important;
  font-weight: 650 !important;
  letter-spacing: .1px;
  color: var(--hp-text) !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-left: 3px solid transparent !important;
  margin: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary::-webkit-details-marker{
  display:none;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover{
  background: rgba(26,115,232,0.07) !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary{
  background: rgba(26,115,232,0.10) !important;
  border-left-color: var(--hp-blue) !important;
}

/* Give an expanded sidebar section a subtle boundary without layout shift */
section[data-testid="stSidebar"] [data-testid="stExpander"] details[open]{
  box-shadow: inset 0 0 0 1px var(--hp-border) !important;
  background: rgba(255,255,255,0.70) !important;
}

/* Sidebar inputs: add borders so empty placeholders don't look blank */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  border: 1px solid var(--hp-border) !important;
  border-radius: 10px !important;
  background: #fff !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"] > div{
  border: 1px solid var(--hp-border) !important;
  border-radius: 10px !important;
  background: #fff !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] textarea{
  border: 1px solid var(--hp-border) !important;
  border-radius: 10px !important;
}

/* Inside-sidebar section separators / headings */
.hp-side-divider{
  height: 1px;
  background: var(--hp-border);
  margin: 10px 0 10px 0;
}
.hp-side-section-title{
  font-size: 12px;
  font-weight: 750;
  color: var(--hp-muted);
  letter-spacing: 0.8px;
  text-transform: uppercase;
  margin: 6px 0 6px 0;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details > div{
  background: transparent !important;
  border: none !important;
  padding: 6px 10px 8px 10px !important;
  margin: 0 !important;
}

/* Header */
.hp-header{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  padding: 12px 14px;
  border: 1px solid var(--hp-border);
  border-radius: 16px;
  background: var(--hp-bg);
}
.hp-brand{
  display:flex;
  align-items:center;
  gap:10px;
}
.hp-logo{
  width: 28px;
  height: 28px;
  border-radius: 10px;
  border: 1px solid var(--hp-border);
  background: linear-gradient(135deg, rgba(26,115,232,0.18), rgba(26,115,232,0.04));
  position: relative;
}
.hp-logo::after{
  content:"";
  position:absolute;
  inset: 8px;
  border-radius: 8px;
  background: var(--hp-blue);
  opacity: 0.85;
}
.hp-header .hp-title{
  font-size: 20px;
  font-weight: 760;
  margin: 0;
  line-height: 1.1;
  letter-spacing: 0.1px;
}
.hp-accent{ color: var(--hp-blue); }
.hp-header .hp-sub{
  margin: 2px 0 0 0;
  color: var(--hp-muted);
  font-size: 13px;
}
.hp-chip{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--hp-border);
  background: var(--hp-sub);
  color: var(--hp-muted);
  font-size: 12px;
}

/* Tabs: make them look like a nav bar */
div[data-testid="stTabs"] [role="tablist"]{
  background: var(--hp-sub);
  border: 1px solid var(--hp-border);
  border-radius: 14px;
  padding: 6px;
  gap: 2px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.06);
  margin-top: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
div[data-testid="stTabs"] button[role="tab"]{
  padding: 8px 12px;
  border-radius: 12px;
  font-weight: 650;
  font-size: 13px;
  color: var(--hp-muted);
  border: 1px solid transparent;
  transition: background 120ms ease, border 120ms ease, color 120ms ease;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  background: var(--hp-bg);
  border: 1px solid var(--hp-border);
  color: var(--hp-blue);
  box-shadow: 0 1px 2px rgba(0,0,0,0.08);
}
div[data-testid="stTabs"] button[role="tab"]:hover{
  background: rgba(26,115,232,0.06);
  color: var(--hp-text);
}
div[data-testid="stTabs"] div[data-baseweb="tab-border"]{
  display:none;
}

/* KPI (st.metric) compact cards */
div[data-testid="stMetric"]{
  border: 1px solid var(--hp-border);
  border-radius: 14px;
  padding: 10px 12px;
  background: var(--hp-bg);
}
div[data-testid="stMetricLabel"] > div{
  font-size: 12px;
  color: var(--hp-muted);
  font-weight: 600;
  letter-spacing: 0.2px;
}
div[data-testid="stMetricValue"] > div{
  font-size: 22px;
  font-weight: 700;
  white-space: nowrap;
}
div[data-testid="stMetricDelta"] > div{
  font-size: 12px;
}

/* Plotly modebar */
.modebar-container{ opacity: 0.35; }
.modebar-container:hover{ opacity: 1.0; }

/* Hide Streamlit footer */
footer{ visibility:hidden; }

</style>
"""

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x*100:.2f}%"


def _fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def _subset_series(s: Optional[pd.Series], start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[pd.Series]:
    if s is None or getattr(s, "empty", True):
        return s
    if start is None or end is None:
        return s
    return s.loc[(s.index >= start) & (s.index <= end)]


def _subset_df(df: Optional[pd.DataFrame], start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
    if df is None or getattr(df, "empty", True):
        return df
    if start is None or end is None:
        return df
    return df.loc[(df.index >= start) & (df.index <= end)]



def run_app():
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            """
<div class="hp-side-brand">
  <div class="hp-side-dot"></div>
  <div class="hp-side-title">Hermis <span class="hp-accent">Prism</span></div>
</div>
<div class="hp-side-sub">Portfolio simulation insights</div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Experiment", expanded=True):
            root = st.text_input("Experiments root", value="experiments")
            rootp = Path(root)
            exp_dirs = load_experiment_list(rootp)

            if not exp_dirs:
                st.warning(f"No valid experiments found in `{root}`. Please run a simulation first.")
                st.stop()

            exp_labels = [p.name for p in exp_dirs]
            sel_label = st.selectbox("Experiment", exp_labels, index=0)
            selected_path = exp_dirs[exp_labels.index(sel_label)]

        with st.expander("Data", expanded=False):
            load_prices = st.checkbox("Prices (slow)", value=False)
            load_weights = st.checkbox("Weights", value=False)
            load_trades = st.checkbox("Trades & costs", value=False)


        with st.expander("Charts", expanded=False):
            sample_max = st.number_input("Max assets to plot", min_value=10, max_value=5000, value=100, step=10)
            normalize_default = st.checkbox("Normalize prices & NAV to 1", value=True)

        with st.expander("Metrics", expanded=False):
            rf_annual = st.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.30, value=0.0, step=0.005)
        with st.expander("Compare", expanded=False):
            enable_cmp = st.checkbox("Compare runs", value=False)
            if enable_cmp:
                default_choices = [sel_label] if sel_label in exp_labels else []
                cmp_choices = st.multiselect("Runs", exp_labels, default=default_choices)
                cmp_paths = [exp_dirs[exp_labels.index(x)] for x in cmp_choices]

                st.markdown("<div class='hp-side-divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='hp-side-section-title'>Benchmarks</div>", unsafe_allow_html=True)
                include_bench = st.checkbox("Include benchmark(s)", value=False)
                bench_labels = []
                custom_bench_ticker, custom_bench_label = "", ""
                if include_bench:
                    bench_labels = st.multiselect(
                        "Benchmarks",
                        list(BENCHMARKS.keys()),
                        default=["NIFTY 50 (Auto: NSE / Yahoo)"] if "NIFTY 50 (Auto: NSE / Yahoo)" in BENCHMARKS else ([] if "NIFTY 50 (Yahoo: ^NSEI)" not in BENCHMARKS else ["NIFTY 50 (Yahoo: ^NSEI)"]),
                    )
                    with st.expander("Custom Yahoo ticker", expanded=False):
                        custom_bench_ticker = st.text_input("Ticker (e.g. ^NSEI)", value="")
                        custom_bench_label = st.text_input("Label", value="")
            else:
                cmp_paths = []
                include_bench = False
                bench_labels = []
                custom_bench_ticker, custom_bench_label = "", ""

    # Header
    st.markdown(
        f'''
        <div class="hp-header">
          <div class="hp-brand">
            <div class="hp-logo"></div>
            <div>
              <div class="hp-title">Hermis <span class="hp-accent">Prism</span></div>
              <div class="hp-sub">Portfolio simulation insights • diagnostics • comparisons</div>
            </div>
          </div>
          <div class="hp-chip">Experiment: <b>{selected_path.name}</b></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Make Plotly charts consistent
    px.defaults.template = "plotly_white"

    # --- Fast load: NAV + params/meta (small) ---
    with st.spinner(f"Loading `{selected_path.name}`…"):
        exp_light = load_nav_only(str(selected_path))
        nav = ensure_series(exp_light.get("nav"))
        meta = exp_light.get("meta", {}) or {}
        params = exp_light.get("params", {}) or {}
        preperf = load_precomputed_perf(selected_path)

    if nav is None or nav.empty:
        st.warning("NAV data not found for this experiment.")
        st.stop()
    # Window (based on NAV)
    nav_min, nav_max = nav.index.min(), nav.index.max()
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)  # gap above timeline slider
    start_end = st.slider(
        "Window",
        min_value=nav_min.to_pydatetime(),
        max_value=nav_max.to_pydatetime(),
        value=(nav_min.to_pydatetime(), nav_max.to_pydatetime()),
        label_visibility="collapsed",
    )
    start = pd.Timestamp(start_end[0])
    end = pd.Timestamp(start_end[1])

    nav_w = _subset_series(nav, start, end)

    pre_metrics = preperf.get("metrics", {}) if isinstance(preperf.get("metrics", {}), dict) else {}
    window_metrics = ann_metrics(nav_w, rf_annual=rf_annual)

    tabs = st.tabs([
        "Dashboard",
        "Allocation",
        "Attribution",
        "Trades",
        "Performance",
        "Calendar",
        "Explorer",
        "Assets",
        "Compare",
        "Diagnostics",
    ])

    # -------------------
    # Dashboard
    # -------------------
    with tabs[0]:
        st.subheader(f"Experiment: `{selected_path.name}`")

        exp_cfg = params.get("experiment", {}) if isinstance(params.get("experiment", {}), dict) else {}
        opt_cfg = params.get("optimizer", {}) if isinstance(params.get("optimizer", {}), dict) else {}
        data_cfg = params.get("data", {}) if isinstance(params.get("data", {}), dict) else {}
        risk_cfg = params.get("risk_model", {}) if isinstance(params.get("risk_model", {}), dict) else {}

        left, right = st.columns([2, 1])
        with left:
            kpi1 = st.columns(3)
            kpi1[0].metric("Total", _fmt_pct(window_metrics.get("total_return")))
            kpi1[1].metric("CAGR", _fmt_pct(window_metrics.get("cagr")))
            kpi1[2].metric("Sharpe", _fmt_num(window_metrics.get("sharpe")))

            kpi2 = st.columns(3)
            kpi2[0].metric("Vol", _fmt_pct(window_metrics.get("ann_vol")))
            kpi2[1].metric("Max DD", _fmt_pct(window_metrics.get("max_drawdown")), delta_color="inverse")
            kpi2[2].metric("Calmar", _fmt_num(window_metrics.get("calmar")))

            st.plotly_chart(plot_nav(nav_w, title="NAV"), use_container_width=True, key="pl_01")

        with right:
            st.markdown("#### Run context")
            c1, c2 = st.columns(2)
            c1.write("**Rebalance**")
            c1.code(str(exp_cfg.get("rebalance", "monthly")))
            c2.write("**Lookback**")
            c2.code(str(risk_cfg.get("lookback", "None")))

            st.write("**Optimizer**")
            st.code(str(opt_cfg.get("type", "mv_reg")))

            st.write("**Universe**")
            st.code(str(meta.get("n_assets", data_cfg.get("n_assets", "—"))))

            with st.expander("Parameters (params.yaml)", expanded=False):
                st.json(params)

            with st.expander("Metadata (metadata.json)", expanded=False):
                st.json(meta)

            with st.expander("How the simulator computes NAV (important)", expanded=False):
                st.markdown(
                    """
- Each day, the simulator **drifts weights** using realized asset returns and updates NAV using yesterday's weights.
- On rebalance dates, it estimates **expected returns** as mean log-returns and **covariance** as log-return covariance over the rolling window.
- It calls the configured optimizer to get **target weights**, then applies **linear transaction costs**: `NAV *= (1 - tc * turnover)`.
- If the optimizer fails, it falls back to **equal weight** across assets with valid prices.

**Note:** rebalance dates are selected as the *first available trading day* of each period (month/week/…).
If you want end-of-month rebalances or to avoid any look-ahead, we should adjust the estimation window to stop at `t-1` and apply weights at `t`.
"""
                )

        st.markdown("#### Precomputed metrics (full run)")
        if pre_metrics:
            st.dataframe(pd.DataFrame([pre_metrics]), use_container_width=True)
        else:
            st.info("No precomputed metrics found in outputs/performance/metrics.json for this run.")

    # -------------------
    # Allocation
    # -------------------
    with tabs[1]:
        st.header("Portfolio Allocation")

        weights = None
        if load_weights:
            with st.spinner("Loading weights…"):
                weights = load_weights_only(str(selected_path))
        if weights is None or weights.empty:
            st.info("Weights not loaded (or not found). Enable **Load weights** in the sidebar.")
        else:
            weights.index = pd.to_datetime(weights.index, errors="coerce")
            weights = _subset_df(weights, start, end)

            st.subheader("Concentration over time")
            conc = concentration_stats(weights)
            if not conc.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=conc.index, y=conc["n_holdings"], mode="lines", name="Holdings"))
                fig.add_trace(go.Scatter(x=conc.index, y=conc["eff_n"], mode="lines", name="Effective N"))
                fig.update_layout(title="Holdings & Effective Diversification", height=380, xaxis_rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True, key="pl_02")
                st.caption("Effective N = 1 / Σ w². Higher is more diversified.")

            st.subheader("Allocation heatmap (top assets by average weight)")
            top_n = st.slider("Top N assets", 5, min(250, weights.shape[1]), 25)
            avg_weights = weights.mean().sort_values(ascending=False)
            top_assets = avg_weights.head(top_n).index.tolist()
            st.plotly_chart(plot_heatmap(weights, top_assets), use_container_width=True, key="pl_03")

            st.subheader("Largest weights")
            last_w = weights.iloc[-1].sort_values(ascending=False)
            st.dataframe(last_w.to_frame("weight").style.format({"weight": "{:.2%}"}), use_container_width=True)

    # -------------------
    # Attribution
    # -------------------
    with tabs[2]:
        st.header("Attribution (what actually drove returns)")

        weights = None
        prices = None
        if load_weights:
            weights = load_weights_only(str(selected_path))
        if load_prices:
            prices = safe_read_prices(selected_path)

        if weights is None or prices is None or weights.empty or prices.empty:
            st.info("Enable **Load weights** and **Load prices** in the sidebar to compute attribution.")
        else:
            weights.index = pd.to_datetime(weights.index, errors="coerce")
            prices.index = pd.to_datetime(prices.index, errors="coerce")
            weights_w = _subset_df(weights, start, end)
            prices_w = _subset_df(prices, start, end)

            out = pnl_contributions(weights_w, prices_w)
            if not out:
                st.warning("Could not compute contributions (dates/universe mismatch).")
            else:
                port_ret = out["portfolio_return"]
                contrib = out["contrib_returns"]

                st.subheader("Portfolio daily returns (from contributions)")
                st.plotly_chart(px.line(port_ret, title="Daily portfolio return"), use_container_width=True, key="pl_04")

                st.subheader("Top contributors (cumulative in window)")
                cum_contrib = contrib.fillna(0.0).sum(axis=0).sort_values(ascending=False)
                top_k = st.slider("Show top K contributors", 5, min(50, len(cum_contrib)), 15)
                top_assets = cum_contrib.head(top_k).index.tolist()
                bar = px.bar(cum_contrib.head(top_k)[::-1], orientation="h", title="Top contributors (sum of daily contributions)")
                st.plotly_chart(bar, use_container_width=True, key="pl_05")

                st.subheader("Contribution heatmap (top contributors)")
                st.plotly_chart(plot_heatmap(contrib.fillna(0.0), top_assets, title="Contribution (w × r)"), use_container_width=True, key="pl_06")

                with st.expander("Raw contribution table", expanded=False):
                    raw_long = contrib.fillna(0.0).stack().reset_index()
                    raw_long.columns = ["date", "asset", "contribution"]
                    raw_long["date"] = pd.to_datetime(raw_long["date"], errors="coerce")
                    st.dataframe(raw_long, use_container_width=True)

    # -------------------
    # Trades & Costs
    # -------------------
    with tabs[3]:
        st.header("Trades, Turnover, Transaction Costs")

        trades = None
        if load_trades:
            with st.spinner("Loading trades…"):
                trades = load_trades_only(str(selected_path))

        if trades is None or trades.empty:
            st.info("Trades not loaded (or not found). Enable **Load trades** in the sidebar.")
        else:
            trades = trades.copy()
            if "date" in trades.columns:
                trades["date"] = pd.to_datetime(trades["date"], errors="coerce")
                trades = trades.sort_values("date")
                trades_w = trades[(trades["date"] >= start) & (trades["date"] <= end)]
            else:
                trades_w = trades

            st.subheader("Turnover per rebalance")
            if "turnover" in trades_w.columns and "date" in trades_w.columns:
                st.plotly_chart(px.bar(trades_w, x="date", y="turnover", title="Turnover"), use_container_width=True, key="pl_07")

            if "cost" in trades_w.columns and "date" in trades_w.columns:
                st.subheader("Transaction costs")
                csum = trades_w.set_index("date")["cost"].cumsum()
                st.plotly_chart(px.line(csum, title="Cumulative cost"), use_container_width=True, key="pl_08")

            if "opt_status" in trades_w.columns:
                st.subheader("Optimizer statuses")
                counts = trades_w["opt_status"].value_counts().reset_index()
                counts.columns = ["opt_status", "count"]
                st.plotly_chart(px.bar(counts, x="opt_status", y="count", title="opt_status counts"), use_container_width=True, key="pl_09")

            st.subheader("Trade blotter")
            if "date" in trades_w.columns:
                st.dataframe(trades_w.sort_values(by="date", ascending=False), use_container_width=True)
            else:
                st.dataframe(trades_w, use_container_width=True)

    # -------------------
    # Performance
    # -------------------
    with tabs[4]:
        st.header("Performance Analytics")

        cum = (nav_w / nav_w.iloc[0]) if normalize_default else nav_w
        st.plotly_chart(px.line(cum, title="Cumulative NAV (window)"), use_container_width=True, key="pl_10")

        r = nav_w.pct_change().dropna()
        cols = st.columns(3)
        cols[0].metric("Hit rate", _fmt_pct(window_metrics.get("hit_rate")))
        cols[1].metric("Best day", _fmt_pct(window_metrics.get("best_day")))
        cols[2].metric("Worst day", _fmt_pct(window_metrics.get("worst_day")))

        col1, col2 = st.columns(2)
        with col1:
            roll_max = nav_w.cummax()
            dd = (nav_w - roll_max) / roll_max
            st.plotly_chart(plot_drawdown(dd), use_container_width=True, key="pl_11")
        with col2:
            win = st.slider("Rolling window (days)", 21, 252, 63, key="roll_win_perf")
            if len(r) >= win:
                rv = r.rolling(win).std() * np.sqrt(252)
                st.plotly_chart(px.line(rv, title=f"Rolling Volatility ({win}d)"), use_container_width=True, key="pl_12")
            else:
                st.info("Not enough data in the selected window for rolling volatility.")

        st.subheader("Return distribution")
        if not r.empty:
            st.plotly_chart(px.histogram(r, nbins=60, title="Histogram of daily returns"), use_container_width=True, key="pl_13")

        st.subheader("Rolling Sharpe")
        if "rolling_sharpe" in preperf and isinstance(preperf["rolling_sharpe"], pd.Series):
            rs = _subset_series(preperf["rolling_sharpe"], start, end)
            st.plotly_chart(plot_rolling_sharpe(rs, title="Rolling Sharpe (precomputed 63d)"), use_container_width=True, key="pl_14")
        else:
            if len(r) >= win:
                rs = (r.rolling(win).mean() / (r.rolling(win).std() + 1e-12)) * np.sqrt(252)
                st.plotly_chart(plot_rolling_sharpe(rs, title=f"Rolling Sharpe ({win}d)"), use_container_width=True, key="pl_15")

    # -------------------
    # Calendar
    # -------------------
    with tabs[5]:
        st.header("Calendar Returns Heatmap")
        heatmap_df = create_calendar_heatmap(nav_w)
        if heatmap_df.empty:
            st.info("Not enough data for calendar heatmap in this window.")
        else:
            st.plotly_chart(plot_calendar_heatmap(heatmap_df), use_container_width=True, key="pl_16")

    # -------------------
    # Parameter Explorer
    # -------------------
    with tabs[6]:
        st.header("Parameter Explorer (across runs)")
        param_df = load_all_experiment_parameters_and_metrics(rootp)

        if param_df.empty:
            st.warning("No experiment data found (need params.yaml + outputs/performance/metrics.json).")
        else:
            if "optimizer.type" in param_df.columns:
                opt_types = sorted([x for x in param_df["optimizer.type"].dropna().unique().tolist()])
                chosen = st.multiselect("Filter optimizer.type", opt_types, default=opt_types)
                if chosen:
                    param_df = param_df[param_df["optimizer.type"].isin(chosen)]

            st.caption(f"Rows: {len(param_df)}")
            x_options = sorted([c for c in param_df.columns if c != "experiment_name"])
            x_default = x_options.index("optimizer.k_cardinality") if "optimizer.k_cardinality" in x_options else 0
            x_axis_val = st.selectbox("X parameter", options=x_options, index=x_default)

            y_options = [c for c in x_options if c != x_axis_val]
            y_default = y_options.index("cagr") if "cagr" in y_options else (y_options.index("ann_return") if "ann_return" in y_options else 0)
            y_axis_val = st.selectbox("Y metric", options=y_options, index=y_default)

            fig = px.scatter(param_df, x=x_axis_val, y=y_axis_val, hover_name="experiment_name", color="optimizer.type" if "optimizer.type" in param_df.columns else None)
            st.plotly_chart(fig, use_container_width=True, key="pl_17")

            with st.expander("Correlation (numeric columns)", expanded=False):
                num = param_df.select_dtypes(include=["number"])
                if num.shape[1] >= 2:
                    st.dataframe(num.corr(numeric_only=True), use_container_width=True)
                else:
                    st.info("Not enough numeric columns for correlation.")

            with st.expander("Raw data", expanded=False):
                st.dataframe(param_df, use_container_width=True)

    # -------------------
    # Asset Inspector
    # -------------------
    with tabs[7]:
        st.header("Asset Inspector")
        if not load_prices:
            st.info("Enable **Load prices** in the sidebar.")
        else:
            prices = safe_read_prices(selected_path)
            if prices is None or prices.empty:
                st.info("No prices snapshot available to inspect assets.")
            else:
                prices.index = pd.to_datetime(prices.index, errors="coerce")
                prices_w = _subset_df(prices, start, end)

                all_tickers = sorted(prices_w.columns.tolist())
                search_query = st.text_input("Filter tickers (comma-separated substrings)", "")
                if search_query:
                    tokens = [t.strip().lower() for t in search_query.split(",") if t.strip()]
                    filtered = [t for t in all_tickers if any(tok in t.lower() for tok in tokens)]
                else:
                    filtered = all_tickers

                selected_tickers = st.multiselect("Select assets", options=filtered, default=filtered[: min(5, len(filtered))])
                if selected_tickers:
                    plot_prices = prices_w[selected_tickers].copy()
                    normalize = st.checkbox("Normalize selected prices", value=True, key="inspector_normalize")
                    if normalize:
                        first_vals = plot_prices.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else np.nan)
                        plot_prices = plot_prices.divide(first_vals, axis=1)

                    st.plotly_chart(px.line(plot_prices, title="Selected Asset Prices"), use_container_width=True, key="pl_18")

                    csv = plot_prices.to_csv().encode("utf-8")
                    st.download_button("Download Selected Prices (CSV)", data=csv, file_name="selected_prices.csv")

    # -------------------
    # Compare
    # -------------------
    with tabs[8]:
        st.header("Compare Experiments")

        if not cmp_paths:
            st.info("Enable Comparison Mode in the sidebar and select multiple experiments.")
        else:
            # NOTE: Comparison metrics and plots should respect the global date-window slider.
            # The previous implementation used cached *full-period* metrics, so changing the
            # date window appeared to "do nothing" in Compare mode.
            @st.cache_data(show_spinner=False)
            def get_comparison_data(
                paths: List[str],
                start_ts: pd.Timestamp,
                end_ts: pd.Timestamp,
                rf: float,
            ) -> pd.DataFrame:
                rows = []
                for p_str in paths:
                    p = Path(p_str)
                    exp_data = load_nav_only(str(p))
                    nav_s = ensure_series(exp_data.get("nav"))
                    nav_s = _subset_series(nav_s, start_ts, end_ts)
                    if nav_s is None or nav_s.dropna().empty or len(nav_s.dropna()) < 2:
                        continue

                    m = ann_metrics(nav_s, rf_annual=rf) or {}
                    rows.append({
                        "Experiment": p.name,
                        "Total Return": m.get("total_return"),
                        "CAGR": m.get("cagr"),
                        "Volatility": m.get("ann_vol"),
                        "Max Drawdown": m.get("max_drawdown"),
                        "Sharpe": m.get("sharpe"),
                    })
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows).set_index("Experiment")

            metrics_df = get_comparison_data([str(p) for p in cmp_paths], start, end, rf_annual)

            # ---- Benchmarks (optional) ----
            bench_rows = []
            bench_navs: List[pd.Series] = []
            bench_names: List[str] = []
            missing_bench: List[str] = []
            bench_errors: Dict[str, str] = {}

            for bl in (bench_labels or []):
                bnav, err = try_get_benchmark_nav(bl, start=start, end=end)
                if bnav is None or bnav.dropna().empty or len(bnav.dropna()) < 2:
                    missing_bench.append(bl)
                    if err:
                        bench_errors[bl] = err
                    continue
                bm = ann_metrics(bnav, rf_annual=rf_annual) or {}
                bench_rows.append({
                    "Experiment": bnav.name or bl,
                    "Total Return": bm.get("total_return"),
                    "CAGR": bm.get("cagr"),
                    "Volatility": bm.get("ann_vol"),
                    "Max Drawdown": bm.get("max_drawdown"),
                    "Sharpe": bm.get("sharpe"),
                })
                bench_navs.append(bnav)
                bench_names.append(bnav.name or bl)

            if (custom_bench_ticker or "").strip():
                bnav = get_custom_yahoo_nav(
                    custom_bench_ticker,
                    start=start,
                    end=end,
                    label=(custom_bench_label or "").strip() or custom_bench_ticker,
                )
                if bnav is None or bnav.dropna().empty or len(bnav.dropna()) < 2:
                    missing_bench.append(f"Custom: {custom_bench_ticker}")
                else:
                    bm = ann_metrics(bnav, rf_annual=rf_annual) or {}
                    bench_rows.append({
                        "Experiment": bnav.name or f"Benchmark: {custom_bench_ticker}",
                        "Total Return": bm.get("total_return"),
                        "CAGR": bm.get("cagr"),
                        "Volatility": bm.get("ann_vol"),
                        "Max Drawdown": bm.get("max_drawdown"),
                        "Sharpe": bm.get("sharpe"),
                    })
                    bench_navs.append(bnav)
                    bench_names.append(bnav.name or f"Benchmark: {custom_bench_ticker}")

            if missing_bench:
                st.info(
                    "Some benchmarks could not be loaded (network / ticker / library issue): "
                    + ", ".join(missing_bench)
                )
                if bench_errors:
                    with st.expander("Show benchmark error details", expanded=False):
                        for k, v in bench_errors.items():
                            st.markdown(f"**{k}**")
                            st.code(v)

            if bench_rows:
                bench_df = pd.DataFrame(bench_rows).set_index("Experiment")
                # merge, keeping experiments first
                metrics_df = pd.concat([metrics_df, bench_df], axis=0)
            if metrics_df.empty:
                st.warning("No valid NAV data found for the selected experiments.")
            else:
                st.subheader("Metrics table")
                st.dataframe(
                    metrics_df.style.format({
                        "Total Return": "{:.2%}",
                        "CAGR": "{:.2%}",
                        "Volatility": "{:.2%}",
                        "Max Drawdown": "{:.2%}",
                        "Sharpe": "{:.2f}",
                    }),
                    use_container_width=True,
                )

                st.subheader("Risk/return map")
                scatter_df = metrics_df.reset_index()
                fig = px.scatter(
                    scatter_df,
                    x="Volatility",
                    y="CAGR",
                    text="Experiment",
                    hover_name="Experiment",
                    title="CAGR vs Volatility",
                )
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True, key="pl_19")

                st.subheader("Normalized NAV comparison")
                nav_series: List[pd.Series] = []
                names: List[str] = []
                for path in cmp_paths:
                    # nav-only is enough here and much faster than loading the full experiment.
                    exp_data = load_nav_only(str(path))
                    nav_s = ensure_series(exp_data.get("nav"))
                    nav_s = _subset_series(nav_s, start, end)
                    if nav_s is not None and not nav_s.dropna().empty:
                        nav_series.append(nav_s)
                        names.append(path.name)

                # Add benchmarks to NAV chart (if available)
                if bench_navs:
                    nav_series.extend(bench_navs)
                    names.extend(bench_names)

                if len(nav_series) < 1:
                    st.info("No NAV data overlaps the selected date window for the chosen experiments.")
                else:
                    
                    st.plotly_chart(plot_compare_navs(nav_series, names), use_container_width=True, key="pl_20")

    # -------------------
    # Diagnostics
    # -------------------
    with tabs[9]:
        st.header("Diagnostics (data quality + stability)")

        st.subheader("Rebalance / optimizer stability")
        trades = None
        if load_trades:
            trades = load_trades_only(str(selected_path))
        if trades is None or trades.empty:
            st.info("Enable **Load trades** to see optimizer/rebalance diagnostics.")
        else:
            t = trades.copy()
            if "date" in t.columns:
                t["date"] = pd.to_datetime(t["date"], errors="coerce")
                t = t.sort_values("date")
                t = t[(t["date"] >= start) & (t["date"] <= end)]
            if "opt_status" in t.columns:
                counts = t["opt_status"].value_counts().reset_index()
                counts.columns = ["opt_status", "count"]
                st.plotly_chart(px.bar(counts, x="opt_status", y="count", title="opt_status counts"), use_container_width=True, key="pl_21")
            st.dataframe(t, use_container_width=True)

        st.subheader("Prices snapshot quality")
        if not load_prices:
            st.info("Enable **Load prices** to run data quality checks.")
        else:
            prices = safe_read_prices(selected_path)
            if prices is None or prices.empty:
                st.info("No prices snapshot found.")
            else:
                prices.index = pd.to_datetime(prices.index, errors="coerce")
                prices_w = _subset_df(prices, start, end)
                miss = prices_w.isna().mean().sort_values(ascending=False)
                st.caption("Missing rate per asset (higher is worse).")
                st.dataframe(miss.to_frame("missing_rate").style.format({"missing_rate": "{:.2%}"}), use_container_width=True)
                st.caption("Tip: high missingness will distort covariances and attribution. Consider filtering assets in preprocessing.")

                st.subheader("Run logs")
                log_path = selected_path / "logs" / "run.log"
                fail_path = selected_path / "logs" / "failure.traceback.txt"
                if log_path.exists():
                    with st.expander("run.log (tail)", expanded=False):
                        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-250:])
                        st.code(tail)
                else:
                    st.info("No run.log found in this experiment folder.")
                if fail_path.exists() and fail_path.read_text(encoding="utf-8", errors="ignore").strip():
                    with st.expander("failure.traceback.txt", expanded=False):
                        st.code(fail_path.read_text(encoding="utf-8", errors="ignore")[-8000:])
