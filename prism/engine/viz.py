# portfolio_viz/viz.py
from typing import Optional, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_nav(nav: pd.Series, title: str = 'Portfolio NAV') -> go.Figure:
    fig = px.area(nav, title=title, labels={'index': 'Date', 'value': 'NAV'})
    fig.update_layout(showlegend=False)
    return fig


def plot_all_prices(prices: pd.DataFrame, nav: Optional[pd.Series] = None, normalize: bool = True) -> go.Figure:
    plot_prices = prices.copy()
    if normalize:
        first_vals = plot_prices.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else None)
        plot_prices = plot_prices.divide(first_vals, axis=1)
    fig = go.Figure()
    for col in plot_prices.columns:
        fig.add_trace(go.Scatter(x=plot_prices.index, y=plot_prices[col], mode='lines', line=dict(width=0.5), opacity=0.5, name=col, showlegend=False))
    median_series = plot_prices.median(axis=1)
    fig.add_trace(go.Scatter(x=median_series.index, y=median_series, mode='lines', line=dict(width=2, dash='dash'), name='Median Asset'))
    if nav is not None:
        nav_aligned = nav.reindex(plot_prices.index, method='ffill')
        if normalize:
            nav_aligned /= nav_aligned.dropna().iloc[0]
        fig.add_trace(go.Scatter(x=nav_aligned.index, y=nav_aligned, mode='lines', line=dict(width=3), name='Portfolio NAV'))
    fig.update_layout(title='Asset Price Paths and Portfolio NAV', height=600, xaxis_rangeslider_visible=True)
    return fig


def plot_heatmap(weights: pd.DataFrame, top_assets: List[str], title: Optional[str] = None) -> go.Figure:
    """
    Plot heatmap of weights for `top_assets`. This version coerces numeric types,
    fills NaNs with 0 and sets a visible colorbar with percent formatting.
    """
    sub = weights[top_assets].copy()
    # Ensure numeric dtype; replace non-numeric with NaN then fill with 0
    sub = sub.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # If rows are dates (index) keep them; ensure columns are assets
    z = sub.T.values  # assets x dates for better y-axis reading

    # Compute sensible color limits
    vmin = float(sub.min().min())
    vmax = float(sub.max().max())
    # if symmetric around zero is desired (e.g., contributions), center at 0
    if vmin < 0 < vmax and abs(vmin) > 0 and abs(vmax) > 0:
        bound = max(abs(vmin), abs(vmax))
        zmin, zmax = -bound, bound
    else:
        zmin, zmax = vmin, vmax

    # Fallback title
    if title is None:
        title = f"Top {len(top_assets)} Asset Weights Over Time"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=sub.index,        # x = dates
            y=sub.columns,      # y = asset names
            colorscale='Viridis',
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Weight", tickformat=".2%"),  # percent ticks
            hovertemplate="Date: %{x}<br>Asset: %{y}<br>Weight: %{z:.2%}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        height=600,
        yaxis=dict(automargin=True),
        xaxis=dict(automargin=True),
    )
    return fig


def plot_drawdown(drawdown: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name='Drawdown'))
    fig.update_layout(title='Portfolio Drawdown', yaxis_tickformat='.2%', height=400)
    return fig


def plot_rolling_sharpe(rs: pd.Series, title: str) -> go.Figure:
    fig = px.line(rs, title=title)
    return fig


def plot_compare_navs(nav_series: List[pd.Series], names: List[str]) -> go.Figure:
    fig = go.Figure()
    for s, name in zip(nav_series, names):
        s_norm = s / s.dropna().iloc[0]
        fig.add_trace(go.Scatter(x=s_norm.index, y=s_norm, mode='lines', name=name))
    fig.update_layout(height=500, title='Normalized NAV Over Time', xaxis_rangeslider_visible=True)
    return fig

def plot_calendar_heatmap(
    heatmap_df: pd.DataFrame,
    title: Optional[str] = "Calendar Returns",
    cell_height: int = 28,
    zmin: float = -0.30,
    zmax: float = 0.30,
) -> go.Figure:
    """
    Monthly returns heatmap with *fixed* color buckets (not relative to sample range).

    Buckets (fixed bins):
      <= -15% : deep red
      -15%..-10% : red
      -10%..-5% : orange
      -5%..0% : yellow
      0%..5% : light green
      5%..10% : medium green
      10%..15% : deep green
      >= 15% : bluish green

    `zmin`/`zmax` control the clamp range for the palette.
    """
    df = heatmap_df.copy()
    if 'Year' in df.columns:
        df_plot = df.drop(columns=['Year'])
    else:
        df_plot = df

    # numeric matrix
    z = df_plot.values.astype(float)
    x = list(df_plot.columns)
    y = [str(i) for i in df_plot.index]

    # Clamp to fixed range for consistent coloring
    z = z.clip(min=zmin, max=zmax)

    # Fixed bucket edges (in decimal returns)
    edges = [zmin, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, zmax]
    colors = [
        "#8e0000",  # deep red
        "#e53935",  # red
        "#fb8c00",  # orange
        "#fdd835",  # yellow
        "#c8e6c9",  # light green
        "#66bb6a",  # medium green
        "#2e7d32",  # deep green
        "#00897b",  # bluish green
    ]
    def _pos(v: float) -> float:
        return 0.0 if zmax == zmin else (v - zmin) / (zmax - zmin)

    # Build a stepped colorscale (discrete buckets)
    # Plotly colorscale expects positions in [0,1]
    cs = []
    for i in range(len(colors)):
        left = _pos(edges[i])
        right = _pos(edges[i + 1])
        cs.append([left, colors[i]])
        cs.append([right, colors[i]])
        if i < len(colors) - 1:
            cs.append([right, colors[i + 1]])

    # Colorbar labels at bucket midpoints
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(colors))]
    tickvals = mids
    ticktext = [
        "≤ -15%",
        "-15% .. -10%",
        "-10% .. -5%",
        "-5% .. 0%",
        "0% .. 5%",
        "5% .. 10%",
        "10% .. 15%",
        "≥ 15%",
    ]
# Figure height: one row per year
    fig_height = max(320, int(cell_height * max(1, len(y)) + 160))

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=zmin,
            zmax=zmax,
            colorscale=cs,
            colorbar=dict(
                title="Monthly return",
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
            ),
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>",
        )
    )

    # Overlay the values as text
    texts, xs, ys, text_colors = [], [], [], []
    for yi, row in enumerate(y):
        for xi, col in enumerate(x):
            val = z[yi, xi]
            if pd.isna(val):
                texts.append("")
                xs.append(col)
                ys.append(row)
                text_colors.append("#202124")
                continue

            texts.append(f"{val:+.2%}")
            xs.append(col)
            ys.append(row)

            # Decide text color for readability based on fixed normalization
            norm = _pos(float(val))
            # Dark text on lighter colors; white text on strong red/green
            text_colors.append("white" if (norm <= 0.18 or norm >= 0.78) else "#202124")

    font_size = 12 if len(y) <= 20 else 10

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="text",
            text=texts,
            textfont=dict(size=font_size, color=text_colors),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        height=fig_height,
        margin=dict(l=70, r=20, t=55, b=110),
        template="plotly_white",
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(automargin=True)

    return fig
