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
    title: Optional[str] = "Monthly Returns Heatmap",
    cell_height: int = 28,
) -> go.Figure:
    """
    Plot a colored heatmap of monthly returns (from create_calendar_heatmap()) and overlay numeric labels.
    `cell_height` controls the vertical pixel height allocated per row (year).
    """
    df = heatmap_df.copy()
    if 'Year' in df.columns:
        df_plot = df.drop(columns=['Year'])
    else:
        df_plot = df

    # axis labels and numeric matrix
    z = df_plot.values.astype(float)  # shape (n_years, n_months)
    x = list(df_plot.columns)         # month names (x axis)
    y = [str(i) for i in df_plot.index]  # years (y axis as strings)

    # Base image (colored heatmap)
    fig = px.imshow(
        z,
        x=x,
        y=y,
        labels={'x': 'Month', 'y': 'Year', 'color': 'Monthly Return'},
        aspect='auto',
        origin='lower',
        color_continuous_scale='RdYlGn',
    )
    fig.update_coloraxes(colorbar_tickformat=".2%")

    # Compute dynamic figure height using cell_height
    # Add some padding for title/axis labels:
    top_bottom_padding = 140  # px for title, x-axis ticks, margins (tweak if needed)
    fig_height = max(300, len(y) * int(cell_height) + top_bottom_padding)

    # Text overlay: format as percent with sign, handle NaN
    texts = []
    text_colors = []
    xs = []
    ys = []

    # Determine normalization for text color contrast
    flat = pd.Series(z.flatten())
    zmin = float(flat.min(skipna=True)) if not flat.empty else 0.0
    zmax = float(flat.max(skipna=True)) if not flat.empty else 0.0
    center = 0.0
    span = max(abs(zmin - center), abs(zmax - center), 1e-9)

    for yi, row in enumerate(y):
        for xi, col in enumerate(x):
            val = z[yi, xi]
            if pd.isna(val):
                text = ""
            else:
                text = f"{val:+.2%}"
            texts.append(text)
            xs.append(col)
            ys.append(row)
            # choose text color for contrast: white on strong color, else black
            if pd.isna(val):
                text_colors.append("black")
            else:
                norm = abs(val - center) / span
                text_colors.append("white" if norm > 0.45 else "black")

    # Compute font size proportional to cell_height (clamped for legibility)
    # Rough heuristic: ~40% of cell height, clamp between 8 and 18
    font_size = int(max(8, min(18, round(cell_height * 0.42))))


    # Add Scatter text layer (overlay). hoverinfo='skip' to keep heatmap hover
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

    # Tweak layout: dynamic height, margins and tick styling
    fig.update_layout(
        title=title,
        height=fig_height,
        margin=dict(l=80, r=20, t=60, b=120),  # more bottom margin for rotated x-ticks
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(automargin=True)

    # Make hover template still informative on the heatmap trace
    # Note: px.imshow created the heatmap trace already; update its hovertemplate
    # The first trace is usually the heatmap; find and update its hovertemplate if possible
    for trace in fig.data:
        if isinstance(trace, go.Heatmap) or (hasattr(trace, 'z') and trace.name in (None, '')):
            trace.hovertemplate = 'Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}'
            break

    return fig
