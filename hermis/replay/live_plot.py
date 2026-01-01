"""hermis/replay/live_plot.py

Live NAV plotter (matplotlib) for long-running simulations.

This is intentionally dependency-light. It gives you:
- a separate interactive window
- a NAV curve updated in (near) real time
- an info panel (last NAV, total return, last daily return, drawdown)

If the environment is headless or matplotlib can't open a GUI window,
instantiate will raise and callers should catch it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np


@dataclass
class NavStats:
    date: Optional[object]
    nav: float
    total_return: float
    daily_return: float
    drawdown: float


class NavLivePlotter:
    """Interactive live plot for NAV."""

    def __init__(
        self,
        title: str = "Hermis Live NAV",
        subtitle: str = "",
        theme: str = "dark",
        window_days: Optional[int] = None,
    ):
        # Local imports so replay runs fine even without GUI support.
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        self._plt = plt
        self._mdates = mdates

        theme = (theme or "").strip().lower()
        if theme in ("dark", "night", "black"):
            try:
                plt.style.use("dark_background")
            except Exception:
                pass
        elif theme in ("light", "day"):
            # default style
            pass
        else:
            # user may pass a matplotlib style name
            try:
                plt.style.use(theme)
            except Exception:
                pass

        # Interactive mode
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(11, 6))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        self.ax.set_title(title + ("\n" + subtitle if subtitle else ""))
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("NAV")

        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(self.ax.xaxis.get_major_locator()))

        self._dates = []
        self._nav = []
        (self.line,) = self.ax.plot([], [], linewidth=1.8)
        (self.marker,) = self.ax.plot([], [], marker="o", markersize=5)

        # info box
        self._info = self.ax.text(
            0.01,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="black" if theme in ("dark", "night", "black") else "white", alpha=0.35),
        )

        self._window_days = window_days
        self._last_nav = None
        self._peak_nav = 1.0

        # Show non-blocking window
        try:
            plt.show(block=False)
        except Exception:
            pass

    def _compute_stats(self) -> NavStats:
        if not self._nav:
            return NavStats(date=None, nav=float("nan"), total_return=float("nan"), daily_return=float("nan"), drawdown=float("nan"))

        nav = float(self._nav[-1])
        start = float(self._nav[0])
        total_return = (nav / start - 1.0) if start > 0 else float("nan")

        if len(self._nav) >= 2 and self._nav[-2] > 0:
            daily_return = nav / float(self._nav[-2]) - 1.0
        else:
            daily_return = float("nan")

        self._peak_nav = max(self._peak_nav, nav) if math.isfinite(nav) else self._peak_nav
        drawdown = (1.0 - nav / self._peak_nav) if self._peak_nav > 0 and math.isfinite(nav) else float("nan")

        return NavStats(date=self._dates[-1], nav=nav, total_return=total_return, daily_return=daily_return, drawdown=drawdown)

    def update_point(self, date, nav: float):
        """Append a single (date, nav) point and refresh the window."""
        try:
            nav = float(nav)
        except Exception:
            return

        self._dates.append(date)
        self._nav.append(nav)

        # Update data
        self.line.set_data(self._dates, self._nav)
        self.marker.set_data([self._dates[-1]], [self._nav[-1]])

        # Windowing (show last N days)
        if self._window_days is not None and len(self._dates) > self._window_days:
            start_i = max(0, len(self._dates) - self._window_days)
            xd = self._dates[start_i:]
            yd = self._nav[start_i:]
            self.line.set_data(xd, yd)
            self.marker.set_data([xd[-1]], [yd[-1]])

        # Autoscale nicely
        self.ax.relim()
        self.ax.autoscale_view()

        st = self._compute_stats()
        info = (
            f"Last: {st.nav:,.4f}\n"
            f"Total: {st.total_return*100:+.2f}%\n"
            f"1D: {st.daily_return*100:+.2f}%\n"
            f"DD: {st.drawdown*100:+.2f}%"
        )
        self._info.set_text(info)

        # Render
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._plt.pause(0.001)
        except Exception:
            # If backend doesn't support interactive updates, do nothing.
            pass


    def update(self, date, nav: float):
        """Compatibility alias (older code calls .update())."""
        return self.update_point(date, nav)

    def pump(self):
        """Process GUI events without adding a point.

        Call this periodically to keep the window responsive even if you
        update the curve only every N steps.
        """
        try:
            self.fig.canvas.flush_events()
            self._plt.pause(0.001)
        except Exception:
            pass

    def wait_until_closed(self, poll: float = 0.1):
        """Keep the process alive until the user closes the plot window.

        Useful at the end of a simulation so the window doesn't disappear
        immediately when the Python process exits.
        """
        try:
            fig = self.fig
            if fig is None:
                return
            plt = self._plt
            # Loop until figure is closed; also pumps GUI events.
            while plt.fignum_exists(fig.number):
                self.pump()
                plt.pause(poll)
        except KeyboardInterrupt:
            # Allow Ctrl+C to exit even if window is open.
            return

    def finalize(self):
        """Switch back to non-interactive mode (best-effort)."""
        try:
            self._plt.ioff()
        except Exception:
            pass
