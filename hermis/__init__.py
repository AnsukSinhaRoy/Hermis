"""Hermis - trading simulation toolkit.

Hermis is the *simulator tool* in this repo.

Design goals:
- Modular and explicit extension points (new optimizers, new strategies, RL policies).
- Keep daily-rebalance portfolio simulation simple and fast.
- Provide a path to event-driven, high-frequency simulation later (1s / derivatives).

Prism (Streamlit UI) and Levitate (scheduler/launcher) live as separate tools in this repo.
"""

from __future__ import annotations

__all__ = [
    "data",
    "optimizers",
    "sim",
    "strategies",
]

__version__ = "0.1.0"
