from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd

OptimizerFn = Callable[[pd.Series, pd.DataFrame], Dict[str, Any]]


@dataclass
class OptimizerSpec:
    name: str
    fn: OptimizerFn
    description: str = ""


_REGISTRY: Dict[str, OptimizerSpec] = {}


def register(name: str, description: str = ""):
    """Decorator to register an optimizer function."""

    def _wrap(fn: OptimizerFn) -> OptimizerFn:
        _REGISTRY[name] = OptimizerSpec(name=name, fn=fn, description=description)
        return fn

    return _wrap


def get(name: str) -> OptimizerSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown optimizer: {name!r}. Known: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_optimizers() -> Dict[str, OptimizerSpec]:
    return dict(_REGISTRY)
