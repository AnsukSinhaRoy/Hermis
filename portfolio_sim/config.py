# portfolio_sim/config.py
import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Config:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str) -> 'Config':
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)
        return Config(raw)
