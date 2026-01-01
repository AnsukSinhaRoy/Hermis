"""
utils/run_logging.py

Lightweight logging / run-metadata helpers expected by run_experiment.py.

The original project had a richer implementation; this provides compatible
APIs for:
- RunPaths dataclass
- setup_logging(log_file, verbose)
- write_metadata(paths, extra)
- PercentProgressLogger
- log_failure(logger, failure_file, exc)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging
import sys
import traceback
from datetime import datetime


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    config_copy: Path
    log_file: Path
    stdio_log_file: Path
    failure_file: Path
    metadata_file: Path


def setup_logging(log_file: str | Path, verbose: bool = True, name: str = "hermis") -> logging.Logger:
    """
    Configure and return a logger writing to `log_file` and (optionally) console.

    - verbose=True: also logs to stdout
    - verbose=False: file-only
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def write_metadata(paths: RunPaths, extra: Optional[Dict[str, Any]] = None) -> None:
    """Best-effort merge into paths.metadata_file."""
    try:
        paths.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        meta: Dict[str, Any] = {}
        if paths.metadata_file.exists():
            meta = json.loads(paths.metadata_file.read_text(encoding="utf-8") or "{}")
        if extra:
            meta.update(extra)
        meta.setdefault("updated_utc", datetime.utcnow().isoformat())
        paths.metadata_file.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    except Exception:
        return


class PercentProgressLogger:
    """Log progress every N percent steps."""

    def __init__(self, logger: logging.Logger, total_steps: int, percent_step: int = 1):
        self.logger = logger
        self.total_steps = max(1, int(total_steps))
        self.percent_step = max(1, int(percent_step))
        self._next_pct = 0

    def maybe_log(self, step: int, **fields: Any) -> None:
        try:
            pct = int((100 * (step + 1)) / self.total_steps)
            if pct >= self._next_pct:
                self._next_pct = pct + self.percent_step
                msg = "Progress: %d%% (%d/%d)" % (pct, step + 1, self.total_steps)
                if fields:
                    msg += " | " + " ".join(f"{k}={v}" for k, v in fields.items())
                self.logger.info(msg)
        except Exception:
            # progress logging must never break a run
            return


def log_failure(logger: logging.Logger, failure_file: str | Path, exc: Exception) -> None:
    """Write exception repr + traceback to failure_file and log it."""
    failure_file = Path(failure_file)
    failure_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        failure_file.write_text(f"{repr(exc)}\n\n{traceback.format_exc()}", encoding="utf-8")
    except Exception:
        pass
    try:
        logger.exception("Run failed: %s", repr(exc))
    except Exception:
        pass
