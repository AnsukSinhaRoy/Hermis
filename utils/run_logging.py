import json
import logging
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

@dataclass
class RunPaths:
    run_dir: Path
    config_copy: Path
    log_file: Path
    stdio_log_file: Path
    failure_file: Path
    metadata_file: Path

def _now_stamp() -> str:
    # sortable + human readable
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def make_run_dir(base_dir: str | Path, config_path: str | Path) -> RunPaths:
    base_dir = Path(base_dir)
    config_path = Path(config_path)

    cfg_stem = config_path.stem.replace(" ", "_")
    run_id = f"{_now_stamp()}_{str(uuid4())[:8]}"
    run_dir = base_dir / cfg_stem / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_copy = run_dir / "config.yaml"
    shutil.copy2(config_path, config_copy)

    return RunPaths(
        run_dir=run_dir,
        config_copy=config_copy,
        log_file=run_dir / "run.log",
        stdio_log_file=run_dir / "stdout_stderr.log",
        failure_file=run_dir / "failure.traceback.txt",
        metadata_file=run_dir / "metadata.json",
    )

def setup_logging(log_path: Path, verbose: bool = True) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("simulation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    # File: super detailed
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console: less noisy
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(ch)

    # Capture warnings into logging too
    logging.captureWarnings(True)

    return logger

def write_metadata(paths: RunPaths, extra: dict | None = None) -> None:
    payload = {
        "run_dir": str(paths.run_dir),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "start_time_unix": time.time(),
    }
    if extra:
        payload.update(extra)
    paths.metadata_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

class PercentProgressLogger:
    """
    Logs at each 1% increment (or any percent_step you choose).
    """
    def __init__(self, logger: logging.Logger, total_steps: int, percent_step: int = 1):
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if percent_step <= 0:
            raise ValueError("percent_step must be > 0")

        self.logger = logger
        self.total_steps = total_steps
        self.percent_step = percent_step
        self._next_percent = percent_step

        # always log the plan first
        self.logger.info(f"Planned total steps: {total_steps}")
        self.logger.info(f"Progress logging every {percent_step}%")

    def update(self, step_index_1_based: int, **details):
        # clamp
        if step_index_1_based < 0:
            step_index_1_based = 0
        if step_index_1_based > self.total_steps:
            step_index_1_based = self.total_steps

        pct = int((step_index_1_based / self.total_steps) * 100)

        # log at each threshold crossing
        while pct >= self._next_percent:
            msg = f"Progress: {self._next_percent}% ({step_index_1_based}/{self.total_steps})"
            if details:
                msg += " | " + " ".join(f"{k}={v}" for k, v in details.items())
            self.logger.info(msg)
            self._next_percent += self.percent_step

def log_failure(logger: logging.Logger, failure_file: Path, exc: BaseException):
    logger.exception("Simulation failed with an exception.")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    failure_file.write_text(tb, encoding="utf-8")
