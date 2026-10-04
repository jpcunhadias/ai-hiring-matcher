import json
import logging
import os
import threading
from pathlib import Path

import joblib
import pandas as pd

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
REQUEST_LOG_PATH = DATA_DIR / "logs" / "requests.jsonl"
# The log is a rolling window: only the most recent rows are kept, so it can't grow forever.
REQUEST_LOG_MAX_ROWS = int(os.getenv("REQUEST_LOG_MAX_ROWS", "10000"))
_ROW_SIZE_GATE = 200  # bytes/row above which we bother counting lines (a row is ~110 bytes)
_request_log_lock = threading.Lock()


def setup_logging(name: str = "ml_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger


logger = setup_logging("ml_pipeline")


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("DataFrame saved to: %s", path)


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_model(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Object saved to: %s", path)


def load_model(path: Path) -> object:
    return joblib.load(path)


def _trim_request_log(path: Path, max_rows: int) -> None:
    """Keep only the newest `max_rows` rows. A cheap size check comes first, so the file
    is only read once it is clearly past the cap; trimming rewrites it atomically."""
    if path.stat().st_size < max_rows * _ROW_SIZE_GATE:
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) <= max_rows:
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(lines[-max_rows:]), encoding="utf-8")
    tmp.replace(path)


def log_request(payload: dict) -> None:
    """Appends one prediction request's features to the local rolling log that
    drift_monitor.py later compares against the training-time reference.

    The log keeps the newest REQUEST_LOG_MAX_ROWS rows. The lock makes append + trim safe
    across threads in one process; several uvicorn workers would need a real store.
    """
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with _request_log_lock:
        REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REQUEST_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
        _trim_request_log(REQUEST_LOG_PATH, REQUEST_LOG_MAX_ROWS)


def read_request_log(path: Path = REQUEST_LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(json.loads(line) for line in lines)
