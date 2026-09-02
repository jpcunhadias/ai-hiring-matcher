import json
import logging
from pathlib import Path

import joblib
import pandas as pd

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
REQUEST_LOG_PATH = DATA_DIR / "logs" / "requests.jsonl"


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
    logger.info("DataFrame salvo em: %s", path)


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_model(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Objeto salvo em: %s", path)


def load_model(path: Path) -> object:
    return joblib.load(path)


def log_request(payload: dict) -> None:
    """Appends one prediction request's features to the local rolling log that
    drift_monitor.py later compares against the training-time reference."""
    REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REQUEST_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_request_log(path: Path = REQUEST_LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(json.loads(line) for line in lines)
