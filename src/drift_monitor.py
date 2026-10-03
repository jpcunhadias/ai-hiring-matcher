import os
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.utils import logger, read_request_log

REFERENCE_PATH = Path("data/processed/reference_features.csv")
DRIFT_REPORT_PATH = Path("drift_report.html")
DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.5"))

# Evidently's per-column statistical tests are noisy on small samples: a 30-row
# window against a 10,000-row reference flagged 100% of columns as "drifted" purely
# from sampling variance, even with near-identical means. Below this size, skip the
# check rather than risk a false alarm.
MIN_WINDOW_SIZE = int(os.getenv("DRIFT_MIN_WINDOW_SIZE", "100"))

# Same feature shape logged per /match request in src/api.py, compared against the
# training-time reference computed in src/train_model.py — a real live-vs-training
# comparison, not the old resample-of-itself placeholder.
DRIFT_COLUMNS = ["cosine_similarity", "skill_overlap", "resume_length", "best_match_proba"]


def load_reference() -> pd.DataFrame:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Training reference not found at {REFERENCE_PATH}. Run "
            "`uv run python -m src.train_model` first."
        )
    return pd.read_csv(REFERENCE_PATH)[DRIFT_COLUMNS]


def load_current_window() -> pd.DataFrame:
    log_df = read_request_log()
    if log_df.empty:
        raise ValueError(
            "No requests logged yet in data/logs/requests.jsonl — "
            "there is no production window to compare. Make some /match calls first."
        )
    if len(log_df) < MIN_WINDOW_SIZE:
        raise ValueError(
            f"Only {len(log_df)} requests logged (minimum: {MIN_WINDOW_SIZE}). "
            "Drift tests are unreliable on small windows — wait for more traffic."
        )
    return log_df[DRIFT_COLUMNS]


def _extract_drift_share(report_dict: dict) -> float:
    for metric in report_dict["metrics"]:
        result = metric.get("result", {})
        if "share_of_drifted_columns" in result:
            return result["share_of_drifted_columns"]
    raise KeyError("share_of_drifted_columns not found in the Evidently report.")


def alert_on_drift(drift_share: float, threshold: float = DRIFT_SHARE_THRESHOLD) -> None:
    if drift_share >= threshold:
        logger.warning(
            "DRIFT ALERT: %.0f%% of monitored features show drift (configured threshold: %.0f%%).",
            drift_share * 100,
            threshold * 100,
        )
    else:
        logger.info(
            "No significant drift: %.0f%% of monitored features (threshold: %.0f%%).",
            drift_share * 100,
            threshold * 100,
        )


def run_drift_check() -> float:
    """Compares real, logged /match requests against the training-time reference
    distribution and raises a threshold-based alert. Wired to `make drift-check` —
    intended to run on a schedule (cron/systemd timer), not just on demand."""
    logger.info("Loading training reference and window of real requests...")
    reference = load_reference()
    current = load_current_window()

    logger.info("Running drift analysis with Evidently...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(DRIFT_REPORT_PATH))

    drift_share = _extract_drift_share(report.as_dict())
    alert_on_drift(drift_share)
    logger.info("Drift report saved to: %s", DRIFT_REPORT_PATH)
    return drift_share


if __name__ == "__main__":
    run_drift_check()
