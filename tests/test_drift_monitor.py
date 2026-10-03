import logging

import pandas as pd
import pytest

from src.drift_monitor import (
    _extract_drift_share,
    alert_on_drift,
    load_current_window,
    load_reference,
)


def test_load_reference_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.drift_monitor.REFERENCE_PATH", tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError):
        load_reference()


def test_load_current_window_raises_when_log_empty(monkeypatch):
    monkeypatch.setattr("src.drift_monitor.read_request_log", lambda: pd.DataFrame())

    with pytest.raises(ValueError):
        load_current_window()


def _fake_log(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cosine_similarity": [0.9] * n,
            "skill_overlap": [0.5] * n,
            "resume_length": [500] * n,
            "best_match_proba": [0.8] * n,
            "extra_field_not_used_for_drift": ["noise"] * n,
        }
    )


def test_load_current_window_selects_drift_columns(monkeypatch):
    monkeypatch.setattr("src.drift_monitor.MIN_WINDOW_SIZE", 1)
    monkeypatch.setattr("src.drift_monitor.read_request_log", lambda: _fake_log(1))

    window = load_current_window()

    assert list(window.columns) == [
        "cosine_similarity",
        "skill_overlap",
        "resume_length",
        "best_match_proba",
    ]


def test_load_current_window_raises_when_below_min_size(monkeypatch):
    monkeypatch.setattr("src.drift_monitor.MIN_WINDOW_SIZE", 100)
    monkeypatch.setattr("src.drift_monitor.read_request_log", lambda: _fake_log(5))

    with pytest.raises(ValueError):
        load_current_window()


def test_extract_drift_share_finds_the_right_metric():
    report_dict = {
        "metrics": [
            {"metric": "SomeOtherMetric", "result": {"unrelated": 1}},
            {"metric": "DatasetDriftMetric", "result": {"share_of_drifted_columns": 0.75}},
        ]
    }

    assert _extract_drift_share(report_dict) == 0.75


def test_extract_drift_share_missing_raises():
    with pytest.raises(KeyError):
        _extract_drift_share({"metrics": [{"metric": "X", "result": {}}]})


def test_alert_on_drift_warns_above_threshold(caplog):
    with caplog.at_level(logging.WARNING, logger="ml_pipeline"):
        alert_on_drift(0.9, threshold=0.5)

    assert any("DRIFT ALERT" in record.message for record in caplog.records)


def test_alert_on_drift_no_warning_below_threshold(caplog):
    with caplog.at_level(logging.WARNING, logger="ml_pipeline"):
        alert_on_drift(0.1, threshold=0.5)

    assert not any("DRIFT ALERT" in record.message for record in caplog.records)
