import json

import pytest

import src.utils as utils


def _write_rows(n: int, tmp_path, monkeypatch, max_rows: int):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(utils, "REQUEST_LOG_PATH", path)
    monkeypatch.setattr(utils, "REQUEST_LOG_MAX_ROWS", max_rows)
    utils._row_counts.pop(path, None)
    for i in range(n):
        utils.log_request({"i": i})  # realistic tiny rows, no padding needed to hit the cap
    return path


def test_request_log_never_exceeds_the_cap_plus_slack_and_keeps_the_newest(tmp_path, monkeypatch):
    path = _write_rows(500, tmp_path, monkeypatch, max_rows=100)

    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert 100 <= len(rows) <= 110  # trimmed back to 100 once it passes 110
    assert rows[-1]["i"] == 499  # the newest row is always kept
    assert [r["i"] for r in rows] == list(range(500 - len(rows), 500))  # ordered, no gaps


def test_request_log_trims_with_realistic_row_sizes(tmp_path, monkeypatch):
    # The earlier size-based gate let ~22,000 real rows through a 10,000 cap.
    path = _write_rows(2000, tmp_path, monkeypatch, max_rows=1000)

    assert len(path.read_text().splitlines()) <= 1100


def test_request_log_below_the_cap_is_untouched(tmp_path, monkeypatch):
    path = _write_rows(10, tmp_path, monkeypatch, max_rows=100)

    assert len(path.read_text().splitlines()) == 10


def test_trim_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    path = _write_rows(500, tmp_path, monkeypatch, max_rows=100)

    assert not path.with_suffix(".tmp").exists()


def test_a_deleted_log_file_restarts_the_count(tmp_path, monkeypatch):
    path = _write_rows(50, tmp_path, monkeypatch, max_rows=100)
    path.unlink()

    utils.log_request({"i": 0})

    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("bad", [0, -5])
def test_a_non_positive_cap_is_rejected_instead_of_keeping_everything(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(utils, "REQUEST_LOG_PATH", tmp_path / "r.jsonl")
    monkeypatch.setattr(utils, "REQUEST_LOG_MAX_ROWS", bad)

    with pytest.raises(ValueError):
        utils.log_request({"i": 0})
