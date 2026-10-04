import json

import src.utils as utils


def _write_rows(n: int, tmp_path, monkeypatch, max_rows: int):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(utils, "REQUEST_LOG_PATH", path)
    monkeypatch.setattr(utils, "REQUEST_LOG_MAX_ROWS", max_rows)
    for i in range(n):
        utils.log_request({"i": i, "pad": "x" * 300})  # rows big enough to pass the size gate
    return path


def test_request_log_keeps_only_the_newest_rows(tmp_path, monkeypatch):
    path = _write_rows(60, tmp_path, monkeypatch, max_rows=20)

    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(rows) <= 40  # bounded: never more than the size gate allows
    assert rows[-1]["i"] == 59  # the newest row is always kept
    assert [r["i"] for r in rows] == sorted(r["i"] for r in rows)  # order preserved, no gaps
    assert rows[0]["i"] == 60 - len(rows)


def test_request_log_below_the_cap_is_untouched(tmp_path, monkeypatch):
    path = _write_rows(10, tmp_path, monkeypatch, max_rows=100)

    assert len(path.read_text().splitlines()) == 10


def test_trim_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    path = _write_rows(60, tmp_path, monkeypatch, max_rows=20)

    assert not path.with_suffix(".tmp").exists()
