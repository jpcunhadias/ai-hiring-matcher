from datetime import UTC, datetime

import pandas as pd

from src import run_report


def test_markdown_table_writes_a_header_a_rule_and_one_line_per_row():
    frame = pd.DataFrame({"method": ["a", "b"], "gain": ["+1.0", "-2.5"]})

    assert run_report.markdown_table(frame).splitlines() == [
        "| method | gain |",
        "|---|---|",
        "| a | +1.0 |",
        "| b | -2.5 |",
    ]
    assert run_report.markdown_table(frame.iloc[0:0]) == "_(no rows)_"


def test_render_puts_the_run_metadata_before_the_sections():
    text = run_report.render(
        "Title",
        {"model seed": 3, "masked-data generation": "abc"},
        [("first", "body one"), ("second", "body two")],
        now=datetime(2026, 11, 15, 22, 5, tzinfo=UTC),
    )

    assert text.splitlines()[:5] == [
        "# Title",
        "",
        "- run: 2026-11-15 22:05 UTC",
        "- model seed: 3",
        "- masked-data generation: abc",
    ]
    assert text.index("## first") < text.index("body one") < text.index("## second")


def test_write_creates_the_reports_directory(tmp_path):
    path = run_report.write(tmp_path / "reports" / "x.md", "hello\n")

    assert path.read_text() == "hello\n"


def test_points_and_intervals_are_signed_percentage_points():
    assert run_report.pts(0.064) == "+6.4"
    assert run_report.pts(-0.04) == "-4.0"
    assert run_report.interval(0.027, 0.101) == "[+2.7, +10.1]"
