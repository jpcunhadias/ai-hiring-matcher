"""Markdown files for the aggregate tables of a real-data run, so a number can be cited.

A report holds aggregates only (hit rates, gains, intervals, counts): no text or id from the
masked tables. Its header records what is needed to reproduce the run: the masked-data
generation it read, the seeds and the command-line options.
"""

import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

NO_GENERATION = "none (the masked tables predate the GENERATION marker; re-run `make ingest`)"


def git_commit() -> str:
    """Short hash of the checked-out commit, with a mark when the tree has local changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit} (with uncommitted changes)" if dirty else commit


def markdown_table(frame: pd.DataFrame) -> str:
    """A GitHub-flavoured table; numbers are written as they are (format them first)."""
    if frame.empty:
        return "_(no rows)_"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def render(
    title: str,
    meta: Mapping[str, object],
    sections: Sequence[tuple[str, str]],
    now: datetime | None = None,
) -> str:
    """The whole report: a title, the run's metadata, then one `## heading` per section."""
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# {title}", "", f"- run: {stamp}"]
    lines += [f"- {key}: {value}" for key, value in meta.items()]
    for heading, body in sections:
        lines += ["", f"## {heading}", "", body]
    return "\n".join(lines) + "\n"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def pts(value: float) -> str:
    """A fraction as signed percentage points."""
    return f"{100 * value:+.1f}"


def interval(low: float, high: float) -> str:
    return f"[{pts(low)}, {pts(high)}]"
