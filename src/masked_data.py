"""Loader for the masked real data: candidacies -> labeled (vacancy, candidate) pairs.

Reads only `data/masked/` (see docs/masking.md); the raw archives are never touched here.

Label. `y = 1` when the candidacy ended in a hire. Most outcomes are still pending (77.7% in
the real data), and a pending candidacy is not a negative, so two settings are offered:

* `resolved_only`: only candidacies with a known outcome. Honest, but small.
* `all_prospects`: pending counted as not hired. Larger, but noisier: some of those candidates
  may still be hired. Report both.

Ranking is judged within a vacancy, so a vacancy is kept only if it has at least two
candidacies and at least one hire (and, when resolved-only, at least one non-hire too).

Split. Vacancies are split by their requested month: train on earlier vacancies, test on later
ones. The split comes first and the labels are then rebuilt *as of the cutoff*: a candidacy that
did not exist yet is dropped, and an outcome recorded after the cutoff (or with no date at all)
counts as still pending, exactly as it would have in production. Only then are the rankable
vacancies chosen, so no future outcome decides what the training set contains. The test period
keeps its final labels, because that is what is being predicted.

Sex, disability and age band live in the separate `sensitive` table. They are not joined into
the pairs, so they cannot become features by accident; `audit_attributes` returns them for
fairness checks only.
"""

import argparse
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.ingest import TABLE_COLUMNS
from src.utils import logger

MASKED_DIR = Path(os.getenv("MASKED_DIR", "data/masked"))
MIN_QUERY_CHARS = 50  # a vacancy text shorter than this carries no usable signal
MIN_CV_CHARS = 200  # same for a CV
Setting = Literal["resolved_only", "all_prospects"]
SETTINGS: tuple[Setting, ...] = ("resolved_only", "all_prospects")

VACANCY_FEATURES = [
    "client_id", "is_sap", "contract_type", "priority", "seniority", "education_level",
    "english_level", "spanish_level", "areas", "state",
]  # fmt: skip
APPLICANT_FEATURES = [
    "education_level", "english_level", "spanish_level", "area", "seniority",
    "professional_title", "technical_skills", "certifications",
]  # fmt: skip


@dataclass(frozen=True)
class MaskedTables:
    vacancies: pd.DataFrame
    applicants: pd.DataFrame
    sensitive: pd.DataFrame
    candidacies: pd.DataFrame


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame  # labels as known at the cutoff, rankable vacancies only
    test: pd.DataFrame  # final labels, rankable vacancies only
    cutoff: str  # last requested month that belongs to the training period
    train_pool: pd.DataFrame  # every candidacy of the train vacancies that existed at the cutoff
    test_pool: pd.DataFrame  # every candidacy of the test vacancies, whatever its outcome


def load_masked(masked_dir: Path | None = None) -> MaskedTables:
    """Read the four masked tables, failing loudly if one is missing or has the wrong columns."""
    directory = masked_dir or MASKED_DIR
    frames: dict[str, pd.DataFrame] = {}
    for name, columns in TABLE_COLUMNS.items():
        path = directory / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `make ingest` first")
        frame = pd.read_parquet(path)
        if list(frame.columns) != list(columns):
            raise ValueError(f"{path} has unexpected columns {list(frame.columns)}")
        frames[name] = frame
    return MaskedTables(**frames)


def _blank_to_na(series: pd.Series) -> pd.Series:
    return series.where(series.notna() & (series != ""))


def build_pairs(
    tables: MaskedTables, min_query_chars: int = MIN_QUERY_CHARS, min_cv_chars: int = MIN_CV_CHARS
) -> pd.DataFrame:
    """One row per (vacancy, candidate) with the label, the texts and the structured fields.

    Only candidacies whose vacancy and applicant records both exist and have enough text are
    kept; duplicate pairs are collapsed. Structured fields are prefixed `v_` (vacancy) and
    `a_` (applicant). The `sensitive` table is deliberately not joined.
    """
    vac = tables.vacancies.copy()
    vac["query"] = (vac["title"] + " " + vac["activities"] + " " + vac["competencies"]).str.strip()
    vac = vac[vac["query"].str.len() >= min_query_chars]
    vac = vac[["vacancy_id", "title", "query", "requested_month", *VACANCY_FEATURES]].rename(
        columns={"title": "v_title", **{c: f"v_{c}" for c in VACANCY_FEATURES}}
    )

    app = tables.applicants
    app = app[app["cv_text"].str.len() >= min_cv_chars]
    app = app[["candidate_id", "cv_text", *APPLICANT_FEATURES]].rename(
        columns={c: f"a_{c}" for c in APPLICANT_FEATURES}
    )

    cand = tables.candidacies.drop_duplicates(["vacancy_id", "candidate_id"])
    cand = cand[
        ["vacancy_id", "candidate_id", "status", "outcome", "candidacy_month", "updated_month"]
    ]
    pairs = cand.merge(vac, on="vacancy_id").merge(app, on="candidate_id")
    pairs["y"] = (pairs["outcome"] == "hired").astype(int)
    pairs["resolved"] = pairs["outcome"] != "pending"
    for column in ("requested_month", "candidacy_month", "updated_month"):
        pairs[column] = _blank_to_na(pairs[column])
    return pairs.reset_index(drop=True)


def select_setting(pairs: pd.DataFrame, setting: Setting) -> pd.DataFrame:
    """Keep the candidacies of a label setting, and only the vacancies that can be ranked."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; choose one of {SETTINGS}")
    frame = pairs[pairs["resolved"]] if setting == "resolved_only" else pairs
    stats = frame.groupby("vacancy_id")["y"].agg(size="size", hires="sum")
    keep = (stats["hires"] >= 1) & (stats["size"] >= 2)
    if setting == "resolved_only":
        keep &= stats["hires"] < stats["size"]  # someone must have been passed over
    return frame[frame["vacancy_id"].isin(stats.index[keep])].reset_index(drop=True)


def labels_as_of(frame: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """The labels as they stood at the end of the cutoff month.

    Candidacies started after the cutoff did not exist yet. An outcome counts only if it was
    recorded by then; one recorded later, or without a date, is pending at the cutoff.
    """
    existing = frame[frame["candidacy_month"] <= cutoff].copy()
    known = existing["resolved"] & existing["updated_month"].notna()
    known &= existing["updated_month"] <= cutoff
    unknown = existing["resolved"] & ~known
    existing.loc[unknown, "outcome"] = "pending"
    existing.loc[unknown, "y"] = 0
    existing["resolved"] = existing["resolved"] & known
    return existing


def temporal_split(
    pairs: pd.DataFrame,
    setting: Setting,
    test_fraction: float = 0.2,
    cutoff: str | None = None,
    test_until: str | None = None,
) -> Split:
    """Train on earlier vacancies, test on later ones; no shuffling, no label leakage.

    The cutoff is the requested month below which `1 - test_fraction` of the vacancies with at
    least two candidacies fall (a label-free criterion, so no outcome influences it), or the
    month given as `cutoff` ("YYYY-MM"). Training labels are rebuilt as of the cutoff before the
    rankable vacancies are selected. With `test_until`, only vacancies requested up to that month
    are tested (a window, for rolling evaluation).
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if pairs["requested_month"].isna().any():
        raise ValueError("every vacancy needs a requested_month to be split by time")
    sizes = pairs.groupby("vacancy_id").agg(n=("y", "size"), month=("requested_month", "first"))
    months = sizes.loc[sizes["n"] >= 2, "month"].sort_values()
    if cutoff is None:
        if months.nunique() < 2:
            raise ValueError("need at least two distinct months to split by time")
        cutoff = str(months.iloc[int((1 - test_fraction) * len(months)) - 1])
    elif not re.fullmatch(r"\d{4}-\d{2}", cutoff):
        raise ValueError("cutoff must look like 'YYYY-MM'")

    early = pairs["requested_month"] <= cutoff
    train_pool = pairs[early & (pairs["candidacy_month"] <= cutoff)]
    late = ~early
    if test_until is not None:
        late &= pairs["requested_month"] <= test_until
    test_pool = pairs[late]
    train = select_setting(labels_as_of(pairs[early], cutoff), setting)
    test = select_setting(test_pool, setting)
    if train.empty or test.empty:
        raise ValueError("the split left an empty train or test set")
    return Split(
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        cutoff,
        train_pool.reset_index(drop=True),
        test_pool.reset_index(drop=True),
    )


def _shift_month(month: str, delta: int) -> str:
    index = int(month[:4]) * 12 + int(month[5:]) - 1 + delta
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def rolling_splits(
    pairs: pd.DataFrame, setting: Setting, first_cutoff: str, step_months: int = 6
) -> Iterator[Split]:
    """Expanding-window folds: train up to a cutoff, test the next `step_months` of vacancies,
    then move the cutoff forward. Every vacancy is tested at most once, by a model that only
    saw the past. Windows with an empty train or test set are skipped."""
    if step_months < 1:
        raise ValueError("step_months must be at least 1")
    last = str(pairs["requested_month"].max())
    cutoff = first_cutoff
    while cutoff < last:
        until = _shift_month(cutoff, step_months)
        try:
            yield temporal_split(pairs, setting, cutoff=cutoff, test_until=until)
        except ValueError:
            logger.info("window %s..%s skipped: nothing to train or test on", cutoff, until)
        cutoff = until


def audit_attributes(tables: MaskedTables) -> pd.DataFrame:
    """Sex, disability and age band by candidate. For fairness audits; never a feature."""
    return tables.sensitive.copy()


def summarize(frame: pd.DataFrame) -> dict:
    """Aggregates only: safe to print or log."""
    return {
        "vacancies": int(frame["vacancy_id"].nunique()),
        "candidacies": len(frame),
        "hires": int(frame["y"].sum()),
        "hire_rate": round(float(frame["y"].mean()), 4) if len(frame) else 0.0,
        "median_candidates_per_vacancy": float(frame.groupby("vacancy_id").size().median())
        if len(frame)
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the masked data per label setting.")
    parser.add_argument("--masked-dir", type=Path, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--cutoff", default=None, help="last training month, e.g. 2022-06")
    args = parser.parse_args()

    pairs = build_pairs(load_masked(args.masked_dir))
    logger.info("pairs with a vacancy text, a CV and an outcome record: %s", summarize(pairs))
    for setting in SETTINGS:
        split = temporal_split(pairs, setting, args.test_fraction, args.cutoff)
        logger.info(
            "%s | split at %s | train %s | test %s",
            setting,
            split.cutoff,
            summarize(split.train),
            summarize(split.test),
        )


if __name__ == "__main__":
    main()
