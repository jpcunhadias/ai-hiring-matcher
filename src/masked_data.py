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
ones. A training row is dropped if its outcome was only recorded after the cutoff, because in
production that label would not be known yet.

Sex, disability and age band live in the separate `sensitive` table. They are not joined into
the pairs, so they cannot become features by accident; `audit_attributes` returns them for
fairness checks only.
"""

import argparse
import os
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
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: str  # last requested month that belongs to the training period


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
    vac = vac[["vacancy_id", "query", "requested_month", *VACANCY_FEATURES]].rename(
        columns={c: f"v_{c}" for c in VACANCY_FEATURES}
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


def temporal_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> Split:
    """Train on earlier vacancies, test on later ones; no random shuffling, no label leakage.

    The cutoff is the requested month below which `1 - test_fraction` of the vacancies fall.
    Vacancies requested after the cutoff are the test set. Training rows whose outcome was
    recorded after the cutoff are dropped: that label would not exist yet at training time.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if frame["requested_month"].isna().any():
        raise ValueError("every vacancy needs a requested_month to be split by time")
    months = frame.drop_duplicates("vacancy_id")["requested_month"].sort_values()
    if months.nunique() < 2:
        raise ValueError("need at least two distinct months to split by time")
    cutoff = str(months.iloc[int((1 - test_fraction) * len(months)) - 1])
    in_train = frame["requested_month"] <= cutoff
    known = frame["updated_month"].isna() | (frame["updated_month"] <= cutoff)
    train, test = frame[in_train & known], frame[~in_train]
    if train.empty or test.empty:
        raise ValueError("the split left an empty train or test set")
    return Split(train.reset_index(drop=True), test.reset_index(drop=True), cutoff)


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
    args = parser.parse_args()

    pairs = build_pairs(load_masked(args.masked_dir))
    logger.info("pairs with a vacancy text, a CV and an outcome record: %s", summarize(pairs))
    for setting in SETTINGS:
        frame = select_setting(pairs, setting)
        split = temporal_split(frame, args.test_fraction)
        logger.info("%s: %s", setting, summarize(frame))
        logger.info(
            "  split at %s | train %s | test %s",
            split.cutoff,
            summarize(split.train),
            summarize(split.test),
        )


if __name__ == "__main__":
    main()
