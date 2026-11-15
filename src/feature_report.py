"""Which features carry signal? Each feature alone, used as a ranking rule on the test period.

    uv run python -m src.feature_report [--embed]

Prints aggregates only (hit@1 gain over a random order, with a 95% bootstrap interval over
vacancies); no text or id leaves the masked tables. A feature that is constant inside a
vacancy (such as the number of candidates) cannot rank anyone and is left out.

The same tables are written to reports/feature_report.md (`--report`), with the masked-data
generation in its header.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import run_report
from src.features import PROCESS_ARTIFACTS, Encoder, FeatureBuilder
from src.masked_data import SETTINGS, build_pairs, load_masked, temporal_split
from src.rank_metrics import gain_with_ci, per_vacancy_metrics, random_metrics
from src.utils import REPORTS_DIR, logger

NOTE = "  <- process artifact, not used by the model"
SINGLE = [
    "tfidf_word", "tfidf_char", "title_sim", "query_coverage", "skills_coverage", "e5_cosine",
    "area_exact", "area_group", "sap_match", "seniority_in_cv", "education_gap", "english_gap",
    "spanish_gap", "client_prior_hire_rate",
    "a_education_filled", "a_english_filled", "a_area_filled", "a_skills_filled", "a_title_filled",
    "cv_len_log", "prior_candidacies", "prior_hires", "lag_months",
]  # fmt: skip


def e5_encoder(model_name: str = "intfloat/multilingual-e5-small") -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    model.max_seq_length = 512

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )

    return encode


RESULT_COLUMNS = ["feature", "hit@1", "gain", "low", "high"]
REPORT_PATH = REPORTS_DIR / "feature_report.md"


def single_feature_table(test: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """One row per feature that can rank candidates inside at least one vacancy.

    A missing value ranks below every observed one, so a feature is judged constant only after
    that rule is applied: a vacancy where one candidate has a value and the rest are missing
    still gets a ranking.
    """
    random = random_metrics(test)["hit@1"]
    rows = []
    for name in SINGLE:
        if name not in feats or feats[name].notna().sum() == 0:
            continue
        values = feats[name]
        scores = values.fillna(values.min() - 1.0)
        if scores.groupby(test["vacancy_id"]).nunique().max() <= 1:
            continue  # constant inside every vacancy: it cannot rank
        hit1 = per_vacancy_metrics(test, scores.to_numpy())["hit@1"]
        gain, low, high = gain_with_ci(hit1, random)
        rows.append({"feature": name, "hit@1": hit1.mean(), "gain": gain, "low": low, "high": high})
    table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return table.sort_values("gain", ascending=False).reset_index(drop=True)


def report_section(setting: str, split_info: dict, table: pd.DataFrame) -> tuple[str, str]:
    """One label setting's single-feature table, with the test period it was measured on."""
    shown = pd.DataFrame(
        {
            "feature": [
                f + (" (process artifact)" if f in PROCESS_ARTIFACTS else "")
                for f in table["feature"]
            ],
            "hit@1": table["hit@1"].map("{:.3f}".format),
            "gain@1 (pts)": table["gain"].map(run_report.pts),
            "95% CI": [run_report.interval(lo, hi) for lo, hi in zip(table["low"], table["high"])],
        }
    )
    intro = (
        f"Test: {split_info['vacancies']} vacancies, {split_info['candidacies']} candidacies, "
        f"after cutoff {split_info['cutoff']}; random hit@1 {split_info['random']:.3f}."
    )
    return setting, intro + "\n\n" + run_report.markdown_table(shown)


def write_report(path: Path, sections: list[tuple[str, str]], generation: str, embed: bool) -> Path:
    meta = {
        "masked-data generation": generation or run_report.NO_GENERATION,
        "commit": run_report.git_commit(),
        "e5 embedding": embed,
        "bootstrap": "2000 resamples of the test vacancies, seed 0",
    }
    return run_report.write(
        path, run_report.render("Single-feature ranking rules (src.feature_report)", meta, sections)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-dir", type=Path, default=None)
    parser.add_argument("--embed", action="store_true", help="add the e5 embedding cosine (slow)")
    parser.add_argument("--report", type=Path, default=REPORT_PATH, help="aggregate tables")
    args = parser.parse_args()

    tables = load_masked(args.masked_dir)
    pairs = build_pairs(tables)
    encoder = e5_encoder() if args.embed else None
    sections = []
    for setting in SETTINGS:
        split = temporal_split(pairs, setting)
        builder = FeatureBuilder(encoder=encoder).fit(split.train)
        feats = builder.transform(split.test, history=pairs, pool=split.test_pool)
        random = random_metrics(split.test)["hit@1"].mean()
        logger.info(
            "%s | test: %d vacancies, %d candidacies | cutoff %s | random hit@1 %.3f",
            setting,
            split.test["vacancy_id"].nunique(),
            len(split.test),
            split.cutoff,
            random,
        )
        table = single_feature_table(split.test, feats)
        info = {
            "vacancies": split.test["vacancy_id"].nunique(),
            "candidacies": len(split.test),
            "cutoff": split.cutoff,
            "random": random,
        }
        sections.append(report_section(setting, info, table))
        for row in table.to_dict("records"):
            gain, low, high = (100 * row[k] for k in ("gain", "low", "high"))
            logger.info(
                "  %-20s hit@1 %.3f  gain %+.1f pts [%+.1f, %+.1f]%s",
                row["feature"], row["hit@1"], gain, low, high,
                NOTE if row["feature"] in PROCESS_ARTIFACTS else "",
            )  # fmt: skip
    written = write_report(args.report, sections, tables.generation, args.embed)
    logger.info("aggregate tables written to %s", written)


if __name__ == "__main__":
    main()
