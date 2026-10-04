"""Which features carry signal? Each feature alone, used as a ranking rule on the test period.

    uv run python -m src.feature_report [--embed]

Prints aggregates only (hit@1 gain over a random order, with a 95% bootstrap interval over
vacancies); no text or id leaves the masked tables. A feature that is constant inside a
vacancy (such as the number of candidates) cannot rank anyone and is left out.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import PROCESS_ARTIFACTS, Encoder, FeatureBuilder
from src.masked_data import SETTINGS, build_pairs, load_masked, select_setting, temporal_split
from src.rank_metrics import gain_with_ci, per_vacancy_metrics, random_metrics
from src.utils import logger

NOTE = "  <- process artifact, not used by the model"
SINGLE = [
    "tfidf_word", "tfidf_char", "title_sim", "query_coverage", "skills_coverage", "e5_cosine",
    "area_exact", "area_group", "sap_match", "seniority_in_cv", "education_gap", "english_gap",
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


def single_feature_table(test: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    random = random_metrics(test)["hit@1"]
    rows = []
    for name in SINGLE:
        if name not in feats:
            continue
        values = feats[name]
        if values.groupby(test["vacancy_id"]).nunique().max() <= 1:
            continue  # constant inside every vacancy: it cannot rank
        scores = values.fillna(values.min() - 1.0).to_numpy()
        hit1 = per_vacancy_metrics(test, scores)["hit@1"]
        gain, low, high = gain_with_ci(hit1, random)
        rows.append({"feature": name, "hit@1": hit1.mean(), "gain": gain, "low": low, "high": high})
    return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-dir", type=Path, default=None)
    parser.add_argument("--embed", action="store_true", help="add the e5 embedding cosine (slow)")
    args = parser.parse_args()

    pairs = build_pairs(load_masked(args.masked_dir))
    encoder = e5_encoder() if args.embed else None
    for setting in SETTINGS:
        split = temporal_split(select_setting(pairs, setting))
        builder = FeatureBuilder(encoder=encoder).fit(split.train)
        feats = builder.transform(split.test, history=pairs)
        random = random_metrics(split.test)["hit@1"].mean()
        logger.info(
            "%s | test: %d vacancies, %d candidacies | cutoff %s | random hit@1 %.3f",
            setting,
            split.test["vacancy_id"].nunique(),
            len(split.test),
            split.cutoff,
            random,
        )
        for row in single_feature_table(split.test, feats).to_dict("records"):
            gain, low, high = (100 * row[k] for k in ("gain", "low", "high"))
            logger.info(
                "  %-20s hit@1 %.3f  gain %+.1f pts [%+.1f, %+.1f]%s",
                row["feature"], row["hit@1"], gain, low, high,
                NOTE if row["feature"] in PROCESS_ARTIFACTS else "",
            )  # fmt: skip


if __name__ == "__main__":
    main()
