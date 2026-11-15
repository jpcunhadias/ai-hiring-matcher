"""Supervised rankers, evaluated over rolling time folds on the masked real data.

    uv run python -m src.ranker --first-cutoff 2021-06 [--step 6] [--embed] [--seed 0]

The aggregate tables are also written to reports/ranker.md (`--report`), with the masked-data
generation, the seeds and the options in its header, so a cited number points at a run.

Each fold trains on every vacancy up to a cutoff, with labels as they stood then, and is tested
on the next window of vacancies with their final labels. Every vacancy is tested at most once,
by a model that only saw the past. Results are pooled over folds (aggregates only) and compared
with a random order, the TF-IDF zero-shot rule and the `lag_months` process artifact, which is
reported as the "+ lag" ablation and never part of the clean model.

Pointwise models: a hire-probability score per (vacancy, candidate), ranked inside the vacancy.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import run_report
from src.feature_report import e5_encoder
from src.features import FeatureBuilder, model_features
from src.masked_data import SETTINGS, Setting, build_pairs, load_masked, rolling_splits
from src.rank_metrics import gain_with_ci, per_vacancy_metrics, random_metrics
from src.utils import REPORTS_DIR, logger

KINDS = ("logistic", "boosted")
REPORT_PATH = REPORTS_DIR / "ranker.md"
PER_FOLD_METHODS = ("boosted trees, centered", "tfidf zero-shot")
FOLD_COLUMNS = ["cutoff", "train_vacancies", "test_vacancies", "test_candidacies"]


class Ranker:
    """A pointwise hire-probability model whose score is used to rank within a vacancy."""

    def __init__(self, kind: str = "logistic", c: float = 0.1, seed: int = 0) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; choose one of {KINDS}")
        self.kind, self.c, self.seed = kind, c, seed
        self.columns: list[str] = []
        self._model: Pipeline | HistGradientBoostingClassifier | None = None

    def _build(self) -> Pipeline | HistGradientBoostingClassifier:
        if self.kind == "logistic":
            return Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=self.c, max_iter=2000)),
                ]
            )
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
            l2_regularization=1.0, random_state=self.seed,
        )  # fmt: skip

    def fit(self, features: pd.DataFrame, y: pd.Series | np.ndarray) -> Self:
        if len(np.unique(y)) < 2:
            raise ValueError("training needs both hired and not-hired candidacies")
        # a column that is missing for every training row carries nothing (and breaks the
        # boosted model), so it is left out; scoring then ignores it too
        self.columns = [c for c in features.columns if features[c].notna().any()]
        if not self.columns:
            raise ValueError("every feature is missing for every training row")
        self._model = self._build().fit(features[self.columns].to_numpy(dtype=float), np.asarray(y))
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() first")
        missing = set(self.columns) - set(features.columns)
        if missing:
            raise ValueError(f"missing feature columns: {sorted(missing)}")
        return self._model.predict_proba(features[self.columns].to_numpy(dtype=float))[:, 1]

    @property
    def coefficients(self) -> pd.Series:
        """Standardized logistic weights (one per feature); only the logistic model has them."""
        if self.kind != "logistic" or self._model is None:
            raise ValueError("coefficients exist only for a fitted logistic ranker")
        weights = self._model.named_steps["model"].coef_[0]  # type: ignore[union-attr]
        return pd.Series(weights, index=self.columns)


@dataclass
class RollingResult:
    folds: list[dict] = field(default_factory=list)
    per_vacancy: dict[str, list[pd.DataFrame]] = field(default_factory=dict)
    coefficients: list[pd.Series] = field(default_factory=list)

    def pooled(self, method: str) -> pd.DataFrame:
        return pd.concat(self.per_vacancy[method])


def center_within_vacancy(features: pd.DataFrame, vacancy_ids: pd.Series) -> pd.DataFrame:
    """Each feature minus its mean over the same vacancy. A pointwise model on raw features also
    learns vacancy-level effects (how many candidates, how often this client hires) that cannot
    help rank candidates inside one vacancy; centering leaves only differences between them."""
    means = features.groupby(vacancy_ids.to_numpy()).transform("mean")
    return features - means


def _method_scores(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    seed: int,
    train_groups: pd.Series | None = None,
    test_groups: pd.Series | None = None,
) -> tuple[dict[str, np.ndarray], pd.Series]:
    """Scores of every compared method on the test rows, plus the clean logistic weights.

    The "centered" variants (features minus their vacancy mean) need the vacancy ids."""
    clean = model_features(train_x)
    with_lag = model_features(train_x, include_process=True)
    scores: dict[str, np.ndarray] = {
        "tfidf zero-shot": test_x["tfidf_word"].to_numpy(),
        "lag_months alone (artifact)": test_x["lag_months"].fillna(-1).to_numpy(),
    }
    weights = pd.Series(dtype=float)
    for kind, label in (("logistic", "logistic"), ("boosted", "boosted trees")):
        for columns, suffix in ((clean, ""), (with_lag, " + lag")):
            ranker = Ranker(kind, seed=seed).fit(train_x[columns], train_y)
            scores[label + suffix] = ranker.score(test_x)
            if kind == "logistic" and not suffix:
                weights = ranker.coefficients
        if train_groups is not None and test_groups is not None:
            centered = Ranker(kind, seed=seed).fit(
                center_within_vacancy(train_x[clean], train_groups), train_y
            )
            scores[f"{label}, centered"] = centered.score(
                center_within_vacancy(test_x[clean], test_groups)
            )
    return scores, weights


def evaluate_rolling(
    pairs: pd.DataFrame,
    setting: Setting,
    first_cutoff: str,
    step_months: int = 6,
    make_builder: Callable[[], FeatureBuilder] = FeatureBuilder,
    seed: int = 0,
) -> RollingResult:
    result = RollingResult()
    for split in rolling_splits(pairs, setting, first_cutoff, step_months):
        if split.train["y"].nunique() < 2:
            continue
        builder = make_builder().fit(split.train)
        train_x = builder.transform(split.train, history=pairs, pool=split.train_pool)
        test_x = builder.transform(split.test, history=pairs, pool=split.test_pool)
        scores, weights = _method_scores(
            train_x,
            split.train["y"],
            test_x,
            seed,
            split.train["vacancy_id"],
            split.test["vacancy_id"],
        )
        result.folds.append(
            {
                "cutoff": split.cutoff,
                "train_vacancies": split.train["vacancy_id"].nunique(),
                "test_vacancies": split.test["vacancy_id"].nunique(),
                "test_candidacies": len(split.test),
            }
        )
        result.per_vacancy.setdefault("random", []).append(random_metrics(split.test))
        for name, values in scores.items():
            result.per_vacancy.setdefault(name, []).append(per_vacancy_metrics(split.test, values))
        result.coefficients.append(weights)
    return result


def summarize(result: RollingResult) -> pd.DataFrame:
    """Pooled hit@1 / hit@3 / MRR per method, with the gain over random and a bootstrap
    interval over all tested vacancies, and the paired difference to the TF-IDF zero-shot rule."""
    columns = ["method", "hit@1", "gain@1", "low", "high", "gain@3", "gain_mrr",
               "vs_tfidf", "vs_low", "vs_high"]  # fmt: skip
    if not result.folds:
        return pd.DataFrame(columns=columns)
    random = result.pooled("random")
    zero_shot = result.pooled("tfidf zero-shot")["hit@1"]
    rows = []
    for method in result.per_vacancy:
        pooled = result.pooled(method)
        gain1, low, high = gain_with_ci(pooled["hit@1"], random["hit@1"])
        versus, v_low, v_high = gain_with_ci(pooled["hit@1"], zero_shot)
        rows.append(
            {
                "method": method,
                "hit@1": pooled["hit@1"].mean(),
                "gain@1": gain1,
                "low": low,
                "high": high,
                "gain@3": float((pooled["hit@3"] - random["hit@3"]).mean()),
                "gain_mrr": float((pooled["mrr"] - random["mrr"]).mean()),
                "vs_tfidf": versus,
                "vs_low": v_low,
                "vs_high": v_high,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def fold_gains(result: RollingResult, method: str) -> list[dict]:
    """hit@1 gain over random inside each fold: is a pooled gain consistent over time?"""
    return [
        {
            "cutoff": fold["cutoff"],
            "vacancies": fold["test_vacancies"],
            "gain": float(
                (
                    result.per_vacancy[method][i]["hit@1"]
                    - result.per_vacancy["random"][i]["hit@1"]
                ).mean()
            ),
        }
        for i, fold in enumerate(result.folds)
    ]


def mean_coefficients(result: RollingResult) -> pd.Series:
    """Average standardized logistic weight per feature over folds, largest magnitude first."""
    frame = pd.concat(result.coefficients, axis=1)
    return frame.mean(axis=1).sort_values(key=np.abs, ascending=False)


def report_sections(setting: str, result: RollingResult) -> list[tuple[str, str]]:
    """The aggregate tables of one label setting, as report sections."""
    folds = pd.DataFrame(result.folds, columns=FOLD_COLUMNS)
    table = summarize(result)
    shown = pd.DataFrame(
        {
            "method": table["method"],
            "hit@1": table["hit@1"].map("{:.3f}".format),
            "gain@1 (pts)": table["gain@1"].map(run_report.pts),
            "95% CI": [run_report.interval(lo, hi) for lo, hi in zip(table["low"], table["high"])],
            "gain@3 (pts)": table["gain@3"].map(run_report.pts),
            "gain MRR": table["gain_mrr"].map("{:+.3f}".format),
            "vs TF-IDF (pts)": table["vs_tfidf"].map(run_report.pts),
            "vs TF-IDF 95% CI": [
                run_report.interval(lo, hi) for lo, hi in zip(table["vs_low"], table["vs_high"])
            ],
        }
    )
    sections = [
        (f"{setting}: folds", run_report.markdown_table(folds)),
        (
            f"{setting}: gain in top-1 hit rate over a random order, pooled over folds",
            run_report.markdown_table(shown),
        ),
    ]
    if result.folds:
        per_fold = pd.DataFrame(
            {
                "cutoff": [f["cutoff"] for f in result.folds],
                "test vacancies": [f["test_vacancies"] for f in result.folds],
                **{
                    f"{m} (pts)": [run_report.pts(g["gain"]) for g in fold_gains(result, m)]
                    for m in PER_FOLD_METHODS
                    if m in result.per_vacancy
                },
            }
        )
        sections.append((f"{setting}: gain per fold", run_report.markdown_table(per_fold)))
    if result.coefficients:
        weights = mean_coefficients(result).head(8)
        sections.append(
            (
                f"{setting}: clean logistic, mean standardized weight over folds",
                run_report.markdown_table(
                    pd.DataFrame(
                        {"feature": weights.index, "weight": weights.map("{:+.3f}".format)}
                    )
                ),
            )
        )
    return sections


def write_report(
    path: Path,
    results: dict[str, RollingResult],
    generation: str,
    options: dict[str, object],
) -> Path:
    meta = {
        "masked-data generation": generation or run_report.NO_GENERATION,
        "commit": run_report.git_commit(),
        **options,
        "bootstrap": "2000 resamples of the test vacancies, seed 0",
    }
    sections = [s for setting, result in results.items() for s in report_sections(setting, result)]
    return run_report.write(
        path, run_report.render("Rolling-fold rankers (src.ranker)", meta, sections)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked-dir", type=Path, default=None)
    parser.add_argument(
        "--first-cutoff", default="2021-06", help="end of the first training period"
    )
    parser.add_argument("--step", type=int, default=6, help="months per test window")
    parser.add_argument("--embed", action="store_true", help="add the e5 embedding cosine (slow)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the boosted models")
    parser.add_argument("--report", type=Path, default=REPORT_PATH, help="aggregate tables")
    args = parser.parse_args()

    tables = load_masked(args.masked_dir)
    pairs = build_pairs(tables)
    encoder = e5_encoder() if args.embed else None
    results: dict[str, RollingResult] = {}
    for setting in SETTINGS:
        result = evaluate_rolling(
            pairs,
            setting,
            args.first_cutoff,
            args.step,
            lambda: FeatureBuilder(encoder=encoder),
            seed=args.seed,
        )
        results[setting] = result
        total = sum(f["test_vacancies"] for f in result.folds)
        logger.info("%s | %d folds, %d test vacancies in total", setting, len(result.folds), total)
        for fold in result.folds:
            logger.info(
                "  cutoff %s: train %d vacancies | test %d vacancies, %d candidacies",
                fold["cutoff"], fold["train_vacancies"], fold["test_vacancies"],
                fold["test_candidacies"],
            )  # fmt: skip
        for row in summarize(result).to_dict("records"):
            gain, low, high = (100 * row[k] for k in ("gain@1", "low", "high"))
            note = "  <- uses the lag artifact" if "lag" in row["method"] else ""
            logger.info(
                "  %-26s hit@1 %.3f  gain %+.1f pts [%+.1f, %+.1f]  hit@3 %+.1f  MRR %+.3f"
                "  | vs tfidf %+.1f [%+.1f, %+.1f]%s",
                row["method"], row["hit@1"], gain, low, high, 100 * row["gain@3"], row["gain_mrr"],
                100 * row["vs_tfidf"], 100 * row["vs_low"], 100 * row["vs_high"], note,
            )  # fmt: skip
        for method in PER_FOLD_METHODS:
            per_fold = [
                f"{g['cutoff']} ({g['vacancies']}v) {100 * g['gain']:+.0f}"
                for g in fold_gains(result, method)
            ]
            logger.info("  per-fold gain, %s: %s", method, "; ".join(per_fold))
        if result.coefficients:
            top = mean_coefficients(result).head(8)
            logger.info("  clean logistic, mean standardized weight: %s", top.round(3).to_dict())
    options = {
        "model seed": args.seed,
        "first cutoff": args.first_cutoff,
        "step (months)": args.step,
        "e5 embedding": args.embed,
    }
    written = write_report(args.report, results, tables.generation, options)
    logger.info("aggregate tables written to %s", written)


if __name__ == "__main__":
    main()
