"""Pairwise features for ranking candidates within a vacancy (masked real data).

Fit on the training period only, then transform any frame from `src.masked_data`:

    builder = FeatureBuilder().fit(split.train)
    X = builder.transform(split.test, history=all_pairs, pool=split.test_pool)

Four families (docs/real-data.md): text match, structured match, within-vacancy context, and
history. History is computed strictly from the past: a row at month t only sees candidacies
started before t, and an outcome only once it had been recorded before t. The within-vacancy
context (count, z-score, rank) is computed over the vacancy's whole pool, whatever the outcome of
its members, so it cannot depend on who happened to be resolved; it describes the completed
pool, which makes the evaluation a retrospective ranking of that pool.

Never features: sex, disability, age band (separate table), status and updated_month (they are
the outcome), the surrogate ids.
"""

import re
from collections.abc import Callable
from typing import Self

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.masked_data import APPLICANT_FEATURES, VACANCY_FEATURES
from src.masking import fold

Encoder = Callable[[list[str]], np.ndarray]  # texts -> L2-normalized embeddings

_TOKEN = re.compile(r"\w{3,}")
MAX_COMMON_VACANCY_SHARE = 0.2  # tokens in more than this share of vacancies are generic
SMOOTHING = 5.0  # pseudo-candidacies pulling a client's hire rate towards the global rate

_LANGUAGE = {
    "nenhum": 0,
    "basico": 1,
    "intermediario": 2,
    "tecnico": 2,
    "avancado": 3,
    "fluente": 4,
}
# most specific first: "Pós Doutorado" is a doctorate, "Ensino Médio Técnico" is technical
_LEVEL = [("doutorado", 6), ("mestrado", 5), ("pos", 4), ("superior", 3), ("tecnico", 2.5),
          ("medio", 2), ("fundamental", 1)]  # fmt: skip
_UNFINISHED = ("cursando", "incompleto")
_TEXT_COLUMNS = [
    "v_title", "query", "cv_text",
    *(f"v_{c}" for c in VACANCY_FEATURES), *(f"a_{c}" for c in APPLICANT_FEATURES),
]  # fmt: skip

TEXT_SCORES = ["tfidf_word", "tfidf_char", "title_sim", "query_coverage", "skills_coverage"]
CONTEXT_SCORES = [*TEXT_SCORES, "cv_len_log"]
STRUCTURED = [
    "education_gap", "english_gap", "spanish_gap", "area_exact", "area_group", "sap_match",
    "seniority_in_cv", "a_education_filled", "a_english_filled", "a_area_filled",
    "a_skills_filled", "a_title_filled", "cv_len_log",
]  # fmt: skip
HISTORY = ["prior_candidacies", "prior_hires", "client_prior_hire_rate", "lag_months"]
# Funnel dynamics, not candidate quality: recruiters keep adding candidates until someone is
# hired, so the hire tends to be the latest one added (68% of the time against a 40% base rate
# in resolved vacancies). Reported as a diagnostic, left out of the model by default.
PROCESS_ARTIFACTS = ["lag_months"]


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(fold(text)))


def month_index(month: str | None) -> float:
    """'2021-03' -> a comparable integer; missing -> NaN."""
    if not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month):
        return float("nan")
    return int(month[:4]) * 12 + int(month[5:]) - 1


def _clean(value: object) -> str:
    """A missing value (None, NaN, pd.NA) is the empty string; anything else is its text."""
    return "" if value is None or pd.isna(value) else str(value)


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy whose text and category columns never hold a missing value."""
    out = frame.copy()
    for column in _TEXT_COLUMNS:
        if column in out:
            out[column] = out[column].astype(object).where(out[column].notna(), "")
    return out


def education_rank(value: object) -> float:
    text = fold(_clean(value))
    if not text:
        return float("nan")
    for key, rank in _LEVEL:
        if key in text:
            return rank - (0.5 if any(w in text for w in _UNFINISHED) else 0.0)
    return float("nan")


def language_rank(value: object) -> float:
    return float(_LANGUAGE.get(fold(_clean(value)), float("nan")))


def _area_parts(areas: object) -> list[str]:
    """'TI - Projetos-TI - SAP-' -> ['ti - projetos', 'ti - sap']: entries are separated by a
    hyphen without surrounding spaces, and a name may itself contain ' - '."""
    return [p.strip() for p in re.split(r"(?<!\s)-(?!\s)", fold(_clean(areas))) if p.strip()]


def _cosine(left, right, rows_left: np.ndarray, rows_right: np.ndarray) -> np.ndarray:
    """Row-wise cosine of two L2-normalized sparse matrices at the given row positions."""
    return np.asarray(left[rows_left].multiply(right[rows_right]).sum(axis=1)).ravel()


def _within(frame: pd.DataFrame, values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Z-score and percentile rank of a score among the candidates of the same vacancy."""
    grouped = values.groupby(frame["vacancy_id"])
    std = grouped.transform("std").fillna(0.0)
    z = ((values - grouped.transform("mean")) / std.where(std > 0, 1.0)).where(std > 0, 0.0)
    return z, grouped.rank(pct=True)


class FeatureBuilder:
    """Fit TF-IDF vocabularies on the training pairs; transform builds the feature frame."""

    def __init__(
        self,
        encoder: Encoder | None = None,
        char_max_features: int = 100_000,
        min_df: int = 2,
        max_df: float = 0.5,
        common_share: float = MAX_COMMON_VACANCY_SHARE,
    ) -> None:
        self.encoder = encoder
        self.char_max_features = char_max_features
        self.min_df = min_df
        self.max_df = max_df
        self.common_share = common_share
        self._word: TfidfVectorizer | None = None
        self._char: TfidfVectorizer | None = None
        self._common: set[str] = set()
        self.global_hire_rate = 0.0

    def fit(self, train: pd.DataFrame) -> Self:
        vacancies = train.drop_duplicates("vacancy_id")["query"].tolist()
        cvs = train.drop_duplicates("candidate_id")["cv_text"].tolist()
        corpus = vacancies + cvs
        self._word = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=150_000,
        ).fit(corpus)
        self._char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True, strip_accents="unicode",
            min_df=self.min_df + 1, max_df=self.max_df, max_features=self.char_max_features,
        ).fit(corpus)  # fmt: skip
        df: dict[str, int] = {}
        for text in vacancies:
            for t in tokens(text):
                df[t] = df.get(t, 0) + 1
        limit = self.common_share * max(len(vacancies), 1)
        self._common = {t for t, n in df.items() if n > limit}
        self.global_hire_rate = float(train["y"].mean())
        return self

    # ------------------------------------------------------------------ text match

    def _text_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self._word is not None and self._char is not None, "call fit() first"
        v_ids, v_pos = np.unique(frame["vacancy_id"], return_inverse=True)
        c_ids, c_pos = np.unique(frame["candidate_id"], return_inverse=True)
        queries = frame.drop_duplicates("vacancy_id").set_index("vacancy_id").loc[v_ids]
        cvs = frame.drop_duplicates("candidate_id").set_index("candidate_id").loc[c_ids]
        out = pd.DataFrame(index=frame.index)
        for name, vec in (("tfidf_word", self._word), ("tfidf_char", self._char)):
            out[name] = _cosine(
                vec.transform(queries["query"]), vec.transform(cvs["cv_text"]), v_pos, c_pos
            )
        titles = frame["a_professional_title"].fillna("")
        out["title_sim"] = np.asarray(
            self._word.transform(frame["v_title"])
            .multiply(self._word.transform(titles))
            .sum(axis=1)
        ).ravel()
        out["query_coverage"] = self._coverage(frame["query"], frame["cv_text"])
        out["skills_coverage"] = self._coverage(frame["query"], frame["a_technical_skills"])
        if self.encoder is not None:
            q = self.encoder(["query: " + t for t in queries["query"]])
            c = self.encoder(["passage: " + t for t in cvs["cv_text"]])
            out["e5_cosine"] = np.einsum("ij,ij->i", q[v_pos], c[c_pos])
        return out

    def _coverage(self, queries: pd.Series, others: pd.Series) -> np.ndarray:
        """Share of a vacancy's distinctive words that the other text also contains."""
        cache: dict[str, set[str]] = {}

        def distinct(text: str) -> set[str]:
            if text not in cache:
                cache[text] = tokens(text) - self._common
            return cache[text]

        result = []
        for query, other in zip(queries, others.fillna(""), strict=True):
            wanted = distinct(query)
            result.append(len(wanted & tokens(other)) / len(wanted) if wanted else 0.0)
        return np.asarray(result)

    # ------------------------------------------------------------- structured match

    @staticmethod
    def _structured_features(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        out["education_gap"] = frame["a_education_level"].map(education_rank) - frame[
            "v_education_level"
        ].map(education_rank)
        for lang in ("english", "spanish"):
            out[f"{lang}_gap"] = frame[f"a_{lang}_level"].map(language_rank) - frame[
                f"v_{lang}_level"
            ].map(language_rank)
        vac_parts = frame["v_areas"].map(_area_parts)
        app_area = frame["a_area"].fillna("").map(fold)
        out["area_exact"] = [
            float(bool(a) and a in parts) for a, parts in zip(app_area, vac_parts, strict=True)
        ]
        groups = app_area.str.split(" - ").str[0]
        out["area_group"] = [
            float(bool(g) and any(p == g or p.startswith(f"{g} - ") for p in parts))
            for g, parts in zip(groups, vac_parts, strict=True)
        ]
        out["sap_match"] = (
            (frame["v_is_sap"] == "Sim") & app_area.str.contains(r"\bsap\b", regex=True)
        ).astype(float)
        cues = frame["v_seniority"].fillna("").map(fold)
        haystack = (frame["cv_text"] + " " + frame["a_professional_title"].fillna("")).map(fold)
        out["seniority_in_cv"] = [
            float(bool(c) and c in text) for c, text in zip(cues, haystack, strict=True)
        ]
        for name, column in (
            ("a_education_filled", "a_education_level"),
            ("a_english_filled", "a_english_level"),
            ("a_area_filled", "a_area"),
            ("a_skills_filled", "a_technical_skills"),
            ("a_title_filled", "a_professional_title"),
        ):
            out[name] = (frame[column].fillna("") != "").astype(float)
        out["cv_len_log"] = np.log1p(frame["cv_text"].str.len())
        return out

    # ----------------------------------------------------------------------- history

    def _history_features(self, frame: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        now = frame["candidacy_month"].map(month_index)
        out["lag_months"] = now - frame["requested_month"].map(month_index)
        if history is None:
            out[["prior_candidacies", "prior_hires"]] = np.nan
            out["client_prior_hire_rate"] = np.nan
            return out
        h = history.assign(
            start=history["candidacy_month"].map(month_index),
            known=history["updated_month"].map(month_index),
        )
        hired = h[h["y"] == 1]
        resolved = h[h["resolved"]]
        out["prior_candidacies"] = _count_before(
            frame["candidate_id"], now, h, "candidate_id", "start"
        )
        out["prior_hires"] = _count_before(
            frame["candidate_id"], now, hired, "candidate_id", "known"
        )
        c_all = _count_before(frame["v_client_id"], now, resolved, "v_client_id", "known")
        c_hired = _count_before(
            frame["v_client_id"], now, resolved[resolved["y"] == 1], "v_client_id", "known"
        )
        rate = self.global_hire_rate
        out["client_prior_hire_rate"] = (c_hired + SMOOTHING * rate) / (c_all + SMOOTHING)
        return out

    # --------------------------------------------------------------------- transform

    def _compute(self, frame: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
        text = self._text_features(frame)
        feats = pd.concat(
            [text, self._structured_features(frame), self._history_features(frame, history)],
            axis=1,
        )
        feats["n_candidates"] = frame.groupby("vacancy_id")["candidate_id"].transform("size")
        scores = [*CONTEXT_SCORES, *(["e5_cosine"] if "e5_cosine" in feats else [])]
        for name in scores:
            feats[f"{name}_z"], feats[f"{name}_rank"] = _within(frame, feats[name])
        return feats

    def transform(
        self,
        frame: pd.DataFrame,
        history: pd.DataFrame | None = None,
        pool: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Feature frame aligned with `frame`'s index.

        `history` is every pair known to the loader (not only the evaluated setting); only its
        strict past is used per row. `pool` is every candidacy of the vacancies in `frame`,
        whatever its outcome: the within-vacancy context is computed over it, so filtering
        `frame` by outcome (resolved-only) cannot change the context. Defaults to `frame`.
        """
        frame = _normalized(frame)
        if pool is None:
            return self._compute(frame, history)
        keys = ["vacancy_id", "candidate_id"]
        members = _normalized(pool[pool["vacancy_id"].isin(frame["vacancy_id"])]).reset_index(
            drop=True
        )
        index = pd.MultiIndex.from_frame(members[keys])
        if not index.is_unique:
            raise ValueError("the pool has duplicate (vacancy, candidate) pairs")
        positions = index.get_indexer(pd.MultiIndex.from_frame(frame[keys]))
        if (positions < 0).any():
            raise ValueError("every evaluated row must be part of the pool")
        feats = self._compute(members, history).iloc[positions]
        feats.index = frame.index
        return feats


def _count_before(
    keys: pd.Series, times: pd.Series, source: pd.DataFrame, key_column: str, time_column: str
) -> np.ndarray:
    """For each (key, time): rows of `source` with that key whose time is strictly earlier."""
    groups = {
        key: np.sort(group[time_column].dropna().to_numpy())
        for key, group in source.groupby(key_column)
    }
    empty = np.array([])
    return np.array(
        [
            np.searchsorted(groups.get(k, empty), t, side="left") if t == t else np.nan
            for k, t in zip(keys, times, strict=True)
        ],
        dtype=float,
    )


def model_features(features: pd.DataFrame, include_process: bool = False) -> list[str]:
    """Columns to hand to a model. Process artifacts only when asked for, as an ablation."""
    drop = set() if include_process else set(PROCESS_ARTIFACTS)
    drop |= {f"{name}_{kind}" for name in PROCESS_ARTIFACTS for kind in ("z", "rank")} & set(
        features
    )
    return [c for c in features.columns if c not in drop]
