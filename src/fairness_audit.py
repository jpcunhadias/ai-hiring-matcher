import math
from dataclasses import dataclass

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression

from src.utils import logger

DEMOGRAPHIC_COLUMNS = ["Gender", "Race", "Ethnicity"]


@dataclass
class FairnessReport:
    label_selection_rate: pd.DataFrame
    gender_gap_by_role: pd.DataFrame
    residual_gap: pd.DataFrame
    bimodality: dict
    proba_range: tuple[float, float]
    retrieval_by_group: pd.DataFrame
    retrieval_pvalues: dict


def label_selection_rates(df: pd.DataFrame, target_col: str = "Best Match") -> pd.DataFrame:
    """Does the *label itself* (independent of any model) skew by demographic group?

    This is a check on the data, not the model — worth running first, since a
    synthetic dataset can encode occupational stereotypes by demographic before
    any model ever sees it.
    """
    rows = []
    for col in DEMOGRAPHIC_COLUMNS:
        rates = df.groupby(col)[target_col].mean()
        for group, rate in rates.items():
            rows.append({"attribute": col, "group": group, "selection_rate": rate})
    return pd.DataFrame(rows)


def gender_gap_by_role(
    df: pd.DataFrame, role_col: str = "Job Roles", target_col: str = "Best Match"
) -> pd.DataFrame:
    """Male-Female selection-rate gap *within each job role*.

    The aggregate rate across all roles understates how extreme this gets: some
    roles show a large gap in one direction, others show none, others show it
    reversed — a role-dependent pattern is a stronger fairness signal than a flat
    global one, since it points at something structural in how the label was
    generated per role rather than a single global skew.
    """
    rates = df.groupby([role_col, "Gender"])[target_col].mean().unstack("Gender")
    if "Male" not in rates or "Female" not in rates:
        return pd.DataFrame(columns=[role_col, "female_rate", "male_rate", "gap"])

    result = pd.DataFrame(
        {
            role_col: rates.index,
            "female_rate": rates["Female"].values,
            "male_rate": rates["Male"].values,
        }
    )
    result["gap"] = result["male_rate"] - result["female_rate"]
    return result.reindex(result["gap"].abs().sort_values(ascending=False).index)


def group_selection_rates(
    df: pd.DataFrame, role_col: str = "Job Roles", target_col: str = "Best Match"
) -> pd.Series:
    """Selection rate of every (job role, gender) group, indexed by (role, gender)."""
    return df.groupby([role_col, "Gender"])[target_col].mean()


def gender_rate_bimodality(
    df: pd.DataFrame, role_col: str = "Job Roles", target_col: str = "Best Match"
) -> dict:
    """Distribution shape of (Job Role, Gender) group selection rates.

    Distinguishes three explanations for a group-level gap: a hard-coded rule
    (rates sit exactly at 0 or 1), incidental noise (rates spread smoothly around
    the overall rate), or biased-but-random generation (rates cluster tightly near
    two poles without ever reaching them) — the last is what this dataset shows.
    """
    rates = group_selection_rates(df, role_col, target_col)
    return {
        "n_groups": int(len(rates)),
        "n_deterministic": int(((rates == 0) | (rates == 1)).sum()),
        "n_near_zero": int((rates <= 0.2).sum()),
        "n_near_one": int((rates >= 0.8).sum()),
        "n_middle": int(((rates > 0.2) & (rates < 0.8)).sum()),
    }


def retrieval_accuracy_by_group(df: pd.DataFrame, rank_col: str = "retrieval_rank") -> pd.DataFrame:
    """Share of resumes whose own job role is ranked first, per demographic group.

    The matcher is built without demographic inputs, but "no protected attributes as
    features" is not the same as "equal accuracy for everyone", so measure it. std_error
    is the binomial standard error, to judge whether a gap is more than sampling noise.
    """
    rows = []
    for col in DEMOGRAPHIC_COLUMNS:
        for group, ranks in df.groupby(col)[rank_col]:
            n = len(ranks)
            p = float((ranks <= 1).mean())
            rows.append(
                {
                    "attribute": col,
                    "group": group,
                    "n": n,
                    "recall_at_1": p,
                    "std_error": math.sqrt(p * (1 - p) / n),
                }
            )
    return pd.DataFrame(rows)


def retrieval_homogeneity_pvalues(
    df: pd.DataFrame, rank_col: str = "retrieval_rank"
) -> dict[str, float]:
    """Chi-square p-value per attribute for "recall@1 is the same in every group".

    A large p-value means the observed gaps are what sampling noise alone produces.
    With no variation in hits (all or none) or a single group there is nothing to
    differ, so p = 1.
    """
    hit = df[rank_col] <= 1
    pvalues = {}
    for col in DEMOGRAPHIC_COLUMNS:
        table = pd.crosstab(df[col], hit)
        pvalues[col] = (
            1.0 if table.shape[0] < 2 or table.shape[1] < 2 else float(chi2_contingency(table)[1])
        )
    return pvalues


def residual_gap_by_group(
    df: pd.DataFrame,
    proba_col: str = "best_match_proba",
    explain_cols: tuple[str, ...] = ("cosine_similarity", "skill_overlap"),
) -> pd.DataFrame:
    """Regresses match probability on the legitimate similarity features, then checks
    whether the *leftover* (unexplained) probability still correlates with demographic
    group membership — bias beyond what skill/semantic match already explains.
    """
    baseline = LinearRegression()
    baseline.fit(df[list(explain_cols)], df[proba_col])
    residuals = df[proba_col] - baseline.predict(df[list(explain_cols)])
    overall_mean = residuals.mean()

    rows = []
    for col in DEMOGRAPHIC_COLUMNS:
        group_means = residuals.groupby(df[col]).mean()
        for group, mean_residual in group_means.items():
            rows.append(
                {
                    "attribute": col,
                    "group": group,
                    "mean_residual": mean_residual,
                    "gap_vs_overall": mean_residual - overall_mean,
                }
            )
    return pd.DataFrame(rows)


def run_fairness_audit(df: pd.DataFrame) -> FairnessReport:
    """`df` must carry Gender/Race/Ethnicity/Job Roles plus cosine_similarity,
    skill_overlap and best_match_proba for every row (i.e. training data scored by
    the trained classifier).
    """
    logger.info("Running fairness audit...")
    report = FairnessReport(
        label_selection_rate=label_selection_rates(df),
        gender_gap_by_role=gender_gap_by_role(df) if "Job Roles" in df.columns else pd.DataFrame(),
        residual_gap=residual_gap_by_group(df),
        bimodality=gender_rate_bimodality(df) if "Job Roles" in df.columns else {},
        proba_range=(float(df["best_match_proba"].min()), float(df["best_match_proba"].max())),
        retrieval_by_group=(
            retrieval_accuracy_by_group(df) if "retrieval_rank" in df.columns else pd.DataFrame()
        ),
        retrieval_pvalues=(
            retrieval_homogeneity_pvalues(df) if "retrieval_rank" in df.columns else {}
        ),
    )

    max_gap = report.label_selection_rate.loc[
        report.label_selection_rate["attribute"] == "Gender", "selection_rate"
    ]
    if len(max_gap) == 2 and (max_gap.max() - max_gap.min()) > 0.1:
        logger.warning(
            "FAIRNESS ALERT: Best Match rate differs by %.0f percentage points "
            "between Gender groups in the raw label (before any model).",
            (max_gap.max() - max_gap.min()) * 100,
        )

    return report


def render_fairness_report(report: FairnessReport) -> str:
    bimodality_line = ""
    if report.bimodality:
        b = report.bimodality
        bimodality_line = (
            f"Across {b['n_groups']} (Job Role, Gender) groups, {b['n_deterministic']} are "
            f"perfectly deterministic (rate = 0 or 1), {b['n_near_zero']} cluster near 0 "
            f"(<=20%), {b['n_near_one']} cluster near 1 (>=80%), and only {b['n_middle']} "
            "fall in between — a bimodal shape consistent with Best Match having been "
            "sampled as Bernoulli(p) per group, with p fixed near 0.1 or 0.9, rather than "
            "either a hard rule or incidental noise."
        )

    lo, hi = report.proba_range
    spread_note = ""
    if hi - lo < 0.1:
        spread_note = (
            f"**Read this table with care:** the classifier's predicted probabilities span "
            f"only {lo:.2f}-{hi:.2f}, so it says almost the same thing for everyone. Residual "
            "gaps near zero are therefore expected and are *not* evidence that the matcher "
            "is fair.\n"
        )

    retrieval_section: list[str] = []
    if not report.retrieval_by_group.empty:
        retrieval_section = [
            "## Retrieval accuracy by group",
            "",
            "Share of resumes whose own job role the matcher ranks first (recall@1), per "
            "group. `std_error` is the binomial standard error: gaps within about two "
            "standard errors are indistinguishable from sampling noise.",
            "",
            report.retrieval_by_group.to_markdown(index=False),
            "",
            "Chi-square test that recall@1 is equal across groups: "
            + ", ".join(f"{attr} p = {p:.2f}" for attr, p in report.retrieval_pvalues.items())
            + ". A large p-value means the gaps are what sampling noise alone would produce.",
            "",
        ]

    lines = [
        "# Fairness Audit",
        "",
        "## Context",
        "",
        "The dataset's own Kaggle card describes it as material for \"HR Analytics & "
        'Hiring Bias Studies" and for analyzing hiring trends/biases by age, gender or '
        "ethnicity — so a bias finding here is not a surprise, it's the dataset's stated "
        "purpose. What's notable is that the same card defines `Best Match` as reflecting "
        '"qualifications and experience" — but the correlation between `Best Match` and '
        "the actual similarity features (cosine_similarity, skill_overlap) computed in "
        "this project is close to zero (see src/train_model.py's logged feature-target "
        "correlation). The label's documented definition doesn't match its empirical "
        "behavior; this audit exists to catch exactly that gap before training on faith "
        "in a label.",
        "",
        bimodality_line,
        "",
        "## Label selection rate by group",
        "",
        "Share of `Best Match = 1` rows per demographic group, in the raw dataset "
        "(no model involved) — a skew here means the data itself, not the model, "
        "encodes it.",
        "",
        report.label_selection_rate.to_markdown(index=False),
        "",
        "## Male-Female selection-rate gap by job role",
        "",
        "The aggregate Gender rate above understates how extreme this gets per role "
        "— sorted by absolute gap, largest first.",
        "",
        report.gender_gap_by_role.head(10).to_markdown(index=False),
        "",
        "## Residual match-probability gap by group",
        "",
        spread_note,
        "Mean leftover match probability per group *after* regressing out "
        "cosine_similarity and skill_overlap — the part of the model's score that "
        "similarity/skill overlap alone doesn't explain.",
        "",
        report.residual_gap.to_markdown(index=False),
        "",
        *retrieval_section,
    ]
    return "\n".join(lines)
