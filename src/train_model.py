import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.data_preparation import (
    build_skill_vocabulary,
    extract_job_skills,
    load_raw_data,
    parse_resume,
    skill_overlap,
)
from src.embeddings import embed_texts
from src.fairness_audit import render_fairness_report, run_fairness_audit
from src.matcher import JobCatalog, build_job_catalog_from_df, evaluate_retrieval
from src.utils import MODELS_DIR, REPORTS_DIR, logger, save_df, save_model

MLFLOW_EXPERIMENT = "ai-hiring-matcher"
REFERENCE_FEATURES_PATH = Path("data/processed/reference_features.csv")


def build_features(
    df: pd.DataFrame, resume_embeddings: np.ndarray, job_embedding_lookup: dict
) -> tuple[pd.DataFrame, set[str]]:
    """[cosine_similarity, skill_overlap] for every resume/job row, plus the skill
    vocabulary used to compute the overlap (needed again at inference time)."""
    job_embeddings = np.stack([job_embedding_lookup[jd] for jd in df["Job Description"]])
    cosine_sim = np.sum(resume_embeddings * job_embeddings, axis=1)

    logger.info("Extracting skills and computing overlap with the jobs...")
    vocabulary = build_skill_vocabulary(df["Resume"])
    overlaps = [
        skill_overlap(
            parse_resume(resume)["skills"], extract_job_skills(job_description, vocabulary)
        )
        for resume, job_description in zip(df["Resume"], df["Job Description"])
    ]

    features = pd.DataFrame({"cosine_similarity": cosine_sim, "skill_overlap": overlaps})
    return features, vocabulary


def build_top1_reference(
    df: pd.DataFrame,
    resume_embeddings: np.ndarray,
    catalog: JobCatalog,
    vocabulary: set[str],
    model: LogisticRegression,
) -> pd.DataFrame:
    """Drift reference must mirror what /match actually computes per request: each
    resume's TOP-1 retrieved job, not the dataset's own (arbitrary) resume/job
    pairing used to train the classifier. Using the paired-job features here instead
    produced a spurious ~75-100% "drift" alert on real traffic — not real drift, just
    comparing two structurally different distributions (best-of-51 vs. one arbitrary
    pairing)."""
    scores_matrix = resume_embeddings @ catalog.embeddings.T  # (n_resumes, n_jobs)
    best_idx = np.argmax(scores_matrix, axis=1)
    best_similarity = scores_matrix[np.arange(len(df)), best_idx]
    best_roles = [catalog.roles[i] for i in best_idx]

    role_to_description = dict(zip(catalog.roles, catalog.descriptions))
    overlaps = [
        skill_overlap(
            parse_resume(resume)["skills"],
            extract_job_skills(role_to_description[role], vocabulary),
        )
        for resume, role in zip(df["Resume"], best_roles)
    ]

    reference = pd.DataFrame({"cosine_similarity": best_similarity, "skill_overlap": overlaps})
    reference["resume_length"] = df["Resume"].str.len().values
    reference["best_match_proba"] = model.predict_proba(
        reference[["cosine_similarity", "skill_overlap"]]
    )[:, 1]
    return reference


def train_classifier(features: pd.DataFrame, target: pd.Series) -> tuple[LogisticRegression, dict]:
    logger.info(
        "Feature-target correlation: %s",
        features.assign(**{"Best Match": target}).corr()["Best Match"].to_dict(),
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Logistic regression, not XGBoost: its coefficients stay interpretable, which
    # the fairness audit above relies on when explaining the residual gap.
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Classifier evaluation:\n%s", classification_report(y_test, y_pred))

    return model, report


def main() -> None:
    # MLflow's filesystem backend (file:./mlruns) is in maintenance mode as of
    # MLflow 3.x, so local tracking defaults to a local sqlite database instead.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_raw_data()

    logger.info("Computing embeddings for %d resumes...", len(df))
    resume_embeddings = embed_texts(df["Resume"].tolist())

    logger.info("Building job catalog (deduplicated)...")
    catalog = build_job_catalog_from_df(df)
    job_embedding_lookup = dict(zip(catalog.descriptions, catalog.embeddings))

    features, vocabulary = build_features(df, resume_embeddings, job_embedding_lookup)
    target = df["Best Match"]

    with mlflow.start_run():
        model, report = train_classifier(features, target)
        retrieval_metrics = evaluate_retrieval(resume_embeddings, df["Job Roles"].tolist(), catalog)

        mlflow.log_metrics(
            {
                "precision_positive": report["1"]["precision"],
                "recall_positive": report["1"]["recall"],
                "f1_positive": report["1"]["f1-score"],
                "accuracy": report["accuracy"],
                **retrieval_metrics,
            }
        )
        mlflow.sklearn.log_model(model, name="model")

        # Fairness audit runs against the full dataset scored by the freshly trained
        # classifier — needs cosine_similarity/skill_overlap/best_match_proba per row.
        fairness_input = features.copy()
        fairness_input["best_match_proba"] = model.predict_proba(features)[:, 1]
        for col in ("Gender", "Race", "Ethnicity", "Job Roles"):
            fairness_input[col] = df[col].values
        fairness_input["Best Match"] = df["Best Match"].values

        fairness_report = run_fairness_audit(fairness_input)
        report_text = render_fairness_report(fairness_report)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "fairness_report.md").write_text(report_text, encoding="utf-8")
        mlflow.log_artifact(str(REPORTS_DIR / "fairness_report.md"))

    save_model(model, MODELS_DIR / "matcher_classifier.joblib")
    save_model(vocabulary, MODELS_DIR / "skill_vocabulary.joblib")
    save_model(catalog, MODELS_DIR / "job_catalog.joblib")

    # Reference distribution for drift_monitor.py: built the same way as a real
    # /match request (top-1 retrieval, not the dataset's own resume/job pairing) so
    # the comparison against live traffic is apples-to-apples.
    reference = build_top1_reference(df, resume_embeddings, catalog, vocabulary, model)
    save_df(reference, REFERENCE_FEATURES_PATH)

    logger.info("Training complete. Retrieval metrics: %s", retrieval_metrics)


if __name__ == "__main__":
    main()
