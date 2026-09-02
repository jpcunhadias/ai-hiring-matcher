import numpy as np

from src.matcher import JobCatalog
from src.predict_model import match_resume

RESUME_TEXT = (
    "Proficient in Care, with mid-level experience in the field. Holds a "
    "Bachelors degree. Skilled in delivering results and adapting to dynamic "
    "environments."
)


class _FakeClassifier:
    """Higher cosine_similarity deterministically means a higher match probability."""

    def predict_proba(self, X):
        positive = X["cosine_similarity"].to_numpy()
        return np.column_stack([1 - positive, positive])


def _unit(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=float)
    return arr / np.linalg.norm(arr)


def test_match_resume_ranks_by_similarity_and_scores_proba(monkeypatch):
    monkeypatch.setattr("src.predict_model.embed_texts", lambda texts: np.stack([_unit([1, 0])]))

    catalog = JobCatalog(
        roles=["Nurse", "Pilot"],
        descriptions=["care for patients", "fly planes"],
        embeddings=np.stack([_unit([1, 0]), _unit([0, 1])]),
    )
    vocabulary = {"Care"}

    ranked = match_resume(RESUME_TEXT, _FakeClassifier(), vocabulary, catalog, top_n=2)

    assert ranked.iloc[0]["job_role"] == "Nurse"
    assert ranked.iloc[1]["job_role"] == "Pilot"
    assert ranked["best_match_proba"].between(0, 1).all()
    # "Care" only appears in the Nurse job description, not Pilot's.
    assert ranked.set_index("job_role").loc["Nurse", "skill_overlap"] == 1.0
    assert ranked.set_index("job_role").loc["Pilot", "skill_overlap"] == 0.0
