import numpy as np

from src.matcher import JobCatalog
from src.predict_model import match_resume

RESUME_TEXT = (
    "Proficient in Care, with mid-level experience in the field. Holds a "
    "Bachelors degree. Skilled in delivering results and adapting to dynamic "
    "environments."
)


def _unit(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=float)
    return arr / np.linalg.norm(arr)


def test_match_resume_ranks_by_similarity_with_skill_overlap(monkeypatch):
    monkeypatch.setattr("src.predict_model.embed_texts", lambda texts: np.stack([_unit([1, 0])]))

    catalog = JobCatalog(
        roles=["Nurse", "Pilot"],
        descriptions=["care for patients", "fly planes"],
        embeddings=np.stack([_unit([1, 0]), _unit([0, 1])]),
    )
    vocabulary = {"Care"}

    ranked = match_resume(RESUME_TEXT, vocabulary, catalog, top_n=2)

    assert ranked.iloc[0]["job_role"] == "Nurse"
    assert ranked.iloc[1]["job_role"] == "Pilot"
    # No match probability: the Best Match classifier is not part of serving.
    assert list(ranked.columns) == ["job_role", "similarity", "skill_overlap"]
    # "Care" only appears in the Nurse job description, not Pilot's.
    assert ranked.set_index("job_role").loc["Nurse", "skill_overlap"] == 1.0
    assert ranked.set_index("job_role").loc["Pilot", "skill_overlap"] == 0.0
