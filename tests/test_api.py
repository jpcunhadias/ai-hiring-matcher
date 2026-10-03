import pytest
from fastapi.testclient import TestClient

from src.utils import MODELS_DIR

REQUIRED_ARTIFACTS = [
    "skill_vocabulary.joblib",
    "job_catalog.joblib",
]

# src.api loads the trained artifacts at import time, so these tests only make
# sense after `make train` — skip (rather than fail at collection) on a fresh clone.
pytestmark = pytest.mark.skipif(
    not all((MODELS_DIR / name).exists() for name in REQUIRED_ARTIFACTS),
    reason="Models not trained — run `make train` first.",
)

SAMPLE_RESUME = (
    "Proficient in Firewalls, Cyber Threats, Penetration Testing, Vulnerability "
    "Analysis, Ethical Hacking, with senior-level experience in the field. Holds "
    "a Masters degree. Holds certifications such as Certified Ethical Hacker "
    "(CEH). Skilled in delivering results and adapting to dynamic environments."
)


@pytest.fixture(scope="module")
def client():
    from src.api import app

    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_match_endpoint_returns_ranked_jobs(client, tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.REQUEST_LOG_PATH", tmp_path / "requests.jsonl")

    response = client.post("/match", json={"resume": SAMPLE_RESUME, "top_n": 3})

    assert response.status_code == 200
    matches = response.json()["matches"]
    assert len(matches) == 3

    top = matches[0]
    assert set(top.keys()) == {"job_role", "similarity", "skill_overlap"}
    # Similarity should be sorted descending.
    assert matches[0]["similarity"] >= matches[1]["similarity"] >= matches[2]["similarity"]


def test_match_endpoint_logs_the_request(client, tmp_path, monkeypatch):
    # Redirected to tmp_path so the test never touches the real data/logs/requests.jsonl
    # (the old version of this test deleted it).
    log_path = tmp_path / "requests.jsonl"
    monkeypatch.setattr("src.utils.REQUEST_LOG_PATH", log_path)

    client.post("/match", json={"resume": SAMPLE_RESUME, "top_n": 1})

    assert log_path.exists()
    assert len(log_path.read_text(encoding="utf-8").strip().splitlines()) == 1
