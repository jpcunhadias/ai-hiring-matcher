import os
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


@pytest.mark.skipif(
    not all([os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY")]),
    reason="Credenciais AWS não encontradas",
)
def test_predict_endpoint():
    payload = {
        "nivel_academico": "Ensino Superior Completo",
        "ingles": "Avançado",
        "espanhol": "Intermediário",
        "area_atuacao": "Ti - Projetos",
        "nivel_profissional": "Sênior",
        "sap": "Não",
        "cliente": "Gonzalez And Sons",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "prob_contratacao" in json_data
    assert isinstance(json_data["prob_contratacao"], float)
    assert 0.0 <= json_data["prob_contratacao"] <= 1.0
