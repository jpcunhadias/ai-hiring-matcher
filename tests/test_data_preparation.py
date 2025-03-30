import pandas as pd
from src.train_model import create_target_variable


def test_create_target_variable():
    df_mock = pd.DataFrame(
        {
            "situacao": [
                "Proposta Aceita",
                "Entrevista Técnica",
                "Contratado como Hunting",
                "Prospect",
                "Contratado pela Decision",
            ],
            "nivel_academico": ["Superior"] * 5,
            "ingles": ["Básico"] * 5,
            "espanhol": ["Nenhum"] * 5,
            "area_atuacao": ["TI"] * 5,
            "nivel_profissional": ["Júnior"] * 5,
            "sap": ["Sim"] * 5,
            "cliente": ["Empresa X"] * 5,
        }
    )

    result = create_target_variable(df_mock)

    assert "target" in result.columns
    assert set(result["target"].unique()) <= {0, 1}
