import pandas as pd
from src.data_preparation import build_pipeline


def test_pipeline_fit_predict():
    # Exemplo com duas amostras (1 sucesso, 1 fracasso)
    df = pd.DataFrame([
        {
            "nivel_academico": "Ensino Superior Completo",
            "ingles": "Avançado",
            "espanhol": "Intermediário",
            "area_atuacao": "Ti - Projetos",
            "nivel_profissional": "Sênior",
            "sap": "Não",
            "cliente": "Gonzalez And Sons"
        },
        {
            "nivel_academico": "Ensino Médio Completo",
            "ingles": "Básico",
            "espanhol": "Nenhum",
            "area_atuacao": "Ti - Suporte",
            "nivel_profissional": "Júnior",
            "sap": "Sim",
            "cliente": "Porter-Wilson"
        }
    ])

    y = [1, 0]  # Sucesso e fracasso

    pipeline = build_pipeline(df.columns.tolist())
    pipeline.fit(df, y)
    probs = pipeline.predict_proba(df)

    assert probs.shape == (2, 2)
    assert all(0.0 <= p <= 1.0 for p in probs[:, 1])
