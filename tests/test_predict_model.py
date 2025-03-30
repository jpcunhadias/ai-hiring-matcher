import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.predict_model import predict_proba


def test_predict_proba_output_format():
    df = pd.DataFrame({
        "nivel_academico": ["Superior", "Médio"],
        "ingles": ["Avançado", "Básico"],
        "espanhol": ["Básico", "Nenhum"],
        "area_atuacao": ["TI", "RH"],
        "nivel_profissional": ["Sênior", "Júnior"],
        "sap": ["Sim", "Não"],
        "cliente": ["Empresa X", "Empresa Y"]
    })

    y = [1, 0]  # Agora temos duas classes

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), df.columns.tolist())
        ])),
        ("classifier", LogisticRegression())
    ])
    pipeline.fit(df, y)

    result = predict_proba(pipeline, df.head(1))
    assert "prob_contratacao" in result.columns
    assert result["prob_contratacao"].between(0, 1).all()
