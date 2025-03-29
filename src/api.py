from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("models/modelo_priorizacao_xgb.pkl")

app = FastAPI(title="API de Priorização de Candidatos", version="1.0")


class CandidatoInput(BaseModel):
    nivel_academico: str
    ingles: str
    espanhol: str
    area_atuacao: str
    nivel_profissional: str
    sap: str
    cliente: str


def priorizar_candidatos(df_input, modelo, top_n=1):
    df_copy = df_input.copy()
    df_copy = df_copy.drop(columns=["target"], errors="ignore")

    # Previsão
    probs = modelo.predict_proba(df_copy)[:, 1]
    df_copy["prob_contratacao"] = probs

    df_ranked = df_copy.sort_values(by="prob_contratacao", ascending=False)
    return df_ranked.head(top_n)


@app.post("/predict")
def predict(data: CandidatoInput):
    df = pd.DataFrame([data.model_dump()])
    resultado = priorizar_candidatos(df, model, top_n=1)
    return resultado[["prob_contratacao"]].to_dict(orient="records")[0]
