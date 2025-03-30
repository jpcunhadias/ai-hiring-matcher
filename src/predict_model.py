import joblib
import pandas as pd

MODEL_PATH = "models/modelo_priorizacao_xgb.pkl"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def predict_proba(model, df_input: pd.DataFrame, top_n=5) -> pd.DataFrame:
    """
    Recebe um DataFrame com candidatos e retorna os top_n com maior probabilidade de contratação.
    """
    df_input = df_input.copy()

    # Remove target se existir
    if "target" in df_input.columns:
        df_input = df_input.drop(columns=["target"])

    # Predição de probabilidade da classe positiva
    probs = model.predict_proba(df_input)[:, 1]
    df_input["prob_contratacao"] = probs

    # Ordena por maior probabilidade
    df_input = df_input.sort_values(by="prob_contratacao", ascending=False)

    return df_input.head(top_n)
