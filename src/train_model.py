import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data_preparation import build_pipeline, get_categorical_features
from src.load_data import load_and_merge_data
from src.utils import save_df_to_s3, save_model_to_s3


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra situações e cria a variável target.
    """
    valores_sucesso = [
        "Contratado pela Decision",
        "Contratado como Hunting",
        "Proposta Aceita",
    ]
    valores_excluir = [
        "Prospect",
        "Inscrito",
        "Encaminhado ao Requisitante",
        "Entrevista Técnica",
        "Entrevista com Cliente",
        "Em avaliação pelo RH",
        "Encaminhar Proposta",
        "Documentação PJ",
        "Documentação CLT",
        "Documentação Cooperado",
    ]

    df = df[~df["situacao"].isin(valores_excluir)].copy()
    df["target"] = df["situacao"].apply(lambda x: 1 if x in valores_sucesso else 0)

    selected_columns = [
        "nivel_academico",
        "ingles",
        "espanhol",
        "area_atuacao",
        "nivel_profissional",
        "sap",
        "cliente",
        "target",
    ]

    return df[selected_columns]


def preprocess_and_train(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    cat_features = get_categorical_features(X)
    pipeline = build_pipeline(cat_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))

    return pipeline


if __name__ == "__main__":
    # 1. Carrega e unifica os dados
    merged_df = load_and_merge_data(save=False)

    # 2. Cria o dataset final com target
    df_model = create_target_variable(merged_df)

    # 3. Salva df_model no S3
    save_df_to_s3(df_model, "data/processed/df_model.csv")

    # 4. Treina modelo
    model = preprocess_and_train(df_model)

    # 5. Salva modelo no S3
    save_model_to_s3(model, "models/modelo_priorizacao_xgb.pkl")
