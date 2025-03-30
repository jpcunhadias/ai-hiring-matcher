import os
from io import StringIO

import boto3
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)

BUCKET = os.getenv("S3_BUCKET_NAME")
PROCESSED_PATH = os.getenv("PROCESSED_PATH")


def get_categorical_features(df: pd.DataFrame):
    return df.select_dtypes(include="object").columns.tolist()


def build_pipeline(cat_features: list) -> Pipeline:
    """
    Cria o pipeline de pré-processamento + modelo.
    """
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)]
    )

    model = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    return pipeline


def save_df_to_s3(df, filename="df_model.csv"):
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    s3.put_object(
        Bucket=BUCKET, Key=f"{PROCESSED_PATH}{filename}", Body=buffer.getvalue()
    )
    print(f"Arquivo salvo em s3://{BUCKET}/{PROCESSED_PATH}{filename}")
