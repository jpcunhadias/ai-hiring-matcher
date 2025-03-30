import logging
import os
from io import StringIO, BytesIO

import boto3
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("S3_BUCKET_NAME")
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)


def load_model_from_s3(s3_path: str):
    """
    Carrega um modelo .pkl armazenado no S3.
    """
    buffer = BytesIO()
    s3.download_fileobj(Bucket=BUCKET, Key=s3_path, Fileobj=buffer)
    buffer.seek(0)
    model = joblib.load(buffer)
    return model


def save_df_to_s3(df: pd.DataFrame, key: str):
    """
    Salva um DataFrame como CSV no S3.
    """
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=csv_buffer.getvalue())
    print(f"DataFrame salvo no S3 em: s3://{BUCKET}/{key}")


def save_model_to_s3(model, key: str):
    """
    Salva um modelo como .pkl no S3.
    """
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    print(f"Modelo salvo no S3 em: s3://{BUCKET}/{key}")


def setup_logging(name: str = "ml_pipeline"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger


logger = setup_logging("ml_pipeline")
