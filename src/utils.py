import os
import boto3
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("S3_BUCKET_NAME")
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)


def save_df_to_s3(df: pd.DataFrame, key: str):
    """
    Salva um DataFrame como CSV no S3.
    """
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=csv_buffer.getvalue())
    print(f"✔️ DataFrame salvo no S3 em: s3://{BUCKET}/{key}")
