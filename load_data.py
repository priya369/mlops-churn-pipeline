from sklearn.datasets import fetch_california_housing
from google.cloud import bigquery
import pandas as pd
import os

# Fetch dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Add ID column
df.insert(0, 'house_id', range(1, len(df) + 1))

print(f"Dataset shape: {df.shape}")
print(df.head())

# Upload to BigQuery
project_id = os.environ['PROJECT_ID']
client = bigquery.Client(project=project_id)

table_id = f"{project_id}.housing_data.houses"

job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_TRUNCATE",
)

job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
job.result()

print(f"✅ Loaded {len(df)} rows to {table_id}")
