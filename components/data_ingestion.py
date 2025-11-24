from kfp.v2.dsl import component, Output, Dataset

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow", "db-dtypes"]
)
def ingest_data(
    project_id: str,
    output_data: Output[Dataset]
):
    """Fetch housing data from BigQuery"""
    from google.cloud import bigquery
    import pandas as pd
    
    client = bigquery.Client(project=project_id)
    
    query = f"""
    SELECT * FROM `{project_id}.housing_data.houses`
    """
    
    df = client.query(query).to_dataframe()
    df.to_csv(output_data.path, index=False)
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"Features: {list(df.columns)}")
