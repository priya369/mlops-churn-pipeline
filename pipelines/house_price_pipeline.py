from kfp.v2 import dsl, compiler
import sys
import os
sys.path.append('..')

from components.data_ingestion import ingest_data
from components.feature_engineering import prepare_features
from components.train_model import train_model
from components.evaluate_model import evaluate_model
from components.deploy_model import deploy_model

@dsl.pipeline(
    name="house-price-mlops-pipeline",
    description="End-to-end MLOps pipeline for house price prediction",
    pipeline_root="gs://data-oasis-472909-u4-mlops/pipeline_root"
)
def house_price_pipeline(
    project_id: str = "data-oasis-472909-u4",
    region: str = "us-central1",
    model_name: str = "house-price-model",
    endpoint_name: str = "house-price-endpoint"
):
    # Step 1: Ingest data from BigQuery
    ingest_task = ingest_data(project_id=project_id)
    
    # Step 2: Feature engineering
    prepare_task = prepare_features(
        input_data=ingest_task.outputs['output_data']
    )
    
    # Step 3: Train model
    train_task = train_model(
        train_data=prepare_task.outputs['train_data']
    )
    
    # Step 4: Evaluate model
    eval_task = evaluate_model(
        model=train_task.outputs['model_output'],
        test_data=prepare_task.outputs['test_data']
    )
    
    # Step 5: Deploy model
    deploy_task = deploy_model(
        model=train_task.outputs['model_output'],
        approval=eval_task.outputs['approval'],
        project_id=project_id,
        region=region,
        model_name=model_name,
        endpoint_name=endpoint_name
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=house_price_pipeline,
        package_path='house_price_pipeline.json'
    )
    print("✅ Pipeline compiled: house_price_pipeline.json")
