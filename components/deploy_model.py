from kfp.v2.dsl import component, Input, Output, Model, Artifact

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform"]
)
def deploy_model(
    model: Input[Model],
    approval: Input[Artifact],
    project_id: str,
    region: str,
    model_name: str,
    endpoint_name: str,
    deployment_info: Output[Artifact]
):
    """Register and deploy model to Vertex AI"""
    from google.cloud import aiplatform
    
    # Check approval
    with open(approval.path, 'r') as f:
        approved = f.read().strip()
    
    if approved != "true":
        print("❌ Model not approved")
        with open(deployment_info.path, 'w') as f:
            f.write("status: rejected")
        return
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    print("📦 Uploading model to Vertex AI Model Registry...")
    uploaded_model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )
    
    print(f"✅ Model registered: {uploaded_model.resource_name}")
    
    # Get or create endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    
    if endpoints:
        endpoint = endpoints[0]
        print(f"📍 Using existing endpoint: {endpoint.display_name}")
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        print(f"📍 Created new endpoint: {endpoint.display_name}")
    
    # Deploy
    print("🚀 Deploying model...")
    uploaded_model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100
    )
    
    # Save info
    info = f"status: deployed\nmodel: {uploaded_model.resource_name}\nendpoint: {endpoint.resource_name}"
    with open(deployment_info.path, 'w') as f:
        f.write(info)
    
    print(f"✅ Deployed to: {endpoint.resource_name}")
