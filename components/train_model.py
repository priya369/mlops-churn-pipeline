from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    train_data: Input[Dataset],
    model_output: Output[Model],
    metrics: Output[Metrics]
):
    """Train Random Forest model"""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    
    # Load data
    df = pd.read_csv(train_data.path)
    X_train = df.drop('MedHouseVal', axis=1)
    y_train = df['MedHouseVal']
    
    # Train model
    print("🔧 Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_output.path + "/model.pkl")
    
    # Log metrics
    metrics.log_metric("n_estimators", 100)
    metrics.log_metric("max_depth", 10)
    metrics.log_metric("training_samples", len(X_train))
    
    print("✅ Model trained successfully")
