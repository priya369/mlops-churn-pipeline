from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate_model(
    model: Input[Model],
    test_data: Input[Dataset],
    eval_metrics: Output[Metrics],
    approval: Output[Artifact]
):
    """Evaluate model performance"""
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import math
    
    # Load model and data
    trained_model = joblib.load(model.path + "/model.pkl")
    df = pd.read_csv(test_data.path)
    
    X_test = df.drop('MedHouseVal', axis=1)
    y_test = df['MedHouseVal']
    
    # Predictions
    y_pred = trained_model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    eval_metrics.log_metric("mse", float(mse))
    eval_metrics.log_metric("rmse", float(rmse))
    eval_metrics.log_metric("mae", float(mae))
    eval_metrics.log_metric("r2_score", float(r2))
    
    print(f"📊 RMSE: {rmse:.4f}")
    print(f"📊 MAE: {mae:.4f}")
    print(f"📊 R²: {r2:.4f}")
    
    # Approval logic (R² > 0.7)
    approved = "true" if r2 > 0.7 else "false"
    
    with open(approval.path, 'w') as f:
        f.write(approved)
    
    status = "✅ APPROVED" if approved == "true" else "❌ REJECTED"
    print(f"{status} (R² threshold: 0.7)")
