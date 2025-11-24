from kfp.v2.dsl import component, Input, Output, Dataset

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn"]
)
def prepare_features(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset]
):
    """Split and prepare data"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(input_data.path)
    
    # Remove ID column
    df = df.drop('house_id', axis=1, errors='ignore')
    
    # Split features and target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save
    train_df = X_train.copy()
    train_df['MedHouseVal'] = y_train
    test_df = X_test.copy()
    test_df['MedHouseVal'] = y_test
    
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    
    print(f"✅ Train: {train_df.shape}, Test: {test_df.shape}")
