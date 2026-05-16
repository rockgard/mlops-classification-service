import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

def load_data():
    """Load the Census Income dataset from the local raw data directory."""
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"
    ]
    local_path = "data/raw/census_raw.csv"
    return pd.read_csv(local_path, names=columns, sep=r'\s*,\s*', engine='python')

def build_pipeline(n_estimators, max_depth):
    """Define preprocessing steps and couple them with the classifier."""
    numeric_features = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    categorical_features = ["workclass", "marital_status", "occupation", "relationship", "race", "sex"]
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1))
    ])
    
    return pipeline

def main():
    # Set the experiment name in MLflow
    mlflow.set_experiment("Census_Income_Classification")
    
    # Define hyperparameters to log
    n_estimators = 100
    max_depth = 12
    
    # Start an MLflow run context
    with mlflow.start_run():
        print("⏳ Loading dataset...")
        df = load_data()
        
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
        X = df.drop(columns=['income', 'fnlwgt', 'education', 'native_country'])
        y = df['income']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("⚙️ Training Random Forest Pipeline...")
        model_pipeline = build_pipeline(n_estimators, max_depth)
        model_pipeline.fit(X_train, y_train)
        
        # Evaluate performance
        predictions = model_pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        
        print(f"\n📊 Accuracy: {acc:.4f}")
        
        # --- MLOps: Log Parameters & Metrics to MLflow ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # Save local copy for DVC tracking
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model_pipeline, "artifacts/model_pipeline.joblib")
        
        # --- MLOps: Log the actual Model Artifact directly to MLflow ---
        mlflow.sklearn.log_model(model_pipeline, artifact_path="model")
        print("💾 Metrics and Model successfully logged to MLflow!")

if __name__ == "__main__":
    main()