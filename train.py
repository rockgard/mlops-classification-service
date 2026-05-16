import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    """Load the Census Income dataset from the local raw data directory."""
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"
    ]
    # Point to the local file we downloaded
    local_path = "data/raw/census_raw.csv"
    return pd.read_csv(local_path, names=columns, sep=r'\s*,\s*', engine='python')

def build_pipeline():
    """Define preprocessing steps and couple them with the classifier."""
    # Define features by type
    numeric_features = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    categorical_features = ["workclass", "marital_status", "occupation", "relationship", "race", "sex"]
    
    # Preprocessing transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Bundle preprocessing into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create the complete execution pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    
    return pipeline

def main():
    print("⏳ Loading dataset from UCI Repository...")
    df = load_data()
    
    # Preprocess target variable (Convert string income thresholds to binary 0 and 1)
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
    
    X = df.drop(columns=['income', 'fnlwgt', 'education', 'native_country'])
    y = df['income']
    
    # Split data securely
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("⚙️ Training Random Forest Pipeline...")
    model_pipeline = build_pipeline()
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate performance
    predictions = model_pipeline.predict(X_test)
    print("\n📊 Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Serialize model and metadata for serving
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model_pipeline, "artifacts/model_pipeline.joblib")
    print("💾 Model artifact successfully saved to 'artifacts/model_pipeline.joblib'")

if __name__ == "__main__":
    main()