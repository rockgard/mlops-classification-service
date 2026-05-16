import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize the FastAPI app
app = FastAPI(
    title="Census Income Prediction Service",
    description="A production-grade API to predict whether an individual makes over $50K/year.",
    version="1.0.0"
)

# 1. Load the model pipeline once on startup
MODEL_PATH = "artifacts/model_pipeline.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model artifact from {MODEL_PATH}. Error: {str(e)}")

# 2. Define the input data schema using Pydantic
class InferencePayload(BaseModel):
    age: int = Field(..., example=35)
    workclass: str = Field(..., example="Private")
    marital_status: str = Field(..., example="Married-civ-spouse")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=5000)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=50)
    education_num: int = Field(..., example=13)

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "workclass": "Private",
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 5000,
                "capital_loss": 0,
                "hours_per_week": 50,
                "education_num": 13
            }
        }

# 3. Define the Health Check endpoint
@app.get("/health", tags=["Utility"])
def health_check():
    """Verify server status and model readiness."""
    return {"status": "healthy", "model_loaded": model is not None}

# 4. Define the Prediction endpoint
@app.post("/predict", tags=["Inference"])
def predict(payload: InferencePayload):
    """Accept raw demographic details and return the income bracket prediction."""
    try:
        # Convert incoming JSON payload to a dictionary format matching the training features
        input_dict = {key: [value] for key, value in payload.dict().items()}
        input_df = pd.DataFrame(input_dict)
        
        # Execute model pipeline prediction (handles scaling and encoding under the hood)
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
        
        # Format the output metrics securely
        predicted_class = int(prediction[0])
        confidence = float(probabilities[0][predicted_class])
        income_bracket = ">50K" if predicted_class == 1 else "<=50K"
        
        return {
            "prediction": predicted_class,
            "income_bracket": income_bracket,
            "confidence": round(confidence, 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")