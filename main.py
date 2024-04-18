from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title="Diabetes Diagnosis Model API",
    version="0.0.1"
)

# Load the AI model
model = joblib.load("model/logistic_regression_model_v01.pkl")

@app.post("/api/v1/predict-diabetes", tags=["apple"])
async def predict(
        Age: float,
        Gender: float,
        BMI: float,
        High_BP: float,
        FBS: float,
        HbA1c_level: float,
        Smoking: float
):

    # Create a dictionary from the input features
    dictionary = {
        'Age': Age,
        'Gender': Gender,
        'BMI': BMI,
        'High_BP': High_BP,
        'FBS': FBS,
        'HbA1c_level': HbA1c_level,
        'Smoking': Smoking
    }

    try:
        # Create a DataFrame from the dictionary
        df = pd.DataFrame(dictionary, index=[0])
        # Make prediction using the loaded model
        prediction = model.predict(df)
        # Return the prediction as JSON response
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=prediction.tolist()  # Convert numpy array to list for JSON compatibility
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )