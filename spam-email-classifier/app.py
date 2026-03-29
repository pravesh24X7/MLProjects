# This file contains the API routes of spam email project

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

# Defines the dtype of message received.
class EmailRequest(BaseModel):
    message: str


# load model
model = joblib.load("model/spam_email_model.pkl")

@app.get("/")
def home():
    return {
        "message": "SPAM email classifier API Running"
    }

@app.post("/predict")
def predict(request: EmailRequest):
    prediction = model.predict([request.message])[0]
    prob = model.predict_proba([request.message])[0][1]

    return {
        "message": request.message,
        "prediction": prediction,
        "spam_probability": prob
    }
