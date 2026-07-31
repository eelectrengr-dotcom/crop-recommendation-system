from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")
le = joblib.load("models/label_encoder.pkl")

# ✅ Define request body
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict")
def predict(data: CropInput):

    print(data)  # ✅ correct indentation

    values = np.array([[   # ✅ FIXED
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]])

    pred = model.predict(values)[0]
    crop = le.inverse_transform([pred])[0]

    return {"recommended_crop": crop}