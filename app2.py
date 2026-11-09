import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel,Field
from model2 import predict
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model2 import predict

app = FastAPI(title="California Housing Price API")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

METRICS = json.loads(Path(__file__).with_name("metrics.json").read_text())


class HouseInput(BaseModel):
    MedInc: float = Field(..., gt=0, le=20, description="Median income, in $10k")
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age, years")
    AveRooms: float = Field(..., gt=0, le=200, description="Average rooms per household")
    AveBedrms: float = Field(..., gt=0, le=50, description="Average bedrooms per household")
    Population: float = Field(..., gt=0, le=40000, description="Block group population")
    AveOccup: float = Field(..., gt=0, le=2000, description="Average occupants per household")
    Latitude: float = Field(..., ge=32, le=42.5, description="Latitude (California)")
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude (California)")

@app.get("/health")
def health():
    return{"status":"ok"}

@app.get("/metrics")
def metrics():
    return METRICS

@app.post("/predict")
def predict_endpoint(house: HouseInput):
    price=predict(house.model_dump())
    return {"predicted_price_usd": round(price,2)}
