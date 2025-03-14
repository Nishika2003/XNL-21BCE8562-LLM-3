from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    description: str

class FraudDetectionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_score: float


@app.post("/detect_fraud", response_model=FraudDetectionResponse)
async def detect_fraud(transaction: Transaction):
    # Dummy Fraud Detection Logic
    fraud_score = 0.8 if transaction.amount > 1000 else 0.2
    is_fraud = fraud_score > 0.5
    
    return FraudDetectionResponse(
        transaction_id=transaction.transaction_id,
        is_fraud=is_fraud,
        fraud_score=fraud_score
    )


@app.get("/health")
async def health_check():
    return {"status": "Running"}
