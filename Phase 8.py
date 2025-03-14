from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import random

app = FastAPI()

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    timestamp: str
    location: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/detect_fraud")
def detect_fraud(transaction: Transaction):
    fraud_score = random.uniform(0, 1)
    return {"transaction_id": transaction.transaction_id, "fraud_score": fraud_score}
