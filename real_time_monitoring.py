import json
import os
import threading
import time
from fastapi import FastAPI, WebSocket
from typing import List
from elasticsearch import Elasticsearch
from prometheus_client import start_http_server, Gauge
from twilio.rest import Client
from bayes_opt import BayesianOptimization
import numpy as np

# FastAPI app setup
app = FastAPI()
es = Elasticsearch("http://localhost:9200")  

alerts_detected = Gauge('fraud_alerts_detected', 'Number of fraud alerts detected')

# Twilio Setup (for SMS alerts)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

active_connections: List[WebSocket] = []

fraud_threshold = 0.8

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except:
        active_connections.remove(websocket)


def send_alerts_to_clients(message: str):
    for connection in active_connections:
        try:
            connection.send_text(message)
        except:
            pass


def log_anomaly(transaction_id: str, details: dict):
    es.index(index="fraud_alerts", body={
        "transaction_id": transaction_id,
        "details": details,
        "timestamp": time.time()
    })


def send_sms_alert(message: str, recipient_number: str):
    message = client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=recipient_number
    )
    print(f"SMS sent: {message.sid}")


def detect_fraud(transaction):
    risk_score = transaction.get('risk_score', 0)
    return risk_score > fraud_threshold


def process_transaction(transaction: dict):
    if detect_fraud(transaction):
        alerts_detected.inc()
        transaction_id = transaction['transaction_id']
        log_anomaly(transaction_id, transaction)
        message = f"Fraud detected in transaction {transaction_id}!"
        send_alerts_to_clients(message)
        send_sms_alert(message, recipient_number="+1234567890")


def run_transaction_monitoring():
    while True:
        transaction = {
            'transaction_id': str(time.time()),
            'risk_score': np.random.uniform(0, 1)  # Random risk score
        }
        process_transaction(transaction)
        time.sleep(5)


def start_prometheus_server():
    start_http_server(8000)


def optimize_threshold():
    def objective(threshold):
        global fraud_threshold
        fraud_threshold = threshold
        return -abs(threshold - 0.85)  # Dummy objective for optimization

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"threshold": (0.5, 0.95)},
        random_state=42,
    )
    optimizer.maximize(init_points=5, n_iter=10)


if __name__ == "__main__":
    threading.Thread(target=run_transaction_monitoring).start()
    threading.Thread(target=start_prometheus_server).start()
    threading.Thread(target=optimize_threshold).start()
