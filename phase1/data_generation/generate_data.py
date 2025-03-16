import pandas as pd
import numpy as np
import random
from faker import Faker
from tqdm import tqdm
import os
import logging

fake = Faker()
tqdm.pandas()

# Create logs directory if it doesn't exist
if not os.path.exists('../logs'):
    os.makedirs('../logs')

# Logging configuration
logging.basicConfig(filename='../logs/generate_data.log', level=logging.INFO, format='%(asctime)s - %(message)s')

try:
    NUM_RECORDS = 10000  # Start with 10K

    def generate_user():
        return {
            "user_id": fake.uuid4(),
            "age": random.randint(18, 70),
            "region": fake.state(),
            "credit_score": random.randint(300, 850),
            "behavioral_history": fake.text(max_nb_chars=50)
        }

    def generate_transaction(user_id, credit_score):
        return {
            "user_id": user_id,
            "credit_score": credit_score,
            "timestamp": fake.date_time_this_year(),
            "amount": round(random.uniform(1, 10000), 2),
            "merchant": fake.company(),
            "ip": fake.ipv4(),
            "device_fingerprint": fake.sha256(),
            "location": fake.city(),
            "velocity_pattern": random.choice(['Low', 'Medium', 'High']),
            "browser_info": fake.user_agent(),
            "geolocation": f"{random.uniform(-90, 90)}, {random.uniform(-180, 180)}",
            "network_latency": random.uniform(10, 500),
            "time_of_day": random.choice(['Morning', 'Afternoon', 'Evening', 'Night']),
            "session_metadata": fake.text(max_nb_chars=50),
            "label": random.choices(['normal', 'suspicious', 'fraudulent'], weights=[85, 10, 5])[0]
        }

    users = [generate_user() for _ in range(1000)]

    transactions = []
    for user in tqdm(users):
        user_id = user["user_id"]
        credit_score = user["credit_score"]
        for _ in range(10):
            transactions.append(generate_transaction(user_id, credit_score))

    df = pd.DataFrame(transactions)
    df.to_csv('../datasets/transactions_1.csv', index=False)
    logging.info("Data generation complete!")

except Exception as e:
    logging.error(f"An error occurred: {e}")
