import pandas as pd
import random

# Load transaction data
df = pd.read_csv('../datasets/transactions.csv')

# Define possible attack types
attack_types = ['Credential Stuffing', 'Phishing', 'Fake Merchant', 'Adversarial Transactions']

# RL Agent Simulation - Making Fraudulent Attacks Smarter
def simulate_rl_attack(row):
    if row['label'] == 'fraudulent':
        if row['amount'] > 5000 and row['credit_score'] < 500:
            return 'Credential Stuffing'
        elif 'Login' in row['session_metadata']:
            return 'Phishing'
        elif random.random() < 0.3:
            return 'Adversarial Transactions'
        else:
            return random.choice(attack_types)
    return 'None'

# Generate attack labels
df['attack_type'] = df.apply(simulate_rl_attack, axis=1)
df.to_csv('../datasets/rl_agent_output.csv', index=False)
print("RL Agent Simulation complete!")
