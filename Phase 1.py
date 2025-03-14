import numpy as np
import pandas as pd
from faker import Faker
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from gensim.models import Word2Vec, FastText
import sentencepiece as spm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Optional
import logging
import os
import json
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransactionConfig:
    """Configuration class for transaction generation"""
    def __init__(self):
        self.MERCHANT_CATEGORIES = [
            'Retail', 'Restaurant', 'Travel', 'Entertainment', 
            'Services', 'Healthcare', 'Technology', 'Education'
        ]
        self.TRANSACTION_TYPES = ['purchase', 'refund', 'transfer', 'payment']
        self.PAYMENT_METHODS = ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet']
        self.BROWSER_TYPES = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera']
        self.DEVICE_TYPES = ['Mobile', 'Desktop', 'Tablet', 'Smart TV']
        self.OS_TYPES = ['Windows', 'MacOS', 'iOS', 'Android', 'Linux']

class UserProfile:
    """Class to generate and maintain user profiles"""
    def __init__(self, fake: Faker):
        self.fake = fake
        self.profile = self._generate_profile()
        self.transaction_history = []
        self.behavioral_patterns = self._generate_behavioral_patterns()

    def _generate_profile(self) -> Dict:
        return {
            'user_id': self.fake.uuid4(),
            'age': random.randint(18, 85),
            'region': self.fake.country(),
            'credit_score': random.randint(300, 850),
            'income_level': random.choice(['low', 'medium', 'high']),
            'account_age_days': random.randint(1, 3650),
            'risk_score': random.uniform(0, 1)
        }

    def _generate_behavioral_patterns(self) -> Dict:
        return {
            'avg_transaction_amount': random.uniform(20, 500),
            'typical_transaction_hours': sorted(random.sample(range(24), 3)),
            'frequent_merchant_categories': random.sample(
                TransactionConfig().MERCHANT_CATEGORIES, 
                k=random.randint(2, 4)
            ),
            'preferred_payment_methods': random.sample(
                TransactionConfig().PAYMENT_METHODS,
                k=random.randint(1, 3)
            )
        }

class TransactionGenerator:
    """Main class for generating synthetic transactions"""
    def __init__(self, num_users: int = 1000):
        self.fake = Faker()
        self.config = TransactionConfig()
        self.users = [UserProfile(self.fake) for _ in range(num_users)]
        self.merchants = self._generate_merchant_list()

    def _generate_merchant_list(self, num_merchants: int = 1000) -> List[Dict]:
        merchants = []
        for _ in range(num_merchants):
            merchant = {
                'id': self.fake.uuid4(),
                'name': self.fake.company(),
                'category': random.choice(self.config.MERCHANT_CATEGORIES),
                'risk_score': random.uniform(0, 1),
                'country': self.fake.country(),
                'avg_transaction_value': random.uniform(10, 1000)
            }
            merchants.append(merchant)
        return merchants

    def generate_transaction(self, user: UserProfile, is_fraudulent: bool = False) -> Dict:
        """Generate a single transaction with realistic patterns"""
        merchant = random.choice(self.merchants)
        
        # Base amount based on user and merchant patterns
        base_amount = random.uniform(
            min(user.behavioral_patterns['avg_transaction_amount'], merchant['avg_transaction_value']),
            max(user.behavioral_patterns['avg_transaction_amount'], merchant['avg_transaction_value'])
        )

        # Modify amount if fraudulent
        if is_fraudulent:
            base_amount *= random.uniform(2, 10)

        # Generate timestamp with realistic patterns
        hour = random.choice(user.behavioral_patterns['typical_transaction_hours'])
        if is_fraudulent:
            hour = (hour + 12) % 24  # Opposite of normal pattern
        
        timestamp = datetime.now().replace(
            hour=hour,
            minute=random.randint(0, 59)
        ) - timedelta(days=random.randint(0, 30))

        # Generate device and browser info
        device_info = self._generate_device_info()

        transaction = {
            'transaction_id': self.fake.uuid4(),
            'user_id': user.profile['user_id'],
            'timestamp': timestamp.isoformat(),
            'amount': round(base_amount, 2),
            'merchant_id': merchant['id'],
            'merchant_name': merchant['name'],
            'merchant_category': merchant['category'],
            'payment_method': random.choice(user.behavioral_patterns['preferred_payment_methods']),
            'device_info': device_info,
            'ip_address': self.fake.ipv4(),
            'location': {
                'latitude': self.fake.latitude(),
                'longitude': self.fake.longitude(),
                'country': user.profile['region']
            },
            'is_fraudulent': is_fraudulent
        }

        # Add transaction to user history
        user.transaction_history.append(transaction)
        return transaction

    def _generate_device_info(self) -> Dict:
        device_type = random.choice(self.config.DEVICE_TYPES)
        return {
            'device_type': device_type,
            'os': random.choice(self.config.OS_TYPES),
            'browser': random.choice(self.config.BROWSER_TYPES),
            'browser_language': self.fake.language_code(),
            'screen_resolution': random.choice(['1920x1080', '1366x768', '2560x1440']),
            'device_fingerprint': self.fake.sha256()
        }

class FeatureEngineering:
    """Class for feature extraction and engineering"""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_embedder = self._initialize_text_embedder()
        self.pca = PCA(n_components=50)
        self.tsne = TSNE(n_components=2, random_state=42)

    def _initialize_text_embedder(self):
        """Initialize text embedding model"""
        try:
            return AutoModelForTokenClassification.from_pretrained('bert-base-uncased')
        except Exception as e:
            logger.error(f"Error initializing text embedder: {e}")
            return None

    def extract_features(self, transaction: Dict) -> np.ndarray:
        """Extract features from a transaction"""
        # Numerical features
        numerical_features = np.array([
            float(transaction['amount']),
            hour_of_day_sin(transaction['timestamp']),
            hour_of_day_cos(transaction['timestamp'])
        ])

        # Text features
        text = f"{transaction['merchant_name']} {transaction['merchant_category']}"
        text_features = self._extract_text_features(text)

        # Combine features
        return np.concatenate([numerical_features, text_features])

    def _extract_text_features(self, text: str) -> np.ndarray:
        """Extract text features using BERT"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            with torch.no_grad():
                outputs = self.text_embedder(**inputs)
            return outputs.logits[0].mean(dim=0).numpy()
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return np.zeros(768)  # BERT hidden size

def generate_dataset(
    num_records: int,
    output_path: str,
    batch_size: int = 10000
) -> None:
    """Generate and save the synthetic dataset"""
    generator = TransactionGenerator(num_users=num_records // 100)
    feature_engineering = FeatureEngineering()
    
    os.makedirs(output_path, exist_ok=True)
    
    batch_num = 0
    transactions = []
    features = []
    
    with tqdm(total=num_records) as pbar:
        while len(transactions) < num_records:
            # Select random user
            user = random.choice(generator.users)
            
            # Determine if transaction should be fraudulent
            is_fraudulent = random.random() < 0.001  # 0.1% fraud rate
            
            # Generate transaction
            transaction = generator.generate_transaction(user, is_fraudulent)
            feature_vector = feature_engineering.extract_features(transaction)
            
            transactions.append(transaction)
            features.append(feature_vector)
            
            # Save batch if reached batch_size
            if len(transactions) >= batch_size:
                save_batch(
                    transactions[:batch_size],
                    features[:batch_size],
                    output_path,
                    batch_num
                )
                transactions = transactions[batch_size:]
                features = features[batch_size:]
                batch_num += 1
            
            pbar.update(1)

def save_batch(
    transactions: List[Dict],
    features: List[np.ndarray],
    output_path: str,
    batch_num: int
) -> None:
    """Save batch of transactions and features"""
    # Save transactions
    df = pd.DataFrame(transactions)
    df.to_parquet(
        os.path.join(output_path, f'transactions_batch_{batch_num}.parquet')
    )
    
    # Save features
    np.save(
        os.path.join(output_path, f'features_batch_{batch_num}.npy'),
        np.array(features)
    )

def hour_of_day_sin(timestamp_str: str) -> float:
    """Convert hour to sine for cyclical feature"""
    hour = datetime.fromisoformat(timestamp_str).hour
    return np.sin(2 * np.pi * hour / 24)

def hour_of_day_cos(timestamp_str: str) -> float:
    """Convert hour to cosine for cyclical feature"""
    hour = datetime.fromisoformat(timestamp_str).hour
    return np.cos(2 * np.pi * hour / 24)

if __name__ == "__main__":
    start_time = time.time()
    
    # Configuration
    NUM_RECORDS = 100_000  # Change this for full dataset
    OUTPUT_PATH = "synthetic_transactions"
    
    try:
        logger.info("Starting synthetic transaction generation...")
        generate_dataset(NUM_RECORDS, OUTPUT_PATH)
        
        duration = time.time() - start_time
        logger.info(f"Generation completed in {duration:.2f} seconds")
        
        # Basic statistics
        total_files = len(os.listdir(OUTPUT_PATH))
        logger.info(f"Generated {total_files} batch files in {OUTPUT_PATH}")
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        raise