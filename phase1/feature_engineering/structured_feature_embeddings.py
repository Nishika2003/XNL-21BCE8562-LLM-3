import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import logging

if not os.path.exists('../logs'):
    os.makedirs('../logs')

logging.basicConfig(filename='../logs/structured_feature_embeddings.log', level=logging.INFO, format='%(asctime)s - %(message)s')

try:
    df = pd.read_csv('../datasets/transactions_1.csv')
    numerical_features = df[['amount', 'credit_score', 'network_latency']]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numerical_features)
    
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(scaled_features)
    
    pd.DataFrame(embeddings, columns=['PC1', 'PC2']).to_parquet('../datasets/embeddings/embeddings.parquet')
    logging.info("Structured embeddings generation completed successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
