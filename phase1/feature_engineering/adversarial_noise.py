import pandas as pd
import numpy as np
import os
import logging

if not os.path.exists('../logs'):
    os.makedirs('../logs')

logging.basicConfig(filename='../logs/adversarial_noise.log', level=logging.INFO, format='%(asctime)s - %(message)s')

try:
    df = pd.read_parquet('../datasets/embeddings/embeddings.parquet')
    noise_level = 0.05
    noisy_embeddings = df.values + noise_level * np.random.randn(*df.values.shape)

    noisy_df = pd.DataFrame(noisy_embeddings, columns=['PC1', 'PC2'])
    noisy_df.to_parquet('../datasets/embeddings/noisy_embeddings.parquet')
    logging.info("Adversarial noise generation complete.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
