import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import logging

if not os.path.exists('../logs'):
    os.makedirs('../logs')

logging.basicConfig(filename='../logs/visualize_embeddings.log', level=logging.INFO, format='%(asctime)s - %(message)s')

try:
    embeddings = pd.read_parquet('../datasets/embeddings/embeddings.parquet').values
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=1, alpha=0.6)
    plt.title('t-SNE Visualization of Transaction Embeddings')
    plt.savefig('../datasets/visualizations/tsne_plot.png')
    plt.show()
    
    logging.info("t-SNE visualization complete.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
