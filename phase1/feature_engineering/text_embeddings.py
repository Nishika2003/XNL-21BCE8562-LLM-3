import pandas as pd
from gensim.models import Word2Vec, FastText
from sklearn.preprocessing import StandardScaler
import os
import logging

# Create logs directory if it doesn't exist
if not os.path.exists('../logs'):
    os.makedirs('../logs')

logging.basicConfig(filename='../logs/text_embeddings.log', level=logging.INFO, format='%(asctime)s - %(message)s')

try:
    # Load transaction data
    df = pd.read_csv('../datasets/transactions_1.csv')

    # Combine text-based columns for text embeddings
    text_data = df['session_metadata'].astype(str) + " " + df['merchant'].astype(str)

    # Tokenize text data
    tokenized_data = [text.split() for text in text_data]

    # Word2Vec Model
    word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_embeddings = [word2vec_model.wv[text].mean(axis=0) if text in word2vec_model.wv else [0]*100 for text in tokenized_data]

    # FastText Model
    fasttext_model = FastText(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
    fasttext_embeddings = [fasttext_model.wv[text].mean(axis=0) if text in fasttext_model.wv else [0]*100 for text in tokenized_data]

    # Combine embeddings with original data
    embeddings_df = pd.DataFrame(word2vec_embeddings, columns=[f'w2v_{i}' for i in range(100)])
    fasttext_df = pd.DataFrame(fasttext_embeddings, columns=[f'ft_{i}' for i in range(100)])

    # Concatenate embeddings
    combined_df = pd.concat([df, embeddings_df, fasttext_df], axis=1)

    # Save embeddings to file
    combined_df.to_parquet('../datasets/embeddings/text_embeddings.parquet')
    logging.info("Word2Vec & FastText embeddings generation completed successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}")
