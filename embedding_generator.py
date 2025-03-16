# fine_tuning_and_embedding_generation.py

import torch
import torch.nn as nn
import openai
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, 
    RobertaTokenizer, RobertaForSequenceClassification, T5ForConditionalGeneration
)
import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


class TransactionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        transaction = self.dataframe.iloc[idx]
        text = transaction['text']
        label = transaction['label']
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def fine_tune_model(model_name, dataset_path, output_dir, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Load dataset
    df = pd.read_parquet(dataset_path)
    train_size = int(0.8 * len(df))
    train_df, val_df = df[:train_size], df[train_size:]

    # Create datasets
    train_dataset = TransactionDataset(train_df, tokenizer)
    val_dataset = TransactionDataset(val_df, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_total_limit=1,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train model
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")


def get_gpt4_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']


def generate_embeddings(model_name, dataset_path, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Load dataset
    df = pd.read_parquet(dataset_path)

    embeddings = []
    results = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = row['text']
            transaction_id = row['transaction_id']
            label = row['label']
            amount = row['amount']

            # Generate RoBERTa embedding
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
            outputs = model(**inputs)
            roberta_embedding = outputs.logits.squeeze().numpy()

            # Generate GPT-4 embedding
            gpt4_embedding = get_gpt4_embedding(text)

            # Combine embeddings (Simple Concatenation)
            combined_embedding = np.concatenate([roberta_embedding, gpt4_embedding])

            results.append({
                'transaction_id': transaction_id,
                'embedding': combined_embedding,
                'label': label,
                'text': text,
                'amount': amount
            })

    # Save results
    embeddings_df = pd.DataFrame(results)
    embeddings_df.to_parquet(output_file)
    logger.info(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    # Fine-tuning example
    fine_tune_model(
        model_name="roberta-base",
        dataset_path="synthetic_transactions/transactions.parquet",
        output_dir="fine_tuned_models/roberta_base"
    )

    # Embedding generation example
    generate_embeddings(
        model_name="fine_tuned_models/roberta_base",
        dataset_path="synthetic_transactions/transactions.parquet",
        output_file="embeddings/combined_embeddings.parquet"
    )