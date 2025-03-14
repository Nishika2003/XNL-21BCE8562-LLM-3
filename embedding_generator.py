import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    T5ForConditionalGeneration, 
    TrainingArguments,
    Trainer
)
from torch_geometric import nn as gnn
from torch_geometric.data import Data
import h5py
import onnx
import onnxruntime as ort
from sklearn.ensemble import VotingClassifier, StackingClassifier
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import logging
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionEmbeddingGenerator:
    def __init__(self):
        self.finbert = AutoModel.from_pretrained('yiyanghkust/finbert-tone')
        self.roberta = AutoModel.from_pretrained('roberta-base')
        self.tokenizer_finbert = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to GPU if available
        self.finbert = self.finbert.to(self.device)
        self.roberta = self.roberta.to(self.device)
        
    def generate_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """GenerateHere's the implementation of Phase 2 focusing on hybrid embeddings and multi-LLM integration:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    T5Tokenizer, T5ForConditionalGeneration,
    RobertaTokenizer, RobertaModel
)
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GAT
import h5py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.ensemble import StackingClassifier, VotingClassifier
import onnx
import onnxruntime
from tqdm import tqdm
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionDataset(Dataset):
    def __init__(self, transactions: List[Dict], tokenizer):
        self.transactions = transactions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, idx):
        transaction = self.transactions[idx]
        text = self._create_transaction_text(transaction)
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'label': torch.tensor(1 if transaction['is_fraudulent'] else 0)
        }

    def _create_transaction_text(self, transaction: Dict) -> str:
        return f"Transaction of ${transaction['amount']} at {transaction['merchant_name']} "\
               f"in category {transaction['merchant_category']}"

class HybridEmbeddingGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize models
        self.finbert = self._init_finbert()
        self.roberta = self._init_roberta()
        self.t5 = self._init_t5()
        self.gnn = self._init_gnn()
        
        # Initialize tokenizers
        self.finbert_tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def _init_finbert(self):
        model = AutoModel.from_pretrained('yiyanghkust/finbert-tone')
        model.to(self.device)
        return model

    def _init_roberta(self):
        model = RobertaModel.from_pretrained('roberta-base')
        model.to(self.device)
        return model

    def _init_t5(self):
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        model.to(self.device)
        return model

    def _init_gnn(self):
        class GNN(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(GNN, self).__init__()
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, output_dim)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                return self.conv2(x, edge_index)

        model = GNN(input_dim=768, hidden_dim=256, output_dim=128)
        model.to(self.device)
        return model

    def generate_embeddings(self, transaction: Dict) -> Dict[str, torch.Tensor]:
        text = f"Transaction of ${transaction['amount']} at {transaction['merchant_name']}"
        
        # Generate FinBERT embeddings
        finbert_inputs = self.finbert_tokenizer(
            text, return_tensors='pt', padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            finbert_outputs = self.finbert(**finbert_inputs)
            finbert_embedding = finbert_outputs.last_hidden_state.mean(dim=1)

        # Generate RoBERTa embeddings
        roberta_inputs = self.roberta_tokenizer(
            text, return_tensors='pt', padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            roberta_outputs = self.roberta(**roberta_inputs)
            roberta_embedding = roberta_outputs.last_hidden_state.mean(dim=1)

        return {
            'finbert': finbert_embedding,
            'roberta': roberta_embedding
        }

class ContrastiveLearning(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.encoder = base_encoder
        self.projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.projector(h)

class MultiLLMFraudDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.embedding_generator = HybridEmbeddingGenerator(device)
        self.models = self._initialize_models()
        self.ensemble = self._create_ensemble()

    def _initialize_models(self):
        return {
            'finbert': AutoModel.from_pretrained('yiyanghkust/finbert-tone'),
            't5': T5ForConditionalGeneration.from_pretrained('t5-base')
        }

    def _create_ensemble(self):
        estimators = [
            ('finbert', self.models['finbert']),
            ('t5', self.models['t5'])
        ]
        return VotingClassifier(estimators=estimators, voting='soft')

    def process_transaction(self, transaction: Dict) -> Dict:
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(transaction)
        
        # Get model predictions
        predictions = {}
        for model_name, embedding in embeddings.items():
            predictions[model_name] = self._get_model_prediction(
                model_name, embedding
            )

        # Combine predictions using ensemble
        ensemble_prediction = self.ensemble.predict_proba(
            torch.cat(list(embeddings.values()), dim=1).cpu().numpy()
        )

        return {
            'fraud_score': float(ensemble_prediction[0][1]),
            'model_predictions': predictions,
            'embeddings': {k: v.cpu().numpy() for k, v in embeddings.items()}
        }

    def _get_model_prediction(self, model_name: str, embedding: torch.Tensor) -> float:
        with torch.no_grad():
            if model_name == 'finbert':
                logits = self.models['finbert'](embedding)[0]
            elif model_name == 't5':
                logits = self.models['t5'].encoder(embedding)[0]
            
            return torch.sigmoid(logits.mean()).item()

class EmbeddingStorage:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.h5_file = None

    def __enter__(self):
        self.h5_file = h5py.File(self.file_path, 'a')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5_file:
            self.h5_file.close()

    def store_embeddings(self, transaction_id: str, embeddings: Dict[str, np.ndarray]):
        group = self.h5_file.create_group(transaction_id)
        for model_name, embedding in embeddings.items():
            group.create_dataset(model_name, data=embedding)

def export_to_onnx(model, sample_input, output_path):
    """Export PyTorch model to ONNX format"""
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )

def main():
    # Initialize components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_generator = HybridEmbeddingGenerator(device)
    fraud_detector = MultiLLMFraudDetector(device)

    # Load sample transactions
    transactions = pd.read_parquet('synthetic_transactions/transactions_batch_0.parquet')

    # Process transactions and store embeddings
    with EmbeddingStorage('embeddings.h5') as storage:
        for _, transaction in tqdm(transactions.iterrows(), total=len(transactions)):
            # Generate embeddings
            embeddings = embedding_generator.generate_embeddings(transaction.to_dict())
            
            # Store embeddings
            storage.store_embeddings(
                transaction['transaction_id'],
                {k: v.cpu().numpy() for k, v in embeddings.items()}
            )

            # Get fraud prediction
            prediction = fraud_detector.process_transaction(transaction.to_dict())
            logger.info(f"Fraud score for transaction {transaction['transaction_id']}: {prediction['fraud_score']}")

    # Export models to ONNX
    sample_text = "Transaction of $100 at Sample Merchant"
    sample_input = embedding_generator.finbert_tokenizer(
        sample_text, return_tensors='pt'
    ).to(device)
    export_to_onnx(embedding_generator.finbert, sample_input, 'finbert.onnx')

if __name__ == "__main__":
    main()