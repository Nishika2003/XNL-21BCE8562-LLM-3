import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from kafka import KafkaProducer, KafkaConsumer
import json
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
import pinecone
from annoy import AnnoyIndex
import scann
import threading
from queue import Queue
import time
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
from collections import defaultdict
from your_file import PINECONE_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    transaction_id: str
    distance: float
    similarity_score: float
    timestamp: datetime

class VectorIndexConfig:
    def __init__(self):
        self.dimension = 768  # BERT embedding dimension
        self.n_lists = 100    # Number of cells for IVF
        self.n_probes = 10    # Number of cells to probe
        self.ef_construction = 200  # HNSW parameter
        self.ef_search = 50    # HNSW parameter
        self.m = 16           # HNSW parameter

class DistributedVectorSearch:
    def __init__(self, config: VectorIndexConfig):
        self.config = config
        self.faiss_index = self._create_faiss_index()
        self.annoy_index = self._create_annoy_index()
        self.scann_index = None  # Will be initialized with data
        self.pinecone_index = self._initialize_pinecone()
        
        # Initialize Cassandra connection
        self.cassandra_cluster = Cluster(['localhost'])
        self.cassandra_session = self.cassandra_cluster.connect()
        
        # Initialize Kafka
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Queue for batch processing
        self.index_queue = Queue()
        self.batch_size = 1000
        self.batch_timeout = 1.0  # seconds
        
        # Start background indexing thread
        self.indexing_thread = threading.Thread(target=self._batch_indexing_worker)
        self.indexing_thread.daemon = True
        self.indexing_thread.start()

    def _create_faiss_index(self) -> faiss.Index:
        """Create HNSW index with IVF for better scaling"""
        # Quantizer for IVF
        quantizer = faiss.IndexHNSWFlat(
            self.config.dimension,
            self.config.m,
            faiss.METRIC_L2
        )
        
        # Create IVF_HNSW index
        index = faiss.IndexIVF_HNSW_Quantizer(
            quantizer,
            self.config.dimension,
            self.config.n_lists,
            faiss.METRIC_L2
        )
        
        # Configure HNSW parameters
        index.hnsw.efConstruction = self.config.ef_construction
        index.hnsw.efSearch = self.config.ef_search
        
        return index

    def _create_annoy_index(self) -> AnnoyIndex:
        """Create Annoy index for backup similarity search"""
        index = AnnoyIndex(self.config.dimension, 'angular')
        index.set_ef(self.config.ef_search)
        return index

    def _initialize_pinecone(self) -> pinecone.Index:
        """Initialize Pinecone index"""
        pinecone.init(api_key=PINECONE_API_KEY)
        return pinecone.Index('fraud-detection')

    async def _initialize_scann(self, initial_vectors: np.ndarray):
        """Initialize ScaNN index with initial vectors"""
        self.scann_index = scann.scann_ops_pybind.builder(
            initial_vectors,
            self.config.n_lists,
            "dot_product"
        ).tree(
            num_leaves=2000,
            num_leaves_to_search=100,
            training_sample_size=250000
        ).score_ah(
            2,
            anisotropic_quantization_threshold=0.2
        ).reorder(100).build()

    def _create_cassandra_schema(self):
        """Create Cassandra schema for vector storage"""
        self.cassandra_session.execute("""
            CREATE KEYSPACE IF NOT EXISTS fraud_detection
            WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3}
        """)
        
        self.cassandra_session.execute("""
            CREATE TABLE IF NOT EXISTS fraud_detection.vectors (
                transaction_id text PRIMARY KEY,
                vector blob,
                metadata text,
                timestamp timestamp
            )
        """)

    async def add_vector(self, transaction_id: str, vector: np.ndarray, metadata: Dict):
        """Add vector to all indices"""
        # Add to queue for batch processing
        self.index_queue.put({
            'transaction_id': transaction_id,
            'vector': vector,
            'metadata': metadata,
            'timestamp': datetime.now()
        })
        
        # Store in Cassandra
        await self._store_in_cassandra(transaction_id, vector, metadata)

    async def _store_in_cassandra(self, transaction_id: str, vector: np.ndarray, metadata: Dict):
        """Store vector in Cassandra"""
        query = """
            INSERT INTO fraud_detection.vectors 
            (transaction_id, vector, metadata, timestamp)
            VALUES (?, ?, ?, ?)
        """
        
        await self.cassandra_session.execute_async(
            query,
            (transaction_id, vector.tobytes(), json.dumps(metadata), datetime.now())
        )

    def _batch_indexing_worker(self):
        """Background worker for batch indexing"""
        while True:
            batch = []
            batch_start = time.time()
            
            while len(batch) < self.batch_size and \
                  time.time() - batch_start < self.batch_timeout:
                try:
                    item = self.index_queue.get(timeout=self.batch_timeout)
                    batch.append(item)
                except Queue.Empty:
                    break
            
            if batch:
                self._process_batch(batch)

    def _process_batch(self, batch: List[Dict]):
        """Process a batch of vectors"""
        vectors = np.vstack([item['vector'] for item in batch])
        
        # Update FAISS index
        self.faiss_index.add(vectors)
        
        # Update Annoy index
        for i, item in enumerate(batch):
            self.annoy_index.add_item(i, item['vector'])
        
        # Update Pinecone
        self.pinecone_index.upsert([
            (item['transaction_id'], item['vector'].tolist())
            for item in batch
        ])
        
        # Notify Kafka about updates
        self.kafka_producer.send(
            'index_updates',
            {'batch_size': len(batch), 'timestamp': datetime.now().isoformat()}
        )

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        timeout: float = 0.01
    ) -> List[SearchResult]:
        """
        Perform parallel search across all indices with timeout
        Returns best results within timeout period
        """
        results_queue = Queue()
        
        # Start parallel searches
        search_tasks = [
            self.thread_pool.submit(self._faiss_search, query_vector, k),
            self.thread_pool.submit(self._annoy_search, query_vector, k),
            self.thread_pool.submit(self._scann_search, query_vector, k),
            self.thread_pool.submit(self._pinecone_search, query_vector, k)
        ]
        
        # Wait for results with timeout
        start_time = time.time()
        results = defaultdict(list)
        
        while time.time() - start_time < timeout and len(results) < len(search_tasks):
            for future in search_tasks:
                if future.done():
                    try:
                        search_results = future.result()
                        results[future].extend(search_results)
                    except Exception as e:
                        logger.error(f"Search error: {e}")
        
        # Merge and deduplicate results
        return self._merge_results(results.values(), k)

    def _merge_results(
        self,
        result_sets: List[List[SearchResult]],
        k: int
    ) -> List[SearchResult]:
        """Merge results from different indices and return top k"""
        # Combine all results
        all_results = []
        seen_ids = set()
        
        for result_set in result_sets:
            for result in result_set:
                if result.transaction_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.transaction_id)
        
        # Sort by similarity score and return top k
        return sorted(
            all_results,
            key=lambda x: x.similarity_score,
            reverse=True
        )[:k]

    def _faiss_search(
        self,
        query_vector: np.ndarray,
        k: int
    ) -> List[SearchResult]:
        """Search using FAISS index"""
        distances, indices = self.faiss_index.search(
            query_vector.reshape(1, -1),
            k
        )
        
        return [
            SearchResult(
                transaction_id=str(idx),
                distance=float(dist),
                similarity_score=1.0 / (1.0 + float(dist)),
                timestamp=datetime.now()
            )
            for idx, dist in zip(indices[0], distances[0])
        ]

class RealTimeDetector:
    def __init__(
        self,
        vector_search: DistributedVectorSearch,
        similarity_threshold: float = 0.8
    ):
        self.vector_search = vector_search
        self.similarity_threshold = similarity_threshold
        self.recent_transactions = {}  # In-memory cache
        
    async def process_transaction(
        self,
        transaction_id: str,
        vector: np.ndarray,
        metadata: Dict
    ) -> Dict:
        """Process a transaction in real-time"""
        start_time = time.time()
        
        # Perform similarity search
        similar_transactions = await self.vector_search.search(
            vector,
            k=10,
            timeout=0.01
        )
        
        # Calculate fraud score based on similarity
        fraud_score = self._calculate_fraud_score(similar_transactions)
        
        # Store transaction
        await self.vector_search.add_vector(transaction_id, vector, metadata)
        
        processing_time = time.time() - start_time
        
        return {
            'transaction_id': transaction_id,
            'fraud_score': fraud_score,
            'processing_time_ms': processing_time * 1000,
            'similar_transactions': similar_transactions
        }
    
    def _calculate_fraud_score(
        self,
        similar_transactions: List[SearchResult]
    ) -> float:
        """Calculate fraud score based on similarity to past transactions"""
        if not similar_transactions:
            return 0.5  # Default score for no similar transactions
        
        # Weight recent transactions more heavily
        weighted_scores = []
        for result in similar_transactions:
            age_hours = (datetime.now() - result.timestamp).total_seconds() / 3600
            time_weight = 1.0 / (1.0 + age_hours)
            weighted_scores.append(result.similarity_score * time_weight)
        
        return sum(weighted_scores) / len(weighted_scores)

async def main():
    # Initialize components
    config = VectorIndexConfig()
    vector_search = DistributedVectorSearch(config)
    detector = RealTimeDetector(vector_search)
    
    # Example usage
    test_vector = np.random.random(config.dimension).astype('float32')
    result = await detector.process_transaction(
        transaction_id="test-123",
        vector=test_vector,
        metadata={'amount': 100.0, 'merchant': 'Test Store'}
    )
    
    logger.info(f"Processing result: {result}")

if __name__ == "__main__":
    asyncio.run(main())