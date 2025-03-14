import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from kafka import KafkaConsumer, KafkaProducer
import dask.dataframe as dd
import dask.array as da
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import json
import logging
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.spark = self._initialize_spark()
        self.kafka_consumer = self._initialize_kafka_consumer()
        self.kafka_producer = self._initialize_kafka_producer()
        self.feature_engineering = FeatureEngineering()
        self.vector_search = VectorSearch()
        self.fraud_scorer = FraudScorer()
        self.alert_system = AlertSystem()
        self.xai_explainer = XAIExplainer()
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # Initialize processing queues
        self.preprocessing_queue = Queue()
        self.embedding_queue = Queue()
        self.scoring_queue = Queue()
        
    def _initialize_spark(self) -> SparkSession:
        """Initialize Spark session with required configurations"""
        return SparkSession.builder \
            .appName("FraudDetectionPipeline") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .getOrCreate()
    
    def _initialize_kafka_consumer(self) -> KafkaConsumer:
        """Initialize Kafka consumer for incoming transactions"""
        return KafkaConsumer(
            'transactions',
            bootstrap_servers=self.config['kafka_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='fraud_detection_group',
            auto_offset_reset='latest'
        )
    
    def _initialize_kafka_producer(self) -> KafkaProducer:
        """Initialize Kafka producer for alerts"""
        return KafkaProducer(
            bootstrap_servers=self.config['kafka_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

    async def process_transaction(self, transaction: Dict) -> Dict:
        """Process a single transaction through the entire pipeline"""
        try:
            # ETL Preprocessing
            preprocessed_data = await self.preprocess_transaction(transaction)
            
            # Feature Engineering
            features = await self.feature_engineering.extract_features(preprocessed_data)
            
            # Generate Embeddings
            embeddings = await self.generate_embeddings(features)
            
            # Vector Search
            similar_transactions = await self.vector_search.find_similar(embeddings)
            
            # Fraud Scoring
            fraud_score, fraud_indicators = await self.fraud_scorer.score_transaction(
                transaction=preprocessed_data,
                features=features,
                embeddings=embeddings,
                similar_transactions=similar_transactions
            )
            
            # Generate XAI Explanation
            explanation = await self.xai_explainer.explain_prediction(
                features=features,
                fraud_score=fraud_score,
                fraud_indicators=fraud_indicators
            )
            
            # Send Alert if Necessary
            if fraud_score > self.config['fraud_threshold']:
                await self.alert_system.send_alert(
                    transaction=preprocessed_data,
                    fraud_score=fraud_score,
                    explanation=explanation
                )
            
            return {
                'transaction_id': transaction['transaction_id'],
                'fraud_score': fraud_score,
                'explanation': explanation,
                'processing_time': datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            raise

    async def preprocess_transaction(self, transaction: Dict) -> Dict:
        """Preprocess transaction using PySpark"""
        # Convert transaction to Spark DataFrame
        df = self.spark.createDataFrame([transaction])
        
        # Apply preprocessing transformations
        preprocessed_df = df.withColumn(
            'amount_normalized',
            (col('amount') - col('amount').mean()) / col('amount').stddev()
        ).withColumn(
            'timestamp_hour',
            hour(col('timestamp'))
        )
        
        return preprocessed_df.toPandas().to_dict('records')[0]

class FeatureEngineering:
    def __init__(self):
        self.dask_client = dd.compute.__enter__()
    
    async def extract_features(self, transaction: Dict) -> Dict:
        """Extract features using Dask for parallel processing"""
        df = dd.from_pandas(pd.DataFrame([transaction]), npartitions=1)
        
        # Parallel feature computation
        features = {
            'numerical_features': await self._compute_numerical_features(df),
            'categorical_features': await self._compute_categorical_features(df),
            'temporal_features': await self._compute_temporal_features(df)
        }
        
        return features
    
    async def _compute_numerical_features(self, df: dd.DataFrame) -> Dict:
        """Compute numerical features in parallel"""
        computations = [
            df['amount'].mean(),
            df['amount'].std(),
            df['amount'].max(),
            df['amount'].min()
        ]
        results = await self.dask_client.compute(*computations)
        return dict(zip(['mean', 'std', 'max', 'min'], results))

class XAIExplainer:
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = None
        
    def initialize_explainers(self, model, feature_names: List[str]):
        """Initialize SHAP and LIME explainers"""
        self.feature_names = feature_names
        self.shap_explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=['legitimate', 'fraudulent'],
            mode='classification'
        )
    
    async def explain_prediction(
        self,
        features: Dict,
        fraud_score: float,
        fraud_indicators: List[str]
    ) -> Dict:
        """Generate both SHAP and LIME explanations"""
        feature_vector = self._prepare_feature_vector(features)
        
        # Generate SHAP values
        shap_values = await self._generate_shap_explanation(feature_vector)
        
        # Generate LIME explanation
        lime_explanation = await self._generate_lime_explanation(feature_vector)
        
        # Combine explanations
        explanation = {
            'shap_values': shap_values.tolist(),
            'lime_explanation': lime_explanation,
            'fraud_indicators': fraud_indicators,
            'feature_importance': self._get_feature_importance(shap_values),
            'human_readable': self._generate_human_readable_explanation(
                shap_values,
                lime_explanation,
                fraud_indicators
            )
        }
        
        return explanation
    
    def _generate_human_readable_explanation(
        self,
        shap_values: np.ndarray,
        lime_explanation: Dict,
        fraud_indicators: List[str]
    ) -> str:
        """Generate human-readable explanation of the fraud prediction"""
        # Get top contributing features
        feature_importances = list(zip(self.feature_names, shap_values))
        top_features = sorted(
            feature_importances,
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        explanation_parts = [
            "Transaction flagged for the following reasons:"
        ]
        
        for feature, importance in top_features:
            direction = "higher" if importance > 0 else "lower"
            explanation_parts.append(
                f"- {feature} is {direction} than normal "
                f"(impact: {abs(importance):.2f})"
            )
        
        if fraud_indicators:
            explanation_parts.append("\nSpecific risk indicators:")
            explanation_parts.extend([f"- {indicator}" for indicator in fraud_indicators])
        
        return "\n".join(explanation_parts)

class DashboardApp:
    def __init__(self, pipeline: FraudDetectionPipeline):
        self.pipeline = pipeline
        self.app = dash.Dash(__name__)
        self.setup_layout()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Fraud Detection Dashboard"),
            
            # Real-time monitoring
            html.Div([
                html.H2("Real-time Transaction Monitoring"),
                dcc.Graph(id='live-transactions-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=1*1000,  # in milliseconds
                    n_intervals=0
                )
            ]),
            
            # XAI Visualization
            html.Div([
                html.H2("Transaction Explanation"),
                html.Div(id='transaction-explanation'),
                dcc.Graph(id='feature-importance-plot')
            ]),
            
            # Transaction Details
            html.Div([
                html.H2("Transaction Details"),
                html.Pre(id='transaction-details')
            ])
        ])
        
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            Output('live-transactions-graph', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_transactions_graph(n):
            # Get recent transactions
            recent_transactions = self.pipeline.get_recent_transactions()
            
            # Create scatter plot
            fig = px.scatter(
                recent_transactions,
                x='timestamp',
                y='fraud_score',
                color='fraud_score',
                hover_data=['transaction_id', 'amount']
            )
            
            return fig
        
        @self.app.callback(
            [Output('transaction-explanation', 'children'),
             Output('feature-importance-plot', 'figure'),
             Output('transaction-details', 'children')],
            Input('live-transactions-graph', 'clickData')
        )
        def display_transaction_details(clickData):
            if not clickData:
                return "Select a transaction to view details.", {}, ""
            
            # Get transaction details
            transaction_id = clickData['points'][0]['customdata'][0]
            transaction = self.pipeline.get_transaction_details(transaction_id)
            
            # Get explanation
            explanation = transaction['explanation']
            
            # Create feature importance plot
            importance_fig = px.bar(
                x=explanation['feature_importance'].keys(),
                y=explanation['feature_importance'].values(),
                title="Feature Importance"
            )
            
            return (
                explanation['human_readable'],
                importance_fig,
                json.dumps(transaction, indent=2)
            )
    
    def run(self, debug=False, port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=port)

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(config)
    
    # Start dashboard
    dashboard = DashboardApp(pipeline)
    
    # Run dashboard in separate thread
    dashboard_thread = threading.Thread(
        target=dashboard.run,
        kwargs={'debug': True}
    )
    dashboard_thread.start()
    
    # Start processing transactions
    asyncio.run(pipeline.start_processing())

if __name__ == "__main__":
    main()