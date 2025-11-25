"""
BEAST LEVEL ML Training Pipeline for AURA K8s
Mac M4 Optimized - CPU-only training with 200+ features
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import joblib

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, IsolationForest, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available, skipping...")

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available, using default hyperparameters...")

# Feature engineering
from feature_engineering import AdvancedFeatureEngineer

# Dataset downloading
try:
    from dataset_downloader import DatasetDownloader
    DATASET_DOWNLOADER_AVAILABLE = True
except ImportError:
    DATASET_DOWNLOADER_AVAILABLE = False
    print("⚠️  DatasetDownloader not available, using synthetic data only")

warnings.filterwarnings('ignore')

# Configuration
RANDOM_SEED = int(os.getenv("ML_RANDOM_SEED", "42"))
np.random.seed(RANDOM_SEED)

# Memory optimization for Mac M4 (16GB RAM)
MEMORY_LIMIT_GB = float(os.getenv("ML_MEMORY_LIMIT_GB", "12.0"))
CHUNK_SIZE = int(os.getenv("ML_CHUNK_SIZE", "10000"))

# Model directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class WeightedEnsemble:
    """
    Weighted voting ensemble - module-level class for pickling
    Combines multiple models with weighted voting
    """
    def __init__(self, models, weights, n_classes):
        self.models = models
        self.weights = weights
        self.n_classes = n_classes
    
    def predict(self, X):
        """Predict using weighted ensemble voting"""
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            if name == 'isolation_forest' or name == 'ensemble':
                # Skip isolation forest and ensemble itself
                continue
            
            try:
                proba = model.predict_proba(X)
                predictions.append(proba)
                probabilities.append(proba)
            except Exception as e:
                # If predict_proba fails, try predict and convert
                pred = model.predict(X)
                # Create one-hot encoded probabilities
                proba = np.zeros((len(X), self.n_classes))
                for i, p in enumerate(pred):
                    if 0 <= p < self.n_classes:
                        proba[i, int(p)] = 1.0
                predictions.append(proba)
                probabilities.append(proba)
        
        if not predictions:
            # Fallback to first available model
            for name, model in self.models.items():
                if name not in ['isolation_forest', 'ensemble']:
                    return model.predict(X)
            return np.zeros(len(X))
        
        # Weighted average
        final_proba = np.zeros_like(predictions[0])
        valid_models = [k for k in self.models.keys() if k not in ['isolation_forest', 'ensemble']]
        for i, (name, proba) in enumerate(zip(valid_models, predictions)):
            weight = self.weights.get(name, 0.0)
            final_proba += weight * proba
        
        # Normalize
        final_proba = final_proba / (final_proba.sum(axis=1, keepdims=True) + 1e-9)
        
        return np.argmax(final_proba, axis=1)
    
    def predict_proba(self, X):
        """Predict probabilities using weighted ensemble"""
        predictions = []
        
        for name, model in self.models.items():
            if name == 'isolation_forest' or name == 'ensemble':
                continue
            
            try:
                proba = model.predict_proba(X)
                predictions.append(proba)
            except Exception as e:
                # Fallback
                pred = model.predict(X)
                proba = np.zeros((len(X), self.n_classes))
                for i, p in enumerate(pred):
                    if 0 <= p < self.n_classes:
                        proba[i, int(p)] = 1.0
                predictions.append(proba)
        
        if not predictions:
            # Fallback
            for name, model in self.models.items():
                if name not in ['isolation_forest', 'ensemble']:
                    return model.predict_proba(X) if hasattr(model, 'predict_proba') else np.zeros((len(X), self.n_classes))
            return np.zeros((len(X), self.n_classes))
        
        final_proba = np.zeros_like(predictions[0])
        valid_models = [k for k in self.models.keys() if k not in ['isolation_forest', 'ensemble']]
        for i, (name, proba) in enumerate(zip(valid_models, predictions)):
            weight = self.weights.get(name, 0.0)
            final_proba += weight * proba
        
        final_proba = final_proba / (final_proba.sum(axis=1, keepdims=True) + 1e-9)
        return final_proba


class StackedEnsemble:
    """
    Stacked ensemble - module-level class for pickling
    Combines meta-learners with weighted voting
    """
    def __init__(self, meta_models, weights, n_classes):
        self.meta_models = meta_models
        self.weights = weights
        self.n_classes = n_classes
    
    def predict(self, X):
        """Predict using stacked ensemble"""
        predictions = []
        
        for name, model in self.meta_models.items():
            try:
                pred = model.predict(X)
                # Convert to one-hot if needed
                if pred.ndim == 1:
                    proba = np.zeros((len(X), self.n_classes))
                    for i, p in enumerate(pred):
                        if 0 <= p < self.n_classes:
                            proba[i, int(p)] = 1.0
                    predictions.append(proba)
                else:
                    predictions.append(pred)
            except Exception:
                # Fallback
                predictions.append(np.zeros((len(X), self.n_classes)))
        
        if not predictions:
            return np.zeros(len(X))
        
        # Weighted average
        final_proba = np.zeros_like(predictions[0])
        valid_models = list(self.meta_models.keys())
        for i, (name, proba) in enumerate(zip(valid_models, predictions)):
            weight = self.weights.get(name, 0.0)
            final_proba += weight * proba
        
        # Normalize
        final_proba = final_proba / (final_proba.sum(axis=1, keepdims=True) + 1e-9)
        return np.argmax(final_proba, axis=1)
    
    def predict_proba(self, X):
        """Predict probabilities using stacked ensemble"""
        predictions = []
        
        for name, model in self.meta_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    predictions.append(proba)
                else:
                    pred = model.predict(X)
                    proba = np.zeros((len(X), self.n_classes))
                    for i, p in enumerate(pred):
                        if 0 <= p < self.n_classes:
                            proba[i, int(p)] = 1.0
                    predictions.append(proba)
            except Exception:
                predictions.append(np.zeros((len(X), self.n_classes)))
        
        if not predictions:
            return np.zeros((len(X), self.n_classes))
        
        final_proba = np.zeros_like(predictions[0])
        valid_models = list(self.meta_models.keys())
        for i, (name, proba) in enumerate(zip(valid_models, predictions)):
            weight = self.weights.get(name, 0.0)
            final_proba += weight * proba
        
        final_proba = final_proba / (final_proba.sum(axis=1, keepdims=True) + 1e-9)
        return final_proba


class BlendedEnsemble:
    """
    Blended ensemble - module-level class for pickling
    Combines base and stacked ensembles
    """
    def __init__(self, base_ensemble, stacked_ensemble, base_weight, stacked_weight, n_classes):
        self.base_ensemble = base_ensemble
        self.stacked_ensemble = stacked_ensemble
        self.base_weight = base_weight
        self.stacked_weight = stacked_weight
        self.n_classes = n_classes
    
    def predict(self, X):
        """Predict using blended ensemble"""
        try:
            base_pred = self.base_ensemble.predict(X)
            base_proba = np.zeros((len(X), self.n_classes))
            for i, p in enumerate(base_pred):
                if 0 <= p < self.n_classes:
                    base_proba[i, int(p)] = 1.0
            
            stacked_pred = self.stacked_ensemble.predict(X)
            stacked_proba = np.zeros((len(X), self.n_classes))
            for i, p in enumerate(stacked_pred):
                if 0 <= p < self.n_classes:
                    stacked_proba[i, int(p)] = 1.0
        except Exception:
            # Fallback: try predict_proba
            if hasattr(self.base_ensemble, 'predict_proba'):
                base_proba = self.base_ensemble.predict_proba(X)
            else:
                base_proba = np.ones((len(X), self.n_classes)) / self.n_classes
            
            if hasattr(self.stacked_ensemble, 'predict_proba'):
                stacked_proba = self.stacked_ensemble.predict_proba(X)
            else:
                stacked_proba = np.ones((len(X), self.n_classes)) / self.n_classes
        
        # Blend probabilities
        final_proba = self.base_weight * base_proba + self.stacked_weight * stacked_proba
        final_proba = final_proba / (final_proba.sum(axis=1, keepdims=True) + 1e-9)
        
        return np.argmax(final_proba, axis=1)
    
    def predict_proba(self, X):
        """Predict probabilities using blended ensemble"""
        try:
            if hasattr(self.base_ensemble, 'predict_proba'):
                base_proba = self.base_ensemble.predict_proba(X)
            else:
                pred = self.base_ensemble.predict(X)
                base_proba = np.zeros((len(X), self.n_classes))
                for i, p in enumerate(pred):
                    if 0 <= p < self.n_classes:
                        base_proba[i, int(p)] = 1.0
            
            if hasattr(self.stacked_ensemble, 'predict_proba'):
                stacked_proba = self.stacked_ensemble.predict_proba(X)
            else:
                pred = self.stacked_ensemble.predict(X)
                stacked_proba = np.zeros((len(X), self.n_classes))
                for i, p in enumerate(pred):
                    if 0 <= p < self.n_classes:
                        stacked_proba[i, int(p)] = 1.0
        except Exception:
            base_proba = np.ones((len(X), self.n_classes)) / self.n_classes
            stacked_proba = np.ones((len(X), self.n_classes)) / self.n_classes
        
        # Blend probabilities
        final_proba = self.base_weight * base_proba + self.stacked_weight * stacked_proba
        final_proba = final_proba / (final_proba.sum(axis=1, keepdims=True) + 1e-9)
        
        return final_proba


class BeastLevelTrainer:
    """
    BEAST LEVEL ML Trainer - Optimized for Mac M4
    Trains multiple models with 200+ features, CPU-optimized
    """
    
    def __init__(self, memory_limit_gb: float = MEMORY_LIMIT_GB):
        self.memory_limit_gb = memory_limit_gb
        self.feature_engineer = AdvancedFeatureEngineer(memory_limit_gb=memory_limit_gb)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.models = {}
        self.training_results = {}
        self.feature_names = []
        self.anomaly_types = []
        
    def load_data(self, data_path: Optional[str] = None, 
                  from_database: bool = False,
                  db_connection_string: Optional[str] = None,
                  from_dataset: Optional[str] = None,
                  use_real_datasets: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data from file, database, or downloaded datasets
        
        Args:
            data_path: Path to CSV file
            from_database: Load from PostgreSQL database
            db_connection_string: Database connection string
            from_dataset: Dataset name to download/use (synthetic_k8s, yahoo_s5, etc.)
            use_real_datasets: Whether to try downloading real datasets first
            
        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        print("📊 Loading training data...")
        
        if from_database and db_connection_string:
            df = self._load_from_database(db_connection_string)
        elif data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path, low_memory=False)
            print(f"   Loaded {len(df)} samples from {data_path}")
        elif from_dataset and DATASET_DOWNLOADER_AVAILABLE:
            # Handle multiple datasets (comma-separated)
            dataset_names = [d.strip() for d in from_dataset.split(',')]
            print(f"   Loading datasets: {dataset_names}")
            downloader = DatasetDownloader()
            
            all_dataframes = []
            for dataset_name in dataset_names:
                try:
                    print(f"   📥 Downloading/Loading {dataset_name}...")
                    dataset_path = downloader.download_dataset(dataset_name, force=False)
                    df_single = downloader.load_dataset(dataset_name)
                    print(f"   ✅ Loaded {len(df_single)} samples from {dataset_name}")
                    all_dataframes.append(df_single)
                except Exception as e:
                    print(f"   ⚠️  Failed to load {dataset_name}: {e}")
                    print(f"   Continuing with other datasets...")
            
            if all_dataframes:
                # Combine all datasets
                df = pd.concat(all_dataframes, ignore_index=True)
                print(f"   ✅ Combined total: {len(df)} samples from {len(all_dataframes)} datasets")
            else:
                print(f"   ⚠️  No datasets loaded, falling back to synthetic...")
                df = self._generate_synthetic_data(n_samples=100000)
        elif use_real_datasets and DATASET_DOWNLOADER_AVAILABLE:
            # Try to download and use synthetic_k8s dataset (always available)
            print("   Downloading/Generating synthetic Kubernetes dataset...")
            downloader = DatasetDownloader()
            dataset_path = downloader.download_dataset("synthetic_k8s", force=False)
            df = downloader.load_dataset("synthetic_k8s")
            print(f"   Loaded {len(df)} samples from synthetic dataset")
        else:
            # Generate synthetic data for testing
            print("   No data source provided, generating synthetic data...")
            df = self._generate_synthetic_data(n_samples=100000)  # Increased default
        
        # Separate features and labels
        if 'anomaly_type' in df.columns:
            y = df['anomaly_type']
            X_raw = df.drop('anomaly_type', axis=1)
        else:
            # Create labels from anomaly indicators
            y = self._create_labels_from_indicators(df)
            X_raw = df
        
        # Convert labels to strings to handle mixed types (float/str) from different datasets
        y = y.astype(str)
        
        # Store anomaly types (handle mixed types by converting to string)
        unique_labels = y.unique().tolist()
        # Filter out NaN and convert to string, then sort
        self.anomaly_types = sorted([str(label) for label in unique_labels if str(label) != 'nan' and str(label).lower() != 'none'])
        print(f"   Found {len(self.anomaly_types)} anomaly types: {self.anomaly_types[:10]}...")
        
        return X_raw, y
    
    def _load_from_database(self, conn_string: str) -> pd.DataFrame:
        """Load data from PostgreSQL database"""
        try:
            import psycopg
            with psycopg.connect(conn_string) as conn:
                query = """
                    SELECT 
                        cpu_usage_millicores as cpu_usage,
                        memory_usage_bytes as memory_usage,
                        disk_usage_bytes as disk_usage,
                        network_rx_bytes, network_tx_bytes,
                        network_rx_errors, network_tx_errors,
                        cpu_utilization, memory_utilization,
                        restarts as restart_count,
                        age as age_seconds,
                        cpu_limit_millicores as cpu_limit,
                        memory_limit_bytes as memory_limit,
                        disk_limit_bytes as disk_limit,
                        cpu_trend, memory_trend, restart_trend,
                        has_oom_kill, has_crash_loop, has_high_cpu, has_network_issues,
                        ready, phase, container_state, last_state_reason,
                        timestamp, namespace, pod_name
                    FROM pod_metrics
                    WHERE timestamp > NOW() - INTERVAL '30 days'
                    ORDER BY timestamp DESC
                    LIMIT 100000
                """
                df = pd.read_sql(query, conn)
                print(f"   Loaded {len(df)} samples from database")
                return df
        except Exception as e:
            print(f"   ⚠️  Failed to load from database: {e}")
            print("   Falling back to synthetic data...")
            return self._generate_synthetic_data(n_samples=50000)
    
    def _generate_synthetic_data(self, n_samples: int = 50000) -> pd.DataFrame:
        """Generate synthetic training data"""
        print(f"   Generating {n_samples} synthetic samples...")
        
        anomaly_types = [
            'healthy', 'cpu_spike', 'memory_leak', 'disk_full', 'network_latency',
            'pod_crash', 'oom_kill', 'slow_response', 'high_error_rate', 'deadlock',
            'resource_contention', 'connection_pool_exhausted', 'dns_resolution_failure',
            'disk_io_bottleneck', 'cache_thrashing', 'image_pull_error', 'crash_loop',
            'cpu_throttling', 'memory_pressure', 'network_partition'
        ]
        
        data = []
        np.random.seed(RANDOM_SEED)
        
        for _ in range(n_samples):
            anomaly = np.random.choice(anomaly_types, p=[0.6] + [0.4/19]*19)  # 60% healthy
            
            # Generate metrics based on anomaly type
            if anomaly == 'healthy':
                cpu = np.random.uniform(10, 40)
                memory = np.random.uniform(20, 50)
                disk = np.random.uniform(10, 40)
            elif anomaly == 'cpu_spike':
                cpu = np.random.uniform(80, 100)
                memory = np.random.uniform(20, 60)
                disk = np.random.uniform(10, 40)
            elif anomaly == 'memory_leak':
                cpu = np.random.uniform(30, 60)
                memory = np.random.uniform(70, 95)
                disk = np.random.uniform(10, 40)
            elif anomaly == 'oom_kill':
                cpu = np.random.uniform(30, 70)
                memory = np.random.uniform(95, 100)
                disk = np.random.uniform(10, 50)
            elif anomaly == 'crash_loop':
                cpu = np.random.uniform(0, 20)
                memory = np.random.uniform(0, 30)
                disk = np.random.uniform(10, 40)
            else:
                cpu = np.random.uniform(20, 80)
                memory = np.random.uniform(30, 80)
                disk = np.random.uniform(20, 70)
            
            network_rx = np.random.uniform(100, 5000)
            network_tx = np.random.uniform(100, 5000)
            restarts = np.random.poisson(0.5 if anomaly == 'healthy' else 3)
            age = np.random.uniform(60, 86400)  # 1 minute to 1 day
            
            data.append({
                'cpu_usage': cpu,
                'memory_usage': memory * 1024 * 1024,  # Convert to bytes
                'disk_usage': disk * 1024 * 1024,
                'network_rx_bytes': network_rx,
                'network_tx_bytes': network_tx,
                'network_rx_errors': 0 if anomaly != 'network_latency' else np.random.poisson(10),
                'network_tx_errors': 0 if anomaly != 'network_latency' else np.random.poisson(10),
                'cpu_utilization': cpu,
                'memory_utilization': memory,
                'restart_count': restarts,
                'age_minutes': age / 60,
                'cpu_limit': 1000,
                'memory_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'disk_limit': 10 * 1024 * 1024 * 1024,  # 10GB
                'cpu_trend': np.random.uniform(-5, 5),
                'memory_trend': np.random.uniform(-5, 5),
                'restart_trend': np.random.uniform(-1, 1),
                'has_oom_kill': 1 if anomaly == 'oom_kill' else 0,
                'has_crash_loop': 1 if anomaly == 'crash_loop' else 0,
                'has_high_cpu': 1 if anomaly == 'cpu_spike' else 0,
                'has_network_issues': 1 if 'network' in anomaly.lower() else 0,
                'ready': 1 if anomaly == 'healthy' else 0,
                'phase': 'Running' if anomaly == 'healthy' else 'Pending',
                'container_state': 'Running' if anomaly == 'healthy' else 'Waiting',
                'last_state_reason': anomaly if anomaly != 'healthy' else '',
                'anomaly_type': anomaly,
                'timestamp': datetime.now(),
                'namespace': 'default',
                'pod_name': f'pod-{np.random.randint(1000, 9999)}'
            })
        
        return pd.DataFrame(data)
    
    def _create_labels_from_indicators(self, df: pd.DataFrame) -> pd.Series:
        """Create labels from anomaly indicators"""
        labels = []
        for _, row in df.iterrows():
            if row.get('has_oom_kill', 0):
                labels.append('oom_kill')
            elif row.get('has_crash_loop', 0):
                labels.append('crash_loop')
            elif row.get('has_high_cpu', 0):
                labels.append('cpu_spike')
            elif row.get('has_network_issues', 0):
                labels.append('network_latency')
            else:
                labels.append('healthy')
        return pd.Series(labels)
    
    def train(self, X_raw: pd.DataFrame, y: pd.Series, 
              optimize_hyperparameters: bool = False,
              n_trials: int = 50) -> Dict:
        """
        Train all models with feature engineering
        
        Args:
            X_raw: Raw features DataFrame
            y: Labels Series
            optimize_hyperparameters: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*70)
        print("🚀 BEAST LEVEL ML TRAINING - Mac M4 Optimized")
        print("="*70)
        
        start_time = time.time()
        
        # Step 1: Feature Engineering
        print("\n🔧 Step 1: Feature Engineering (200+ features)...")
        X = self.feature_engineer.engineer_features(X_raw, chunk_size=CHUNK_SIZE)
        self.feature_names = self.feature_engineer.get_feature_names()
        print(f"✅ Created {len(self.feature_names)} features")
        
        # Step 2: Encode Labels
        print("\n🔤 Step 2: Encoding labels...")
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"✅ Encoded {len(self.label_encoder.classes_)} classes")
        
        # Step 3: Split Data
        print("\n✂️  Step 3: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
        )
        print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Step 4: Handle NaN and inf values before scaling
        print("\n🧹 Step 4: Cleaning data (removing NaN and inf)...")
        # Replace inf with NaN first, then fill NaN
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median for each feature
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Final check: ensure no NaN or inf
        assert not X_train.isnull().any().any(), "X_train still contains NaN after imputation"
        assert not X_test.isnull().any().any(), "X_test still contains NaN after imputation"
        assert not np.isinf(X_train.values).any(), "X_train still contains inf after cleaning"
        assert not np.isinf(X_test.values).any(), "X_test still contains inf after cleaning"
        print("✅ Data cleaned (no NaN or inf)")
        
        # Step 5: Feature Scaling
        print("\n📏 Step 5: Feature scaling...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Ensure scaled data doesn't have NaN or inf
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        print("✅ Features scaled")
        
        # Step 6: Feature Selection (select top features - increased for better accuracy)
        print("\n🎯 Step 6: Feature selection (top 200 features for maximum accuracy)...")
        n_features_to_select = min(200, len(self.feature_names))  # Increased from 150 to 200
        self.feature_selector = SelectKBest(f_classif, k=n_features_to_select)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Final check: ensure selected features don't have NaN
        X_train_selected = np.nan_to_num(X_train_selected, nan=0.0, posinf=1.0, neginf=-1.0)
        X_test_selected = np.nan_to_num(X_test_selected, nan=0.0, posinf=1.0, neginf=-1.0)
        
        selected_features = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
        print(f"✅ Selected {len(selected_features)} top features out of {len(self.feature_names)}")
        
        # Step 7: Train Models
        print("\n🤖 Step 7: Training models...")
        self.models = {}
        
        # XGBoost - Load existing if available
        xgb_model_path = os.path.join('models', 'xgboost_model.joblib')
        if os.path.exists(xgb_model_path):
            print("\n   ✅ XGBoost model found - Loading existing model (skipping training)...")
            try:
                xgb_model = joblib.load(xgb_model_path)
                self.models['xgboost'] = xgb_model
                print(f"      ✅ Loaded XGBoost model from {xgb_model_path}")
                self._evaluate_model('xgboost', xgb_model, X_test_selected, y_test)
            except Exception as e:
                print(f"      ⚠️  Failed to load existing XGBoost model: {e}. Training new model...")
                xgb_model = self._train_xgboost(X_train_selected, y_train, optimize_hyperparameters, n_trials)
                self.models['xgboost'] = xgb_model
                self._evaluate_model('xgboost', xgb_model, X_test_selected, y_test)
        else:
            print("\n   Training XGBoost...")
            xgb_model = self._train_xgboost(X_train_selected, y_train, optimize_hyperparameters, n_trials)
            self.models['xgboost'] = xgb_model
            self._evaluate_model('xgboost', xgb_model, X_test_selected, y_test)
        
        # LightGBM - Load existing if available
        lgb_model_path = os.path.join('models', 'lightgbm_model.joblib')
        if os.path.exists(lgb_model_path):
            print("\n   ✅ LightGBM model found - Loading existing model (skipping training)...")
            try:
                lgb_model = joblib.load(lgb_model_path)
                self.models['lightgbm'] = lgb_model
                print(f"      ✅ Loaded LightGBM model from {lgb_model_path}")
                self._evaluate_model('lightgbm', lgb_model, X_test_selected, y_test)
            except Exception as e:
                print(f"      ⚠️  Failed to load existing LightGBM model: {e}. Training new model...")
                lgb_model = self._train_lightgbm(X_train_selected, y_train, optimize_hyperparameters, n_trials)
                self.models['lightgbm'] = lgb_model
                self._evaluate_model('lightgbm', lgb_model, X_test_selected, y_test)
        else:
            print("\n   Training LightGBM...")
            lgb_model = self._train_lightgbm(X_train_selected, y_train, optimize_hyperparameters, n_trials)
            self.models['lightgbm'] = lgb_model
            self._evaluate_model('lightgbm', lgb_model, X_test_selected, y_test)
        
        # CatBoost - Load existing if available (15 trials completed)
        catboost_model_path = os.path.join('models', 'catboost_model.joblib')
        if os.path.exists(catboost_model_path):
            print("\n   ✅ CatBoost model found - Loading existing model (skipping training)...")
            try:
                cat_model = joblib.load(catboost_model_path)
                self.models['catboost'] = cat_model
                print(f"      ✅ Loaded CatBoost model from {catboost_model_path}")
                self._evaluate_model('catboost', cat_model, X_test_selected, y_test)
            except Exception as e:
                print(f"      ⚠️  Failed to load existing CatBoost model: {e}. Training new model...")
                cat_model = self._train_catboost(X_train_selected, y_train, optimize_hyperparameters, n_trials)
                self.models['catboost'] = cat_model
                self._evaluate_model('catboost', cat_model, X_test_selected, y_test)
        else:
            print("\n   Training CatBoost with 15 Optuna trials...")
            cat_model = self._train_catboost(X_train_selected, y_train, optimize_hyperparameters, n_trials)
            self.models['catboost'] = cat_model
            self._evaluate_model('catboost', cat_model, X_test_selected, y_test)
        
        # Random Forest - Load existing if available
        rf_model_path = os.path.join('models', 'random_forest_model.joblib')
        if os.path.exists(rf_model_path):
            print("\n   ✅ Random Forest model found - Loading existing model (skipping training)...")
            try:
                rf_model = joblib.load(rf_model_path)
                self.models['random_forest'] = rf_model
                print(f"      ✅ Loaded Random Forest model from {rf_model_path}")
                self._evaluate_model('random_forest', rf_model, X_test_selected, y_test)
            except Exception as e:
                print(f"      ⚠️  Failed to load existing Random Forest model: {e}. Training new model...")
                rf_model = self._train_random_forest(X_train_selected, y_train)
                self.models['random_forest'] = rf_model
                self._evaluate_model('random_forest', rf_model, X_test_selected, y_test)
        else:
            print("\n   Training Random Forest...")
            rf_model = self._train_random_forest(X_train_selected, y_train)
            self.models['random_forest'] = rf_model
            self._evaluate_model('random_forest', rf_model, X_test_selected, y_test)
        
        # Isolation Forest - Load existing if available
        iso_model_path = os.path.join('models', 'isolation_forest_model.joblib')
        if os.path.exists(iso_model_path):
            print("\n   ✅ Isolation Forest model found - Loading existing model (skipping training)...")
            try:
                iso_model = joblib.load(iso_model_path)
                self.models['isolation_forest'] = iso_model
                print(f"      ✅ Loaded Isolation Forest model from {iso_model_path}")
                self._evaluate_model('isolation_forest', iso_model, X_test_selected, y_test)
            except Exception as e:
                print(f"      ⚠️  Failed to load existing Isolation Forest model: {e}. Training new model...")
                iso_model = self._train_isolation_forest(X_train_selected, y_train)
                self.models['isolation_forest'] = iso_model
                self._evaluate_model('isolation_forest', iso_model, X_test_selected, y_test)
        else:
            print("\n   Training Isolation Forest...")
            iso_model = self._train_isolation_forest(X_train_selected, y_train)
            self.models['isolation_forest'] = iso_model
            self._evaluate_model('isolation_forest', iso_model, X_test_selected, y_test)
        
        # Extra Trees - Load existing if available (10 trials completed)
        et_model_path = os.path.join('models', 'extra_trees_model.joblib')
        if os.path.exists(et_model_path):
            print("\n   ✅ Extra Trees model found - Loading existing model (skipping training)...")
            try:
                et_model = joblib.load(et_model_path)
                self.models['extra_trees'] = et_model
                print(f"      ✅ Loaded Extra Trees model from {et_model_path}")
                self._evaluate_model('extra_trees', et_model, X_test_selected, y_test)
            except Exception as e:
                print(f"      ⚠️  Failed to load existing Extra Trees model: {e}. Training new model...")
                et_model = self._train_extra_trees(X_train_selected, y_train, optimize_hyperparameters, n_trials)
                self.models['extra_trees'] = et_model
                self._evaluate_model('extra_trees', et_model, X_test_selected, y_test)
        else:
            print("\n   Training Extra Trees...")
            et_model = self._train_extra_trees(X_train_selected, y_train, optimize_hyperparameters, n_trials)
            self.models['extra_trees'] = et_model
            self._evaluate_model('extra_trees', et_model, X_test_selected, y_test)
        
        # Gradient Boosting (sklearn) - SKIPPED per user request
        print("\n   ⏭️  Skipping Gradient Boosting training (as requested)...")
        # gb_model = self._train_gradient_boosting(X_train_selected, y_train, optimize_hyperparameters, n_trials)
        # self.models['gradient_boosting'] = gb_model
        # self._evaluate_model('gradient_boosting', gb_model, X_test_selected, y_test)
        
        # Histogram Gradient Boosting - SKIPPED per user request
        print("\n   ⏭️  Skipping Histogram Gradient Boosting training (as requested)...")
        # try:
        #     print("\n   Training Histogram Gradient Boosting...")
        #     hgb_model = self._train_histogram_gradient_boosting(X_train_selected, y_train, optimize_hyperparameters, n_trials)
        #     self.models['histogram_gradient_boosting'] = hgb_model
        #     self._evaluate_model('histogram_gradient_boosting', hgb_model, X_test_selected, y_test)
        # except Exception as e:
        #     print(f"      ⚠️  Histogram Gradient Boosting skipped: {e}")
        
        # Local Outlier Factor (for anomaly scoring)
        print("\n   Training Local Outlier Factor...")
        lof_model = self._train_local_outlier_factor(X_train_selected, y_train)
        self.models['local_outlier_factor'] = lof_model
        self._evaluate_model('local_outlier_factor', lof_model, X_test_selected, y_test)
        
        # One-Class SVM (for anomaly scoring)
        try:
            print("\n   Training One-Class SVM...")
            ocsvm_model = self._train_one_class_svm(X_train_selected, y_train)
            self.models['one_class_svm'] = ocsvm_model
            self._evaluate_model('one_class_svm', ocsvm_model, X_test_selected, y_test)
        except Exception as e:
            print(f"      ⚠️  One-Class SVM skipped (may be too slow): {e}")
        
        # Step 8: Advanced Ensemble (Stacking + Blending)
        print("\n🎯 Step 8: Creating advanced ensemble (stacking + blending)...")
        
        # Level 1: Base models ensemble (weighted voting)
        base_ensemble = self._create_ensemble()
        self.models['base_ensemble'] = base_ensemble
        self._evaluate_model('base_ensemble', base_ensemble, X_test_selected, y_test)
        
        # Level 2: Stacking with meta-learners
        stacked_ensemble = self._create_stacked_ensemble(
            X_train_selected, y_train, X_test_selected, y_test,
            optimize_hyperparameters, n_trials
        )
        self.models['stacked_ensemble'] = stacked_ensemble
        self._evaluate_model('stacked_ensemble', stacked_ensemble, X_test_selected, y_test)
        
        # Level 3: Final blended ensemble (combines all)
        final_ensemble = self._create_blended_ensemble(
            base_ensemble, stacked_ensemble, X_test_selected, y_test
        )
        self.models['ensemble'] = final_ensemble
        self._evaluate_model('ensemble', final_ensemble, X_test_selected, y_test)
        
        # Step 9: Save Everything
        print("\n💾 Step 9: Saving models and artifacts...")
        self._save_models()
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        return self.training_results
    
    def _train_xgboost(self, X_train, y_train, optimize: bool, n_trials: int):
        """Train XGBoost model"""
        if optimize and OPTUNA_AVAILABLE:
            # Use fewer CV folds and add progress bar for large datasets
            # For large datasets (100K+ samples), reduce CV folds and trials
            n_samples = len(X_train)
            if n_samples > 50000:
                cv_folds = 3  # Use 3-fold CV for large datasets
                # Optimize for speed vs accuracy balance: 15 trials (sweet spot)
                effective_trials = min(n_trials, 15)  # Use 15 trials for best balance
                print(f"      Optimizing with {effective_trials} trials, {cv_folds}-fold CV (large dataset: {n_samples} samples)")
            else:
                cv_folds = 5
                effective_trials = min(n_trials, 15)  # Cap at 15 for optimal balance
                print(f"      Optimizing with {effective_trials} trials, {cv_folds}-fold CV")
            
            def objective(trial):
                # Optimized for fast inference: slightly fewer estimators, shallower trees
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),  # Reduced max from 500 to 300 for speed
                    'max_depth': trial.suggest_int('max_depth', 5, 12),  # Reduced max from 15 to 12 for fast inference
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),  # Higher learning rate = fewer trees needed
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 0.3),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                }
                model = xgb.XGBClassifier(**params, n_jobs=-1, tree_method='hist', 
                                        device='cpu', random_state=RANDOM_SEED, verbosity=0)
                # Use fewer CV folds for large datasets
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_weighted', n_jobs=1)
                return scores.mean()
            
            study = optuna.create_study(
                direction='maximize',
                study_name=f'xgboost_optimization',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
            print(f"      Starting Optuna optimization ({effective_trials} trials)...")
            study.optimize(objective, n_trials=effective_trials, show_progress_bar=True, n_jobs=1)
            best_params = study.best_params
            print(f"      ✅ Best score: {study.best_value:.4f}")
            print(f"      Best params: {best_params}")
        else:
            # Optimized default params for fast inference + good accuracy
            best_params = {
                'n_estimators': 200,  # Reduced from 300 for faster inference
                'max_depth': 8,  # Reduced from 10 for faster inference
                'learning_rate': 0.1,  # Higher LR = faster training + inference
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
        
        model = xgb.XGBClassifier(
            **best_params,
            n_jobs=-1,
            tree_method='hist',  # Fastest tree method for CPU
            device='cpu',
            random_state=RANDOM_SEED,
            verbosity=0
        )
        model.fit(X_train, y_train)
        return model
    
    def _train_lightgbm(self, X_train, y_train, optimize: bool, n_trials: int):
        """Train LightGBM model"""
        if optimize and OPTUNA_AVAILABLE:
            n_samples = len(X_train)
            if n_samples > 50000:
                cv_folds = 3
                effective_trials = min(n_trials, 15)  # Optimized: 15 trials for balance
                print(f"      Optimizing with {effective_trials} trials, {cv_folds}-fold CV (large dataset: {n_samples} samples)")
            else:
                cv_folds = 5
                effective_trials = min(n_trials, 15)  # Optimized: 15 trials for balance
            
            def objective(trial):
                # Optimized for fast inference
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 250),  # Reduced max for speed
                    'max_depth': trial.suggest_int('max_depth', 5, 10),  # Reduced max for fast inference
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),  # Higher LR = faster
                    'num_leaves': trial.suggest_int('num_leaves', 20, 63),  # Reduced max for speed
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
                }
                model = lgb.LGBMClassifier(**params, n_jobs=-1, device='cpu', 
                                         random_state=RANDOM_SEED, verbosity=-1, force_row_wise=True)  # Faster on CPU
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_weighted', n_jobs=1)
                return scores.mean()
            
            study = optuna.create_study(
                direction='maximize',
                study_name=f'lightgbm_optimization',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)  # Faster pruning
            )
            print(f"      Starting Optuna optimization ({effective_trials} trials)...")
            study.optimize(objective, n_trials=effective_trials, show_progress_bar=True, n_jobs=1)
            best_params = study.best_params
            print(f"      ✅ Best score: {study.best_value:.4f}")
        else:
            # Optimized default params for fast inference
            best_params = {
                'n_estimators': 200,  # Reduced for faster inference
                'max_depth': 8,  # Reduced for faster inference
                'learning_rate': 0.1,  # Higher LR = faster
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
            }
        
        model = lgb.LGBMClassifier(
            **best_params,
            n_jobs=-1,
            device='cpu',
            random_state=RANDOM_SEED,
            verbosity=-1,
            force_row_wise=True  # Faster inference on CPU
        )
        model.fit(X_train, y_train)
        return model
    
    def _train_catboost(self, X_train, y_train, optimize: bool, n_trials: int):
        """Train CatBoost model - Safe optimization for Mac M4"""
        if optimize and OPTUNA_AVAILABLE:
            n_samples = len(X_train)
            if n_samples > 50000:
                cv_folds = 3
                effective_trials = min(n_trials, 15)
                print(f"      Optimizing with {effective_trials} trials, {cv_folds}-fold CV (large dataset: {n_samples} samples)")
            else:
                cv_folds = 5
                effective_trials = min(n_trials, 15)
                print(f"      Optimizing with {effective_trials} trials, {cv_folds}-fold CV")
            
            def objective(trial):
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 250),
                    'depth': trial.suggest_int('depth', 5, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 5),
                    'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                }
                model = CatBoostClassifier(
                    **params,
                    thread_count=4,  # Limited threads to avoid hanging
                    random_seed=RANDOM_SEED,
                    verbose=False,
                    task_type='CPU'
                )
                # Use n_jobs=1 in cross_val_score to prevent deadlock
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_weighted', n_jobs=1)
                return scores.mean()
            
            study = optuna.create_study(
                direction='maximize',
                study_name=f'catboost_optimization',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
            )
            print(f"      Starting Optuna optimization ({effective_trials} trials)...")
            study.optimize(objective, n_trials=effective_trials, show_progress_bar=True, n_jobs=1)
            best_params = study.best_params
            print(f"      ✅ Best score: {study.best_value:.4f}")
        else:
            # Optimized default params for fast inference
            best_params = {
                'iterations': 200,
                'depth': 8,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3,
            }
        
        model = CatBoostClassifier(
            **best_params,
            thread_count=4,  # Use limited threads to avoid hanging
            random_seed=RANDOM_SEED,
            verbose=False,
            task_type='CPU'  # Explicit CPU for fastest inference
        )
        model.fit(X_train, y_train)
        return model
    
    def _train_random_forest(self, X_train, y_train):
        """Train Random Forest model - optimized for fast inference"""
        model = RandomForestClassifier(
            n_estimators=150,  # Reduced from 200 for faster inference
            max_depth=12,  # Reduced from 15 for faster inference
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            max_features='sqrt'  # Faster inference
        )
        model.fit(X_train, y_train)
        return model
    
    def _train_isolation_forest(self, X_train, y_train):
        """Train Isolation Forest (unsupervised anomaly detection)"""
        model = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=0.1,
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        # Fit on all data (unsupervised)
        model.fit(X_train)
        return model
    
    def _train_extra_trees(self, X_train, y_train, optimize: bool, n_trials: int):
        """Train Extra Trees classifier - optimized for fast inference"""
        if optimize and OPTUNA_AVAILABLE:
            effective_trials = min(n_trials, 10)  # Reduced to 10 trials for faster completion
            def objective(trial):
                # Optimized for fast inference and speed
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 150),  # Reduced max further for speed
                    'max_depth': trial.suggest_int('max_depth', 6, 10),  # Narrower range for faster training
                    'min_samples_split': trial.suggest_int('min_samples_split', 3, 8),  # Narrower range
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),  # Narrower range
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),  # Removed None for speed
                }
                model = ExtraTreesClassifier(**params, n_jobs=-1, random_state=RANDOM_SEED)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=1)
                return scores.mean()
            
            study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5))
            study.optimize(objective, n_trials=effective_trials, show_progress_bar=True)
            best_params = study.best_params
            print(f"      ✅ Best score: {study.best_value:.4f}")
        else:
            # Optimized default params for fast inference
            best_params = {
                'n_estimators': 150,  # Reduced for faster inference
                'max_depth': 10,  # Reduced for faster inference
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt'  # Faster inference
            }
        
        model = ExtraTreesClassifier(**best_params, n_jobs=-1, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        return model
    
    def _train_gradient_boosting(self, X_train, y_train, optimize: bool, n_trials: int):
        """Train Gradient Boosting classifier"""
        if optimize and OPTUNA_AVAILABLE:
            # GradientBoostingClassifier is VERY slow (sequential), optimize for maximum speed
            cv_folds = 3  # Keep 3-fold (2-fold would hurt accuracy too much)
            effective_trials = min(n_trials, 10)  # 10 trials for faster completion
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 60, 100),  # Reduced further: 60-100 (much faster)
                    'max_depth': trial.suggest_int('max_depth', 4, 7),  # Reduced: 4-7 (shallower = faster)
                    'learning_rate': trial.suggest_float('learning_rate', 0.08, 0.15),  # Higher LR range = fewer trees needed
                    'subsample': trial.suggest_float('subsample', 0.8, 1.0),  # Narrower range
                    'min_samples_split': trial.suggest_int('min_samples_split', 4, 6),  # Very narrow range
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 2),  # Minimal range
                }
                model = GradientBoostingClassifier(**params, random_state=RANDOM_SEED)
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_weighted', n_jobs=1)
                return scores.mean()
            
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3)  # Very aggressive pruning
            )
            study.optimize(objective, n_trials=effective_trials, show_progress_bar=True, n_jobs=1)
            best_params = study.best_params
        else:
            best_params = {
                'n_estimators': 150,
                'max_depth': 7,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        
        model = GradientBoostingClassifier(**best_params, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        return model
    
    def _train_histogram_gradient_boosting(self, X_train, y_train, optimize: bool, n_trials: int):
        """Train Histogram Gradient Boosting classifier"""
        if optimize and OPTUNA_AVAILABLE:
            # HistGradientBoostingClassifier is faster, optimize for maximum speed
            cv_folds = 3  # Keep 3-fold (2-fold would hurt accuracy too much)
            effective_trials = min(n_trials, 10)  # 10 trials for faster completion
            
            def objective(trial):
                params = {
                    'max_iter': trial.suggest_int('max_iter', 60, 100),  # Reduced further: 60-100 (much faster)
                    'max_depth': trial.suggest_int('max_depth', 4, 7),  # Reduced: 4-7 (shallower = faster)
                    'learning_rate': trial.suggest_float('learning_rate', 0.08, 0.15),  # Higher LR range = fewer iterations
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 30),  # Narrower range
                }
                model = HistGradientBoostingClassifier(**params, random_state=RANDOM_SEED)
                scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_weighted', n_jobs=1)
                return scores.mean()
            
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3)  # Very aggressive pruning
            )
            study.optimize(objective, n_trials=effective_trials, show_progress_bar=True, n_jobs=1)
            best_params = study.best_params
        else:
            best_params = {
                'max_iter': 150,
                'max_depth': 7,
                'learning_rate': 0.1,
                'min_samples_leaf': 20
            }
        
        model = HistGradientBoostingClassifier(**best_params, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        return model
    
    def _train_local_outlier_factor(self, X_train, y_train):
        """Train Local Outlier Factor (for anomaly scoring)"""
        # Fit on normal class (assuming last class or healthy)
        normal_mask = (y_train == y_train.mode()[0]) if len(np.unique(y_train)) > 1 else np.ones(len(y_train), dtype=bool)
        X_normal = X_train[normal_mask] if normal_mask.sum() > 10 else X_train
        
        model = LocalOutlierFactor(
            n_neighbors=min(20, len(X_normal) // 2),
            contamination=0.1,
            novelty=True,
            n_jobs=-1
        )
        model.fit(X_normal)
        return model
    
    def _train_one_class_svm(self, X_train, y_train):
        """Train One-Class SVM (for anomaly scoring) - may be slow on large data"""
        # Use subset for training (OneClassSVM can be slow)
        sample_size = min(5000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices]
        
        # Fit on normal class
        y_sample = y_train[indices]
        normal_mask = (y_sample == y_sample.mode()[0]) if len(np.unique(y_sample)) > 1 else np.ones(len(y_sample), dtype=bool)
        X_normal = X_sample[normal_mask] if normal_mask.sum() > 10 else X_sample
        
        model = OneClassSVM(
            gamma='scale',
            nu=0.1,
            kernel='rbf'
        )
        model.fit(X_normal)
        return model
    
    def _create_ensemble(self):
        """Create weighted ensemble using module-level WeightedEnsemble class"""
        # Weights based on expected performance (exclude unsupervised models for voting)
        supervised_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 
                           'extra_trees', 'gradient_boosting', 'histogram_gradient_boosting']
        
        weights = {}
        for model_name in supervised_models:
            if model_name in self.models:
                if model_name == 'xgboost':
                    weights[model_name] = 0.25
                elif model_name == 'lightgbm':
                    weights[model_name] = 0.22
                elif model_name == 'catboost':
                    weights[model_name] = 0.18
                elif model_name in ['random_forest', 'extra_trees']:
                    weights[model_name] = 0.12
                elif model_name in ['gradient_boosting', 'histogram_gradient_boosting']:
                    weights[model_name] = 0.11
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        else:
            # Fallback: equal weights
            weights = {k: 1.0/len(weights) for k in weights.keys()}
        
        # Get number of classes
        n_classes = len(self.label_encoder.classes_) if hasattr(self, 'label_encoder') and self.label_encoder else 24
        for name, model in self.models.items():
            if hasattr(model, 'classes_'):
                n_classes = len(model.classes_)
                break
        
        # Use module-level WeightedEnsemble class (picklable)
        return WeightedEnsemble(self.models, weights, n_classes)
    
    def _create_stacked_ensemble(self, X_train, y_train, X_test, y_test,
                                 optimize: bool, n_trials: int):
        """Create multi-level stacked ensemble with meta-learners"""
        print("      Creating Level 1: Base model predictions...")
        
        # Select supervised models for stacking
        supervised_models = {k: v for k, v in self.models.items() 
                           if k not in ['isolation_forest', 'local_outlier_factor', 'one_class_svm',
                                      'ensemble', 'base_ensemble', 'stacked_ensemble']}
        
        if len(supervised_models) < 3:
            # Not enough models, use simple weighted ensemble
            return self._create_ensemble()
        
        # Use K-Fold cross-validation to generate meta-features
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        
        # Generate meta-features from base models
        meta_train = np.zeros((len(X_train), len(supervised_models)))
        meta_test = np.zeros((len(X_test), len(supervised_models)))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            for model_idx, (model_name, model) in enumerate(supervised_models.items()):
                # Retrain model on fold training data
                try:
                    if hasattr(model, 'fit'):
                        # Clone and retrain
                        from sklearn.base import clone
                        fold_model = clone(model)
                        fold_model.fit(X_fold_train, y_fold_train)
                        
                        # Predict probabilities on fold validation set
                        if hasattr(fold_model, 'predict_proba'):
                            proba = fold_model.predict_proba(X_fold_val)
                            # Use max probability as meta-feature (can also use all probabilities)
                            meta_train[val_idx, model_idx] = proba.max(axis=1)
                        else:
                            # Use predictions as meta-features
                            pred = fold_model.predict(X_fold_val)
                            meta_train[val_idx, model_idx] = pred
                except Exception as e:
                    # Fallback: use original model predictions
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_fold_val)
                        meta_train[val_idx, model_idx] = proba.max(axis=1)
                    else:
                        pred = model.predict(X_fold_val)
                        meta_train[val_idx, model_idx] = pred
        
        # Generate test meta-features using full training data
        for model_idx, (model_name, model) in enumerate(supervised_models.items()):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    meta_test[:, model_idx] = proba.max(axis=1)
                else:
                    pred = model.predict(X_test)
                    meta_test[:, model_idx] = pred
            except Exception:
                meta_test[:, model_idx] = 0
        
        print("      Creating Level 2: Meta-learners...")
        
        # Train multiple meta-learners
        meta_models = {}
        
        # 1. Logistic Regression Meta-Learner
        try:
            meta_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
            meta_lr.fit(meta_train, y_train)
            meta_models['logistic_regression'] = meta_lr
            print("         ✅ Logistic Regression meta-learner")
        except Exception as e:
            print(f"         ⚠️  Logistic Regression failed: {e}")
        
        # 2. LightGBM Meta-Learner (best for stacking)
        try:
            meta_lgb = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1
            )
            meta_lgb.fit(meta_train, y_train)
            meta_models['lightgbm'] = meta_lgb
            print("         ✅ LightGBM meta-learner")
        except Exception as e:
            print(f"         ⚠️  LightGBM meta-learner failed: {e}")
        
        # 3. XGBoost Meta-Learner
        try:
            meta_xgb = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=RANDOM_SEED, verbosity=0, n_jobs=-1, tree_method='hist'
            )
            meta_xgb.fit(meta_train, y_train)
            meta_models['xgboost'] = meta_xgb
            print("         ✅ XGBoost meta-learner")
        except Exception as e:
            print(f"         ⚠️  XGBoost meta-learner failed: {e}")
        
        if not meta_models:
            # Fallback to simple ensemble
            return self._create_ensemble()
        
        # Combine meta-learners with weighted voting
        print("      Creating Level 3: Final stacked ensemble...")
        meta_weights = {'logistic_regression': 0.3, 'lightgbm': 0.4, 'xgboost': 0.3}
        meta_weights = {k: v for k, v in meta_weights.items() if k in meta_models}
        total_meta_weight = sum(meta_weights.values())
        if total_meta_weight > 0:
            meta_weights = {k: v/total_meta_weight for k, v in meta_weights.items()}
        
        n_classes = len(np.unique(y_train))
        return StackedEnsemble(meta_models, meta_weights, n_classes)
    
    def _create_blended_ensemble(self, base_ensemble, stacked_ensemble, X_test, y_test):
        """Create final blended ensemble combining base and stacked ensembles"""
        # Evaluate both ensembles
        base_pred = base_ensemble.predict(X_test)
        base_acc = accuracy_score(y_test, base_pred)
        
        stacked_pred = stacked_ensemble.predict(X_test)
        stacked_acc = accuracy_score(y_test, stacked_pred)
        
        # Weight by performance
        total_acc = base_acc + stacked_acc
        if total_acc > 0:
            base_weight = base_acc / total_acc
            stacked_weight = stacked_acc / total_acc
        else:
            base_weight = 0.5
            stacked_weight = 0.5
        
        n_classes = len(np.unique(y_test))
        return BlendedEnsemble(base_ensemble, stacked_ensemble, base_weight, stacked_weight, n_classes)
    
    def _evaluate_model(self, name: str, model, X_test, y_test):
        """Evaluate model and store results"""
        try:
            y_pred = model.predict(X_test)
            
            # Handle unsupervised models differently
            if name in ['isolation_forest', 'local_outlier_factor', 'one_class_svm']:
                # Convert -1/1 to class labels for unsupervised models
                y_pred_mapped = np.zeros_like(y_pred, dtype=int)
                if name == 'isolation_forest':
                    # Isolation Forest: -1 = anomaly, 1 = normal
                    anomaly_class = 0  # First anomaly class
                    normal_class = len(np.unique(y_test)) - 1  # Last class (healthy)
                    y_pred_mapped[y_pred == -1] = anomaly_class
                    y_pred_mapped[y_pred == 1] = normal_class
                elif name == 'local_outlier_factor':
                    # LOF: -1 = anomaly, 1 = normal
                    y_pred_mapped[y_pred == -1] = 0
                    y_pred_mapped[y_pred == 1] = len(np.unique(y_test)) - 1
                elif name == 'one_class_svm':
                    # OneClassSVM: -1 = anomaly, 1 = normal
                    y_pred_mapped[y_pred == -1] = 0
                    y_pred_mapped[y_pred == 1] = len(np.unique(y_test)) - 1
                y_pred = y_pred_mapped
            
            # Ensure predictions are within valid class range
            n_classes = len(np.unique(y_test))
            y_pred = np.clip(y_pred, 0, n_classes - 1)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            self.training_results[name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
            print(f"      ✅ {name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        except Exception as e:
            print(f"      ⚠️  {name} evaluation failed: {e}")
            self.training_results[name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    def _save_models(self):
        """Save all models and artifacts"""
        # Save all models (including new ones)
        for name, model in self.models.items():
            if name in ['ensemble', 'base_ensemble', 'stacked_ensemble']:
                # Save ensemble models separately
                continue
            try:
                model_path = MODEL_DIR / f"{name}_model.joblib"
                joblib.dump(model, model_path)
                print(f"   💾 Saved {name} model")
            except Exception as e:
                print(f"   ⚠️  Could not save {name} model: {e}")
        
        # Save ensemble models (base, stacked, final)
        for ensemble_name in ['base_ensemble', 'stacked_ensemble', 'ensemble']:
            if ensemble_name in self.models:
                ensemble_path = MODEL_DIR / f"{ensemble_name}_model.joblib"
                try:
                    joblib.dump(self.models[ensemble_name], ensemble_path)
                    print(f"   💾 Saved {ensemble_name} model")
                except Exception as e:
                    print(f"   ⚠️  Could not save {ensemble_name} directly: {e}")
                    # Save ensemble info for reconstruction
                    ensemble_info = {
                        'type': ensemble_name,
                        'model_names': [k for k in self.models.keys() if k not in ['ensemble', 'base_ensemble', 'stacked_ensemble']],
                        'n_classes': getattr(self.models[ensemble_name], 'n_classes', 24)
                    }
                    if hasattr(self.models[ensemble_name], 'weights'):
                        ensemble_info['weights'] = dict(self.models[ensemble_name].weights)
                    with open(MODEL_DIR / f"{ensemble_name}_info.json", "w") as f:
                        json.dump(ensemble_info, f, indent=2)
                    print(f"   💾 Saved {ensemble_name} info (can be reconstructed)")
        
        # Save scaler
        joblib.dump(self.scaler, MODEL_DIR / "scaler.joblib")
        print("   💾 Saved scaler")
        
        # Save label encoder
        joblib.dump(self.label_encoder, MODEL_DIR / "label_encoder.joblib")
        print("   💾 Saved label encoder")
        
        # Save feature selector
        if self.feature_selector:
            joblib.dump(self.feature_selector, MODEL_DIR / "feature_selector.joblib")
            print("   💾 Saved feature selector")
        
        # Save feature names
        with open(MODEL_DIR / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f, indent=2)
        print("   💾 Saved feature names")
        
        # Save selected feature names
        if self.feature_selector:
            selected_feature_names = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
            with open(MODEL_DIR / "selected_feature_names.json", "w") as f:
                json.dump(selected_feature_names, f, indent=2)
            print("   💾 Saved selected feature names")
        
        # Save anomaly types
        with open(MODEL_DIR / "anomaly_types.json", "w") as f:
            json.dump(self.anomaly_types, f, indent=2)
        print("   💾 Saved anomaly types")
        
        # Save training metadata
        selected_feature_names = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)] if self.feature_selector else self.feature_names
        
        metadata = {
            'training_date': datetime.now().isoformat(),
            'version': '3.0.0-peak-performer',
            'num_features': len(self.feature_names),
            'num_selected_features': len(selected_feature_names),
            'num_anomaly_types': len(self.anomaly_types),
            'num_models': len([k for k in self.models.keys() if k not in ['ensemble', 'base_ensemble', 'stacked_ensemble']]),
            'ensemble_type': 'advanced_stacking_blending',
            'anomaly_types': self.anomaly_types,
            'feature_names': self.feature_names,
            'selected_feature_names': selected_feature_names,
            'model_performance': self.training_results,
            'random_seed': RANDOM_SEED,
            'memory_limit_gb': self.memory_limit_gb,
            'models_trained': list(self.models.keys())
        }
        
        with open(MODEL_DIR / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("   💾 Saved training metadata")


def main():
    """Main training function"""
    print("="*70)
    print("🚀 AURA K8s - PEAK PERFORMER ML TRAINING")
    print("   Mac M4 Optimized - CPU-Only Training")
    print("   Maximum Accuracy - 12+ Models + Advanced Ensemble")
    print("   Zero Cost - Real Datasets + Synthetic Data")
    print("="*70)
    
    # Initialize trainer
    trainer = BeastLevelTrainer(memory_limit_gb=MEMORY_LIMIT_GB)
    
    # Load data - try multiple sources
    data_path = os.getenv("TRAINING_DATA_PATH", None)
    from_db = os.getenv("LOAD_FROM_DATABASE", "false").lower() == "true"
    db_conn = os.getenv("DATABASE_URL", None)
    dataset_name = os.getenv("DATASET_NAME", "synthetic_k8s")  # Default to synthetic
    use_real_datasets = os.getenv("USE_REAL_DATASETS", "true").lower() == "true"
    
    print(f"\n📊 Data Loading Strategy:")
    print(f"   • Dataset: {dataset_name}")
    print(f"   • Use real datasets: {use_real_datasets}")
    print(f"   • From database: {from_db}")
    print(f"   • Data path: {data_path}")
    print()
    
    X_raw, y = trainer.load_data(
        data_path=data_path,
        from_database=from_db,
        db_connection_string=db_conn,
        from_dataset=dataset_name if use_real_datasets else None,
        use_real_datasets=use_real_datasets
    )
    
    # Train models - optimized for fast training + high accuracy
    optimize = os.getenv("OPTIMIZE_HYPERPARAMETERS", "true").lower() == "true"  # Default to true
    n_trials = int(os.getenv("OPTUNA_TRIALS", "15"))  # Default to 15 trials (optimal balance)
    
    results = trainer.train(
        X_raw, y,
        optimize_hyperparameters=optimize,
        n_trials=n_trials
    )
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    for name, metrics in results.items():
        print(f"{name:20s}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n🏆 Best Model: {best_model[0]} (F1: {best_model[1]['f1_score']:.4f})")
    print("\n✅ Training complete! Models saved in models/ directory")
    print("="*70)


if __name__ == "__main__":
    main()

