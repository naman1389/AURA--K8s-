"""
Forecasting service for predictive anomaly detection
Provides time-series forecasting capabilities using ML models
Uses existing ML model for anomaly probability prediction
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import hashlib
import json

# Import existing ML model components from predictor (lazy import)
# We'll import these when needed to avoid circular dependency
def _get_ml_components():
    """Lazy import of ML model components to avoid circular dependency"""
    try:
        # Try to import from the same module
        from .predictor import models, scaler, label_encoder, feature_names, anomaly_types
        return models, scaler, label_encoder, feature_names, anomaly_types
    except ImportError:
        try:
            # Fallback to absolute import
            from ml.serve.predictor import models, scaler, label_encoder, feature_names, anomaly_types
            return models, scaler, label_encoder, feature_names, anomaly_types
        except ImportError:
            # Last resort - direct import
            import sys
            from pathlib import Path
            predictor_path = Path(__file__).parent / "predictor.py"
            if predictor_path.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("predictor", predictor_path)
                predictor = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(predictor)
                return predictor.models, predictor.scaler, predictor.label_encoder, predictor.feature_names, predictor.anomaly_types
            else:
                return {}, None, None, [], []

logger = logging.getLogger(__name__)


class ForecastRequest(BaseModel):
    """Request model for forecasting endpoint"""
    pod_name: str = Field(..., description="Name of the pod")
    namespace: str = Field(..., description="Namespace of the pod")
    metrics: Dict[str, List[float]] = Field(..., description="Historical metrics data")
    horizon_seconds: int = Field(..., ge=60, le=3600, description="Prediction horizon in seconds (1min to 1hour)")
    metrics_to_forecast: List[str] = Field(default=["cpu_utilization", "memory_utilization"], 
                                          description="List of metrics to forecast")


class ForecastResponse(BaseModel):
    """Response model for forecasting endpoint"""
    predictions: Dict[str, Dict] = Field(..., description="Forecast for each metric")
    anomaly_probabilities: Dict[str, float] = Field(..., description="Anomaly probability for each metric")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Overall risk score (0-100)")
    time_to_anomaly: Optional[int] = Field(None, description="Estimated seconds until anomaly")
    severity: str = Field(..., description="Severity level: Critical, High, Medium, Low")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in forecast")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended preventive actions")
    timestamp: datetime = Field(default_factory=datetime.now, description="When forecast was made")


class FuturePredictionRequest(BaseModel):
    """Request model for future prediction endpoint"""
    pod_name: str = Field(..., description="Name of the pod")
    namespace: str = Field(..., description="Namespace of the pod")
    historical_data: List[Dict] = Field(..., description="Historical metrics data points")
    steps: int = Field(..., ge=1, le=60, description="Number of future steps to predict")
    step_size_seconds: int = Field(default=60, ge=1, description="Size of each step in seconds")


class FuturePredictionResponse(BaseModel):
    """Response model for future prediction endpoint"""
    predictions: List[Dict] = Field(..., description="Predictions for each future step")
    anomaly_detected: bool = Field(..., description="Whether anomaly is predicted")
    anomaly_type: Optional[str] = Field(None, description="Type of predicted anomaly")
    time_to_anomaly: Optional[int] = Field(None, description="Estimated seconds until anomaly")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction")
    timestamp: datetime = Field(default_factory=datetime.now, description="When prediction was made")


class ForecastingService:
    """Service for generating forecasts using ML models"""
    
    def __init__(self):
        """Initialize forecasting service with existing ML model"""
        # Lazy load ML components
        ml_models, ml_scaler, ml_label_encoder, ml_feature_names, ml_anomaly_types = _get_ml_components()
        self.models = ml_models
        self.scaler = ml_scaler
        self.label_encoder = ml_label_encoder
        self.feature_names = ml_feature_names
        self.anomaly_types = ml_anomaly_types
        
        self.use_existing_model = len(self.models) > 0 and self.scaler is not None
        if self.use_existing_model:
            logger.info(f"ForecastingService initialized with existing ML model ({len(self.models)} models)")
        else:
            logger.warning("ForecastingService initialized without ML models - using placeholder mode")
        
        # Initialize forecast cache (simple in-memory cache)
        self.forecast_cache: Dict[str, tuple] = {}  # key -> (response, timestamp)
        self.cache_ttl = 60  # Cache for 60 seconds
    
    def _generate_cache_key(self, request: ForecastRequest) -> str:
        """Generate cache key from forecast request"""
        # Create a stable key from pod, namespace, metrics, and horizon
        key_data = {
            'pod': request.pod_name,
            'namespace': request.namespace,
            'horizon': request.horizon_seconds,
            'metrics': sorted(request.metrics_to_forecast),
            # Use last 10 values of each metric for cache key (recent trend)
            'cpu_tail': request.metrics.get('cpu_utilization', [])[-10:] if 'cpu_utilization' in request.metrics else [],
            'memory_tail': request.metrics.get('memory_utilization', [])[-10:] if 'memory_utilization' in request.metrics else [],
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _get_cached_forecast(self, cache_key: str) -> Optional[ForecastResponse]:
        """Get cached forecast if available and not expired"""
        if cache_key in self.forecast_cache:
            response, timestamp = self.forecast_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.debug(f"Cache hit for forecast: {cache_key[:8]}...")
                return response
            else:
                # Remove expired entry
                del self.forecast_cache[cache_key]
        return None
    
    def _cache_forecast(self, cache_key: str, response: ForecastResponse):
        """Cache forecast result"""
        # Limit cache size to 1000 entries
        if len(self.forecast_cache) >= 1000:
            # Remove oldest 10% of entries
            sorted_entries = sorted(self.forecast_cache.items(), key=lambda x: x[1][1])
            to_remove = max(1, len(sorted_entries) // 10)
            for i in range(to_remove):
                del self.forecast_cache[sorted_entries[i][0]]
        
        self.forecast_cache[cache_key] = (response, time.time())
    
    def _build_features_from_forecast(self, cpu_forecast: float, memory_forecast: float, 
                                     historical_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Build feature vector from forecasted values for ML model prediction"""
        # Extract historical values for derived features
        cpu_data = historical_data.get('cpu_utilization', [])
        memory_data = historical_data.get('memory_utilization', [])
        
        # Calculate derived features
        disk_usage = 50.0  # Default, would come from historical data if available
        network_bytes_sec = 1000.0  # Default, would come from historical data
        error_rate = 0.0  # Default
        latency_ms = 10.0  # Base latency
        restart_count = 0.0  # Would come from pod metrics
        age_minutes = 120.0  # Would come from pod metrics
        
        # Calculate derived metrics
        cpu_memory_ratio = cpu_forecast / (memory_forecast + 1) if memory_forecast > 0 else 0
        resource_pressure = (cpu_forecast + memory_forecast) / 2
        error_latency_product = error_rate * latency_ms
        network_per_cpu = network_bytes_sec / (cpu_forecast + 1) if cpu_forecast > 0 else 0
        is_critical = 1.0 if cpu_forecast > 80 or memory_forecast > 80 else 0.0
        
        # Estimate latency based on CPU
        if cpu_forecast > 80:
            latency_ms = 10.0 + (cpu_forecast - 80) * 1.5
            if latency_ms > 500:
                latency_ms = 500
        
        # Build feature vector matching the 13 required features
        features = {
            "cpu_usage": cpu_forecast,
            "memory_usage": memory_forecast,
            "disk_usage": disk_usage,
            "network_bytes_sec": network_bytes_sec,
            "error_rate": error_rate,
            "latency_ms": latency_ms,
            "restart_count": restart_count,
            "age_minutes": age_minutes,
            "cpu_memory_ratio": cpu_memory_ratio,
            "resource_pressure": resource_pressure,
            "error_latency_product": error_latency_product,
            "network_per_cpu": network_per_cpu,
            "is_critical": is_critical,
        }
        
        return features
    
    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        """Generate forecasts for future metrics using existing ML model
        
        Args:
            request: Forecast request with historical data and parameters
            
        Returns:
            ForecastResponse with predictions and anomaly probabilities
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_response = self._get_cached_forecast(cache_key)
        if cached_response is not None:
            logger.debug(f"Returning cached forecast (saved {time.time() - start_time:.3f}s)")
            # Record cache hit (if metrics available)
            try:
                import requests
                metrics_url = os.getenv("COLLECTOR_URL", "http://localhost:9090")
                # Prometheus metrics recorded in Go code
            except:
                pass
            return cached_response
        
        try:
            predictions = {}
            anomaly_probabilities = {}
            ml_predictions = {}  # Store ML model predictions for each metric
            
            # Step 1: Calculate trend-based forecasts for metrics (~5-10ms)
            forecast_start = time.time()
            cpu_forecast = None
            memory_forecast = None
            
            for metric_name in request.metrics_to_forecast:
                if metric_name in request.metrics:
                    metric_data = request.metrics[metric_name]
                    if len(metric_data) > 0:
                        # Trend-based forecast calculation
                        recent_avg = np.mean(metric_data[-10:]) if len(metric_data) >= 10 else np.mean(metric_data)
                        recent_trend = (metric_data[-1] - metric_data[0]) / len(metric_data) if len(metric_data) > 1 else 0
                        
                        # Project forward based on trend
                        horizon_steps = request.horizon_seconds // 60  # Convert to minutes
                        predicted_value = recent_avg + (recent_trend * horizon_steps)
                        
                        # Clamp values to reasonable ranges
                        if metric_name == "cpu_utilization" or metric_name == "memory_utilization":
                            predicted_value = max(0.0, min(100.0, predicted_value))
                            if metric_name == "cpu_utilization":
                                cpu_forecast = predicted_value
                            else:
                                memory_forecast = predicted_value
                        
                        predictions[metric_name] = {
                            "predicted_value": float(predicted_value),
                            "confidence_lower": float(predicted_value * 0.9),
                            "confidence_upper": float(predicted_value * 1.1),
                            "trend": float(recent_trend),
                        }
                    else:
                        predictions[metric_name] = {
                            "predicted_value": 0.0,
                            "confidence_lower": 0.0,
                            "confidence_upper": 0.0,
                            "trend": 0.0,
                        }
            
            forecast_time = (time.time() - forecast_start) * 1000  # Convert to ms
            
            # Step 2: Use existing ML model to predict anomaly probabilities (~50-100ms)
            ml_start = time.time()
            if self.use_existing_model and cpu_forecast is not None and memory_forecast is not None:
                try:
                    # Build feature vector from forecasted values
                    features = self._build_features_from_forecast(
                        cpu_forecast, 
                        memory_forecast, 
                        request.metrics
                    )
                    
                    # Prepare feature vector for ML model
                    feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
                    
                    if len(feature_vector) == len(self.feature_names):
                        # Create DataFrame
                        feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)
                        
                        # Scale features
                        if self.scaler:
                            feature_df = pd.DataFrame(self.scaler.transform(feature_df), columns=self.feature_names)
                        
                        # Get ensemble predictions
                        probabilities_list = []
                        for model_name, model in self.models.items():
                            pred_proba = model.predict_proba(feature_df)[0]
                            probabilities_list.append(pred_proba)
                        
                        # Weighted ensemble averaging
                        weights = np.ones(len(probabilities_list)) / len(probabilities_list)
                        avg_proba = np.average(probabilities_list, axis=0, weights=weights)
                        
                        # Calculate anomaly probability (1 - probability of "healthy")
                        if self.label_encoder:
                            healthy_idx = None
                            for i, label in enumerate(self.label_encoder.classes_):
                                if label == "healthy":
                                    healthy_idx = i
                                    break
                            if healthy_idx is not None:
                                anomaly_prob = 1.0 - float(avg_proba[healthy_idx])
                            else:
                                # If no healthy class, use max non-zero probability
                                anomaly_prob = float(max(avg_proba[1:]) if len(avg_proba) > 1 else avg_proba[0])
                        else:
                            # Fallback: use probability of non-first class (assuming first is healthy)
                            anomaly_prob = float(sum(avg_proba[1:]) if len(avg_proba) > 1 else 0.0)
                        
                        # Get predicted anomaly type
                        final_prediction = np.argmax(avg_proba)
                        if self.label_encoder:
                            predicted_anomaly_type = self.label_encoder.inverse_transform([final_prediction])[0]
                        else:
                            predicted_anomaly_type = self.anomaly_types[final_prediction] if final_prediction < len(self.anomaly_types) else "unknown"
                        
                        confidence = float(avg_proba[final_prediction])
                        
                        # Store ML prediction results
                        ml_predictions['anomaly_type'] = predicted_anomaly_type
                        ml_predictions['confidence'] = confidence
                        ml_predictions['anomaly_probability'] = anomaly_prob
                        
                        # Set anomaly probabilities for each metric
                        for metric_name in request.metrics_to_forecast:
                            if metric_name in ["cpu_utilization", "memory_utilization"]:
                                # Higher probability if forecasted value is high
                                metric_weight = predictions[metric_name]["predicted_value"] / 100.0
                                anomaly_probabilities[metric_name] = float(anomaly_prob * metric_weight)
                            else:
                                anomaly_probabilities[metric_name] = float(anomaly_prob * 0.5)
                    else:
                        logger.warning(f"Feature vector length mismatch: {len(feature_vector)} != {len(self.feature_names)}")
                        # Fallback to simple threshold-based probability
                        for metric_name in request.metrics_to_forecast:
                            pred_val = predictions[metric_name]["predicted_value"]
                            anomaly_probabilities[metric_name] = float(max(0.0, (pred_val - 70) / 30.0)) if pred_val > 70 else 0.0
                
                except Exception as e:
                    logger.error(f"ML model prediction failed: {e}", exc_info=True)
                    # Fallback to threshold-based probability
                    for metric_name in request.metrics_to_forecast:
                        pred_val = predictions[metric_name]["predicted_value"]
                        anomaly_probabilities[metric_name] = float(max(0.0, (pred_val - 70) / 30.0)) if pred_val > 70 else 0.0
            else:
                # Fallback: threshold-based anomaly probability
                for metric_name in request.metrics_to_forecast:
                    pred_val = predictions[metric_name]["predicted_value"]
                    anomaly_probabilities[metric_name] = float(max(0.0, (pred_val - 70) / 30.0)) if pred_val > 70 else 0.0
            
            ml_time = (time.time() - ml_start) * 1000  # Convert to ms
            
            # Step 3: Calculate risk score and severity (~1-2ms)
            risk_start = time.time()
            max_anomaly_prob = max(anomaly_probabilities.values()) if anomaly_probabilities else 0.0
            avg_anomaly_prob = np.mean(list(anomaly_probabilities.values())) if anomaly_probabilities else 0.0
            
            # Risk score combines ML prediction confidence and anomaly probability
            if ml_predictions:
                ml_confidence = ml_predictions.get('confidence', 0.5)
                ml_anomaly_prob = ml_predictions.get('anomaly_probability', 0.0)
                risk_score = (ml_anomaly_prob * 0.7 + max_anomaly_prob * 0.3) * 100 * ml_confidence
            else:
                risk_score = max_anomaly_prob * 100
            
            risk_score = min(100.0, max(0.0, risk_score))
            
            # Determine severity based on risk score
            if risk_score >= 80:
                severity = "Critical"
            elif risk_score >= 60:
                severity = "High"
            elif risk_score >= 40:
                severity = "Medium"
            else:
                severity = "Low"
            
            # Calculate time to anomaly based on trend
            time_to_anomaly = None
            if cpu_forecast is not None and memory_forecast is not None:
                # Estimate time until threshold (80%) is reached
                threshold = 80.0
                cpu_trend = predictions.get("cpu_utilization", {}).get("trend", 0.0)
                mem_trend = predictions.get("memory_utilization", {}).get("trend", 0.0)
                
                if cpu_trend > 0 and cpu_forecast < threshold:
                    time_to_cpu = (threshold - cpu_forecast) / cpu_trend * 60  # Convert to seconds
                    time_to_anomaly = int(time_to_cpu) if time_to_cpu > 0 else None
                elif mem_trend > 0 and memory_forecast < threshold:
                    time_to_mem = (threshold - memory_forecast) / mem_trend * 60
                    time_to_anomaly = int(time_to_mem) if time_to_mem > 0 else None
            
            # Generate recommended actions
            recommended_actions = []
            if risk_score >= 80:
                recommended_actions = ["scale_up_immediately", "increase_resources"]
            elif risk_score >= 60:
                recommended_actions = ["scale_up", "monitor_closely"]
            elif risk_score >= 40:
                recommended_actions = ["increase_resources", "monitor"]
            else:
                recommended_actions = ["monitor"]
            
            # Overall confidence from ML model
            confidence = ml_predictions.get('confidence', 0.75) if ml_predictions else 0.5
            
            risk_time = (time.time() - risk_start) * 1000  # Convert to ms
            total_time = (time.time() - start_time) * 1000  # Total in ms
            
            logger.debug(f"Forecast timing: forecast={forecast_time:.2f}ms, ml={ml_time:.2f}ms, risk={risk_time:.2f}ms, total={total_time:.2f}ms")
            
            response = ForecastResponse(
                predictions=predictions,
                anomaly_probabilities=anomaly_probabilities,
                risk_score=float(risk_score),
                time_to_anomaly=time_to_anomaly,
                severity=severity,
                confidence=confidence,
                recommended_actions=recommended_actions,
                timestamp=datetime.now()
            )
            
            # Cache the response
            self._cache_forecast(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")
    
    def predict_future(self, request: FuturePredictionRequest) -> FuturePredictionResponse:
        """Predict future state and detect anomalies using existing ML model
        
        Args:
            request: Future prediction request with historical data
            
        Returns:
            FuturePredictionResponse with predictions and anomaly detection
        """
        start_time = time.time()
        try:
            if not request.historical_data:
                raise ValueError("Historical data is required")
            
            # Extract metrics from historical data
            cpu_data = [d.get('cpu_utilization', 0) for d in request.historical_data if 'cpu_utilization' in d]
            memory_data = [d.get('memory_utilization', 0) for d in request.historical_data if 'memory_utilization' in d]
            
            if not cpu_data or not memory_data:
                raise ValueError("Historical data must contain cpu_utilization and memory_utilization")
            
            predictions = []
            max_anomaly_prob = 0.0
            predicted_anomaly_type = None
            max_confidence = 0.0
            
            # Predict for each future step
            for step in range(1, request.steps + 1):
                # Calculate forecasted values using trend
                cpu_avg = np.mean(cpu_data[-10:]) if len(cpu_data) >= 10 else np.mean(cpu_data)
                cpu_trend = (cpu_data[-1] - cpu_data[0]) / len(cpu_data) if len(cpu_data) > 1 else 0
                cpu_forecast = cpu_avg + (cpu_trend * step)
                cpu_forecast = max(0.0, min(100.0, cpu_forecast))
                
                memory_avg = np.mean(memory_data[-10:]) if len(memory_data) >= 10 else np.mean(memory_data)
                memory_trend = (memory_data[-1] - memory_data[0]) / len(memory_data) if len(memory_data) > 1 else 0
                memory_forecast = memory_avg + (memory_trend * step)
                memory_forecast = max(0.0, min(100.0, memory_forecast))
                
                # Use ML model to predict anomaly probability
                anomaly_prob = 0.0
                if self.use_existing_model:
                    try:
                        features = self._build_features_from_forecast(
                            cpu_forecast,
                            memory_forecast,
                            {'cpu_utilization': cpu_data, 'memory_utilization': memory_data}
                        )
                        
                        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
                        if len(feature_vector) == len(self.feature_names):
                            feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)
                            if self.scaler:
                                feature_df = pd.DataFrame(self.scaler.transform(feature_df), columns=self.feature_names)
                            
                            probabilities_list = []
                            for model_name, model in self.models.items():
                                pred_proba = model.predict_proba(feature_df)[0]
                                probabilities_list.append(pred_proba)
                            
                            weights = np.ones(len(probabilities_list)) / len(probabilities_list)
                            avg_proba = np.average(probabilities_list, axis=0, weights=weights)
                            
                            # Calculate anomaly probability
                            if self.label_encoder:
                                healthy_idx = None
                                for i, label in enumerate(self.label_encoder.classes_):
                                    if label == "healthy":
                                        healthy_idx = i
                                        break
                                if healthy_idx is not None:
                                    anomaly_prob = 1.0 - float(avg_proba[healthy_idx])
                                else:
                                    anomaly_prob = float(max(avg_proba[1:]) if len(avg_proba) > 1 else avg_proba[0])
                            else:
                                anomaly_prob = float(sum(avg_proba[1:]) if len(avg_proba) > 1 else 0.0)
                            
                            if anomaly_prob > max_anomaly_prob:
                                max_anomaly_prob = anomaly_prob
                                final_prediction = np.argmax(avg_proba)
                                if self.label_encoder:
                                    predicted_anomaly_type = self.label_encoder.inverse_transform([final_prediction])[0]
                                else:
                                    predicted_anomaly_type = self.anomaly_types[final_prediction] if final_prediction < len(self.anomaly_types) else "unknown"
                                max_confidence = float(avg_proba[final_prediction])
                    except Exception as e:
                        logger.warning(f"ML prediction failed for step {step}: {e}")
                        # Fallback
                        anomaly_prob = max(0.0, (cpu_forecast - 70) / 30.0) if cpu_forecast > 70 else 0.0
                
                prediction = {
                    "step": step,
                    "timestamp": datetime.now() + timedelta(seconds=step * request.step_size_seconds),
                    "cpu_utilization": float(cpu_forecast),
                    "memory_utilization": float(memory_forecast),
                    "anomaly_probability": float(anomaly_prob),
                }
                predictions.append(prediction)
            
            # Calculate time to anomaly
            time_to_anomaly = None
            if max_anomaly_prob > 0.7 and cpu_data:
                cpu_trend = (cpu_data[-1] - cpu_data[0]) / len(cpu_data) if len(cpu_data) > 1 else 0
                if cpu_trend > 0:
                    current_cpu = cpu_data[-1]
                    threshold = 80.0
                    if current_cpu < threshold:
                        time_to_anomaly = int((threshold - current_cpu) / cpu_trend * 60)
            
            total_time = (time.time() - start_time) * 1000
            logger.debug(f"Future prediction timing: {total_time:.2f}ms for {request.steps} steps")
            
            return FuturePredictionResponse(
                predictions=predictions,
                anomaly_detected=max_anomaly_prob > 0.7,
                anomaly_type=predicted_anomaly_type if max_anomaly_prob > 0.7 else None,
                time_to_anomaly=time_to_anomaly,
                confidence=max_confidence if max_confidence > 0 else 0.5,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Future prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Future prediction failed: {str(e)}")


# Global service instance
_forecasting_service: Optional[ForecastingService] = None


def get_forecasting_service() -> ForecastingService:
    """Get or create forecasting service instance"""
    global _forecasting_service
    if _forecasting_service is None:
        _forecasting_service = ForecastingService()
    return _forecasting_service

