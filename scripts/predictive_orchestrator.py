#!/usr/bin/env python3
"""
Predictive Orchestrator for AURA K8s
Processes metrics → Forecasts → Early Warnings → Preventive Actions
Runs continuously to provide proactive anomaly detection
"""

import os
import asyncio
import time
import psycopg2
import psycopg2.extensions
import psycopg2.pool
import httpx
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

# Import config helper for service discovery
import sys
from pathlib import Path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from config_helper import get_database_url, get_service_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
# Configuration with validation - use config helper for environment-aware URLs
DATABASE_URL = get_database_url()
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is required")
    raise ValueError("DATABASE_URL environment variable is required")

ML_SERVICE_URL = get_service_url("ML_SERVICE", "8001")
if not ML_SERVICE_URL:
    logger.error("ML_SERVICE_URL environment variable is required")
    raise ValueError("ML_SERVICE_URL environment variable is required")

FORECAST_INTERVAL = int(os.getenv("FORECAST_INTERVAL", "5"))  # seconds
if FORECAST_INTERVAL < 1:
    logger.warning(f"FORECAST_INTERVAL too low ({FORECAST_INTERVAL}s), setting to minimum 1s")
    FORECAST_INTERVAL = 1

PREDICTION_HORIZON = int(os.getenv("PREDICTION_HORIZON", "900"))  # 15 minutes in seconds
if PREDICTION_HORIZON < 60:
    logger.warning(f"PREDICTION_HORIZON too low ({PREDICTION_HORIZON}s), setting to minimum 60s")
    PREDICTION_HORIZON = 60

ENABLE_PREVENTIVE_ACTIONS = os.getenv("ENABLE_PREVENTIVE_ACTIONS", "true").lower() == "true"

# Connection pool
db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


class PredictiveOrchestrator:
    """Main orchestrator for predictive anomaly detection"""
    
    def __init__(self):
        self.ml_service_url = ML_SERVICE_URL
        self.forecast_interval = FORECAST_INTERVAL
        self.prediction_horizon = PREDICTION_HORIZON
        self.enable_preventive = ENABLE_PREVENTIVE_ACTIONS
        self.running = False
        
    async def run_predictive_loop(self):
        """Main async loop for predictive processing"""
        self.running = True
        logger.info("🚀 Starting predictive orchestrator loop")
        logger.info(f"   Forecast interval: {self.forecast_interval}s")
        logger.info(f"   Prediction horizon: {self.prediction_horizon}s")
        logger.info(f"   Preventive actions: {'✅ enabled' if self.enable_preventive else '❌ disabled'}")
        logger.info("   ⚡ Fast detection mode: Issues detected BEFORE they occur")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Get recent metrics
                metrics = await self.get_recent_metrics()
                if not metrics:
                    logger.debug("No recent metrics found, waiting...")
                    await asyncio.sleep(self.forecast_interval)
                    continue
                
                logger.info(f"📊 Processing {len(metrics)} pod metrics")
                
                # Generate forecasts for all pods (fast parallel processing)
                forecast_start = time.time()
                forecasts = await self.generate_forecasts(metrics)
                forecast_time = time.time() - forecast_start
                
                # Log forecast generation results
                logger.info(f"📈 Generated {len(forecasts)} forecasts from {len(metrics)} metrics in {forecast_time:.3f}s")
                if len(forecasts) > 0:
                    sample_forecast = forecasts[0]
                    logger.debug(f"Sample forecast: pod={sample_forecast.get('pod_name')}, risk={sample_forecast.get('risk_score', 0):.1f}, cpu={sample_forecast.get('predictions', {}).get('cpu_utilization', {}).get('predicted_value', 0):.1f}%")
                
                # Detect future anomalies (very fast - threshold checks)
                detection_start = time.time()
                warnings = await self.detect_future_anomalies(forecasts)
                detection_time = time.time() - detection_start
                
                if len(warnings) > 0:
                    logger.info(f"⚠️  Detected {len(warnings)} future anomalies in {detection_time:.3f}s")
                elif len(forecasts) > 0:
                    logger.debug(f"No warnings generated from {len(forecasts)} forecasts (all below thresholds)")
                
                # Trigger preventive actions (saves to DB, remediator processes automatically)
                if self.enable_preventive and warnings:
                    action_start = time.time()
                    await self.trigger_preventive_actions(warnings)
                    action_time = time.time() - action_start
                    logger.info(f"⚠️  Generated {len(warnings)} early warnings (action time: {action_time:.3f}s)")
                
                elapsed = time.time() - start_time
                logger.info(f"✅ Predictive cycle completed in {elapsed:.2f}s (forecast: {forecast_time:.3f}s, detection: {detection_time:.3f}s)")
                
                # Sleep until next cycle
                sleep_time = max(0, self.forecast_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in predictive loop: {e}", exc_info=True)
                await asyncio.sleep(self.forecast_interval)
    
    async def get_recent_metrics(self) -> List[Dict]:
        """Get recent metrics - tries circular buffer first, falls back to database"""
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from config_helper import get_service_url
        collector_url = get_service_url("COLLECTOR", "9090")
        use_buffer = os.getenv("USE_CIRCULAR_BUFFER", "true").lower() == "true"
        
        # Try circular buffer first (faster)
        if use_buffer:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{collector_url}/api/v1/buffer/metrics?limit=1000")
                    if response.status_code == 200:
                        data = response.json()
                        buffer_metrics = data.get("metrics", [])
                        if buffer_metrics:
                            logger.debug(f"Using {len(buffer_metrics)} metrics from circular buffer")
                            # Convert buffer format to our format
                            metrics = []
                            for m in buffer_metrics:
                                metrics.append({
                                    'pod_name': m['pod_name'],
                                    'namespace': m['namespace'],
                                    'timestamp': datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00')),
                                    'cpu_utilization': float(m.get('cpu_utilization', 0)),
                                    'memory_utilization': float(m.get('memory_utilization', 0)),
                                    'network_rx_bytes': int(m.get('network_rx_bytes', 0)),
                                    'network_tx_bytes': int(m.get('network_tx_bytes', 0)),
                                    'network_rx_errors': 0,  # Buffer may not have this
                                    'network_tx_errors': 0,  # Buffer may not have this
                                    'restarts': int(m.get('restarts', 0)),
                                    'age': 0,  # Will calculate if needed
                                })
                            return metrics
            except Exception as e:
                logger.debug(f"Circular buffer unavailable, falling back to database: {e}")
        
        # Fallback to database query
        try:
            conn = db_pool.getconn()
            try:
                cur = conn.cursor()
                
                # Get metrics from last 5 minutes
                query = """
                    SELECT DISTINCT ON (pod_name, namespace)
                        pod_name, namespace, timestamp,
                        cpu_utilization, memory_utilization,
                        network_rx_bytes, network_tx_bytes,
                        network_rx_errors, network_tx_errors,
                        restarts, age
                    FROM pod_metrics
                    WHERE timestamp > NOW() - INTERVAL '5 minutes'
                        AND namespace NOT IN ('kube-system', 'kube-public', 'kube-node-lease', 'local-path-storage')
                    ORDER BY pod_name, namespace, timestamp DESC
                """
                
                cur.execute(query)
                rows = cur.fetchall()
                
                metrics = []
                for row in rows:
                    metrics.append({
                        'pod_name': row[0],
                        'namespace': row[1],
                        'timestamp': row[2],
                        'cpu_utilization': float(row[3] or 0),
                        'memory_utilization': float(row[4] or 0),
                        'network_rx_bytes': int(row[5] or 0),
                        'network_tx_bytes': int(row[6] or 0),
                        'network_rx_errors': int(row[7] or 0),
                        'network_tx_errors': int(row[8] or 0),
                        'restarts': int(row[9] or 0),
                        'age': int(row[10] or 0),
                    })
                
                cur.close()
                return metrics
                
            finally:
                db_pool.putconn(conn)
                
        except Exception as e:
            logger.error(f"Failed to get recent metrics: {e}", exc_info=True)
            return []
    
    async def generate_forecast_for_pod(self, metric: Dict, client: httpx.AsyncClient, retry_count: int = 0) -> Optional[Dict]:
        """Generate forecast for a single pod (used for parallel processing) with retry logic"""
        max_retries = 3
        retry_delay = 1.0  # seconds
        
        try:
            # Get historical data for this pod
            historical = await self.get_historical_data(metric['pod_name'], metric['namespace'])
            
            if len(historical) < 3:
                # Not enough data for reliable forecast (lowered from 5 to 3 to allow earlier forecasts)
                logger.debug(f"Insufficient historical data for {metric['pod_name']}: {len(historical)} points (need 3+)")
                return None
            
            # Prepare forecast request
            forecast_request = {
                'pod_name': metric['pod_name'],
                'namespace': metric['namespace'],
                'metrics': {
                    'cpu_utilization': [h['cpu_utilization'] for h in historical],
                    'memory_utilization': [h['memory_utilization'] for h in historical],
                },
                'horizon_seconds': self.prediction_horizon,
                'metrics_to_forecast': ['cpu_utilization', 'memory_utilization'],
            }
            
            # Call forecasting service asynchronously with retry
            try:
                response = await client.post(
                    f"{self.ml_service_url}/v1/forecast",
                    json=forecast_request,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    forecast = response.json()
                    forecast['pod_name'] = metric['pod_name']
                    forecast['namespace'] = metric['namespace']
                    return forecast
                elif response.status_code >= 500 and retry_count < max_retries:
                    # Server error - retry
                    logger.warning(f"Forecast service error {response.status_code} for {metric['pod_name']}, retrying ({retry_count + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay * (retry_count + 1))
                    return await self.generate_forecast_for_pod(metric, client, retry_count + 1)
                else:
                    logger.warning(f"Forecast failed for {metric['pod_name']}: {response.status_code}")
                    return None
                    
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                # Network error - retry
                if retry_count < max_retries:
                    logger.warning(f"Forecast service connection error for {metric['pod_name']}, retrying ({retry_count + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay * (retry_count + 1))
                    return await self.generate_forecast_for_pod(metric, client, retry_count + 1)
                else:
                    logger.error(f"Forecast service unavailable after {max_retries} retries for {metric['pod_name']}: {e}")
                    return None
                
        except Exception as e:
            logger.error(f"Error generating forecast for {metric['pod_name']}: {e}")
            return None
    
    async def generate_forecasts(self, metrics: List[Dict]) -> List[Dict]:
        """Generate forecasts for all pods in parallel"""
        forecasts = []
        
        # Use async HTTP client for parallel requests
        async with httpx.AsyncClient() as client:
            # Create tasks for all pods
            tasks = [self.generate_forecast_for_pod(metric, client) for metric in metrics]
            
            # Execute all forecasts in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful forecasts
            for result in results:
                if isinstance(result, dict) and result is not None:
                    forecasts.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Forecast task failed: {result}")
        
        return forecasts
    
    async def get_historical_data(self, pod_name: str, namespace: str, limit: int = 100) -> List[Dict]:
        """Get historical metrics for a pod"""
        try:
            conn = db_pool.getconn()
            try:
                cur = conn.cursor()
                
                query = """
                    SELECT timestamp, cpu_utilization, memory_utilization
                    FROM pod_metrics
                    WHERE pod_name = %s AND namespace = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """
                
                cur.execute(query, (pod_name, namespace, limit))
                rows = cur.fetchall()
                
                historical = []
                for row in rows:
                    historical.append({
                        'timestamp': row[0],
                        'cpu_utilization': float(row[1] or 0),
                        'memory_utilization': float(row[2] or 0),
                    })
                
                # Reverse to get chronological order
                historical.reverse()
                
                cur.close()
                return historical
                
            finally:
                db_pool.putconn(conn)
                
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    async def detect_future_anomalies(self, forecasts: List[Dict]) -> List[Dict]:
        """Detect anomalies in forecasts"""
        warnings = []
        
        for forecast in forecasts:
            try:
                # Check if anomaly is predicted
                risk_score = forecast.get('risk_score', 0.0)
                anomaly_probs = forecast.get('anomaly_probabilities', {})
                max_prob = max(anomaly_probs.values()) if anomaly_probs else 0.0
                
                # Generate warning if risk is high (lowered threshold for better detection)
                # Risk score > 50 OR anomaly probability > 0.6 OR forecasted values indicate issues
                predictions = forecast.get('predictions', {})
                cpu_forecast = predictions.get('cpu_utilization', {}).get('predicted_value', 0) if isinstance(predictions.get('cpu_utilization'), dict) else 0
                mem_forecast = predictions.get('memory_utilization', {}).get('predicted_value', 0) if isinstance(predictions.get('memory_utilization'), dict) else 0
                has_high_forecast = cpu_forecast > 70 or mem_forecast > 70
                
                # Lower thresholds for better detection
                should_warn = risk_score > 40 or max_prob > 0.5 or has_high_forecast
                logger.debug(f"Forecast check for {forecast['pod_name']}: risk={risk_score:.1f} (>{40}?), prob={max_prob:.2f} (>{0.5}?), cpu={cpu_forecast:.1f}% (>{70}?), mem={mem_forecast:.1f}% (>{70}?) -> warn={should_warn}")
                
                if should_warn:
                    logger.info(f"🔮 Early warning triggered for {forecast['pod_name']}: risk={risk_score:.1f}, prob={max_prob:.2f}, cpu={cpu_forecast:.1f}%, mem={mem_forecast:.1f}%")
                    # Flatten predicted_metrics to match Go's expected format (map[string]float64)
                    predictions = forecast.get('predictions', {})
                    flattened_metrics = {}
                    for metric_name, metric_data in predictions.items():
                        if isinstance(metric_data, dict):
                            # Extract predicted_value as the main metric value
                            flattened_metrics[metric_name] = metric_data.get('predicted_value', 0.0)
                        else:
                            # Already a float value
                            flattened_metrics[metric_name] = float(metric_data) if metric_data is not None else 0.0
                    
                    warning = {
                        'pod_name': forecast['pod_name'],
                        'namespace': forecast['namespace'],
                        'warning_type': 'anomaly_predicted',
                        'severity': forecast.get('severity', 'Medium'),
                        'risk_score': risk_score,
                        'time_to_anomaly': forecast.get('time_to_anomaly'),
                        'confidence': forecast.get('confidence', 0.5),
                        'recommended_action': self._get_recommended_action(risk_score, forecast.get('severity')),
                        'predicted_metrics': flattened_metrics,  # Use flattened version
                        'timestamp': datetime.now(),
                    }
                    warnings.append(warning)
                    
            except Exception as e:
                logger.error(f"Error detecting anomaly: {e}")
                continue
        
        return warnings
    
    def _get_recommended_action(self, risk_score: float, severity: str) -> str:
        """Get recommended action based on risk and severity"""
        if severity == "Critical" or risk_score > 80:
            return "scale_up_immediately"
        elif severity == "High" or risk_score > 60:
            return "scale_up"
        elif severity == "Medium" or risk_score > 40:
            return "increase_resources"
        return "monitor"
    
    async def trigger_preventive_actions(self, warnings: List[Dict]):
        """Trigger preventive actions for warnings"""
        remediator_url = get_service_url("REMEDIATOR", "9091")
        
        for warning in warnings:
            try:
                # Save warning to database first
                await self.save_warning(warning)
                
                # Log warning
                logger.info(f"⚠️  Early Warning: {warning['namespace']}/{warning['pod_name']} - "
                          f"{warning['severity']} risk ({warning['risk_score']:.1f}) - "
                          f"Action: {warning['recommended_action']}")
                
                # Trigger preventive action via remediator API (immediate action)
                remediator_url = get_service_url("REMEDIATOR", "9091")
                use_immediate_trigger = os.getenv("USE_IMMEDIATE_PREVENTIVE_TRIGGER", "true").lower() == "true"
                
                if use_immediate_trigger:
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.post(f"{remediator_url}/api/v1/trigger-preventive")
                            if response.status_code == 200:
                                logger.info(f"✅ Preventive action triggered immediately via API")
                            else:
                                logger.warning(f"Remediator API returned {response.status_code}, will process from DB")
                    except Exception as api_err:
                        logger.debug(f"Remediator API unavailable (will process from DB): {api_err}")
                else:
                    logger.info(f"✅ Warning saved - remediator will process automatically (polling mode)")
                
            except Exception as e:
                logger.error(f"Error triggering preventive action: {e}")
                continue
    
    async def save_warning(self, warning: Dict):
        """Save early warning to database with duplicate prevention"""
        try:
            conn = db_pool.getconn()
            try:
                cur = conn.cursor()
                
                # Check for existing active warning for this pod (duplicate prevention)
                check_query = """
                    SELECT id FROM early_warnings
                    WHERE pod_name = %s AND namespace = %s
                        AND warning_type = %s
                        AND (expires_at IS NULL OR expires_at > NOW())
                        AND (acknowledged IS NULL OR acknowledged = FALSE)
                    LIMIT 1
                """
                cur.execute(check_query, (
                    warning['pod_name'],
                    warning['namespace'],
                    warning['warning_type']
                ))
                existing = cur.fetchone()
                
                if existing:
                    # Update existing warning instead of creating duplicate
                    warning_id = existing[0]
                    predicted_metrics_json = json.dumps(warning.get('predicted_metrics', {}))
                    time_to_anomaly_seconds = warning.get('time_to_anomaly')
                    
                    update_query = """
                        UPDATE early_warnings SET
                            severity = %s,
                            risk_score = %s,
                            time_to_anomaly_seconds = %s,
                            confidence = %s,
                            recommended_action = %s,
                            predicted_metrics = %s,
                            created_at = %s
                        WHERE id = %s
                    """
                    cur.execute(update_query, (
                        warning['severity'],
                        warning['risk_score'],
                        time_to_anomaly_seconds,
                        warning['confidence'],
                        warning['recommended_action'],
                        predicted_metrics_json,
                        warning['timestamp'],
                        warning_id,
                    ))
                    logger.debug(f"Updated existing warning for {warning['pod_name']}")
                else:
                    # Create new warning
                    warning_id = str(uuid.uuid4())
                    predicted_metrics_json = json.dumps(warning.get('predicted_metrics', {}))
                    time_to_anomaly_seconds = warning.get('time_to_anomaly')
                    
                    insert_query = """
                        INSERT INTO early_warnings (
                            id, pod_name, namespace, warning_type, severity, risk_score,
                            time_to_anomaly_seconds, confidence, recommended_action,
                            predicted_metrics, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    cur.execute(insert_query, (
                        warning_id,
                        warning['pod_name'],
                        warning['namespace'],
                        warning['warning_type'],
                        warning['severity'],
                        warning['risk_score'],
                        time_to_anomaly_seconds,
                        warning['confidence'],
                        warning['recommended_action'],
                        predicted_metrics_json,
                        warning['timestamp'],
                    ))
                    logger.debug(f"Created new warning for {warning['pod_name']}")
                
                conn.commit()
                cur.close()
                
            finally:
                db_pool.putconn(conn)
                
        except Exception as e:
            logger.error(f"Failed to save warning: {e}")
            if conn:
                conn.rollback()


def init_db_pool():
    """Initialize database connection pool"""
    global db_pool
    try:
        db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        sys.exit(1)


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info("Shutting down predictive orchestrator...")
    global db_pool
    if db_pool:
        db_pool.closeall()
    sys.exit(0)


async def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize database pool
    init_db_pool()
    
    # Create orchestrator
    orchestrator = PredictiveOrchestrator()
    
    # Run predictive loop
    try:
        await orchestrator.run_predictive_loop()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        logger.info("Predictive orchestrator stopped")


if __name__ == '__main__':
    asyncio.run(main())

