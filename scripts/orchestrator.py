#!/usr/bin/env python3
"""
AURA K8s Orchestrator
Processes metrics → ML predictions → issues → remediation
Correctly matches feature engineering with training script
"""

import os
import time
import psycopg2
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://aura:aura_password@timescaledb:5432/aura_metrics")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8001")
PREDICTION_INTERVAL = int(os.getenv("PREDICTION_INTERVAL", "30"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

# Feature names must match training script exactly
FEATURE_NAMES = [
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "network_bytes_sec",
    "error_rate",
    "latency_ms",
    "restart_count",
    "age_minutes",
    "cpu_memory_ratio",
    "resource_pressure",
    "error_latency_product",
    "network_per_cpu",
    "is_critical"
]


def get_db_connection():
    """Get database connection with retry logic"""
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            return conn
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to database after {max_retries} attempts")
                raise


def generate_predictions(conn, ml_service_url: str) -> int:
    """
    Generate ML predictions for recent metrics
    Properly engineers features to match training script
    """
    logger.info("🤖 Generating ML predictions...")

    cur = conn.cursor()

    try:
        # Get recent metrics without predictions
        cur.execute("""
            SELECT DISTINCT ON (pm.pod_name, pm.namespace)
                pm.pod_name, pm.namespace, pm.timestamp,
                pm.cpu_utilization, pm.memory_utilization,
                COALESCE(pm.disk_usage_bytes::float / NULLIF(pm.disk_limit_bytes, 0) * 100, 0) as disk_usage_percent,
                COALESCE(pm.network_rx_bytes + pm.network_tx_bytes, 0) as network_bytes,
                COALESCE(pm.network_rx_errors + pm.network_tx_errors, 0) as error_rate,
                pm.restarts,
                EXTRACT(EPOCH FROM (NOW() - pm.timestamp)) / 60.0 as age_minutes,
                pm.cpu_trend, pm.memory_trend, pm.restart_trend,
                pm.has_oom_kill, pm.has_crash_loop, pm.has_high_cpu, pm.has_network_issues
            FROM pod_metrics pm
            LEFT JOIN ml_predictions mp ON 
                pm.pod_name = mp.pod_name 
                AND pm.namespace = mp.namespace 
                AND pm.timestamp = mp.timestamp
            WHERE pm.timestamp > NOW() - INTERVAL '1 hour'
                AND mp.timestamp IS NULL
            ORDER BY pm.pod_name, pm.namespace, pm.timestamp DESC
            LIMIT 50
        """)

        metrics = cur.fetchall()
        logger.info(f"   Found {len(metrics)} metrics to analyze")

        if len(metrics) == 0:
            return 0

        predictions_made = 0

        for metric in metrics:
            (pod_name, namespace, timestamp, cpu_util, mem_util, disk_pct,
             network_bytes, err_rate, restarts, age_minutes, cpu_trend,
             mem_trend, restart_trend, has_oom, has_crash, has_high_cpu, has_network) = metric

            try:
                # Engineer features to match training script EXACTLY
                cpu_usage = float(cpu_util or 0)
                memory_usage = float(mem_util or 0)
                disk_usage = float(disk_pct or 0)
                network_bytes_sec = float(network_bytes or 0) / 60.0  # Convert to per-second
                error_rate_val = float(err_rate or 0)
                latency_ms = 0  # Would come from actual metrics if available
                restart_count = int(restarts or 0)
                age_minutes_val = float(age_minutes or 0)

                # Engineered features (must match simple_train.py)
                cpu_memory_ratio = cpu_usage / max(memory_usage, 0.1)  # Avoid division by zero
                resource_pressure = (cpu_usage + memory_usage + disk_usage) / 3
                error_latency_product = error_rate_val * (latency_ms + 1)
                network_per_cpu = network_bytes_sec / max(cpu_usage, 0.1)
                is_critical = 1.0 if (has_oom or has_crash or has_high_cpu) else 0.0

                # Build feature vector in correct order
                features = {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage,
                    "network_bytes_sec": network_bytes_sec,
                    "error_rate": error_rate_val,
                    "latency_ms": latency_ms,
                    "restart_count": restart_count,
                    "age_minutes": age_minutes_val,
                    "cpu_memory_ratio": cpu_memory_ratio,
                    "resource_pressure": resource_pressure,
                    "error_latency_product": error_latency_product,
                    "network_per_cpu": network_per_cpu,
                    "is_critical": is_critical,
                }

                # Validate feature count matches
                if len(features) != len(FEATURE_NAMES):
                    logger.error(f"Feature count mismatch: {len(features)} vs {len(FEATURE_NAMES)}")
                    continue

                # Call ML service
                try:
                    response = requests.post(
                        f"{ml_service_url}/predict",
                        json={"features": features},
                        timeout=10
                    )

                    if response.status_code != 200:
                        logger.warning(f"ML service returned {response.status_code}")
                        continue

                    prediction = response.json()

                    predicted_issue = prediction.get('anomaly_type', 'healthy')
                    confidence = float(prediction.get('confidence', 0.5))

                    # Save prediction
                    # Note: explanation field from ML service, fallback to empty string
                    explanation = prediction.get('explanation', prediction.get('reasoning', ''))
                    cur.execute("""
                        INSERT INTO ml_predictions (
                            pod_name, namespace, timestamp,
                            predicted_issue, confidence, time_horizon_seconds,
                            top_features, explanation,
                            resource_type, resource_name, prediction_type,
                            prediction_value, model_version, features
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp, pod_name, namespace) DO NOTHING
                    """, (
                        pod_name, namespace, timestamp,
                        predicted_issue, confidence, 3600,
                        json.dumps(list(features.keys())), explanation,
                        'pod', pod_name, predicted_issue,
                        1.0 if predicted_issue != 'healthy' else 0.0,
                        prediction.get('model_used', 'ensemble'),
                        json.dumps(features)
                    ))

                    predictions_made += 1

                    if predicted_issue != 'healthy' and confidence > CONFIDENCE_THRESHOLD:
                        logger.info(f"      🔴 {namespace}/{pod_name}: {predicted_issue} ({confidence:.1%})")

                except requests.exceptions.RequestException as e:
                    logger.warning(f"ML service request failed for {pod_name}: {e}")
                    # Continue to next metric instead of failing
                    continue

                except (KeyError, ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Invalid prediction response for {pod_name}: {e}")
                    continue

            except psycopg2.Error as e:
                logger.warning(f"Database error for {pod_name}: {e}")
                conn.rollback()
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {pod_name}: {e}")
                continue

        conn.commit()
        logger.info(f"   ✅ Generated {predictions_made} predictions")
        return predictions_made

    except psycopg2.Error as e:
        logger.error(f"Database error in generate_predictions: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()


def create_issues_from_predictions(conn) -> int:
    """Create issues from anomaly predictions"""
    logger.info("📋 Creating issues from predictions...")

    cur = conn.cursor()
    issues_created = 0

    try:
        # Get recent anomaly predictions without issues
        cur.execute("""
            SELECT DISTINCT ON (mp.pod_name, mp.namespace, mp.predicted_issue)
                mp.pod_name, mp.namespace, mp.predicted_issue, mp.confidence, mp.timestamp,
                pm.cpu_utilization, pm.memory_utilization, pm.has_oom_kill,
                pm.has_crash_loop, pm.has_high_cpu, pm.has_network_issues
            FROM ml_predictions mp
            JOIN pod_metrics pm ON
                mp.pod_name = pm.pod_name
                AND mp.namespace = pm.namespace
                AND mp.timestamp = pm.timestamp
            LEFT JOIN issues i ON
                mp.pod_name = i.pod_name
                AND mp.namespace = i.namespace
                AND mp.predicted_issue = i.issue_type
                AND i.status IN ('Open', 'InProgress')
            WHERE mp.timestamp > NOW() - INTERVAL '1 hour'
                AND mp.predicted_issue != 'healthy'
                AND mp.confidence > %s
                AND i.id IS NULL
            ORDER BY mp.pod_name, mp.namespace, mp.predicted_issue, mp.timestamp DESC
            LIMIT 20
        """, (CONFIDENCE_THRESHOLD,))

        predictions = cur.fetchall()
        logger.info(f"   Found {len(predictions)} anomaly predictions")

        for pred in predictions:
            (pod_name, namespace, issue_type, confidence, timestamp,
             cpu_util, mem_util, has_oom, has_crash, has_high_cpu, has_network) = pred

            try:
                # Determine severity based on confidence and issue flags
                if confidence > 0.9 or (has_oom or has_crash):
                    severity = "critical"
                elif confidence > 0.8:
                    severity = "high"
                else:
                    severity = "medium"

                # Build description
                description_parts = [f"ML model detected {issue_type} with {confidence:.1%} confidence"]
                if cpu_util:
                    description_parts.append(f"CPU: {cpu_util:.1f}%")
                if mem_util:
                    description_parts.append(f"Memory: {mem_util:.1f}%")
                description = ". ".join(description_parts)

                issue_id = str(uuid.uuid4())

                cur.execute("""
                    INSERT INTO issues (
                        id, pod_name, namespace, issue_type, severity,
                        description, created_at, status, confidence, predicted_time_horizon
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    issue_id, pod_name, namespace, issue_type, severity,
                    description, timestamp, 'Open', confidence, 3600
                ))

                issues_created += 1
                logger.info(f"      🔴 Issue created: {namespace}/{pod_name} - {issue_type} ({severity})")

            except psycopg2.Error as e:
                logger.warning(f"Database error creating issue: {e}")
                conn.rollback()
                continue

        conn.commit()
        logger.info(f"   ✅ Created {issues_created} issues")
        return issues_created

    except psycopg2.Error as e:
        logger.error(f"Database error in create_issues_from_predictions: {e}")
        conn.rollback()
        return 0
    finally:
        cur.close()


def main():
    """Main orchestrator loop"""
    logger.info("=" * 70)
    logger.info("   AURA K8s Orchestrator")
    logger.info("   Processing: Metrics → Predictions → Issues → Remediation")
    logger.info("=" * 70)

    iteration = 0

    while True:
        iteration += 1
        logger.info(f"\n🔄 Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            conn = get_db_connection()

            # Step 1: Generate predictions from recent metrics
            predictions = generate_predictions(conn, ML_SERVICE_URL)

            # Step 2: Create issues from predictions (only if we have predictions)
            if predictions > 0:
                issues = create_issues_from_predictions(conn)
                logger.info(f"   📊 Cycle complete: {predictions} predictions, {issues} new issues")
            else:
                logger.info("   ⏭️  No new predictions to process")

            conn.close()

            # Wait before next iteration
            logger.info(f"   ⏳ Waiting {PREDICTION_INTERVAL}s before next iteration...")
            time.sleep(PREDICTION_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n⏹️  Stopping orchestrator...")
            break

        except psycopg2.OperationalError as e:
            logger.error(f"   ❌ Database connection error: {e}")
            logger.info(f"   ⏳ Retrying in {PREDICTION_INTERVAL}s...")
            time.sleep(PREDICTION_INTERVAL)

        except requests.exceptions.RequestException as e:
            logger.error(f"   ❌ Service request error: {e}")
            logger.info(f"   ⏳ Retrying in {PREDICTION_INTERVAL}s...")
            time.sleep(PREDICTION_INTERVAL)

        except Exception as e:
            logger.error(f"   ❌ Unexpected error in iteration: {e}")
            logger.info(f"   ⏳ Retrying in {PREDICTION_INTERVAL}s...")
            time.sleep(PREDICTION_INTERVAL)


if __name__ == "__main__":
    main()
