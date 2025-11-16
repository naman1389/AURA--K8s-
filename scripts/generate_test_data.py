#!/usr/bin/env python3
"""
Realistic Test Data Generator for AURA K8s
Generates workload patterns that simulate real Kubernetes applications
"""

import os
import time
import random
import logging
import psycopg2
from datetime import datetime, timedelta
from psycopg2.extras import execute_values
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://aura:aura_password@timescaledb:5432/aura_metrics")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Set seed for reproducibility
random.seed(RANDOM_SEED)

# Realistic pod definitions
PODS = [
    {"name": "web-frontend", "base_cpu": 300, "base_memory": 256, "cpu_variance": 150, "mem_variance": 128},
    {"name": "api-backend", "base_cpu": 500, "base_memory": 512, "cpu_variance": 200, "mem_variance": 200},
    {"name": "database", "base_cpu": 1000, "base_memory": 2048, "cpu_variance": 300, "mem_variance": 512},
    {"name": "cache-redis", "base_cpu": 100, "base_memory": 256, "cpu_variance": 50, "mem_variance": 100},
    {"name": "message-queue", "base_cpu": 200, "base_memory": 512, "cpu_variance": 100, "mem_variance": 200},
    {"name": "worker-1", "base_cpu": 800, "base_memory": 1024, "cpu_variance": 300, "mem_variance": 400},
    {"name": "monitoring", "base_cpu": 150, "base_memory": 256, "cpu_variance": 75, "mem_variance": 128},
]

NAMESPACES = ["production", "staging", "development"]

# Workload patterns (time-of-day effects)
def get_load_multiplier():
    """
    Simulate time-of-day traffic patterns
    Business hours (9-17): high load
    Night time (0-6): low load
    """
    hour = datetime.now().hour
    if 9 <= hour <= 17:
        return 1.0 + random.uniform(0, 0.3)  # Peak hours: 100-130% load
    elif 6 <= hour < 9:
        return 0.5 + random.uniform(0, 0.2)  # Morning ramp: 50-70%
    elif 17 <= hour < 21:
        return 0.8 + random.uniform(0, 0.2)  # Evening: 80-100%
    else:
        return 0.2 + random.uniform(0, 0.1)  # Night: 20-30%


def simulate_realistic_pod_metrics(pod_def: dict, namespace: str, timestamp: datetime):
    """
    Generate realistic pod metrics based on workload patterns
    """
    # Base load
    load_multiplier = get_load_multiplier()

    # Calculate CPU and Memory
    cpu_millicores = pod_def["base_cpu"] * load_multiplier + random.gauss(0, pod_def["cpu_variance"])
    memory_usage = pod_def["base_memory"] * load_multiplier + random.gauss(0, pod_def["mem_variance"])

    # Ensure values are positive
    cpu_millicores = max(0, cpu_millicores)
    memory_usage = max(0, memory_usage)

    # Limit to reasonable max (containers have limits)
    cpu_limit = pod_def["base_cpu"] * 5
    memory_limit = pod_def["base_memory"] * 4

    cpu_utilization = (cpu_millicores / cpu_limit) * 100
    memory_utilization = (memory_usage / memory_limit) * 100

    # Network traffic (correlated with CPU)
    network_rx_bytes = cpu_millicores * 1000 + random.uniform(100000, 500000)
    network_tx_bytes = cpu_millicores * 800 + random.uniform(50000, 300000)

    # Error rate (low in normal conditions, spikes during issues)
    base_error_rate = 0.01  # 1% normal
    if random.random() < 0.05:  # 5% chance of spike
        error_rate = random.uniform(0.05, 0.20)
    else:
        error_rate = base_error_rate

    network_rx_errors = int(max(0, random.gauss(error_rate * 100, 5)))
    network_tx_errors = int(max(0, random.gauss(error_rate * 100, 5)))

    # Pod restart behavior (usually 0, sometimes 1-2 on issues)
    if random.random() < 0.02:  # 2% chance of restart
        restarts = random.randint(1, 3)
    else:
        restarts = 0

    # Simulate issues
    has_oom_kill = False
    has_crash_loop = False
    has_high_cpu = cpu_utilization > 80
    has_network_issues = network_rx_errors + network_tx_errors > 50

    # OOM kill simulation
    if memory_utilization > 90 and random.random() < 0.1:  # 10% chance if memory high
        has_oom_kill = True
        memory_utilization = 95
        restarts = max(restarts, 1)

    # Crash loop simulation
    if restarts > 2 and random.random() < 0.15:  # Likely after multiple restarts
        has_crash_loop = True

    # Pod readiness
    ready = not (has_oom_kill or has_crash_loop) and restarts <= 2

    disk_usage_bytes = random.uniform(100 * 1024 * 1024, 500 * 1024 * 1024)  # 100-500 MB
    disk_limit_bytes = 10 * 1024 * 1024 * 1024  # 10 GB

    # Trends (rate of change)
    cpu_trend = random.uniform(-5, 10)  # -5 to +10 percentage points per minute
    memory_trend = random.uniform(-3, 5)
    restart_trend = random.uniform(0, 2)

    return {
        "cpu_usage_millicores": cpu_millicores,
        "cpu_limit_millicores": cpu_limit,
        "cpu_utilization": cpu_utilization,
        "memory_usage_bytes": int(memory_usage * 1024 * 1024),
        "memory_limit_bytes": int(memory_limit * 1024 * 1024),
        "memory_utilization": memory_utilization,
        "disk_usage_bytes": int(disk_usage_bytes),
        "disk_limit_bytes": int(disk_limit_bytes),
        "network_rx_bytes": int(network_rx_bytes),
        "network_tx_bytes": int(network_tx_bytes),
        "network_rx_errors": network_rx_errors,
        "network_tx_errors": network_tx_errors,
        "restarts": restarts,
        "ready": ready,
        "phase": "Running" if ready else "Pending",
        "container_ready": ready,
        "container_state": "running" if ready else "waiting",
        "has_oom_kill": has_oom_kill,
        "has_crash_loop": has_crash_loop,
        "has_high_cpu": has_high_cpu,
        "has_network_issues": has_network_issues,
        "cpu_trend": cpu_trend,
        "memory_trend": memory_trend,
        "restart_trend": restart_trend,
    }


def insert_metrics(conn, metrics_batch):
    """Insert metrics into database"""
    query = """
    INSERT INTO pod_metrics (
        pod_name, namespace, node_name, container_name, timestamp,
        cpu_usage_millicores, memory_usage_bytes, memory_limit_bytes, cpu_limit_millicores,
        cpu_utilization, memory_utilization, network_rx_bytes, network_tx_bytes,
        network_rx_errors, network_tx_errors, disk_usage_bytes, disk_limit_bytes,
        phase, ready, restarts, age, container_ready, container_state,
        has_oom_kill, has_crash_loop, has_high_cpu, has_network_issues,
        cpu_trend, memory_trend, restart_trend
    ) VALUES %s
    ON CONFLICT (timestamp, pod_name, namespace) DO NOTHING
    """

    values = [
        (
            m["pod_name"], m["namespace"], m["node_name"], m["container_name"], m["timestamp"],
            m["metrics"]["cpu_usage_millicores"], m["metrics"]["memory_usage_bytes"],
            m["metrics"]["memory_limit_bytes"], m["metrics"]["cpu_limit_millicores"],
            m["metrics"]["cpu_utilization"], m["metrics"]["memory_utilization"],
            m["metrics"]["network_rx_bytes"], m["metrics"]["network_tx_bytes"],
            m["metrics"]["network_rx_errors"], m["metrics"]["network_tx_errors"],
            m["metrics"]["disk_usage_bytes"], m["metrics"]["disk_limit_bytes"],
            m["metrics"]["phase"], m["metrics"]["ready"], m["metrics"]["restarts"],
            3600, m["metrics"]["container_ready"], m["metrics"]["container_state"],
            m["metrics"]["has_oom_kill"], m["metrics"]["has_crash_loop"],
            m["metrics"]["has_high_cpu"], m["metrics"]["has_network_issues"],
            m["metrics"]["cpu_trend"], m["metrics"]["memory_trend"], m["metrics"]["restart_trend"],
        )
        for m in metrics_batch
    ]

    with conn.cursor() as cur:
        execute_values(cur, query, values, page_size=100)
        conn.commit()


def generate_and_store_data():
    """Main data generation loop"""
    logger.info("🚀 Starting realistic data generation...")
    logger.info(f"   Using seed: {RANDOM_SEED} (for reproducibility)")

    try:
        conn = psycopg2.connect(DATABASE_URL)
        logger.info("✅ Connected to database")
    except psycopg2.OperationalError as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        return

    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now()

            logger.info(f"\n📊 Iteration {iteration} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

            metrics_batch = []

            for namespace in NAMESPACES:
                for pod_def in PODS:
                    # Create realistic pod name (with hash suffix)
                    pod_hash = random.choice(["abc", "def", "ghi", "jkl", "mno"])
                    pod_id = random.randint(1000, 9999)
                    full_pod_name = f"{pod_def['name']}-{pod_id}-{pod_hash}"

                    metrics = simulate_realistic_pod_metrics(pod_def, namespace, timestamp)

                    metrics_batch.append({
                        "pod_name": full_pod_name,
                        "namespace": namespace,
                        "node_name": f"node-{random.randint(1, 5)}",
                        "container_name": pod_def["name"],
                        "timestamp": timestamp,
                        "metrics": metrics,
                    })

            # Insert into database
            try:
                insert_metrics(conn, metrics_batch)

                # Count anomalies
                anomalies = [m for m in metrics_batch if (
                    m["metrics"]["has_oom_kill"] or
                    m["metrics"]["has_crash_loop"] or
                    m["metrics"]["has_high_cpu"] or
                    m["metrics"]["has_network_issues"]
                )]

                logger.info(f"   ✅ Stored {len(metrics_batch)} metrics ({len(anomalies)} with issues)")

                # Show anomalies
                if anomalies:
                    for anom in anomalies[:3]:
                        issues = []
                        if anom["metrics"]["has_oom_kill"]:
                            issues.append("OOM")
                        if anom["metrics"]["has_crash_loop"]:
                            issues.append("CrashLoop")
                        if anom["metrics"]["has_high_cpu"]:
                            issues.append("HighCPU")
                        if anom["metrics"]["has_network_issues"]:
                            issues.append("NetErrors")

                        logger.info(f"      🔴 {anom['namespace']}/{anom['pod_name']}: {', '.join(issues)}")

            except psycopg2.Error as e:
                logger.error(f"Database error: {e}")
                conn.rollback()
                # Reconnect
                try:
                    conn.close()
                except:
                    pass
                try:
                    conn = psycopg2.connect(DATABASE_URL)
                except psycopg2.OperationalError:
                    logger.warning("Failed to reconnect to database, retrying...")
                    time.sleep(5)
                    continue

            # Sleep before next iteration
            logger.info(f"   ⏳ Sleeping 10s before next iteration...")
            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping data generation...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        try:
            conn.close()
        except:
            pass


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("   AURA K8s Realistic Data Generator")
    logger.info("   Generates workload patterns matching real Kubernetes apps")
    logger.info("=" * 70)

    generate_and_store_data()
