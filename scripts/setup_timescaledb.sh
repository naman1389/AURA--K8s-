#!/bin/bash

# TimescaleDB Setup and Optimization Script
# This script sets up TimescaleDB with hypertables, compression, and retention policies

set -e

echo "🚀 Setting up TimescaleDB with optimizations..."

# Get database connection details
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-aura}"
DB_NAME="${DB_NAME:-aura_metrics}"

# Connect to database and set up TimescaleDB
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert existing tables to hypertables if not already converted
DO \$\$
BEGIN
    -- Convert pod_metrics to hypertable
    IF NOT EXISTS (
        SELECT 1 FROM _timescaledb_catalog.hypertable 
        WHERE table_name = 'pod_metrics'
    ) THEN
        PERFORM create_hypertable(
            'pod_metrics',
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
        RAISE NOTICE 'Created hypertable for pod_metrics';
    ELSE
        RAISE NOTICE 'Hypertable for pod_metrics already exists';
    END IF;

    -- Convert node_metrics to hypertable
    IF NOT EXISTS (
        SELECT 1 FROM _timescaledb_catalog.hypertable 
        WHERE table_name = 'node_metrics'
    ) THEN
        PERFORM create_hypertable(
            'node_metrics',
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
        RAISE NOTICE 'Created hypertable for node_metrics';
    ELSE
        RAISE NOTICE 'Hypertable for node_metrics already exists';
    END IF;

    -- Convert ml_predictions to hypertable
    IF NOT EXISTS (
        SELECT 1 FROM _timescaledb_catalog.hypertable 
        WHERE table_name = 'ml_predictions'
    ) THEN
        PERFORM create_hypertable(
            'ml_predictions',
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );
        RAISE NOTICE 'Created hypertable for ml_predictions';
    ELSE
        RAISE NOTICE 'Hypertable for ml_predictions already exists';
    END IF;

    -- Convert remediations to hypertable
    IF NOT EXISTS (
        SELECT 1 FROM _timescaledb_catalog.hypertable 
        WHERE table_name = 'remediations'
    ) THEN
        PERFORM create_hypertable(
            'remediations',
            'executed_at',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
        RAISE NOTICE 'Created hypertable for remediations';
    ELSE
        RAISE NOTICE 'Hypertable for remediations already exists';
    END IF;
END
\$\$;

-- Enable compression on hypertables
DO \$\$
BEGIN
    -- Enable compression on pod_metrics (compress data older than 7 days)
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs 
        WHERE proc_name = 'policy_compression'
        AND hypertable_name = 'pod_metrics'
    ) THEN
        ALTER TABLE pod_metrics SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'pod_name, namespace',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
        
        SELECT add_compression_policy('pod_metrics', INTERVAL '7 days');
        RAISE NOTICE 'Enabled compression for pod_metrics';
    ELSE
        RAISE NOTICE 'Compression for pod_metrics already enabled';
    END IF;

    -- Enable compression on node_metrics
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs 
        WHERE proc_name = 'policy_compression'
        AND hypertable_name = 'node_metrics'
    ) THEN
        ALTER TABLE node_metrics SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'node_name',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
        
        SELECT add_compression_policy('node_metrics', INTERVAL '7 days');
        RAISE NOTICE 'Enabled compression for node_metrics';
    ELSE
        RAISE NOTICE 'Compression for node_metrics already enabled';
    END IF;

    -- Enable compression on ml_predictions
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs 
        WHERE proc_name = 'policy_compression'
        AND hypertable_name = 'ml_predictions'
    ) THEN
        ALTER TABLE ml_predictions SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'pod_name, namespace',
            timescaledb.compress_orderby = 'timestamp DESC'
        );
        
        SELECT add_compression_policy('ml_predictions', INTERVAL '7 days');
        RAISE NOTICE 'Enabled compression for ml_predictions';
    ELSE
        RAISE NOTICE 'Compression for ml_predictions already enabled';
    END IF;
END
\$\$;

-- Set up retention policies (keep data for 90 days)
DO \$\$
BEGIN
    -- Retention policy for pod_metrics
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs 
        WHERE proc_name = 'policy_retention'
        AND hypertable_name = 'pod_metrics'
    ) THEN
        SELECT add_retention_policy('pod_metrics', INTERVAL '90 days');
        RAISE NOTICE 'Added retention policy for pod_metrics (90 days)';
    ELSE
        RAISE NOTICE 'Retention policy for pod_metrics already exists';
    END IF;

    -- Retention policy for node_metrics
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs 
        WHERE proc_name = 'policy_retention'
        AND hypertable_name = 'node_metrics'
    ) THEN
        SELECT add_retention_policy('node_metrics', INTERVAL '90 days');
        RAISE NOTICE 'Added retention policy for node_metrics (90 days)';
    ELSE
        RAISE NOTICE 'Retention policy for node_metrics already exists';
    END IF;

    -- Retention policy for ml_predictions
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs 
        WHERE proc_name = 'policy_retention'
        AND hypertable_name = 'ml_predictions'
    ) THEN
        SELECT add_retention_policy('ml_predictions', INTERVAL '90 days');
        RAISE NOTICE 'Added retention policy for ml_predictions (90 days)';
    ELSE
        RAISE NOTICE 'Retention policy for ml_predictions already exists';
    END IF;
END
\$\$;

-- Create continuous aggregates for fast queries
DO \$\$
BEGIN
    -- Hourly aggregation for pod_metrics
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.continuous_aggregates 
        WHERE view_name = 'pod_metrics_hourly'
    ) THEN
        CREATE MATERIALIZED VIEW pod_metrics_hourly
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', timestamp) AS hour,
            pod_name,
            namespace,
            AVG(cpu_utilization) AS avg_cpu,
            MAX(cpu_utilization) AS max_cpu,
            AVG(memory_utilization) AS avg_memory,
            MAX(memory_utilization) AS max_memory,
            AVG(cpu_usage_millicores) AS avg_cpu_usage,
            AVG(memory_usage_bytes) AS avg_memory_usage,
            COUNT(*) AS metric_count,
            COUNT(*) FILTER (WHERE is_anomaly = true) AS anomaly_count
        FROM pod_metrics
        GROUP BY hour, pod_name, namespace;

        -- Add refresh policy
        SELECT add_continuous_aggregate_policy('pod_metrics_hourly',
            start_offset => INTERVAL '3 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
        
        RAISE NOTICE 'Created continuous aggregate pod_metrics_hourly';
    ELSE
        RAISE NOTICE 'Continuous aggregate pod_metrics_hourly already exists';
    END IF;
END
\$\$;

-- Optimize indexes for performance
CREATE INDEX IF NOT EXISTS idx_pod_metrics_pod_time_optimized 
    ON pod_metrics (pod_name, namespace, timestamp DESC) 
    WHERE namespace NOT IN ('kube-system', 'kube-public', 'kube-node-lease', 'local-path-storage', 'default');

CREATE INDEX IF NOT EXISTS idx_pod_metrics_anomaly_time 
    ON pod_metrics (timestamp DESC) 
    WHERE is_anomaly = true;

-- Analyze tables for query optimization
ANALYZE pod_metrics;
ANALYZE node_metrics;
ANALYZE ml_predictions;
ANALYZE remediations;

RAISE NOTICE 'TimescaleDB setup completed successfully!';
EOF

echo "✅ TimescaleDB setup completed!"
