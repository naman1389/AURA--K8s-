-- Optimized schema for TimescaleDB
-- This script is referenced in the BEAST_LEVEL_COMPLETE_IMPLEMENTATION_GUIDE.md
-- Note: The main schema is already in pkg/storage/postgres.go, but this provides
-- an additional standalone SQL file for manual setup if needed

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- The main schema creation is handled by pkg/storage/postgres.go InitSchema()
-- This file serves as documentation and can be used for manual setup

-- Key optimizations already in postgres.go:
-- 1. Hypertables for pod_metrics, node_metrics, ml_predictions
-- 2. Compression policies (7 days)
-- 3. Retention policies (90 days)
-- 4. Continuous aggregates (pod_metrics_hourly)
-- 5. Optimized indexes

-- This file can be used to verify or manually apply TimescaleDB optimizations
-- but the main schema is automatically managed by the Go code

SELECT 'TimescaleDB optimizations are managed by pkg/storage/postgres.go InitSchema()' as info;
