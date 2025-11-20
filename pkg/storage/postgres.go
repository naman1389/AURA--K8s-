package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/google/uuid"
	_ "github.com/lib/pq"
	"github.com/namansh70747/aura-k8s/pkg/metrics"
	"github.com/namansh70747/aura-k8s/pkg/utils"
)

type PostgresDB struct {
	db *sql.DB
}

// NewPostgresDB creates a new PostgreSQL database connection
func NewPostgresDB(connStr string) (*PostgresDB, error) {
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(100)                // Maximum number of open connections
	db.SetMaxIdleConns(25)                 // Maximum number of idle connections
	db.SetConnMaxLifetime(5 * time.Minute) // Maximum connection lifetime (5 minutes)

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	utils.Log.Info("Connected to PostgreSQL database with connection pool configured")
	return &PostgresDB{db: db}, nil
}

// InitSchema initializes the database schema with TimescaleDB optimizations
func (p *PostgresDB) InitSchema(ctx context.Context) error {
	if os.Getenv("SKIP_SCHEMA_INIT") == "true" {
		utils.Log.Info("Skipping schema initialization (SKIP_SCHEMA_INIT=true)")
		return nil
	}

	useTimescale := p.enableTimescale(ctx)
	if useTimescale {
		utils.Log.Info("Initializing database schema with TimescaleDB optimizations")
	} else {
		utils.Log.Warn("TimescaleDB not available. Falling back to plain PostgreSQL schema (dashboards will still work).")
	}

	schema := postgresSchemaSQL
	if useTimescale {
		schema = timescaleSchemaSQL
	}

	if _, err := p.db.ExecContext(ctx, schema); err != nil {
		return fmt.Errorf("failed to initialize schema: %w", err)
	}

	if _, err := p.db.ExecContext(ctx, triggersSQL); err != nil {
		return fmt.Errorf("failed to create triggers: %w", err)
	}

	if useTimescale {
		utils.Log.Info("Database schema initialized successfully with TimescaleDB features")
	} else {
		utils.Log.Info("Database schema initialized successfully in compatibility mode")
	}
	return nil
}

func (p *PostgresDB) enableTimescale(ctx context.Context) bool {
	// First check if extension files are available on the system
	var extensionAvailable bool
	checkQuery := `
	SELECT EXISTS (
		SELECT 1
		FROM pg_available_extensions
		WHERE name = 'timescaledb'
	)`

	if err := p.db.QueryRowContext(ctx, checkQuery).Scan(&extensionAvailable); err != nil {
		utils.Log.Warnf("Unable to detect TimescaleDB availability: %v", err)
		return false
	}

	if !extensionAvailable {
		utils.Log.Warn("TimescaleDB extension is not available on this PostgreSQL installation")
		return false
	}

	// Check if extension is already created
	var extensionCreated bool
	createdQuery := `
	SELECT EXISTS (
		SELECT 1
		FROM pg_extension
		WHERE extname = 'timescaledb'
	)`

	if err := p.db.QueryRowContext(ctx, createdQuery).Scan(&extensionCreated); err != nil {
		utils.Log.Warnf("Unable to check if TimescaleDB extension is created: %v", err)
		return false
	}

	if extensionCreated {
		utils.Log.Info("TimescaleDB extension is already created")
		return true
	}

	// Try to create the extension
	if _, err := p.db.ExecContext(ctx, "CREATE EXTENSION IF NOT EXISTS timescaledb"); err != nil {
		utils.Log.Warnf("Unable to enable TimescaleDB extension: %v. Falling back to PostgreSQL compatibility mode", err)
		return false
	}

	utils.Log.Info("TimescaleDB extension successfully created")
	return true
}

const timescaleSchemaSQL = `
-- Pod metrics table + hypertable
CREATE TABLE IF NOT EXISTS pod_metrics (
	id SERIAL,
	pod_name TEXT NOT NULL,
	namespace TEXT NOT NULL,
	node_name TEXT,
	container_name TEXT,
	timestamp TIMESTAMPTZ NOT NULL,
	cpu_usage_millicores DOUBLE PRECISION,
	memory_usage_bytes BIGINT,
	memory_limit_bytes BIGINT,
	cpu_limit_millicores DOUBLE PRECISION,
	cpu_utilization DOUBLE PRECISION,
	memory_utilization DOUBLE PRECISION,
	network_rx_bytes BIGINT,
	network_tx_bytes BIGINT,
	network_rx_errors BIGINT,
	network_tx_errors BIGINT,
	disk_usage_bytes BIGINT,
	disk_limit_bytes BIGINT,
	phase TEXT,
	ready BOOLEAN,
	restarts INTEGER,
	age BIGINT,
	container_ready BOOLEAN,
	container_state TEXT,
	last_state_reason TEXT,
	cpu_trend DOUBLE PRECISION,
	memory_trend DOUBLE PRECISION,
	restart_trend DOUBLE PRECISION,
	has_oom_kill BOOLEAN,
	has_crash_loop BOOLEAN,
	has_high_cpu BOOLEAN,
	has_network_issues BOOLEAN,
	PRIMARY KEY (timestamp, pod_name, namespace)
);

SELECT create_hypertable('pod_metrics', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_pod_metrics_pod ON pod_metrics(pod_name, namespace, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pod_metrics_node ON pod_metrics(node_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pod_metrics_issues ON pod_metrics(timestamp DESC) WHERE has_oom_kill OR has_crash_loop OR has_high_cpu;

-- Node metrics table + hypertable
CREATE TABLE IF NOT EXISTS node_metrics (
	id SERIAL,
	node_name TEXT NOT NULL,
	timestamp TIMESTAMPTZ NOT NULL,
	cpu_usage_millicores DOUBLE PRECISION,
	cpu_capacity_millicores DOUBLE PRECISION,
	memory_usage_bytes BIGINT,
	memory_capacity_bytes BIGINT,
	cpu_utilization DOUBLE PRECISION,
	memory_utilization DOUBLE PRECISION,
	pod_count INTEGER,
	pod_capacity INTEGER,
	disk_pressure BOOLEAN,
	memory_pressure BOOLEAN,
	network_unavailable BOOLEAN,
	ready BOOLEAN,
	PRIMARY KEY (timestamp, node_name)
);

SELECT create_hypertable('node_metrics', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_node_metrics_node ON node_metrics(node_name, timestamp DESC);

-- Issues table
CREATE TABLE IF NOT EXISTS issues (
	id TEXT PRIMARY KEY,
	pod_name TEXT NOT NULL,
	namespace TEXT NOT NULL,
	issue_type TEXT NOT NULL,
	severity TEXT NOT NULL,
	description TEXT,
	created_at TIMESTAMPTZ NOT NULL,
	resolved_at TIMESTAMPTZ,
	status TEXT NOT NULL,
	confidence DOUBLE PRECISION,
	predicted_time_horizon INTEGER,
	root_cause TEXT
);

CREATE INDEX IF NOT EXISTS idx_issues_pod ON issues(pod_name, namespace, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_issues_type ON issues(issue_type, created_at DESC);

-- Remediations table
CREATE TABLE IF NOT EXISTS remediations (
	id TEXT PRIMARY KEY,
	issue_id TEXT REFERENCES issues(id),
	pod_name TEXT NOT NULL,
	namespace TEXT NOT NULL,
	action TEXT NOT NULL,
	action_details TEXT,
	executed_at TIMESTAMPTZ NOT NULL,
	success BOOLEAN NOT NULL,
	error_message TEXT,
	ai_recommendation TEXT,
	time_to_resolve INTEGER,
	timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	completed_at TIMESTAMPTZ,
	strategy TEXT
);

CREATE INDEX IF NOT EXISTS idx_remediations_issue ON remediations(issue_id);
CREATE INDEX IF NOT EXISTS idx_remediations_pod ON remediations(pod_name, namespace, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_remediations_timestamp ON remediations(timestamp DESC);

-- ML predictions table + hypertable
CREATE TABLE IF NOT EXISTS ml_predictions (
	id SERIAL,
	pod_name TEXT NOT NULL,
	namespace TEXT NOT NULL,
	timestamp TIMESTAMPTZ NOT NULL,
	predicted_issue TEXT,
	confidence DOUBLE PRECISION,
	time_horizon_seconds INTEGER,
	xgboost_prediction TEXT,
	random_forest_prediction TEXT,
	gradient_boost_prediction TEXT,
	neural_net_prediction TEXT,
	oom_score DOUBLE PRECISION,
	crash_loop_score DOUBLE PRECISION,
	high_cpu_score DOUBLE PRECISION,
	disk_pressure_score DOUBLE PRECISION,
	network_error_score DOUBLE PRECISION,
	top_features JSONB,
	explanation TEXT,
	resource_type TEXT DEFAULT 'pod',
	resource_name TEXT,
	prediction_type TEXT,
	prediction_value DOUBLE PRECISION,
	model_version TEXT DEFAULT 'ensemble',
	features JSONB,
	is_anomaly INTEGER DEFAULT 0,
	anomaly_type TEXT,
	PRIMARY KEY (timestamp, pod_name, namespace)
);

SELECT create_hypertable('ml_predictions', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_pod ON ml_predictions(pod_name, namespace, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_issue ON ml_predictions(predicted_issue, timestamp DESC);

-- Continuous aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS pod_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
	time_bucket('1 hour', timestamp) AS bucket,
	pod_name,
	namespace,
	AVG(cpu_utilization) AS avg_cpu,
	MAX(cpu_utilization) AS max_cpu,
	AVG(memory_utilization) AS avg_memory,
	MAX(memory_utilization) AS max_memory,
	SUM(restarts) AS total_restarts
FROM pod_metrics
GROUP BY bucket, pod_name, namespace
WITH NO DATA;

SELECT add_continuous_aggregate_policy('pod_metrics_hourly',
	start_offset => INTERVAL '3 hours',
	end_offset => INTERVAL '1 hour',
	schedule_interval => INTERVAL '1 hour',
	if_not_exists => TRUE);

-- Views for Grafana
CREATE OR REPLACE VIEW predictions AS 
SELECT 
	timestamp,
	timestamp AS time,
	pod_name,
	namespace,
	COALESCE(predicted_issue, prediction_type, 'unknown') AS anomaly_type,
	CASE 
		WHEN predicted_issue IS NOT NULL AND predicted_issue != 'healthy' THEN 1
		WHEN prediction_value > 0.5 THEN 1 
		WHEN oom_score > 0.7 OR crash_loop_score > 0.7 OR high_cpu_score > 0.7 THEN 1
		WHEN disk_pressure_score > 0.7 OR network_error_score > 0.7 THEN 1
		ELSE 0 
	END AS is_anomaly,
	COALESCE(confidence, prediction_value, 0.5) AS confidence,
	COALESCE(model_version, 'ensemble') AS model_version,
	COALESCE(features, top_features, '{}'::jsonb) AS features_used,
	COALESCE(oom_score, 0) AS oom_score,
	COALESCE(crash_loop_score, 0) AS crash_loop_score,
	COALESCE(high_cpu_score, 0) AS high_cpu_score,
	COALESCE(disk_pressure_score, 0) AS disk_pressure_score,
	COALESCE(network_error_score, 0) AS network_error_score
FROM ml_predictions;

CREATE OR REPLACE VIEW metrics AS 
SELECT 
	timestamp,
	pod_name,
	namespace,
	node_name,
	cpu_utilization,
	memory_utilization,
	network_rx_bytes,
	network_tx_bytes,
	restarts AS pod_restarts,
	restarts,
	COALESCE(disk_usage_bytes, 0) AS disk_io_read,
	COALESCE(disk_usage_bytes, 0) AS disk_io_write,
	cpu_usage_millicores,
	memory_usage_bytes,
	phase,
	ready,
	has_oom_kill,
	has_crash_loop,
	has_high_cpu,
	has_network_issues
FROM pod_metrics;

-- Retention policies
SELECT add_retention_policy('pod_metrics', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_retention_policy('node_metrics', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_retention_policy('ml_predictions', INTERVAL '30 days', if_not_exists => TRUE);
`

const postgresSchemaSQL = `
-- TimescaleDB compatibility helpers
CREATE OR REPLACE FUNCTION time_bucket(bucket_width INTERVAL, ts TIMESTAMPTZ)
RETURNS TIMESTAMPTZ AS $$
DECLARE
	bucket_seconds DOUBLE PRECISION;
	epoch_seconds DOUBLE PRECISION;
	bucket_number BIGINT;
BEGIN
	-- Convert bucket width to seconds
	bucket_seconds := EXTRACT(EPOCH FROM bucket_width);
	
	-- Get timestamp as epoch seconds
	epoch_seconds := EXTRACT(EPOCH FROM ts);
	
	-- Calculate which bucket this timestamp falls into
	bucket_number := FLOOR(epoch_seconds / bucket_seconds);
	
	-- Return the start of the bucket
	RETURN TO_TIMESTAMP(bucket_number * bucket_seconds);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Pod metrics table
CREATE TABLE IF NOT EXISTS pod_metrics (
	id BIGSERIAL,
	pod_name TEXT NOT NULL,
	namespace TEXT NOT NULL,
	node_name TEXT,
	container_name TEXT,
	timestamp TIMESTAMPTZ NOT NULL,
	cpu_usage_millicores DOUBLE PRECISION,
	memory_usage_bytes BIGINT,
	memory_limit_bytes BIGINT,
	cpu_limit_millicores DOUBLE PRECISION,
	cpu_utilization DOUBLE PRECISION,
	memory_utilization DOUBLE PRECISION,
	network_rx_bytes BIGINT,
	network_tx_bytes BIGINT,
	network_rx_errors BIGINT,
	network_tx_errors BIGINT,
	disk_usage_bytes BIGINT,
	disk_limit_bytes BIGINT,
	phase TEXT,
	ready BOOLEAN,
	restarts INTEGER,
	age BIGINT,
	container_ready BOOLEAN,
	container_state TEXT,
	last_state_reason TEXT,
	cpu_trend DOUBLE PRECISION,
	memory_trend DOUBLE PRECISION,
	restart_trend DOUBLE PRECISION,
	has_oom_kill BOOLEAN,
	has_crash_loop BOOLEAN,
	has_high_cpu BOOLEAN,
	has_network_issues BOOLEAN,
	PRIMARY KEY (timestamp, pod_name, namespace)
);

CREATE INDEX IF NOT EXISTS idx_pod_metrics_timestamp ON pod_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pod_metrics_pod_namespace ON pod_metrics(pod_name, namespace);

-- Node metrics table
CREATE TABLE IF NOT EXISTS node_metrics (
	id BIGSERIAL,
	node_name TEXT NOT NULL,
	timestamp TIMESTAMPTZ NOT NULL,
	cpu_usage_millicores DOUBLE PRECISION,
	cpu_capacity_millicores DOUBLE PRECISION,
	cpu_utilization DOUBLE PRECISION,
	memory_usage_bytes BIGINT,
	memory_capacity_bytes BIGINT,
	memory_utilization DOUBLE PRECISION,
	disk_usage_bytes BIGINT,
	disk_capacity_bytes BIGINT,
	disk_utilization DOUBLE PRECISION,
	network_rx_bytes BIGINT,
	network_tx_bytes BIGINT,
	pod_count INTEGER,
	pod_capacity INTEGER,
	disk_pressure BOOLEAN,
	memory_pressure BOOLEAN,
	network_unavailable BOOLEAN,
	ready BOOLEAN,
	PRIMARY KEY (timestamp, node_name)
);

CREATE INDEX IF NOT EXISTS idx_node_metrics_timestamp ON node_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_node_metrics_node ON node_metrics(node_name, timestamp DESC);

-- ML predictions table
CREATE TABLE IF NOT EXISTS ml_predictions (
	id BIGSERIAL,
	pod_name TEXT NOT NULL,
	namespace TEXT NOT NULL,
	timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	predicted_issue TEXT,
	confidence DOUBLE PRECISION,
	time_horizon_seconds INTEGER,
	xgboost_prediction TEXT,
	random_forest_prediction TEXT,
	gradient_boost_prediction TEXT,
	neural_net_prediction TEXT,
	oom_score DOUBLE PRECISION,
	crash_loop_score DOUBLE PRECISION,
	high_cpu_score DOUBLE PRECISION,
	disk_pressure_score DOUBLE PRECISION,
	network_error_score DOUBLE PRECISION,
	top_features JSONB,
	explanation TEXT,
	resource_type TEXT DEFAULT 'pod',
	resource_name TEXT,
	prediction_type TEXT,
	prediction_value DOUBLE PRECISION,
	model_version TEXT DEFAULT 'ensemble',
	features JSONB,
	is_anomaly INTEGER DEFAULT 0,
	anomaly_type TEXT,
	PRIMARY KEY (timestamp, pod_name, namespace)
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON ml_predictions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_pod ON ml_predictions(pod_name, namespace);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_issue ON ml_predictions(predicted_issue, timestamp DESC);

-- Issues table
CREATE TABLE IF NOT EXISTS issues (
	id TEXT PRIMARY KEY,
	pod_name TEXT NOT NULL,
	namespace TEXT NOT NULL,
	issue_type TEXT NOT NULL,
	severity TEXT NOT NULL,
	description TEXT,
	created_at TIMESTAMPTZ NOT NULL,
	resolved_at TIMESTAMPTZ,
	status TEXT NOT NULL,
	confidence DOUBLE PRECISION,
	predicted_time_horizon INTEGER,
	root_cause TEXT
);

CREATE INDEX IF NOT EXISTS idx_issues_pod ON issues(pod_name, namespace, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_issues_type ON issues(issue_type, created_at DESC);

-- Remediations table
CREATE TABLE IF NOT EXISTS remediations (
	id TEXT PRIMARY KEY,
	issue_id TEXT REFERENCES issues(id),
	pod_name TEXT NOT NULL,
	namespace TEXT NOT NULL,
	action TEXT NOT NULL,
	action_details TEXT,
	executed_at TIMESTAMPTZ NOT NULL,
	success BOOLEAN NOT NULL,
	error_message TEXT,
	ai_recommendation TEXT,
	time_to_resolve INTEGER,
	timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
	completed_at TIMESTAMPTZ,
	strategy TEXT
);

CREATE INDEX IF NOT EXISTS idx_remediations_issue ON remediations(issue_id);
CREATE INDEX IF NOT EXISTS idx_remediations_pod ON remediations(pod_name, namespace, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_remediations_timestamp ON remediations(timestamp DESC);

-- Metrics + predictions views for Grafana
CREATE OR REPLACE VIEW metrics AS 
SELECT 
	timestamp,
	pod_name,
	namespace,
	node_name,
	cpu_utilization,
	memory_utilization,
	network_rx_bytes,
	network_tx_bytes,
	restarts AS pod_restarts,
	restarts,
	COALESCE(disk_usage_bytes, 0) AS disk_io_read,
	COALESCE(disk_usage_bytes, 0) AS disk_io_write,
	cpu_usage_millicores,
	memory_usage_bytes,
	phase,
	ready,
	has_oom_kill,
	has_crash_loop,
	has_high_cpu,
	has_network_issues
FROM pod_metrics;

CREATE OR REPLACE VIEW predictions AS 
SELECT 
	timestamp,
	timestamp AS time,
	pod_name,
	namespace,
	COALESCE(predicted_issue, prediction_type, 'unknown') AS anomaly_type,
	CASE 
		WHEN confidence IS NOT NULL THEN confidence
		WHEN prediction_value IS NOT NULL THEN prediction_value
		ELSE 0.0
	END AS confidence,
	is_anomaly,
	COALESCE(oom_score, 0) AS oom_score,
	COALESCE(crash_loop_score, 0) AS crash_loop_score,
	COALESCE(high_cpu_score, 0) AS high_cpu_score,
	COALESCE(disk_pressure_score, 0) AS disk_pressure_score,
	COALESCE(network_error_score, 0) AS network_error_score,
	model_version,
	explanation,
	features,
	top_features
FROM ml_predictions;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aura;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aura;
`

const triggersSQL = `
CREATE OR REPLACE FUNCTION update_ml_prediction_fields()
RETURNS TRIGGER AS $$
BEGIN
	IF NEW.anomaly_type IS NULL THEN
		NEW.anomaly_type := COALESCE(NEW.predicted_issue, NEW.prediction_type);
	END IF;

	IF NEW.is_anomaly IS NULL OR NEW.is_anomaly = 0 THEN
		NEW.is_anomaly := CASE
			WHEN NEW.predicted_issue IS NOT NULL AND NEW.predicted_issue != 'healthy' THEN 1
			WHEN NEW.prediction_value > 0.5 THEN 1
			WHEN NEW.oom_score > 0.7 OR NEW.crash_loop_score > 0.7 OR NEW.high_cpu_score > 0.7 THEN 1
			WHEN NEW.disk_pressure_score > 0.7 OR NEW.network_error_score > 0.7 THEN 1
			ELSE 0
		END;
	END IF;

	IF NEW.model_version IS NULL THEN
		NEW.model_version := 'ensemble';
	END IF;

	IF NEW.features IS NULL AND NEW.top_features IS NOT NULL THEN
		NEW.features := NEW.top_features;
	END IF;

	RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_ml_prediction_fields ON ml_predictions;
CREATE TRIGGER set_ml_prediction_fields
	BEFORE INSERT OR UPDATE ON ml_predictions
	FOR EACH ROW
	EXECUTE FUNCTION update_ml_prediction_fields();

CREATE OR REPLACE FUNCTION update_remediation_strategy()
RETURNS TRIGGER AS $$
BEGIN
	NEW.strategy := CASE
		WHEN NEW.action LIKE '%scale_down%' OR NEW.action LIKE '%scale%' THEN 'scale_down'
		WHEN NEW.action LIKE '%replica%' OR NEW.action LIKE '%reduce%' THEN 'reduce_replicas'
		WHEN NEW.action LIKE '%optimize%' OR NEW.action LIKE '%rightsize%' THEN 'optimize_resources'
		WHEN NEW.action LIKE '%restart%' THEN 'restart_pod'
		WHEN NEW.action LIKE '%evict%' THEN 'evict_pod'
		ELSE 'other'
	END;
	IF NEW.timestamp IS NULL THEN
		NEW.timestamp := NEW.executed_at;
	END IF;
	IF NEW.completed_at IS NULL AND NEW.success = true AND NEW.time_to_resolve IS NOT NULL THEN
		NEW.completed_at := NEW.executed_at + (NEW.time_to_resolve::text || ' seconds')::INTERVAL;
	ELSIF NEW.completed_at IS NULL AND NEW.success = true THEN
		NEW.completed_at := NEW.executed_at;
	END IF;
	RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_remediation_strategy ON remediations;
CREATE TRIGGER set_remediation_strategy
	BEFORE INSERT OR UPDATE ON remediations
	FOR EACH ROW
	EXECUTE FUNCTION update_remediation_strategy();
`

// SavePodMetrics saves pod metrics to the database
func (p *PostgresDB) SavePodMetrics(ctx context.Context, m *metrics.PodMetrics) error {
	query := `
	INSERT INTO pod_metrics (
		pod_name, namespace, node_name, container_name, timestamp,
		cpu_usage_millicores, memory_usage_bytes, memory_limit_bytes, cpu_limit_millicores,
		cpu_utilization, memory_utilization,
		network_rx_bytes, network_tx_bytes, network_rx_errors, network_tx_errors,
		disk_usage_bytes, disk_limit_bytes,
		phase, ready, restarts, age,
		container_ready, container_state, last_state_reason,
		cpu_trend, memory_trend, restart_trend,
		has_oom_kill, has_crash_loop, has_high_cpu, has_network_issues
	) VALUES (
		$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17,
		$18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31
	) ON CONFLICT (timestamp, pod_name, namespace) DO NOTHING
	`

	_, err := p.db.ExecContext(ctx, query,
		m.PodName, m.Namespace, m.NodeName, m.ContainerName, m.Timestamp,
		m.CPUUsageMillicores, m.MemoryUsageBytes, m.MemoryLimitBytes, m.CPULimitMillicores,
		m.CPUUtilization, m.MemoryUtilization,
		m.NetworkRxBytes, m.NetworkTxBytes, m.NetworkRxErrors, m.NetworkTxErrors,
		m.DiskUsageBytes, m.DiskLimitBytes,
		m.Phase, m.Ready, m.Restarts, m.Age,
		m.ContainerReady, m.ContainerState, m.LastStateReason,
		m.CPUTrend, m.MemoryTrend, m.RestartTrend,
		m.HasOOMKill, m.HasCrashLoop, m.HasHighCPU, m.HasNetworkIssues,
	)

	return err
}

// SaveNodeMetrics saves node metrics to the database
func (p *PostgresDB) SaveNodeMetrics(ctx context.Context, m *metrics.NodeMetrics) error {
	query := `
	INSERT INTO node_metrics (
		node_name, timestamp,
		cpu_usage_millicores, cpu_capacity_millicores,
		memory_usage_bytes, memory_capacity_bytes,
		cpu_utilization, memory_utilization,
		pod_count, pod_capacity,
		disk_pressure, memory_pressure, network_unavailable, ready
	) VALUES (
		$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
	) ON CONFLICT (timestamp, node_name) DO NOTHING
	`

	_, err := p.db.ExecContext(ctx, query,
		m.NodeName, m.Timestamp,
		m.CPUUsageMillicores, m.CPUCapacityMillicores,
		m.MemoryUsageBytes, m.MemoryCapacityBytes,
		m.CPUUtilization, m.MemoryUtilization,
		m.PodCount, m.PodCapacity,
		m.DiskPressure, m.MemoryPressure, m.NetworkUnavailable, m.Ready,
	)

	return err
}

// GetRecentPodMetrics returns recent metrics for a pod
func (p *PostgresDB) GetRecentPodMetrics(ctx context.Context, podName, namespace string, limit int) ([]*metrics.PodMetrics, error) {
	query := `
	SELECT pod_name, namespace, node_name, container_name, timestamp,
		cpu_usage_millicores, memory_usage_bytes, memory_limit_bytes, cpu_limit_millicores,
		cpu_utilization, memory_utilization,
		network_rx_bytes, network_tx_bytes, network_rx_errors, network_tx_errors,
		disk_usage_bytes, disk_limit_bytes,
		phase, ready, restarts, age,
		container_ready, container_state, last_state_reason,
		cpu_trend, memory_trend, restart_trend,
		has_oom_kill, has_crash_loop, has_high_cpu, has_network_issues
	FROM pod_metrics
	WHERE pod_name = $1 AND namespace = $2
	ORDER BY timestamp DESC
	LIMIT $3
	`

	rows, err := p.db.QueryContext(ctx, query, podName, namespace, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []*metrics.PodMetrics
	for rows.Next() {
		m := &metrics.PodMetrics{}
		err := rows.Scan(
			&m.PodName, &m.Namespace, &m.NodeName, &m.ContainerName, &m.Timestamp,
			&m.CPUUsageMillicores, &m.MemoryUsageBytes, &m.MemoryLimitBytes, &m.CPULimitMillicores,
			&m.CPUUtilization, &m.MemoryUtilization,
			&m.NetworkRxBytes, &m.NetworkTxBytes, &m.NetworkRxErrors, &m.NetworkTxErrors,
			&m.DiskUsageBytes, &m.DiskLimitBytes,
			&m.Phase, &m.Ready, &m.Restarts, &m.Age,
			&m.ContainerReady, &m.ContainerState, &m.LastStateReason,
			&m.CPUTrend, &m.MemoryTrend, &m.RestartTrend,
			&m.HasOOMKill, &m.HasCrashLoop, &m.HasHighCPU, &m.HasNetworkIssues,
		)
		if err != nil {
			return nil, err
		}
		results = append(results, m)
	}

	return results, rows.Err()
}

// SaveIssue saves an issue to the database
func (p *PostgresDB) SaveIssue(ctx context.Context, issue *metrics.Issue) error {
	if issue.ID == "" {
		issue.ID = uuid.New().String()
	}

	query := `
	INSERT INTO issues (
		id, pod_name, namespace, issue_type, severity, description,
		created_at, resolved_at, status, confidence, predicted_time_horizon, root_cause
	) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
	ON CONFLICT (id) DO UPDATE SET
		severity = EXCLUDED.severity,
		description = EXCLUDED.description,
		resolved_at = EXCLUDED.resolved_at,
		status = EXCLUDED.status,
		confidence = EXCLUDED.confidence,
		predicted_time_horizon = EXCLUDED.predicted_time_horizon,
		root_cause = EXCLUDED.root_cause
	`

	_, err := p.db.ExecContext(ctx, query,
		issue.ID, issue.PodName, issue.Namespace, issue.IssueType, issue.Severity,
		issue.Description, issue.CreatedAt, issue.ResolvedAt, issue.Status,
		issue.Confidence, issue.PredictedTimeHorizon, issue.RootCause,
	)

	return err
}

// GetOpenIssues returns all open issues
func (p *PostgresDB) GetOpenIssues(ctx context.Context) ([]*metrics.Issue, error) {
	query := `
	SELECT id, pod_name, namespace, issue_type, severity, description,
		created_at, resolved_at, status, confidence, predicted_time_horizon, root_cause
	FROM issues
	WHERE status IN ('Open', 'open', 'InProgress')
	ORDER BY created_at DESC
	`

	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []*metrics.Issue
	for rows.Next() {
		issue := &metrics.Issue{}
		err := rows.Scan(
			&issue.ID, &issue.PodName, &issue.Namespace, &issue.IssueType,
			&issue.Severity, &issue.Description, &issue.CreatedAt, &issue.ResolvedAt,
			&issue.Status, &issue.Confidence, &issue.PredictedTimeHorizon, &issue.RootCause,
		)
		if err != nil {
			return nil, err
		}
		results = append(results, issue)
	}

	return results, rows.Err()
}

// SaveRemediation saves a remediation action to the database
func (p *PostgresDB) SaveRemediation(ctx context.Context, r *metrics.Remediation) error {
	if r.ID == "" {
		r.ID = uuid.New().String()
	}

	if r.Strategy == "" {
		r.Strategy = r.Action
	}

	if r.Timestamp.IsZero() {
		r.Timestamp = r.ExecutedAt
	}

	var completedAt interface{}
	if r.CompletedAt != nil {
		completedAt = r.CompletedAt
	} else if r.TimeToResolve > 0 {
		t := r.ExecutedAt.Add(time.Duration(r.TimeToResolve) * time.Second)
		completedAt = t
	} else if r.Success {
		t := r.ExecutedAt
		completedAt = t
	} else {
		completedAt = nil
	}

	query := `
	INSERT INTO remediations (
		id, issue_id, pod_name, namespace, action, action_details,
		executed_at, success, error_message, ai_recommendation, time_to_resolve,
		timestamp, completed_at, strategy
	) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
	`

	_, err := p.db.ExecContext(ctx, query,
		r.ID, r.IssueID, r.PodName, r.Namespace, r.Action, r.ActionDetails,
		r.ExecutedAt, r.Success, r.ErrorMessage, r.AIRecommendation, r.TimeToResolve,
		r.Timestamp, completedAt, r.Strategy,
	)

	return err
}

// SaveMLPrediction saves an ML prediction to the database
func (p *PostgresDB) SaveMLPrediction(ctx context.Context, pred *metrics.MLPrediction) error {
	topFeaturesJSON, err := json.Marshal(pred.TopFeatures)
	if err != nil {
		return fmt.Errorf("failed to marshal top features: %w", err)
	}

	featuresPayload := map[string]interface{}{
		"pod_name":  pred.PodName,
		"namespace": pred.Namespace,
		"issue":     pred.PredictedIssue,
	}
	featuresJSON, err := json.Marshal(featuresPayload)
	if err != nil {
		return fmt.Errorf("failed to marshal features payload: %w", err)
	}

	// Determine if anomaly and type
	isAnomaly := 0
	anomalyType := pred.PredictedIssue
	predictionValue := 0.0
	if pred.PredictedIssue != "" && pred.PredictedIssue != "healthy" {
		isAnomaly = 1
		predictionValue = 1.0
	}

	query := `
	INSERT INTO ml_predictions (
		pod_name, namespace, timestamp, predicted_issue, confidence, time_horizon_seconds,
		xgboost_prediction, random_forest_prediction, gradient_boost_prediction, neural_net_prediction,
		oom_score, crash_loop_score, high_cpu_score, disk_pressure_score, network_error_score,
		top_features, explanation,
		resource_type, resource_name, prediction_type, prediction_value, model_version, features,
		is_anomaly, anomaly_type
	) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
	ON CONFLICT (timestamp, pod_name, namespace) DO NOTHING
	`

	_, execErr := p.db.ExecContext(ctx, query,
		pred.PodName, pred.Namespace, pred.Timestamp, pred.PredictedIssue, pred.Confidence, pred.TimeHorizonSeconds,
		pred.XGBoostPrediction, pred.RandomForestPred, pred.GradientBoostPred, pred.NeuralNetPrediction,
		pred.OOMScore, pred.CrashLoopScore, pred.HighCPUScore, pred.DiskPressureScore, pred.NetworkErrorScore,
		topFeaturesJSON, pred.Explanation,
		"pod", pred.PodName, pred.PredictedIssue, predictionValue, "ensemble", featuresJSON,
		isAnomaly, anomalyType,
	)

	return execErr
}

// Close closes the database connection
func (p *PostgresDB) Close() error {
	return p.db.Close()
}

// ExecRaw executes a raw SQL query
func (p *PostgresDB) ExecRaw(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	return p.db.ExecContext(ctx, query, args...)
}
