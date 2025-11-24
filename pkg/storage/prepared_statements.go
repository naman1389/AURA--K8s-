package storage

import (
	"context"
	"database/sql"
	"sync"
	"time"

	"github.com/namansh70747/aura-k8s/pkg/utils"
)

// PreparedStatements holds prepared statements for common queries
type PreparedStatements struct {
	insertPodMetrics   *sql.Stmt
	insertNodeMetrics  *sql.Stmt
	getRecentMetrics   *sql.Stmt
	getPodMetrics      *sql.Stmt
	mu                 sync.RWMutex
	initialized        bool
}

var (
	preparedStmts *PreparedStatements
	stmtOnce      sync.Once
)

// InitPreparedStatements initializes prepared statements for better performance
func InitPreparedStatements(ctx context.Context, db *sql.DB) error {
	var initErr error
	stmtOnce.Do(func() {
		preparedStmts = &PreparedStatements{}

		// Prepare insert pod metrics statement
		preparedStmts.insertPodMetrics, initErr = db.PrepareContext(ctx, `
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
		`)
		if initErr != nil {
			return
		}

		// Prepare insert node metrics statement
		preparedStmts.insertNodeMetrics, initErr = db.PrepareContext(ctx, `
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
		`)
		if initErr != nil {
			return
		}

		// Prepare get recent metrics statement
		preparedStmts.getRecentMetrics, initErr = db.PrepareContext(ctx, `
			SELECT 
				pod_name, namespace, node_name, container_name, timestamp,
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
		`)
		if initErr != nil {
			return
		}

		// Prepare get pod metrics statement
		preparedStmts.getPodMetrics, initErr = db.PrepareContext(ctx, `
			SELECT 
				pod_name, namespace, node_name, container_name, timestamp,
				cpu_usage_millicores, memory_usage_bytes, memory_limit_bytes, cpu_limit_millicores,
				cpu_utilization, memory_utilization,
				network_rx_bytes, network_tx_bytes, network_rx_errors, network_tx_errors,
				disk_usage_bytes, disk_limit_bytes,
				phase, ready, restarts, age,
				container_ready, container_state, last_state_reason,
				cpu_trend, memory_trend, restart_trend,
				has_oom_kill, has_crash_loop, has_high_cpu, has_network_issues
			FROM pod_metrics
			WHERE pod_name = $1 AND namespace = $2 AND timestamp >= $3
			ORDER BY timestamp DESC
		`)
		if initErr != nil {
			return
		}

		preparedStmts.initialized = true
		utils.Log.Info("Prepared statements initialized successfully")
	})

	return initErr
}

// GetPreparedStatements returns the prepared statements instance
func GetPreparedStatements() *PreparedStatements {
	return preparedStmts
}

// ClosePreparedStatements closes all prepared statements
func ClosePreparedStatements() {
	if preparedStmts == nil {
		return
	}

	preparedStmts.mu.Lock()
	defer preparedStmts.mu.Unlock()

	if preparedStmts.insertPodMetrics != nil {
		preparedStmts.insertPodMetrics.Close()
	}
	if preparedStmts.insertNodeMetrics != nil {
		preparedStmts.insertNodeMetrics.Close()
	}
	if preparedStmts.getRecentMetrics != nil {
		preparedStmts.getRecentMetrics.Close()
	}
	if preparedStmts.getPodMetrics != nil {
		preparedStmts.getPodMetrics.Close()
	}

	preparedStmts.initialized = false
	utils.Log.Info("Prepared statements closed")
}

// IsInitialized returns true if prepared statements are initialized
func (ps *PreparedStatements) IsInitialized() bool {
	if ps == nil {
		return false
	}
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	return ps.initialized
}

// RetryPrepareStatement retries preparing a statement with exponential backoff
func RetryPrepareStatement(ctx context.Context, db *sql.DB, query string, maxRetries int) (*sql.Stmt, error) {
	var stmt *sql.Stmt
	var err error

	for attempt := 0; attempt < maxRetries; attempt++ {
		stmt, err = db.PrepareContext(ctx, query)
		if err == nil {
			return stmt, nil
		}

		if attempt < maxRetries-1 {
			backoff := time.Duration(attempt+1) * 100 * time.Millisecond
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
				// Retry
			}
		}
	}

	return nil, err
}
