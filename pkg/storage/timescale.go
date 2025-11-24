// +build !with_pgx

package storage

import (
	"context"
	"fmt"

	"github.com/namansh70747/aura-k8s/pkg/metrics"
	"github.com/namansh70747/aura-k8s/pkg/utils"
)

// TimescaleDB provides optimized TimescaleDB operations
// Note: Full pgx/pgxpool support requires adding github.com/jackc/pgx/v5 to go.mod
// For now, this uses the existing PostgresDB with batch operations
type TimescaleDB struct {
	db *PostgresDB
}

// NewTimescaleDB creates a new TimescaleDB wrapper (uses existing PostgresDB)
// To enable full pgx COPY support, add pgx to dependencies and use build tag "with_pgx"
func NewTimescaleDB(connString string) (*TimescaleDB, error) {
	db, err := NewPostgresDB(connString)
	if err != nil {
		return nil, fmt.Errorf("failed to create TimescaleDB wrapper: %w", err)
	}

	utils.Log.Info("TimescaleDB wrapper initialized (using PostgresDB batch operations)")
	utils.Log.Warn("For optimal performance with COPY operations, add pgx/v5 dependency and use 'with_pgx' build tag")

	return &TimescaleDB{db: db}, nil
}

// BulkInsertMetrics performs batch insert using existing batch methods
func (ts *TimescaleDB) BulkInsertMetrics(ctx context.Context, metricsList []*metrics.PodMetrics) error {
	if len(metricsList) == 0 {
		return nil
	}

	// Use existing batch insert method
	return ts.db.SavePodMetricsBatch(ctx, metricsList)
}

// BatchInsertWithTx performs batch insert with transaction
func (ts *TimescaleDB) BatchInsertWithTx(ctx context.Context, metricsList []*metrics.PodMetrics, batchSize int) error {
	if len(metricsList) == 0 {
		return nil
	}

	// Use existing batch method which already handles transactions
	return ts.db.SavePodMetricsBatch(ctx, metricsList)
}

// Close closes the database connection
func (ts *TimescaleDB) Close() {
	if ts.db != nil {
		ts.db.Close()
	}
}

// Ping checks database connection health
func (ts *TimescaleDB) Ping(ctx context.Context) error {
	if ts.db == nil {
		return fmt.Errorf("database not initialized")
	}
	return ts.db.Ping(ctx)
}

// GetStats returns connection pool statistics
func (ts *TimescaleDB) GetStats() map[string]interface{} {
	if ts.db == nil {
		return map[string]interface{}{}
	}
	return ts.db.GetConnectionPoolStats()
}
