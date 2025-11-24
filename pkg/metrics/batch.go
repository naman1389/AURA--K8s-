package metrics

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/namansh70747/aura-k8s/pkg/utils"
)

// MetricBatch represents a batch of metrics to be processed
type MetricBatch struct {
	Metrics   []*PodMetrics
	Timestamp time.Time
	BatchID   string
	Size      int
}

// BatchProcessor handles batch processing of metrics
type BatchProcessor struct {
	batchSize     int
	flushInterval time.Duration
	batches       chan *MetricBatch
	db            Database
	ctx           context.Context
	cancel        context.CancelFunc
	wg            sync.WaitGroup
	mu            sync.Mutex
	stopped       bool
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor(batchSize int, flushInterval time.Duration, db Database) *BatchProcessor {
	ctx, cancel := context.WithCancel(context.Background())

	return &BatchProcessor{
		batchSize:     batchSize,
		flushInterval: flushInterval,
		batches:       make(chan *MetricBatch, 100), // Buffered channel
		db:            db,
		ctx:           ctx,
		cancel:        cancel,
	}
}

// Start starts the batch processor worker
func (bp *BatchProcessor) Start() {
	bp.wg.Add(1)
	go bp.processBatches()
}

// Stop stops the batch processor
func (bp *BatchProcessor) Stop() {
	bp.mu.Lock()
	if bp.stopped {
		bp.mu.Unlock()
		return
	}
	bp.stopped = true
	bp.mu.Unlock()
	
	bp.cancel()
	close(bp.batches)
	bp.wg.Wait()
}

// AddMetrics adds metrics to the batch queue
func (bp *BatchProcessor) AddMetrics(metrics []*PodMetrics) error {
	select {
	case <-bp.ctx.Done():
		return bp.ctx.Err()
	default:
		batch := &MetricBatch{
			Metrics:   metrics,
			Timestamp: time.Now(),
			BatchID:   generateBatchID(),
			Size:      len(metrics),
		}
		bp.batches <- batch
		return nil
	}
}

// processBatches processes batches from the queue
func (bp *BatchProcessor) processBatches() {
	defer bp.wg.Done()

	ticker := time.NewTicker(bp.flushInterval)
	defer ticker.Stop()

	var currentBatch []*PodMetrics

	for {
		select {
		case <-bp.ctx.Done():
			// Flush remaining metrics
			if len(currentBatch) > 0 {
				bp.flushBatch(currentBatch)
			}
			return

		case batch := <-bp.batches:
			if batch == nil {
				// Channel closed or nil batch received
				continue
			}
			if batch.Metrics == nil {
				// Empty metrics, skip
				continue
			}
			currentBatch = append(currentBatch, batch.Metrics...)

			// Flush if batch size reached
			if len(currentBatch) >= bp.batchSize {
				bp.flushBatch(currentBatch)
				currentBatch = currentBatch[:0]
			}

		case <-ticker.C:
			// Flush on interval
			if len(currentBatch) > 0 {
				bp.flushBatch(currentBatch)
				currentBatch = currentBatch[:0]
			}
		}
	}
}

// flushBatch flushes a batch of metrics to the database
func (bp *BatchProcessor) flushBatch(metrics []*PodMetrics) {
	if len(metrics) == 0 {
		return
	}

	if bp.db == nil {
		utils.Log.Error("Database is nil, cannot flush batch")
		return
	}

	// Create a new context with timeout for database operations
	// This ensures database operations don't fail due to canceled context
	dbCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Use batch insert if available
	if batchDB, ok := bp.db.(interface {
		SavePodMetricsBatch(ctx context.Context, metricsList []*PodMetrics) error
	}); ok {
		if err := batchDB.SavePodMetricsBatch(dbCtx, metrics); err != nil {
			utils.Log.WithError(err).Errorf("Failed to save batch of %d metrics", len(metrics))
			return
		}
		utils.Log.Debugf("Successfully saved batch of %d metrics", len(metrics))
	} else {
		// Fallback to individual saves
		for _, m := range metrics {
			if err := bp.db.SavePodMetrics(dbCtx, m); err != nil {
				utils.Log.WithError(err).WithField("pod", m.PodName).Warn("Failed to save metrics")
			}
		}
	}
}

// generateBatchID generates a unique batch ID
func generateBatchID() string {
	return fmt.Sprintf("batch-%d", time.Now().UnixNano())
}
