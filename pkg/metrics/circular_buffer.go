package metrics

import (
	"sync"
	"time"
)

// CircularBuffer is a thread-safe circular buffer for storing PodMetrics
// It overwrites the oldest entry when full
//
// STATUS: INTEGRATED - Buffer is populated (in collector.go) and exposed via HTTP API.
// The predictive orchestrator can access it via /api/v1/buffer/metrics endpoint for faster access.
// Falls back to database queries if buffer is unavailable.
//
// This buffer provides ultra-fast streaming access to recent metrics (last 10k entries),
// reducing database load and improving forecast generation speed.
type CircularBuffer struct {
	buffer []*PodMetrics
	size   int
	head   int // Points to the next write position
	tail   int // Points to the oldest entry
	count  int // Current number of entries
	mu     sync.RWMutex
}

// NewCircularBuffer creates a new circular buffer with the specified size
func NewCircularBuffer(size int) *CircularBuffer {
	if size <= 0 {
		size = 10000 // Default size
	}
	return &CircularBuffer{
		buffer: make([]*PodMetrics, size),
		size:   size,
		head:   0,
		tail:   0,
		count:  0,
	}
}

// Push adds a metric to the buffer, overwriting the oldest entry if full
func (cb *CircularBuffer) Push(metric *PodMetrics) {
	if metric == nil {
		return
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	// Add metric at head position
	cb.buffer[cb.head] = metric
	cb.head = (cb.head + 1) % cb.size

	// If buffer is full, move tail forward (overwrite oldest)
	if cb.count < cb.size {
		cb.count++
	} else {
		// Buffer is full, tail moves forward
		cb.tail = (cb.tail + 1) % cb.size
	}
}

// GetRecent returns the most recent N metrics
// Returns fewer if buffer has fewer entries
func (cb *CircularBuffer) GetRecent(count int) []*PodMetrics {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if count <= 0 || cb.count == 0 {
		return []*PodMetrics{}
	}

	// Limit count to available entries
	if count > cb.count {
		count = cb.count
	}

	result := make([]*PodMetrics, 0, count)

	// Start from the most recent entry (head - 1) and go backwards
	startPos := (cb.head - 1 + cb.size) % cb.size
	for i := 0; i < count; i++ {
		pos := (startPos - i + cb.size) % cb.size
		if cb.buffer[pos] != nil {
			result = append(result, cb.buffer[pos])
		}
	}

	return result
}

// GetAll returns all metrics in the buffer in chronological order (oldest first)
func (cb *CircularBuffer) GetAll() []*PodMetrics {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if cb.count == 0 {
		return []*PodMetrics{}
	}

	result := make([]*PodMetrics, 0, cb.count)

	// Start from tail and go forward
	for i := 0; i < cb.count; i++ {
		pos := (cb.tail + i) % cb.size
		if cb.buffer[pos] != nil {
			result = append(result, cb.buffer[pos])
		}
	}

	return result
}

// Clear removes all entries from the buffer
func (cb *CircularBuffer) Clear() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.buffer = make([]*PodMetrics, cb.size)
	cb.head = 0
	cb.tail = 0
	cb.count = 0
}

// Count returns the current number of entries in the buffer
func (cb *CircularBuffer) Count() int {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.count
}

// Size returns the maximum capacity of the buffer
func (cb *CircularBuffer) Size() int {
	return cb.size
}

// IsFull returns true if the buffer is full
func (cb *CircularBuffer) IsFull() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.count >= cb.size
}

// GetMetricsInTimeRange returns all metrics within the specified time range
func (cb *CircularBuffer) GetMetricsInTimeRange(start, end time.Time) []*PodMetrics {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if cb.count == 0 {
		return []*PodMetrics{}
	}

	result := make([]*PodMetrics, 0)

	// Iterate through all entries
	for i := 0; i < cb.count; i++ {
		pos := (cb.tail + i) % cb.size
		metric := cb.buffer[pos]
		if metric != nil {
			if !metric.Timestamp.Before(start) && !metric.Timestamp.After(end) {
				result = append(result, metric)
			}
		}
	}

	return result
}
