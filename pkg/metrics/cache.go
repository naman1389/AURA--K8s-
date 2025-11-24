package metrics

import (
	"sync"
	"time"
)

// cacheEntry represents a cached metric entry
type cacheEntry struct {
	metrics   *PodMetrics
	expiresAt time.Time
}

// MetricsCache provides in-memory caching for pod metrics with TTL
type MetricsCache struct {
	cache      map[string]*cacheEntry
	mu         sync.RWMutex
	hitCount   int64
	missCount  int64
	hitRate    float64
	defaultTTL time.Duration
	stopCleanup chan struct{}
}

// NewMetricsCache creates a new metrics cache with specified TTL and cleanup interval
func NewMetricsCache(defaultTTL, cleanupInterval time.Duration) *MetricsCache {
	mc := &MetricsCache{
		cache:      make(map[string]*cacheEntry),
		hitCount:   0,
		missCount:  0,
		hitRate:    0.0,
		defaultTTL: defaultTTL,
		stopCleanup: make(chan struct{}),
	}

	// Start cleanup goroutine
	go mc.cleanup(cleanupInterval)

	return mc
}

// GetCachedMetrics retrieves cached metrics for a pod if available
func (mc *MetricsCache) GetCachedMetrics(podKey string) (*PodMetrics, bool) {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	entry, found := mc.cache[podKey]
	if !found {
		mc.missCount++
		mc.updateHitRate()
		return nil, false
	}

	// Check if expired
	if time.Now().After(entry.expiresAt) {
		mc.missCount++
		mc.updateHitRate()
		delete(mc.cache, podKey)
		return nil, false
	}

	mc.hitCount++
	mc.updateHitRate()
	return entry.metrics, true
}

// SetCachedMetrics stores metrics in cache with TTL
func (mc *MetricsCache) SetCachedMetrics(podKey string, metrics *PodMetrics, ttl time.Duration) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	if ttl <= 0 {
		ttl = mc.defaultTTL
	}

	mc.cache[podKey] = &cacheEntry{
		metrics:   metrics,
		expiresAt: time.Now().Add(ttl),
	}
}

// Clear clears all cached metrics
func (mc *MetricsCache) Clear() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.cache = make(map[string]*cacheEntry)
	mc.hitCount = 0
	mc.missCount = 0
	mc.hitRate = 0.0
}

// Stop stops the cleanup goroutine
func (mc *MetricsCache) Stop() {
	close(mc.stopCleanup)
}

// GetHitRate returns the current cache hit rate
func (mc *MetricsCache) GetHitRate() float64 {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	return mc.hitRate
}

// GetStats returns cache statistics
func (mc *MetricsCache) GetStats() map[string]interface{} {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	total := mc.hitCount + mc.missCount
	return map[string]interface{}{
		"hit_count":  mc.hitCount,
		"miss_count": mc.missCount,
		"hit_rate":   mc.hitRate,
		"total":      total,
		"item_count":  len(mc.cache),
	}
}

// updateHitRate calculates and updates the hit rate
func (mc *MetricsCache) updateHitRate() {
	total := mc.hitCount + mc.missCount
	if total > 0 {
		mc.hitRate = float64(mc.hitCount) / float64(total)
	} else {
		mc.hitRate = 0.0
	}
}

// cleanup removes expired entries periodically
func (mc *MetricsCache) cleanup(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-mc.stopCleanup:
			return
		case <-ticker.C:
			mc.mu.Lock()
			now := time.Now()
			for key, entry := range mc.cache {
				if now.After(entry.expiresAt) {
					delete(mc.cache, key)
				}
			}
			mc.mu.Unlock()
		}
	}
}

// GeneratePodKey generates a unique cache key for a pod
func GeneratePodKey(namespace, podName string) string {
	return namespace + "/" + podName
}