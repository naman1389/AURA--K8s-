package ml

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/namansh70747/aura-k8s/pkg/metrics"
)

// MLClient handles communication with the ML prediction service
type MLClient struct {
	baseURL string
	client  *http.Client
}

// PredictionRequest represents the request payload for ML predictions
type PredictionRequest struct {
	Features map[string]float64 `json:"features"`
}

// PredictionResponse represents the response from ML service
type PredictionResponse struct {
	AnomalyType   string             `json:"anomaly_type"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	ModelUsed     string             `json:"model_used"`
	Explanation   string             `json:"explanation"`
}

// NewMLClient creates a new ML client
func NewMLClient(baseURL string) *MLClient {
	return &MLClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Predict sends pod metrics to ML service and returns prediction
func (c *MLClient) Predict(ctx context.Context, podMetrics *metrics.PodMetrics) (*metrics.MLPrediction, error) {
	// Convert pod metrics to ML features (13 features expected by model)
	features := map[string]float64{
		"cpu_usage":             podMetrics.CPUUtilization,
		"memory_usage":          podMetrics.MemoryUtilization,
		"disk_usage":            0, // Not available from K8s API
		"network_bytes_sec":     0, // Not available from K8s API
		"error_rate":            0, // Would need app metrics
		"latency_ms":            0, // Would need app metrics
		"restart_count":         float64(podMetrics.Restarts),
		"age_minutes":           float64(podMetrics.Age) / 60,
		"cpu_memory_ratio":      podMetrics.CPUUtilization / (podMetrics.MemoryUtilization + 1),
		"resource_pressure":     (podMetrics.CPUUtilization + podMetrics.MemoryUtilization) / 2,
		"error_latency_product": 0,
		"network_per_cpu":       0,
		"is_critical":           boolToFloat(podMetrics.HasHighCPU || podMetrics.HasOOMKill),
	}

	reqBody := PredictionRequest{Features: features}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/predict", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status: %d", resp.StatusCode)
	}

	var predResp PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&predResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Convert to MLPrediction
	prediction := &metrics.MLPrediction{
		PodName:             podMetrics.PodName,
		Namespace:           podMetrics.Namespace,
		Timestamp:           time.Now(),
		PredictedIssue:      predResp.AnomalyType,
		Confidence:          predResp.Confidence,
		TimeHorizonSeconds:  300, // 5 minutes default
		Explanation:         predResp.Explanation,
		TopFeatures:         []string{"cpu_usage", "memory_usage", "restart_count"},
		XGBoostPrediction:   predResp.AnomalyType,
		RandomForestPred:    predResp.AnomalyType,
		GradientBoostPred:   predResp.AnomalyType,
		NeuralNetPrediction: predResp.AnomalyType,
	}

	// Extract individual scores from probabilities
	if predResp.Probabilities != nil {
		prediction.OOMScore = predResp.Probabilities["oom_kill"]
		prediction.CrashLoopScore = predResp.Probabilities["pod_crash"]
		prediction.HighCPUScore = predResp.Probabilities["cpu_spike"]
		prediction.DiskPressureScore = predResp.Probabilities["disk_full"]
		prediction.NetworkErrorScore = predResp.Probabilities["network_latency"]
	}

	return prediction, nil
}

// boolToFloat converts boolean to float64
func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}
