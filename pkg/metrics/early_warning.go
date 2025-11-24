package metrics

import (
	"context"
	"fmt"
	"math"
	"time"
)

// EarlyWarningSystem generates early warnings based on forecasts
type EarlyWarningSystem struct {
	riskCalculator     *RiskCalculator
	severityClassifier *SeverityClassifier
	timeEstimator      *TimeToAnomalyEstimator
}

// RiskCalculator calculates risk scores (0-100)
type RiskCalculator struct {
	// Weights for different risk factors
	cpuWeight         float64
	memoryWeight      float64
	errorRateWeight   float64
	networkWeight     float64
	trendWeight       float64
	anomalyProbWeight float64
}

// SeverityClassifier classifies severity levels
type SeverityClassifier struct {
	criticalThreshold float64 // >80
	highThreshold     float64 // 60-80
	mediumThreshold   float64 // 40-60
	lowThreshold      float64 // <40
}

// TimeToAnomalyEstimator estimates time until anomaly occurs
type TimeToAnomalyEstimator struct {
	// Estimation parameters
	minTimeEstimate time.Duration
	maxTimeEstimate time.Duration
}

// NewEarlyWarningSystem creates a new early warning system
func NewEarlyWarningSystem() *EarlyWarningSystem {
	return &EarlyWarningSystem{
		riskCalculator: &RiskCalculator{
			cpuWeight:         0.25,
			memoryWeight:      0.25,
			errorRateWeight:   0.15,
			networkWeight:     0.15,
			trendWeight:       0.10,
			anomalyProbWeight: 0.10,
		},
		severityClassifier: &SeverityClassifier{
			criticalThreshold: 80.0,
			highThreshold:     60.0,
			mediumThreshold:   40.0,
			lowThreshold:      0.0,
		},
		timeEstimator: &TimeToAnomalyEstimator{
			minTimeEstimate: 1 * time.Minute,
			maxTimeEstimate: 30 * time.Minute,
		},
	}
}

// CalculateRiskScore calculates a risk score (0-100) based on forecast and historical data
func (rc *RiskCalculator) CalculateRiskScore(forecast *ForecastResult, historicalData []*PodMetrics) float64 {
	if forecast == nil {
		return 0.0
	}

	riskScore := 0.0

	// Factor 1: Anomaly probability (0-100)
	riskScore += forecast.AnomalyProbability * 100.0 * rc.anomalyProbWeight

	// Factor 2: CPU trend
	if len(historicalData) >= 2 {
		cpuTrend := historicalData[len(historicalData)-1].CPUUtilization - historicalData[0].CPUUtilization
		if cpuTrend > 0 {
			cpuRisk := math.Min(cpuTrend*2, 100.0) // Scale trend to 0-100
			riskScore += cpuRisk * rc.cpuWeight
		}
	}

	// Factor 3: Memory trend
	if len(historicalData) >= 2 {
		memTrend := historicalData[len(historicalData)-1].MemoryUtilization - historicalData[0].MemoryUtilization
		if memTrend > 0 {
			memRisk := math.Min(memTrend*2, 100.0)
			riskScore += memRisk * rc.memoryWeight
		}
	}

	// Factor 4: Error rate
	if len(historicalData) > 0 {
		latest := historicalData[len(historicalData)-1]
		if latest.HasNetworkIssues {
			riskScore += 50.0 * rc.errorRateWeight
		}
	}

	// Factor 5: Network issues
	if len(historicalData) > 0 {
		latest := historicalData[len(historicalData)-1]
		if latest.HasNetworkIssues {
			riskScore += 40.0 * rc.networkWeight
		}
	}

	// Factor 6: Overall trend (if predicted value is increasing)
	if forecast.PredictedValue > 0 {
		// Simple trend risk based on predicted value
		if forecast.PredictedValue > 80 {
			trendRisk := (forecast.PredictedValue - 80) * 2 // Scale above 80%
			riskScore += math.Min(trendRisk, 100.0) * rc.trendWeight
		}
	}

	// Normalize to 0-100 range
	riskScore = math.Min(riskScore, 100.0)
	riskScore = math.Max(riskScore, 0.0)

	return riskScore
}

// ClassifySeverity classifies severity based on risk score and forecast
func (sc *SeverityClassifier) ClassifySeverity(riskScore float64, forecast *ForecastResult) string {
	// Adjust thresholds based on confidence
	confidenceMultiplier := 1.0
	if forecast != nil {
		confidenceMultiplier = forecast.Confidence
	}

	adjustedScore := riskScore * confidenceMultiplier

	if adjustedScore >= sc.criticalThreshold {
		return "Critical"
	} else if adjustedScore >= sc.highThreshold {
		return "High"
	} else if adjustedScore >= sc.mediumThreshold {
		return "Medium"
	}
	return "Low"
}

// EstimateTimeToAnomaly estimates time until anomaly occurs
func (tte *TimeToAnomalyEstimator) EstimateTimeToAnomaly(forecast *ForecastResult, historicalData []*PodMetrics) time.Duration {
	if forecast == nil || len(historicalData) < 2 {
		return tte.maxTimeEstimate // Default to max if insufficient data
	}

	// Simple linear extrapolation
	latest := historicalData[len(historicalData)-1]
	previous := historicalData[len(historicalData)-2]

	// Calculate rate of change per second
	// Use time delta between metrics, not count of data points
	timeDelta := latest.Timestamp.Sub(previous.Timestamp).Seconds()
	if timeDelta <= 0 {
		timeDelta = 1.0 // Avoid division by zero
	}
	cpuChangeRate := (latest.CPUUtilization - previous.CPUUtilization) / timeDelta
	memChangeRate := (latest.MemoryUtilization - previous.MemoryUtilization) / timeDelta

	// Estimate time to reach threshold (80%)
	threshold := 80.0
	timeToThreshold := tte.maxTimeEstimate

	if cpuChangeRate > 0 {
		cpuTime := time.Duration((threshold-latest.CPUUtilization)/cpuChangeRate) * time.Second
		if cpuTime > 0 && cpuTime < timeToThreshold {
			timeToThreshold = cpuTime
		}
	}

	if memChangeRate > 0 {
		memTime := time.Duration((threshold-latest.MemoryUtilization)/memChangeRate) * time.Second
		if memTime > 0 && memTime < timeToThreshold {
			timeToThreshold = memTime
		}
	}

	// Clamp to reasonable bounds
	if timeToThreshold < tte.minTimeEstimate {
		timeToThreshold = tte.minTimeEstimate
	}
	if timeToThreshold > tte.maxTimeEstimate {
		timeToThreshold = tte.maxTimeEstimate
	}

	return timeToThreshold
}

// GenerateWarning generates an early warning from a forecast
func (ews *EarlyWarningSystem) GenerateWarning(ctx context.Context, forecast *ForecastResult) (*EarlyWarning, error) {
	if forecast == nil {
		return nil, nil
	}

	// Get historical data from context if available
	var historicalData []*PodMetrics
	if data, ok := ctx.Value("historical_data").([]*PodMetrics); ok {
		historicalData = data
	}

	// Calculate risk score
	riskScore := ews.riskCalculator.CalculateRiskScore(forecast, historicalData)

	// Classify severity
	severity := ews.severityClassifier.ClassifySeverity(riskScore, forecast)

	// Estimate time to anomaly
	timeToAnomaly := ews.timeEstimator.EstimateTimeToAnomaly(forecast, historicalData)

	// Determine warning type based on forecast
	warningType := "unknown"
	if forecast.PredictedValue > 80 {
		warningType = "resource_exhaustion"
	} else if forecast.AnomalyProbability > 0.7 {
		warningType = "anomaly_predicted"
	} else if riskScore > 70 {
		warningType = "high_risk"
	}

	// Generate recommended action
	recommendedAction := ews.generateRecommendedAction(warningType, severity, riskScore)

	// Build predicted metrics map
	predictedMetrics := make(map[string]float64)
	predictedMetrics["cpu_utilization"] = forecast.PredictedValue
	predictedMetrics["risk_score"] = riskScore
	predictedMetrics["anomaly_probability"] = forecast.AnomalyProbability

	// Get pod info from context if available
	podName := ""
	namespace := ""
	if name, ok := ctx.Value("pod_name").(string); ok {
		podName = name
	}
	if ns, ok := ctx.Value("namespace").(string); ok {
		namespace = ns
	}

	warning := &EarlyWarning{
		ID:                "", // Will be set by caller if needed
		PodName:           podName,
		Namespace:         namespace,
		WarningType:       warningType,
		Severity:          severity,
		RiskScore:         riskScore,
		TimeToAnomaly:     timeToAnomaly,
		Confidence:        forecast.Confidence,
		RecommendedAction: recommendedAction,
		PredictedMetrics:  predictedMetrics,
		Description:       fmt.Sprintf("%s: %s (Risk: %.1f, Time to Anomaly: %s)", warningType, severity, riskScore, timeToAnomaly),
		CreatedAt:         time.Now(),
	}

	return warning, nil
}

// generateRecommendedAction generates a recommended action based on warning
func (ews *EarlyWarningSystem) generateRecommendedAction(_ string, severity string, riskScore float64) string {
	if severity == "Critical" || riskScore > 80 {
		return "scale_up_immediately"
	} else if severity == "High" || riskScore > 60 {
		return "scale_up"
	} else if severity == "Medium" || riskScore > 40 {
		return "increase_resources"
	}
	return "monitor"
}
