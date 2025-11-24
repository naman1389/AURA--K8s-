package remediation

import (
	"context"
	"time"

	"github.com/namansh70747/aura-k8s/pkg/utils"
	corev1 "k8s.io/api/core/v1"
)

// ResourceUsage represents resource usage for cost calculation
type ResourceUsage struct {
	CPUMillicores float64
	MemoryMiB     float64
	Replicas      int32
	Duration      time.Duration
}

// CostEstimate represents cost estimate for a remediation action
type CostEstimate struct {
	BaseCost     float64
	ResourceCost float64
	TimeCost     float64
	TotalCost    float64
	CostPerHour  float64
	Savings      float64
}

// CostCalculator calculates costs of remediation actions
type CostCalculator struct {
	// Cost per unit (in USD)
	CPUCostPerMillicoreHour    float64
	MemoryCostPerMiBHour        float64
	BaseActionCost              float64
	DowntimeCostPerMinute       float64
}

// NewCostCalculator creates a new cost calculator with default values
func NewCostCalculator() *CostCalculator {
	return &CostCalculator{
		CPUCostPerMillicoreHour:  0.00001,  // $0.01 per 1000 millicores per hour
		MemoryCostPerMiBHour:      0.000001, // $0.001 per MiB per hour
		BaseActionCost:            0.10,     // Base cost for any action
		DowntimeCostPerMinute:     1.0,      // $1 per minute of downtime
	}
}

// CalculateActionCost calculates cost of a remediation action
func (c *CostCalculator) CalculateActionCost(
	actionType string,
	operation string,
	parameters map[string]interface{},
	currentResources *ResourceUsage,
	duration time.Duration,
) float64 {
	// Base cost
	baseCost := c.BaseActionCost

	// Operation multiplier
	multiplier := c.getOperationMultiplier(operation)
	baseCost *= multiplier

	// Resource cost calculation
	resourceCost := 0.0
	if currentResources != nil {
		cpuCost := currentResources.CPUMillicores * c.CPUCostPerMillicoreHour * duration.Hours()
		memoryCost := currentResources.MemoryMiB * c.MemoryCostPerMiBHour * duration.Hours()
		resourceCost = cpuCost + memoryCost
	}

	// Calculate change in resource cost
	switch operation {
	case "increase_memory", "increase_cpu":
		if factor, ok := parameters["factor"].(float64); ok {
			increase := (factor - 1.0) * resourceCost
			resourceCost += increase
		}
	case "decrease_memory", "decrease_cpu":
		if factor, ok := parameters["factor"].(float64); ok {
			decrease := (1.0 - factor) * resourceCost
			resourceCost -= decrease // Savings
		}
	case "scale", "scale_up":
		if replicas, ok := parameters["replicas"].(float64); ok {
			resourceCost *= (1 + replicas) // Additional replicas
		}
	case "scale_down":
		if replicas, ok := parameters["replicas"].(float64); ok {
			resourceCost *= (1 - replicas*0.1) // Reduction
		}
	}

	// Time cost (downtime cost)
	timeCost := 0.0
	if operation == "restart" || operation == "delete" || operation == "force_delete" || operation == "drain" {
		downtimeMinutes := duration.Minutes()
		timeCost = downtimeMinutes * c.DowntimeCostPerMinute
	}

	// Total cost
	totalCost := baseCost + resourceCost + timeCost
	if totalCost < 0 {
		totalCost = 0 // Ensure non-negative
	}

	return totalCost
}

// getOperationMultiplier returns cost multiplier for operation
func (c *CostCalculator) getOperationMultiplier(operation string) float64 {
	multipliers := map[string]float64{
		"restart":         0.5,
		"delete":          0.3,
		"force_delete":    0.2,
		"increase_memory": 1.2,
		"increase_cpu":    1.2,
		"decrease_memory": -0.8, // Negative = savings
		"decrease_cpu":    -0.8,
		"scale":           1.5,
		"scale_up":        1.5,
		"scale_down":      -0.5,
		"update_image":    0.8,
		"rollback":        0.6,
		"drain":           2.0,
		"uncordon":        0.3,
	}
	if multiplier, ok := multipliers[operation]; ok {
		return multiplier
	}
	return 1.0
}

// CalculateSavings calculates cost savings from remediation
func (c *CostCalculator) CalculateSavings(
	beforeState ResourceUsage,
	afterState ResourceUsage,
	durationHours float64,
) float64 {
	beforeCost := (beforeState.CPUMillicores*c.CPUCostPerMillicoreHour +
		beforeState.MemoryMiB*c.MemoryCostPerMiBHour) *
		float64(beforeState.Replicas) * durationHours

	afterCost := (afterState.CPUMillicores*c.CPUCostPerMillicoreHour +
		afterState.MemoryMiB*c.MemoryCostPerMiBHour) *
		float64(afterState.Replicas) * durationHours

	savings := beforeCost - afterCost
	return savings
}

// EstimateMonthlyCost estimates monthly cost for resource usage
func (c *CostCalculator) EstimateMonthlyCost(
	resourceUsage ResourceUsage,
) float64 {
	hoursPerMonth := 730.0 // Average hours per month

	cpuCost := resourceUsage.CPUMillicores * c.CPUCostPerMillicoreHour * hoursPerMonth * float64(resourceUsage.Replicas)
	memoryCost := resourceUsage.MemoryMiB * c.MemoryCostPerMiBHour * hoursPerMonth * float64(resourceUsage.Replicas)

	return cpuCost + memoryCost
}

// GetCostBreakdown gets detailed cost breakdown for an action
func (c *CostCalculator) GetCostBreakdown(
	actionType string,
	operation string,
	parameters map[string]interface{},
	currentResources *ResourceUsage,
) CostEstimate {
	duration := 5 * time.Minute // Default 5 minutes

	baseCost := c.BaseActionCost
	multiplier := c.getOperationMultiplier(operation)
	baseCost *= multiplier

	resourceCost := 0.0
	if currentResources != nil {
		resourceCost = c.CalculateActionCost(actionType, operation, parameters, currentResources, duration) - baseCost
	}

	timeCost := 0.0
	if operation == "restart" || operation == "delete" || operation == "force_delete" {
		downtimeMinutes := duration.Minutes()
		timeCost = downtimeMinutes * c.DowntimeCostPerMinute
	}

	totalCost := baseCost + resourceCost + timeCost
	costPerHour := totalCost / duration.Hours()

	// Calculate potential savings (if optimization action)
	savings := 0.0
	if operation == "decrease_memory" || operation == "decrease_cpu" || operation == "scale_down" {
		if currentResources != nil {
			if factor, ok := parameters["factor"].(float64); ok {
				savings = resourceCost * (1.0 - factor)
			}
		}
	}

	return CostEstimate{
		BaseCost:     baseCost,
		ResourceCost: resourceCost,
		TimeCost:     timeCost,
		TotalCost:    totalCost,
		CostPerHour:  costPerHour,
		Savings:      savings,
	}
}

// CostOptimizedPlanner generates cost-optimized remediation plans
type CostOptimizedPlanner struct {
	calculator *CostCalculator
}

// NewCostOptimizedPlanner creates a new cost-optimized planner
func NewCostOptimizedPlanner(calculator *CostCalculator) *CostOptimizedPlanner {
	return &CostOptimizedPlanner{
		calculator: calculator,
	}
}

// OptimizeForCost selects strategy with best cost efficiency
func (p *CostOptimizedPlanner) OptimizeForCost(
	strategies []RemediationStrategy,
	budgetConstraint *float64,
) *RemediationStrategy {
	if len(strategies) == 0 {
		return nil
	}

	// Filter by budget if provided
	filteredStrategies := strategies
	if budgetConstraint != nil {
		newFiltered := []RemediationStrategy{}
		for _, s := range filteredStrategies {
			// Calculate total cost for strategy
			totalCost := 0.0
			for range s.Actions {
				// Estimate cost for each action
				cost := p.calculator.BaseActionCost
				totalCost += cost
			}
			if totalCost <= *budgetConstraint {
				newFiltered = append(newFiltered, s)
			}
		}
		filteredStrategies = newFiltered
	}

	if len(filteredStrategies) == 0 {
		utils.Log.Warn("No strategies within budget constraint")
		return nil
	}

	// Score strategies by cost efficiency (success rate / cost)
	bestStrategy := &filteredStrategies[0]
	bestEfficiency := 0.0

	for i := range filteredStrategies {
		strategy := &filteredStrategies[i]
		totalCost := 0.0
		for range strategy.Actions {
			cost := p.calculator.BaseActionCost
			totalCost += cost
		}

		if totalCost > 0 {
			efficiency := strategy.ExpectedSuccessRate / totalCost
			if efficiency > bestEfficiency {
				bestEfficiency = efficiency
				bestStrategy = strategy
			}
		}
	}

	return bestStrategy
}

// BalanceCostAndPerformance balances cost vs performance
func (p *CostOptimizedPlanner) BalanceCostAndPerformance(
	strategies []RemediationStrategy,
) *RemediationStrategy {
	if len(strategies) == 0 {
		return nil
	}

	bestStrategy := &strategies[0]
	bestScore := 0.0

	for i := range strategies {
		strategy := &strategies[i]
		totalCost := 0.0
		for range strategy.Actions {
			cost := p.calculator.BaseActionCost
			totalCost += cost
		}

		// Normalize metrics
		costScore := 1.0 / (1.0 + totalCost/100.0)
		performanceScore := strategy.ExpectedSuccessRate
		timeScore := 1.0 / (1.0 + float64(strategy.EstimatedRecoveryTime)/300.0)

		// Weighted combination
		combinedScore := 0.3*costScore + 0.5*performanceScore + 0.2*timeScore

		if combinedScore > bestScore {
			bestScore = combinedScore
			bestStrategy = strategy
		}
	}

	return bestStrategy
}

// RemediationStrategy represents a remediation strategy (placeholder for integration)
type RemediationStrategy struct {
	Actions                []RemediationAction
	ExpectedSuccessRate    float64
	EstimatedCostImpact    float64
	EstimatedRecoveryTime  int
	RiskLevel              string
	HistoricalSuccessRate  *float64
	Reasoning              string
	Confidence             float64
}

// GetResourceUsageFromPod extracts resource usage from pod
func GetResourceUsageFromPod(pod *corev1.Pod) *ResourceUsage {
	usage := &ResourceUsage{
		Replicas: 1,
	}

	for _, container := range pod.Spec.Containers {
		if container.Resources.Requests != nil {
			if cpu := container.Resources.Requests[corev1.ResourceCPU]; !cpu.IsZero() {
				usage.CPUMillicores += float64(cpu.MilliValue())
			}
			if memory := container.Resources.Requests[corev1.ResourceMemory]; !memory.IsZero() {
				usage.MemoryMiB += float64(memory.Value()) / (1024 * 1024) // Convert to MiB
			}
		} else if container.Resources.Limits != nil {
			if cpu := container.Resources.Limits[corev1.ResourceCPU]; !cpu.IsZero() {
				usage.CPUMillicores += float64(cpu.MilliValue())
			}
			if memory := container.Resources.Limits[corev1.ResourceMemory]; !memory.IsZero() {
				usage.MemoryMiB += float64(memory.Value()) / (1024 * 1024)
			}
		}
	}

	return usage
}

// AnalyzeResourceUsage analyzes resource usage patterns
func (c *CostCalculator) AnalyzeResourceUsage(
	ctx context.Context,
	pod *corev1.Pod,
	metrics map[string]interface{},
) map[string]interface{} {
	cpuUsagePercent := 0.0
	memoryUsagePercent := 0.0

	if cpuUsage, ok := metrics["cpu_usage_percent"].(float64); ok {
		cpuUsagePercent = cpuUsage
	}
	if memoryUsage, ok := metrics["memory_usage_percent"].(float64); ok {
		memoryUsagePercent = memoryUsage
	}

	analysis := map[string]interface{}{
		"cpu_utilization":    cpuUsagePercent,
		"memory_utilization": memoryUsagePercent,
		"recommendations":    []map[string]interface{}{},
	}

	recommendations := []map[string]interface{}{}

	if cpuUsagePercent < 30 {
		recommendations = append(recommendations, map[string]interface{}{
			"type":   "cpu",
			"action": "decrease",
			"factor": 0.7,
			"reason": "CPU usage is low, can reduce requests",
		})
	}

	if memoryUsagePercent < 30 {
		recommendations = append(recommendations, map[string]interface{}{
			"type":   "memory",
			"action": "decrease",
			"factor": 0.7,
			"reason": "Memory usage is low, can reduce requests",
		})
	}

	if cpuUsagePercent > 80 {
		recommendations = append(recommendations, map[string]interface{}{
			"type":   "cpu",
			"action": "increase",
			"factor": 1.5,
			"reason": "CPU usage is high, should increase limits",
		})
	}

	if memoryUsagePercent > 80 {
		recommendations = append(recommendations, map[string]interface{}{
			"type":   "memory",
			"action": "increase",
			"factor": 1.5,
			"reason": "Memory usage is high, should increase limits",
		})
	}

	analysis["recommendations"] = recommendations
	return analysis
}

