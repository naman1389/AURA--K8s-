package remediation

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/namansh70747/aura-k8s/pkg/k8s"
	"github.com/namansh70747/aura-k8s/pkg/metrics"
	"github.com/namansh70747/aura-k8s/pkg/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// PreventiveAction represents a preventive action to be taken
type PreventiveAction struct {
	ActionType    string                 // Type of action: scale_up, increase_resources, etc.
	Parameters    map[string]interface{} // Action-specific parameters
	TimeToExecute time.Duration          // Estimated time to execute
	Confidence    float64                // Confidence in the action (0-1)
}

// PreventiveRemediator handles preventive remediation actions
type PreventiveRemediator struct {
	k8sClient *k8s.Client
	actions   map[string]PreventiveAction
	dryRun    bool
}

// NewPreventiveRemediator creates a new preventive remediator
func NewPreventiveRemediator(k8sClient *k8s.Client) *PreventiveRemediator {
	return &PreventiveRemediator{
		k8sClient: k8sClient,
		actions:   make(map[string]PreventiveAction),
		dryRun:   false,
	}
}

// SetDryRun sets dry-run mode
func (pr *PreventiveRemediator) SetDryRun(dryRun bool) {
	pr.dryRun = dryRun
}

// ExecutePreventiveAction executes a preventive action based on an early warning
func (pr *PreventiveRemediator) ExecutePreventiveAction(ctx context.Context, warning *metrics.EarlyWarning) error {
	if warning == nil {
		return fmt.Errorf("warning cannot be nil")
	}

	// Validate action before execution
	if err := pr.validateAction(warning); err != nil {
		return fmt.Errorf("action validation failed: %w", err)
	}

	// Execute based on recommended action
	switch warning.RecommendedAction {
	case "scale_up_immediately", "scale_up":
		return pr.scaleUpPod(ctx, warning)
	case "increase_resources":
		return pr.increaseResources(ctx, warning)
	case "load_balance":
		return pr.loadBalance(ctx, warning)
	case "enable_circuit_breaker":
		return pr.enableCircuitBreaker(ctx, warning)
	case "monitor":
		// Just log, no action needed
		utils.Log.WithFields(map[string]interface{}{
			"pod":      warning.PodName,
			"namespace": warning.Namespace,
			"severity": warning.Severity,
		}).Info("Preventive action: monitoring (no action required)")
		return nil
	default:
		utils.Log.WithFields(map[string]interface{}{
			"pod":      warning.PodName,
			"namespace": warning.Namespace,
			"action":   warning.RecommendedAction,
		}).Warn("Unknown preventive action, skipping")
		return nil
	}
}

// validateAction validates that an action can be executed
func (pr *PreventiveRemediator) validateAction(warning *metrics.EarlyWarning) error {
	if warning.PodName == "" {
		return fmt.Errorf("pod name is required")
	}
	if warning.Namespace == "" {
		return fmt.Errorf("namespace is required")
	}
	if warning.Confidence < 0.5 {
		return fmt.Errorf("confidence too low: %.2f (minimum: 0.5)", warning.Confidence)
	}
	return nil
}

// scaleUpPod scales up a pod's deployment/statefulset
func (pr *PreventiveRemediator) scaleUpPod(ctx context.Context, warning *metrics.EarlyWarning) error {
	if pr.dryRun {
		utils.Log.WithFields(map[string]interface{}{
			"pod":      warning.PodName,
			"namespace": warning.Namespace,
		}).Info("[DRY RUN] Would scale up pod")
		return nil
	}

	// Get the pod to find its owner - validate pod exists first
	pod, err := pr.k8sClient.GetPod(ctx, warning.Namespace, warning.PodName)
	if err != nil {
		// If pod doesn't exist, this is expected - log and skip
		if strings.Contains(err.Error(), "not found") || strings.Contains(err.Error(), "NotFound") {
			utils.Log.WithFields(map[string]interface{}{
				"pod":       warning.PodName,
				"namespace": warning.Namespace,
			}).Debug("Pod no longer exists, skipping preventive remediation")
			return nil // Don't treat as error - pod was deleted
		}
		return fmt.Errorf("failed to get pod: %w", err)
	}

	// Find the owner (Deployment, StatefulSet, etc.)
	ownerRef := getOwnerReference(pod)
	if ownerRef == nil {
		return fmt.Errorf("pod has no owner reference")
	}

	// Get deployment for pod
	deployment, err := pr.k8sClient.GetDeploymentForPod(ctx, warning.Namespace, warning.PodName)
	if err != nil {
		return fmt.Errorf("failed to get deployment for pod: %w", err)
	}
	
	if deployment == nil {
		return fmt.Errorf("pod is not owned by a deployment")
	}
	
	currentReplicas := *deployment.Spec.Replicas
	newReplicas := currentReplicas + 1
	if newReplicas > 10 {
		newReplicas = 10 // Cap at 10 replicas
	}
	
	err = pr.k8sClient.ScaleDeployment(ctx, warning.Namespace, deployment.Name, newReplicas)
	if err != nil {
		return fmt.Errorf("failed to scale up deployment: %w", err)
	}
	utils.Log.WithFields(map[string]interface{}{
		"pod":           warning.PodName,
		"namespace":     warning.Namespace,
		"deployment":    deployment.Name,
		"old_replicas":  currentReplicas,
		"new_replicas":  newReplicas,
	}).Info("Scaled up deployment preventively")

	return nil
}

// increaseResources increases resource limits for a pod
func (pr *PreventiveRemediator) increaseResources(ctx context.Context, warning *metrics.EarlyWarning) error {
	if pr.dryRun {
		utils.Log.WithFields(map[string]interface{}{
			"pod":      warning.PodName,
			"namespace": warning.Namespace,
		}).Info("[DRY RUN] Would increase resources")
		return nil
	}

	// Get the pod to find its owner - validate pod exists first
	pod, err := pr.k8sClient.GetPod(ctx, warning.Namespace, warning.PodName)
	if err != nil {
		// If pod doesn't exist, this is expected - log and skip
		if strings.Contains(err.Error(), "not found") || strings.Contains(err.Error(), "NotFound") {
			utils.Log.WithFields(map[string]interface{}{
				"pod":       warning.PodName,
				"namespace": warning.Namespace,
			}).Debug("Pod no longer exists, skipping preventive remediation")
			return nil // Don't treat as error - pod was deleted
		}
		return fmt.Errorf("failed to get pod: %w", err)
	}

	ownerRef := getOwnerReference(pod)
	if ownerRef == nil {
		return fmt.Errorf("pod has no owner reference")
	}

	// Get container name from pod
	containerName := ""
	if len(pod.Spec.Containers) > 0 {
		containerName = pod.Spec.Containers[0].Name
	} else {
		return fmt.Errorf("pod has no containers")
	}

	// Get deployment to read current resource limits
	deployment, err := pr.k8sClient.GetDeploymentForPod(ctx, warning.Namespace, warning.PodName)
	if err != nil {
		return fmt.Errorf("failed to get deployment: %w", err)
	}

	// Find container in deployment and get current limits
	var currentCPU, currentMemory *resource.Quantity
	for _, container := range deployment.Spec.Template.Spec.Containers {
		if container.Name == containerName {
			if limits := container.Resources.Limits; limits != nil {
				if cpu, ok := limits[corev1.ResourceCPU]; ok {
					currentCPU = &cpu
				}
				if mem, ok := limits[corev1.ResourceMemory]; ok {
					currentMemory = &mem
				}
			}
			break
		}
	}

	// Calculate new limits (20% increase) or use defaults if no limits set
	var cpuLimit, memoryLimit string
	if currentCPU != nil {
		// Calculate 20% increase
		newCPUMilli := currentCPU.MilliValue() * 120 / 100
		cpuLimit = fmt.Sprintf("%dm", newCPUMilli)
	} else {
		// Default: 1000m if no limit set
		cpuLimit = "1000m"
		utils.Log.WithFields(map[string]interface{}{
			"pod":       warning.PodName,
			"namespace": warning.Namespace,
		}).Warn("No CPU limit found, using default 1000m")
	}

	if currentMemory != nil {
		// Calculate 20% increase
		newMemoryBytes := currentMemory.Value() * 120 / 100
		// Convert to human-readable format (Mi, Gi, etc.)
		if newMemoryBytes < 1024*1024 {
			memoryLimit = fmt.Sprintf("%dKi", newMemoryBytes/(1024))
		} else if newMemoryBytes < 1024*1024*1024 {
			memoryLimit = fmt.Sprintf("%dMi", newMemoryBytes/(1024*1024))
		} else {
			memoryLimit = fmt.Sprintf("%dGi", newMemoryBytes/(1024*1024*1024))
		}
	} else {
		// Default: 512Mi if no limit set
		memoryLimit = "512Mi"
		utils.Log.WithFields(map[string]interface{}{
			"pod":       warning.PodName,
			"namespace": warning.Namespace,
		}).Warn("No memory limit found, using default 512Mi")
	}

	err = pr.k8sClient.UpdatePodResourceLimits(ctx, warning.Namespace, warning.PodName, containerName, cpuLimit, memoryLimit)
	if err != nil {
		return fmt.Errorf("failed to update pod resource limits: %w", err)
	}

	utils.Log.WithFields(map[string]interface{}{
		"pod":           warning.PodName,
		"namespace":     warning.Namespace,
		"container":     containerName,
		"cpu_limit":     cpuLimit,
		"memory_limit":  memoryLimit,
	}).Info("Increased pod resources preventively")

	return nil
}

// loadBalance redistributes load (placeholder - would integrate with service mesh)
func (pr *PreventiveRemediator) loadBalance(_ context.Context, warning *metrics.EarlyWarning) error {
	if pr.dryRun {
		utils.Log.WithFields(map[string]interface{}{
			"pod":      warning.PodName,
			"namespace": warning.Namespace,
		}).Info("[DRY RUN] Would load balance")
		return nil
	}

	// Placeholder - would integrate with service mesh (Istio, Linkerd, etc.)
	utils.Log.WithFields(map[string]interface{}{
		"pod":      warning.PodName,
		"namespace": warning.Namespace,
	}).Info("Load balancing action (placeholder - requires service mesh integration)")
	return nil
}

// enableCircuitBreaker enables circuit breaker (placeholder)
func (pr *PreventiveRemediator) enableCircuitBreaker(_ context.Context, warning *metrics.EarlyWarning) error {
	if pr.dryRun {
		utils.Log.WithFields(map[string]interface{}{
			"pod":      warning.PodName,
			"namespace": warning.Namespace,
		}).Info("[DRY RUN] Would enable circuit breaker")
		return nil
	}

	// Placeholder - would integrate with service mesh
	utils.Log.WithFields(map[string]interface{}{
		"pod":      warning.PodName,
		"namespace": warning.Namespace,
	}).Info("Circuit breaker action (placeholder - requires service mesh integration)")
	return nil
}

// getOwnerReference gets the owner reference from a pod
func getOwnerReference(pod *corev1.Pod) *metav1.OwnerReference {
	if len(pod.OwnerReferences) == 0 {
		return nil
	}
	return &pod.OwnerReferences[0]
}

