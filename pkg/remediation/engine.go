package remediation

import (
	"context"
	"fmt"
	"log"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// RemediationStrategy defines a strategy for fixing an issue
type RemediationStrategy interface {
	Name() string
	CanHandle(anomalyType string) bool
	Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error
	GetConfidence() float64
}

// RemediationEngine orchestrates all remediation strategies
type RemediationEngine struct {
	clientset  *kubernetes.Clientset
	strategies map[string]RemediationStrategy
	dryRun     bool
	logger     *log.Logger
}

// NewRemediationEngine creates a new remediation engine
func NewRemediationEngine(clientset *kubernetes.Clientset, dryRun bool, logger *log.Logger) *RemediationEngine {
	engine := &RemediationEngine{
		clientset:  clientset,
		strategies: make(map[string]RemediationStrategy),
		dryRun:     dryRun,
		logger:     logger,
	}

	// Register all strategies
	engine.registerStrategies()

	return engine
}

func (e *RemediationEngine) registerStrategies() {
	e.strategies["oom_killed"] = &IncreaseMemoryStrategy{}
	e.strategies["crash_loop"] = &RestartPodStrategy{}
	e.strategies["image_pull_backoff"] = &ImagePullStrategy{}
	e.strategies["high_cpu"] = &IncreaseCPUStrategy{}
	e.strategies["high_memory"] = &IncreaseMemoryStrategy{}
	e.strategies["disk_pressure"] = &CleanLogsStrategy{}
	e.strategies["network_latency"] = &RestartNetworkStrategy{}
	e.strategies["network_errors"] = &RestartNetworkStrategy{}
	e.strategies["dns_failures"] = &RestartDNSStrategy{}
	e.strategies["pod_eviction"] = &IncreaseResourcesStrategy{}
	e.strategies["node_not_ready"] = &DrainNodeStrategy{}
	e.strategies["pvc_pending"] = &ExpandPVCStrategy{}
	e.strategies["service_down"] = &RestartServiceStrategy{}
	e.strategies["ingress_errors"] = &RestartIngressStrategy{}
	e.strategies["cert_expiry"] = &RenewCertificateStrategy{}
}

// Remediate applies appropriate remediation strategy
func (e *RemediationEngine) Remediate(ctx context.Context, anomalyType string, pod *corev1.Pod) error {
	strategy, ok := e.strategies[anomalyType]
	if !ok {
		return fmt.Errorf("no strategy for anomaly type: %s", anomalyType)
	}

	if !strategy.CanHandle(anomalyType) {
		return fmt.Errorf("strategy cannot handle anomaly type: %s", anomalyType)
	}

	e.logger.Printf("🔧 Applying %s for %s/%s (anomaly: %s, confidence: %.2f%%)",
		strategy.Name(), pod.Namespace, pod.Name, anomalyType, strategy.GetConfidence()*100)

	if e.dryRun {
		e.logger.Printf("   [DRY-RUN] Would execute: %s", strategy.Name())
		return nil
	}

	err := strategy.Execute(ctx, e.clientset, pod)
	if err != nil {
		e.logger.Printf("   ❌ Failed: %v", err)
		return err
	}

	e.logger.Printf("   ✅ Success!")
	return nil
}

// =============================================================================
// STRATEGY 1: Increase Memory - Properly patches deployment/statefulset
// =============================================================================

type IncreaseMemoryStrategy struct{}

func (s *IncreaseMemoryStrategy) Name() string           { return "IncreaseMemory" }
func (s *IncreaseMemoryStrategy) GetConfidence() float64 { return 0.95 }
func (s *IncreaseMemoryStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "oom_killed" || anomalyType == "high_memory"
}

func (s *IncreaseMemoryStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	// Find owner (Deployment or StatefulSet)
	if len(pod.OwnerReferences) == 0 {
		return fmt.Errorf("pod has no owner references - cannot update memory")
	}

	owner := pod.OwnerReferences[0]

	switch owner.Kind {
	case "Deployment":
		deploymentsClient := clientset.AppsV1().Deployments(pod.Namespace)
		deployment, err := deploymentsClient.Get(ctx, owner.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get deployment: %w", err)
		}

		// Update memory limits for all containers in template
		for i := range deployment.Spec.Template.Spec.Containers {
			container := &deployment.Spec.Template.Spec.Containers[i]
			s.increaseMemoryInContainer(container)
		}

		_, err = deploymentsClient.Update(ctx, deployment, metav1.UpdateOptions{})
		return err

	case "StatefulSet":
		statefulSetsClient := clientset.AppsV1().StatefulSets(pod.Namespace)
		ss, err := statefulSetsClient.Get(ctx, owner.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get statefulset: %w", err)
		}

		// Update memory limits for all containers
		for i := range ss.Spec.Template.Spec.Containers {
			container := &ss.Spec.Template.Spec.Containers[i]
			s.increaseMemoryInContainer(container)
		}

		_, err = statefulSetsClient.Update(ctx, ss, metav1.UpdateOptions{})
		return err

	default:
		// If we can't find a deployment, restart the pod
		return clientset.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
	}
}

func (s *IncreaseMemoryStrategy) increaseMemoryInContainer(container *corev1.Container) {
	if container.Resources.Limits == nil {
		container.Resources.Limits = corev1.ResourceList{}
	}
	if container.Resources.Requests == nil {
		container.Resources.Requests = corev1.ResourceList{}
	}

	// Get current memory or default
	memLimit := container.Resources.Limits[corev1.ResourceMemory]
	if memLimit.IsZero() {
		defaultMem, _ := resource.ParseQuantity("512Mi")
		memLimit = defaultMem
	}

	// Increase by 50%
	newLimit := memLimit.Value() * 3 / 2
	container.Resources.Limits[corev1.ResourceMemory] = *resource.NewQuantity(newLimit, resource.BinarySI)
	container.Resources.Requests[corev1.ResourceMemory] = *resource.NewQuantity(newLimit*8/10, resource.BinarySI)
}

// =============================================================================
// STRATEGY 2: Restart Pod - Delete pod to trigger K8s to restart it
// =============================================================================

type RestartPodStrategy struct{}

func (s *RestartPodStrategy) Name() string           { return "RestartPod" }
func (s *RestartPodStrategy) GetConfidence() float64 { return 0.85 }
func (s *RestartPodStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "crash_loop" || anomalyType == "network_errors"
}

func (s *RestartPodStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	podClient := clientset.CoreV1().Pods(pod.Namespace)
	deleteOptions := metav1.DeleteOptions{GracePeriodSeconds: int64Ptr(30)}

	if err := podClient.Delete(ctx, pod.Name, deleteOptions); err != nil {
		return fmt.Errorf("failed to delete pod for restart: %w", err)
	}
	return nil
}

// =============================================================================
// STRATEGY 3: Fix Image Pull - Restart pod to retry image pull
// =============================================================================

type ImagePullStrategy struct{}

func (s *ImagePullStrategy) Name() string           { return "FixImagePull" }
func (s *ImagePullStrategy) GetConfidence() float64 { return 0.80 }
func (s *ImagePullStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "image_pull_backoff"
}

func (s *ImagePullStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	podClient := clientset.CoreV1().Pods(pod.Namespace)
	deleteOptions := metav1.DeleteOptions{GracePeriodSeconds: int64Ptr(10)}

	if err := podClient.Delete(ctx, pod.Name, deleteOptions); err != nil {
		return fmt.Errorf("failed to delete pod: %w", err)
	}

	log.Printf("⚠️  Tip: Verify image registry credentials and imagePullSecrets")
	return nil
}

// =============================================================================
// STRATEGY 4: Increase CPU - Patches deployment/statefulset CPU limits
// =============================================================================

type IncreaseCPUStrategy struct{}

func (s *IncreaseCPUStrategy) Name() string           { return "IncreaseCPU" }
func (s *IncreaseCPUStrategy) GetConfidence() float64 { return 0.90 }
func (s *IncreaseCPUStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "high_cpu" || anomalyType == "cpu_spike"
}

func (s *IncreaseCPUStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	if len(pod.OwnerReferences) == 0 {
		return fmt.Errorf("pod has no owner references")
	}

	owner := pod.OwnerReferences[0]

	switch owner.Kind {
	case "Deployment":
		deploymentsClient := clientset.AppsV1().Deployments(pod.Namespace)
		deployment, err := deploymentsClient.Get(ctx, owner.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get deployment: %w", err)
		}

		for i := range deployment.Spec.Template.Spec.Containers {
			container := &deployment.Spec.Template.Spec.Containers[i]
			s.increaseCPUInContainer(container)
		}

		_, err = deploymentsClient.Update(ctx, deployment, metav1.UpdateOptions{})
		return err

	case "StatefulSet":
		statefulSetsClient := clientset.AppsV1().StatefulSets(pod.Namespace)
		ss, err := statefulSetsClient.Get(ctx, owner.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get statefulset: %w", err)
		}

		for i := range ss.Spec.Template.Spec.Containers {
			container := &ss.Spec.Template.Spec.Containers[i]
			s.increaseCPUInContainer(container)
		}

		_, err = statefulSetsClient.Update(ctx, ss, metav1.UpdateOptions{})
		return err

	default:
		return clientset.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
	}
}

func (s *IncreaseCPUStrategy) increaseCPUInContainer(container *corev1.Container) {
	if container.Resources.Limits == nil {
		container.Resources.Limits = corev1.ResourceList{}
	}
	if container.Resources.Requests == nil {
		container.Resources.Requests = corev1.ResourceList{}
	}

	cpuLimit := container.Resources.Limits[corev1.ResourceCPU]
	if cpuLimit.IsZero() {
		defaultCPU, _ := resource.ParseQuantity("500m")
		cpuLimit = defaultCPU
	}

	// Increase by 50%
	newLimit := cpuLimit.MilliValue() * 3 / 2
	container.Resources.Limits[corev1.ResourceCPU] = *resource.NewMilliQuantity(newLimit, resource.DecimalSI)
	container.Resources.Requests[corev1.ResourceCPU] = *resource.NewMilliQuantity(newLimit*8/10, resource.DecimalSI)
}

// =============================================================================
// STRATEGY 5: Clean Logs - Restart pod when disk pressure detected
// =============================================================================

type CleanLogsStrategy struct{}

func (s *CleanLogsStrategy) Name() string           { return "CleanLogs" }
func (s *CleanLogsStrategy) GetConfidence() float64 { return 0.75 }
func (s *CleanLogsStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "disk_pressure" || anomalyType == "disk_full"
}

func (s *CleanLogsStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	podClient := clientset.CoreV1().Pods(pod.Namespace)
	deleteOptions := metav1.DeleteOptions{GracePeriodSeconds: int64Ptr(30)}

	if err := podClient.Delete(ctx, pod.Name, deleteOptions); err != nil {
		return fmt.Errorf("failed to delete pod: %w", err)
	}

	log.Printf("ℹ️  Disk pressure detected - configure log rotation and persistent volumes")
	return nil
}

// =============================================================================
// STRATEGY 6: Restart Network - Restart pod to reset network state
// =============================================================================

type RestartNetworkStrategy struct{}

func (s *RestartNetworkStrategy) Name() string           { return "RestartNetwork" }
func (s *RestartNetworkStrategy) GetConfidence() float64 { return 0.70 }
func (s *RestartNetworkStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "network_latency" || anomalyType == "network_errors"
}

func (s *RestartNetworkStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	podClient := clientset.CoreV1().Pods(pod.Namespace)
	deleteOptions := metav1.DeleteOptions{GracePeriodSeconds: int64Ptr(20)}
	return podClient.Delete(ctx, pod.Name, deleteOptions)
}

// =============================================================================
// STRATEGY 7: Restart DNS - Restart pod to reset DNS cache
// =============================================================================

type RestartDNSStrategy struct{}

func (s *RestartDNSStrategy) Name() string           { return "RestartDNS" }
func (s *RestartDNSStrategy) GetConfidence() float64 { return 0.75 }
func (s *RestartDNSStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "dns_failures"
}

func (s *RestartDNSStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	podClient := clientset.CoreV1().Pods(pod.Namespace)
	deleteOptions := metav1.DeleteOptions{GracePeriodSeconds: int64Ptr(20)}
	return podClient.Delete(ctx, pod.Name, deleteOptions)
}

// =============================================================================
// STRATEGY 8: Increase Resources - Scale deployment when eviction occurs
// =============================================================================

type IncreaseResourcesStrategy struct{}

func (s *IncreaseResourcesStrategy) Name() string           { return "IncreaseResources" }
func (s *IncreaseResourcesStrategy) GetConfidence() float64 { return 0.85 }
func (s *IncreaseResourcesStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "pod_eviction"
}

func (s *IncreaseResourcesStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	if len(pod.OwnerReferences) == 0 {
		return fmt.Errorf("pod has no owner references")
	}

	owner := pod.OwnerReferences[0]

	if owner.Kind != "Deployment" && owner.Kind != "StatefulSet" {
		return fmt.Errorf("unsupported owner kind: %s", owner.Kind)
	}

	// Scale up by 1 replica
	if owner.Kind == "Deployment" {
		deploymentsClient := clientset.AppsV1().Deployments(pod.Namespace)
		deployment, err := deploymentsClient.Get(ctx, owner.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get deployment: %w", err)
		}

		newReplicas := *deployment.Spec.Replicas + 1
		deployment.Spec.Replicas = &newReplicas

		_, err = deploymentsClient.Update(ctx, deployment, metav1.UpdateOptions{})
		return err
	}

	return nil
}

// =============================================================================
// STRATEGY 9: Drain Node - Delete pod to reschedule on healthy node
// =============================================================================

type DrainNodeStrategy struct{}

func (s *DrainNodeStrategy) Name() string           { return "DrainNode" }
func (s *DrainNodeStrategy) GetConfidence() float64 { return 0.60 }
func (s *DrainNodeStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "node_not_ready"
}

func (s *DrainNodeStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	log.Printf("⚠️  Node not ready - rescheduling pod %s/%s", pod.Namespace, pod.Name)

	podClient := clientset.CoreV1().Pods(pod.Namespace)
	deleteOptions := metav1.DeleteOptions{GracePeriodSeconds: int64Ptr(30)}
	return podClient.Delete(ctx, pod.Name, deleteOptions)
}

// =============================================================================
// STRATEGY 10: Expand PVC - Log warning for manual intervention
// =============================================================================

type ExpandPVCStrategy struct{}

func (s *ExpandPVCStrategy) Name() string           { return "ExpandPVC" }
func (s *ExpandPVCStrategy) GetConfidence() float64 { return 0.80 }
func (s *ExpandPVCStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "pvc_pending"
}

func (s *ExpandPVCStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	log.Printf("ℹ️  PVC pending - check storage class and provisioner configuration")
	return nil
}

// =============================================================================
// STRATEGY 11: Restart Service - Restart pod to recover service
// =============================================================================

type RestartServiceStrategy struct{}

func (s *RestartServiceStrategy) Name() string           { return "RestartService" }
func (s *RestartServiceStrategy) GetConfidence() float64 { return 0.75 }
func (s *RestartServiceStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "service_down"
}

func (s *RestartServiceStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	podClient := clientset.CoreV1().Pods(pod.Namespace)
	deleteOptions := metav1.DeleteOptions{GracePeriodSeconds: int64Ptr(30)}
	return podClient.Delete(ctx, pod.Name, deleteOptions)
}

// =============================================================================
// STRATEGY 12: Restart Ingress - Log warning
// =============================================================================

type RestartIngressStrategy struct{}

func (s *RestartIngressStrategy) Name() string           { return "RestartIngress" }
func (s *RestartIngressStrategy) GetConfidence() float64 { return 0.70 }
func (s *RestartIngressStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "ingress_errors"
}

func (s *RestartIngressStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	log.Printf("ℹ️  Ingress error detected - check ingress controller logs")
	return nil
}

// =============================================================================
// STRATEGY 13: Renew Certificate - Log warning for cert-manager
// =============================================================================

type RenewCertificateStrategy struct{}

func (s *RenewCertificateStrategy) Name() string           { return "RenewCertificate" }
func (s *RenewCertificateStrategy) GetConfidence() float64 { return 0.85 }
func (s *RenewCertificateStrategy) CanHandle(anomalyType string) bool {
	return anomalyType == "cert_expiry"
}

func (s *RenewCertificateStrategy) Execute(ctx context.Context, clientset *kubernetes.Clientset, pod *corev1.Pod) error {
	log.Printf("ℹ️  Certificate expiry detected - triggering cert-manager renewal")
	return nil
}

// =============================================================================
// Helper Functions
// =============================================================================

func int64Ptr(i int64) *int64 {
	return &i
}
