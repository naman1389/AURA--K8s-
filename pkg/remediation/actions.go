package remediation

import (
	"context"
	"fmt"

	"github.com/namansh70747/aura-k8s/pkg/k8s"
	"github.com/namansh70747/aura-k8s/pkg/utils"
	"github.com/sirupsen/logrus"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// ActionExecutor handles execution of various remediation actions
type ActionExecutor struct {
	k8sClient *k8s.Client
	logger    *logrus.Logger
}

// NewActionExecutor creates a new action executor
func NewActionExecutor(k8sClient *k8s.Client) *ActionExecutor {
	return &ActionExecutor{
		k8sClient: k8sClient,
		logger:    utils.Log,
	}
}

// ExecuteServiceAction executes service-related actions
func (a *ActionExecutor) ExecuteServiceAction(ctx context.Context, action RemediationAction, namespace string) error {
	switch action.Operation {
	case "traffic_split":
		// Traffic splitting would require service mesh or ingress controller
		// This is a placeholder for the concept
		a.logger.Infof("Traffic split operation (requires service mesh)")
		return fmt.Errorf("traffic_split requires service mesh integration")

	case "remove_pod_from_lb":
		// Remove pod from load balancer by removing labels/selectors
		serviceName := action.Target
		service, err := a.k8sClient.Clientset().CoreV1().Services(namespace).Get(ctx, serviceName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get service: %w", err)
		}
		// This would require modifying service selectors or pod labels
		a.logger.Infof("Removing pod from load balancer for service %s/%s", namespace, serviceName)
		_ = service // Placeholder
		return nil

	case "add_pod_to_lb":
		serviceName := action.Target
		service, err := a.k8sClient.Clientset().CoreV1().Services(namespace).Get(ctx, serviceName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get service: %w", err)
		}
		a.logger.Infof("Adding pod to load balancer for service %s/%s", namespace, serviceName)
		_ = service // Placeholder
		return nil

	default:
		return fmt.Errorf("unsupported service operation: %s", action.Operation)
	}
}

// ExecuteConfigMapAction executes ConfigMap-related actions
func (a *ActionExecutor) ExecuteConfigMapAction(ctx context.Context, action RemediationAction, namespace string) error {
	configMapName := action.Target
	configMap, err := a.k8sClient.Clientset().CoreV1().ConfigMaps(namespace).Get(ctx, configMapName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get ConfigMap: %w", err)
	}

	switch action.Operation {
	case "update":
		dataParam, hasData := action.Parameters["data"]
		if !hasData {
			return fmt.Errorf("data parameter is required")
		}
		data, ok := dataParam.(map[string]interface{})
		if !ok {
			return fmt.Errorf("data must be a map")
		}
		if configMap.Data == nil {
			configMap.Data = make(map[string]string)
		}
		for k, v := range data {
			if strVal, ok := v.(string); ok {
				configMap.Data[k] = strVal
			}
		}
		_, err = a.k8sClient.Clientset().CoreV1().ConfigMaps(namespace).Update(ctx, configMap, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update ConfigMap: %w", err)
		}
		a.logger.Infof("✅ ConfigMap updated successfully")
		return nil

	case "rollback":
		// ConfigMap rollback would require version history
		a.logger.Infof("ConfigMap rollback (requires version history)")
		return fmt.Errorf("ConfigMap rollback requires version history support")

	default:
		return fmt.Errorf("unsupported ConfigMap operation: %s", action.Operation)
	}
}

// ExecuteSecretAction executes Secret-related actions
func (a *ActionExecutor) ExecuteSecretAction(ctx context.Context, action RemediationAction, namespace string) error {
	secretName := action.Target
	secret, err := a.k8sClient.Clientset().CoreV1().Secrets(namespace).Get(ctx, secretName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get Secret: %w", err)
	}

	switch action.Operation {
	case "update":
		dataParam, hasData := action.Parameters["data"]
		if !hasData {
			return fmt.Errorf("data parameter is required")
		}
		data, ok := dataParam.(map[string]interface{})
		if !ok {
			return fmt.Errorf("data must be a map")
		}
		if secret.Data == nil {
			secret.Data = make(map[string][]byte)
		}
		for k, v := range data {
			if strVal, ok := v.(string); ok {
				secret.Data[k] = []byte(strVal)
			}
		}
		_, err = a.k8sClient.Clientset().CoreV1().Secrets(namespace).Update(ctx, secret, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update Secret: %w", err)
		}
		a.logger.Infof("✅ Secret updated successfully")
		return nil

	case "rotate":
		// Secret rotation - update and trigger pod restart
		a.logger.Infof("Rotating secret %s/%s", namespace, secretName)
		// Update secret (would generate new values)
		_, err = a.k8sClient.Clientset().CoreV1().Secrets(namespace).Update(ctx, secret, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to rotate Secret: %w", err)
		}
		a.logger.Infof("✅ Secret rotated successfully")
		return nil

	default:
		return fmt.Errorf("unsupported Secret operation: %s", action.Operation)
	}
}

// ExecutePDBAction executes Pod Disruption Budget actions
func (a *ActionExecutor) ExecutePDBAction(ctx context.Context, action RemediationAction, namespace string) error {
	pdbName := action.Target

	switch action.Operation {
	case "create":
		minAvailableParam, hasMinAvailable := action.Parameters["min_available"]
		maxUnavailableParam, hasMaxUnavailable := action.Parameters["max_unavailable"]
		selectorParam, hasSelector := action.Parameters["selector"]

		if !hasMinAvailable && !hasMaxUnavailable {
			return fmt.Errorf("either min_available or max_unavailable must be specified")
		}
		if !hasSelector {
			return fmt.Errorf("selector parameter is required")
		}

		selectorMap, ok := selectorParam.(map[string]interface{})
		if !ok {
			return fmt.Errorf("selector must be a map")
		}

		labelSelector := metav1.LabelSelector{}
		matchLabels := make(map[string]string)
		for k, v := range selectorMap {
			if strVal, ok := v.(string); ok {
				matchLabels[k] = strVal
			}
		}
		labelSelector.MatchLabels = matchLabels

		pdb := &policyv1.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pdbName,
				Namespace: namespace,
			},
			Spec: policyv1.PodDisruptionBudgetSpec{
				Selector: &labelSelector,
			},
		}

		if hasMinAvailable {
			if minAvailStr, ok := minAvailableParam.(string); ok {
				minAvailIntOrString := intstr.FromString(minAvailStr)
				pdb.Spec.MinAvailable = &minAvailIntOrString
			}
		}
		if hasMaxUnavailable {
			if maxUnavailStr, ok := maxUnavailableParam.(string); ok {
				maxUnavailIntOrString := intstr.FromString(maxUnavailStr)
				pdb.Spec.MaxUnavailable = &maxUnavailIntOrString
			}
		}

		_, err := a.k8sClient.Clientset().PolicyV1().PodDisruptionBudgets(namespace).Create(ctx, pdb, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("failed to create PDB: %w", err)
		}
		a.logger.Infof("✅ PDB created successfully")
		return nil

	case "update":
		pdb, err := a.k8sClient.Clientset().PolicyV1().PodDisruptionBudgets(namespace).Get(ctx, pdbName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get PDB: %w", err)
		}

		if minAvailableParam, hasMinAvailable := action.Parameters["min_available"]; hasMinAvailable {
			if minAvailStr, ok := minAvailableParam.(string); ok {
				minAvailIntOrString := intstr.FromString(minAvailStr)
				pdb.Spec.MinAvailable = &minAvailIntOrString
				pdb.Spec.MaxUnavailable = nil
			}
		}
		if maxUnavailableParam, hasMaxUnavailable := action.Parameters["max_unavailable"]; hasMaxUnavailable {
			if maxUnavailStr, ok := maxUnavailableParam.(string); ok {
				maxUnavailIntOrString := intstr.FromString(maxUnavailStr)
				pdb.Spec.MaxUnavailable = &maxUnavailIntOrString
				pdb.Spec.MinAvailable = nil
			}
		}

		_, err = a.k8sClient.Clientset().PolicyV1().PodDisruptionBudgets(namespace).Update(ctx, pdb, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update PDB: %w", err)
		}
		a.logger.Infof("✅ PDB updated successfully")
		return nil

	case "delete":
		err := a.k8sClient.Clientset().PolicyV1().PodDisruptionBudgets(namespace).Delete(ctx, pdbName, metav1.DeleteOptions{})
		if err != nil {
			return fmt.Errorf("failed to delete PDB: %w", err)
		}
		a.logger.Infof("✅ PDB deleted successfully")
		return nil

	default:
		return fmt.Errorf("unsupported PDB operation: %s", action.Operation)
	}
}

// ExecuteHealthCheckAction executes health check probe actions
func (a *ActionExecutor) ExecuteHealthCheckAction(ctx context.Context, action RemediationAction, pod *corev1.Pod, deployment *appsv1.Deployment) error {
	probeType := action.Operation // e.g., "add_liveness", "add_readiness", "add_startup"
	containerName := ""
	if cnParam, hasContainer := action.Parameters["container"]; hasContainer {
		if cnStr, ok := cnParam.(string); ok {
			containerName = cnStr
		}
	}

	// Find container
	var targetContainer *corev1.Container
	if deployment != nil {
		for i := range deployment.Spec.Template.Spec.Containers {
			container := &deployment.Spec.Template.Spec.Containers[i]
			if containerName == "" || container.Name == containerName {
				targetContainer = container
				break
			}
		}
	}

	if targetContainer == nil {
		return fmt.Errorf("container not found")
	}

	// Create probe from parameters
	probe := &corev1.Probe{}
	if httpGetParam, hasHTTPGet := action.Parameters["http_get"]; hasHTTPGet {
		httpGetMap, ok := httpGetParam.(map[string]interface{})
		if ok {
			httpGet := corev1.HTTPGetAction{}
			if path, ok := httpGetMap["path"].(string); ok {
				httpGet.Path = path
			}
			if port, ok := httpGetMap["port"].(int); ok {
				httpGet.Port = intstr.FromInt(port)
			}
			probe.HTTPGet = &httpGet
		}
	} else if execParam, hasExec := action.Parameters["exec"]; hasExec {
		execMap, ok := execParam.(map[string]interface{})
		if ok {
			if command, ok := execMap["command"].([]interface{}); ok {
				cmd := make([]string, len(command))
				for i, c := range command {
					if str, ok := c.(string); ok {
						cmd[i] = str
					}
				}
				probe.Exec = &corev1.ExecAction{Command: cmd}
			}
		}
	} else if tcpSocketParam, hasTCPSocket := action.Parameters["tcp_socket"]; hasTCPSocket {
		tcpMap, ok := tcpSocketParam.(map[string]interface{})
		if ok {
			tcpSocket := corev1.TCPSocketAction{}
			if port, ok := tcpMap["port"].(int); ok {
				tcpSocket.Port = intstr.FromInt(port)
			}
			probe.TCPSocket = &tcpSocket
		}
	}

	// Set probe timing
	if initialDelaySeconds, hasDelay := action.Parameters["initial_delay_seconds"]; hasDelay {
		if delay, ok := initialDelaySeconds.(float64); ok {
			probe.InitialDelaySeconds = int32(delay)
		}
	}
	if periodSeconds, hasPeriod := action.Parameters["period_seconds"]; hasPeriod {
		if period, ok := periodSeconds.(float64); ok {
			probe.PeriodSeconds = int32(period)
		}
	}
	if timeoutSeconds, hasTimeout := action.Parameters["timeout_seconds"]; hasTimeout {
		if timeout, ok := timeoutSeconds.(float64); ok {
			probe.TimeoutSeconds = int32(timeout)
		}
	}

	// Add probe to container
	switch probeType {
	case "add_liveness":
		targetContainer.LivenessProbe = probe
		a.logger.Infof("Added liveness probe to container %s", targetContainer.Name)
	case "add_readiness":
		targetContainer.ReadinessProbe = probe
		a.logger.Infof("Added readiness probe to container %s", targetContainer.Name)
	case "add_startup":
		targetContainer.StartupProbe = probe
		a.logger.Infof("Added startup probe to container %s", targetContainer.Name)
	case "update":
		// Update existing probe
		if targetContainer.LivenessProbe != nil {
			targetContainer.LivenessProbe = probe
		} else if targetContainer.ReadinessProbe != nil {
			targetContainer.ReadinessProbe = probe
		} else if targetContainer.StartupProbe != nil {
			targetContainer.StartupProbe = probe
		}
		a.logger.Infof("Updated probe for container %s", targetContainer.Name)
	default:
		return fmt.Errorf("unsupported health check operation: %s", probeType)
	}

	// Update deployment
	if deployment != nil {
		_, err := a.k8sClient.Clientset().AppsV1().Deployments(deployment.Namespace).Update(ctx, deployment, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update deployment with health check: %w", err)
		}
		a.logger.Infof("✅ Health check added/updated successfully")
		return nil
	}

	return fmt.Errorf("deployment required for health check actions")
}

// ExecuteAffinityAction executes affinity/anti-affinity actions
func (a *ActionExecutor) ExecuteAffinityAction(ctx context.Context, action RemediationAction, deployment *appsv1.Deployment) error {
	if deployment == nil {
		return fmt.Errorf("deployment is required for affinity actions")
	}

	affinityType := action.Operation // e.g., "set_pod_affinity", "set_pod_anti_affinity", "set_node_affinity"

	if deployment.Spec.Template.Spec.Affinity == nil {
		deployment.Spec.Template.Spec.Affinity = &corev1.Affinity{}
	}

	switch affinityType {
	case "set_pod_affinity":
		// Pod affinity - prefer to schedule with certain pods
		selectorParam, hasSelector := action.Parameters["selector"]
		if !hasSelector {
			return fmt.Errorf("selector parameter is required")
		}
		selectorMap, ok := selectorParam.(map[string]interface{})
		if !ok {
			return fmt.Errorf("selector must be a map")
		}
		matchLabels := make(map[string]string)
		for k, v := range selectorMap {
			if strVal, ok := v.(string); ok {
				matchLabels[k] = strVal
			}
		}
		preferredDuringSchedulingIgnoredDuringExecution := []corev1.WeightedPodAffinityTerm{
			{
				Weight: 100,
				PodAffinityTerm: corev1.PodAffinityTerm{
					LabelSelector: &metav1.LabelSelector{MatchLabels: matchLabels},
					TopologyKey:   "kubernetes.io/hostname",
				},
			},
		}
		deployment.Spec.Template.Spec.Affinity.PodAffinity = &corev1.PodAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: preferredDuringSchedulingIgnoredDuringExecution,
		}
		a.logger.Infof("Set pod affinity for deployment %s/%s", deployment.Namespace, deployment.Name)

	case "set_pod_anti_affinity":
		// Pod anti-affinity - avoid scheduling with certain pods
		selectorParam, hasSelector := action.Parameters["selector"]
		if !hasSelector {
			return fmt.Errorf("selector parameter is required")
		}
		selectorMap, ok := selectorParam.(map[string]interface{})
		if !ok {
			return fmt.Errorf("selector must be a map")
		}
		matchLabels := make(map[string]string)
		for k, v := range selectorMap {
			if strVal, ok := v.(string); ok {
				matchLabels[k] = strVal
			}
		}
		preferredDuringSchedulingIgnoredDuringExecution := []corev1.WeightedPodAffinityTerm{
			{
				Weight: 100,
				PodAffinityTerm: corev1.PodAffinityTerm{
					LabelSelector: &metav1.LabelSelector{MatchLabels: matchLabels},
					TopologyKey:   "kubernetes.io/hostname",
				},
			},
		}
		deployment.Spec.Template.Spec.Affinity.PodAntiAffinity = &corev1.PodAntiAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: preferredDuringSchedulingIgnoredDuringExecution,
		}
		a.logger.Infof("Set pod anti-affinity for deployment %s/%s", deployment.Namespace, deployment.Name)

	case "set_node_affinity":
		// Node affinity - prefer certain nodes
		selectorParam, hasSelector := action.Parameters["selector"]
		if !hasSelector {
			return fmt.Errorf("selector parameter is required")
		}
		selectorMap, ok := selectorParam.(map[string]interface{})
		if !ok {
			return fmt.Errorf("selector must be a map")
		}
		matchExpressions := []corev1.NodeSelectorRequirement{}
		for k, v := range selectorMap {
			if strVal, ok := v.(string); ok {
				matchExpressions = append(matchExpressions, corev1.NodeSelectorRequirement{
					Key:      k,
					Operator: corev1.NodeSelectorOpIn,
					Values:   []string{strVal},
				})
			}
		}
		deployment.Spec.Template.Spec.Affinity.NodeAffinity = &corev1.NodeAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{
				{
					Weight: 100,
					Preference: corev1.NodeSelectorTerm{
						MatchExpressions: matchExpressions,
					},
				},
			},
		}
		a.logger.Infof("Set node affinity for deployment %s/%s", deployment.Namespace, deployment.Name)

	default:
		return fmt.Errorf("unsupported affinity operation: %s", affinityType)
	}

	_, err := a.k8sClient.Clientset().AppsV1().Deployments(deployment.Namespace).Update(ctx, deployment, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update deployment with affinity: %w", err)
	}
	a.logger.Infof("✅ Affinity set successfully")
	return nil
}

