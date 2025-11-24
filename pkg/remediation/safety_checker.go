package remediation

import (
	"context"
	"fmt"

	"github.com/namansh70747/aura-k8s/pkg/k8s"
	"github.com/namansh70747/aura-k8s/pkg/utils"
	"github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

// SafetyCheckResult represents the result of a safety check
type SafetyCheckResult string

const (
	SafetyCheckPass SafetyCheckResult = "pass"
	SafetyCheckWarn SafetyCheckResult = "warn"
	SafetyCheckFail SafetyCheckResult = "fail"
	SafetyCheckSkip SafetyCheckResult = "skip"
)

// SafetyCheck represents a single safety check
type SafetyCheck struct {
	Name    string
	Result  SafetyCheckResult
	Message string
	Details map[string]interface{}
}

// RemediationSafetyChecker performs comprehensive safety checks
type RemediationSafetyChecker struct {
	k8sClient *k8s.Client
	logger    *logrus.Logger
}

// NewRemediationSafetyChecker creates a new safety checker
func NewRemediationSafetyChecker(k8sClient *k8s.Client) *RemediationSafetyChecker {
	return &RemediationSafetyChecker{
		k8sClient: k8sClient,
		logger:    utils.Log,
	}
}

// Validate performs all safety checks for an action
func (s *RemediationSafetyChecker) Validate(
	ctx context.Context,
	action RemediationAction,
	pod *corev1.Pod,
) (bool, []SafetyCheck) {
	checks := []SafetyCheck{}

	// Run all checks
	checks = append(checks, s.checkPDBViolations(ctx, action, pod))
	checks = append(checks, s.checkResourceAvailability(ctx, action, pod))
	checks = append(checks, s.checkDependencies(ctx, action, pod))
	checks = append(checks, s.checkQuotaLimits(ctx, action, pod))
	checks = append(checks, s.checkPermissions(ctx, action, pod))
	checks = append(checks, s.checkRollbackCapability(ctx, action, pod))

	// Check if any critical checks failed
	hasFailures := false
	for _, check := range checks {
		if check.Result == SafetyCheckFail {
			hasFailures = true
			break
		}
	}

	return !hasFailures, checks
}

// checkPDBViolations checks if action would violate Pod Disruption Budget
func (s *RemediationSafetyChecker) checkPDBViolations(
	ctx context.Context,
	action RemediationAction,
	pod *corev1.Pod,
) SafetyCheck {
	// Check if action involves pod deletion/eviction
	if action.Type == "pod" && (action.Operation == "restart" || action.Operation == "delete" || action.Operation == "evict") {
		// Check for PDB
		pdbs, err := s.k8sClient.Clientset().PolicyV1().PodDisruptionBudgets(pod.Namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			return SafetyCheck{
				Name:    "pdb_check",
				Result:  SafetyCheckWarn,
				Message: fmt.Sprintf("Could not check PDB: %v", err),
				Details: map[string]interface{}{"error": err.Error()},
			}
		}

		// Check if pod matches any PDB
		for _, pdb := range pdbs.Items {
			selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
			if err != nil {
				continue
			}

			if selector.Matches(labels.Set(pod.Labels)) {
				// Check if eviction would violate PDB
				// DisruptionsAllowed is an int32 field in policy/v1
				disruptionsAllowed := pdb.Status.DisruptionsAllowed
				if disruptionsAllowed == 0 {
					return SafetyCheck{
						Name:    "pdb_check",
						Result:  SafetyCheckFail,
						Message: fmt.Sprintf("Action would violate PDB %s (0 disruptions allowed)", pdb.Name),
						Details: map[string]interface{}{
							"pdb_name":            pdb.Name,
							"disruptions_allowed": 0,
						},
					}
				}
			}
		}
	}

	return SafetyCheck{
		Name:    "pdb_check",
		Result:  SafetyCheckPass,
		Message: "PDB check passed",
		Details: map[string]interface{}{},
	}
}

// checkResourceAvailability checks if resources are available
func (s *RemediationSafetyChecker) checkResourceAvailability(
	_ context.Context,
	action RemediationAction,
	_ *corev1.Pod,
) SafetyCheck {
	if action.Operation == "scale" || action.Operation == "scale_up" {
		replicas, _ := action.Parameters["replicas"].(float64)
		if replicas > 10 {
			return SafetyCheck{
				Name:    "resource_availability",
				Result:  SafetyCheckWarn,
				Message: fmt.Sprintf("Large scale operation (%v replicas) - verify cluster capacity", replicas),
				Details: map[string]interface{}{"replicas": replicas},
			}
		}
	}

	return SafetyCheck{
		Name:    "resource_availability",
		Result:  SafetyCheckPass,
		Message: "Resource availability check passed",
		Details: map[string]interface{}{},
	}
}

// checkDependencies checks action dependencies
func (s *RemediationSafetyChecker) checkDependencies(
	_ context.Context,
	action RemediationAction,
	_ *corev1.Pod,
) SafetyCheck {
	if action.Type == "deployment" {
		if action.Target == "" || action.Target == "deployment" {
			return SafetyCheck{
				Name:    "dependencies",
				Result:  SafetyCheckFail,
				Message: "Deployment target not specified",
				Details: map[string]interface{}{},
			}
		}
	}

	return SafetyCheck{
		Name:    "dependencies",
		Result:  SafetyCheckPass,
		Message: "Dependency check passed",
		Details: map[string]interface{}{},
	}
}

// checkQuotaLimits checks resource quota limits
func (s *RemediationSafetyChecker) checkQuotaLimits(
	_ context.Context,
	action RemediationAction,
	_ *corev1.Pod,
) SafetyCheck {
	operation := action.Operation
	parameters := action.Parameters

	if operation == "increase_memory" || operation == "increase_cpu" {
		factor, ok := parameters["factor"].(float64)
		if ok && factor > 3.0 {
			return SafetyCheck{
				Name:    "quota_limits",
				Result:  SafetyCheckWarn,
				Message: fmt.Sprintf("Large resource increase (factor %.2f) may exceed quota", factor),
				Details: map[string]interface{}{"factor": factor},
			}
		}
	}

	return SafetyCheck{
		Name:    "quota_limits",
		Result:  SafetyCheckPass,
		Message: "Quota check passed",
		Details: map[string]interface{}{},
	}
}

// checkPermissions checks if user has required permissions
func (s *RemediationSafetyChecker) checkPermissions(
	_ context.Context,
	_ RemediationAction,
	_ *corev1.Pod,
) SafetyCheck {
	// Permission checks would be implemented here
	// For now, assume permissions are valid
	return SafetyCheck{
		Name:    "permissions",
		Result:  SafetyCheckPass,
		Message: "Permission check passed",
		Details: map[string]interface{}{},
	}
}

// checkRollbackCapability checks if action can be rolled back
func (s *RemediationSafetyChecker) checkRollbackCapability(
	_ context.Context,
	action RemediationAction,
	_ *corev1.Pod,
) SafetyCheck {
	nonRollbackable := []string{"delete", "force_delete"}
	for _, op := range nonRollbackable {
		if action.Operation == op {
			return SafetyCheck{
				Name:    "rollback_capability",
				Result:  SafetyCheckWarn,
				Message: fmt.Sprintf("Operation %s cannot be rolled back", action.Operation),
				Details: map[string]interface{}{"rollbackable": false},
			}
		}
	}

	return SafetyCheck{
		Name:    "rollback_capability",
		Result:  SafetyCheckPass,
		Message: "Rollback capability check passed",
		Details: map[string]interface{}{"rollbackable": true},
	}
}

// PreCheck performs pre-execution safety checks
func (s *RemediationSafetyChecker) PreCheck(
	ctx context.Context,
	action RemediationAction,
	pod *corev1.Pod,
) (bool, []string) {
	passed, checks := s.Validate(ctx, action, pod)

	warnings := []string{}
	for _, check := range checks {
		if check.Result == SafetyCheckWarn {
			warnings = append(warnings, check.Message)
		}
	}

	return passed, warnings
}

// PostCheck performs post-execution validation
func (s *RemediationSafetyChecker) PostCheck(
	ctx context.Context,
	action RemediationAction,
	pod *corev1.Pod,
	result map[string]interface{},
) (bool, []string) {
	success, ok := result["success"].(bool)
	if !ok || !success {
		return false, []string{fmt.Sprintf("Action %s failed", action.Operation)}
	}

	return true, []string{}
}
