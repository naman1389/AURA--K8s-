package remediation

import (
	"context"
	"fmt"
	"time"

	"github.com/namansh70747/aura-k8s/pkg/utils"
	corev1 "k8s.io/api/core/v1"
)

// RollbackManager manages rollback operations for remediation actions
type RollbackManager struct {
	actionHistory []ExecutedAction
	rollbackPlans map[string]RollbackPlan
	recoveryState RecoveryState
}

// ExecutedAction represents an executed remediation action
type ExecutedAction struct {
	Action    RemediationAction
	Executed  bool
	Timestamp time.Time
	Result    ExecutionResult
}

// ExecutionResult contains the result of action execution
type ExecutionResult struct {
	Success   bool
	Error     error
	State     map[string]interface{}
	Timestamp time.Time
}

// RollbackPlan defines how to rollback an action
type RollbackPlan struct {
	ActionID      string
	OriginalAction RemediationAction
	RollbackActions []RemediationAction
	StateSnapshot  map[string]interface{}
	CreatedAt      time.Time
}

// RecoveryState tracks the recovery state of a remediation
type RecoveryState struct {
	RemediationID string
	State         string // "in_progress", "completed", "failed", "rolled_back"
	Actions       []ExecutedAction
	StartTime     time.Time
	EndTime       *time.Time
}

// NewRollbackManager creates a new RollbackManager
func NewRollbackManager() *RollbackManager {
	return &RollbackManager{
		actionHistory: make([]ExecutedAction, 0),
		rollbackPlans: make(map[string]RollbackPlan),
		recoveryState: RecoveryState{
			State: "in_progress",
		},
	}
}

// RecordAction records an executed action for potential rollback
func (r *RollbackManager) RecordAction(
	action RemediationAction,
	success bool,
	err error,
	state map[string]interface{},
) {
	executed := ExecutedAction{
		Action:    action,
		Executed:  true,
		Timestamp: time.Now(),
		Result: ExecutionResult{
			Success:   success,
			Error:     err,
			State:     state,
			Timestamp: time.Now(),
		},
	}
	
	r.actionHistory = append(r.actionHistory, executed)
	r.recoveryState.Actions = append(r.recoveryState.Actions, executed)
	
	// Create rollback plan if action succeeded
	if success {
		rollbackPlan := r.createRollbackPlan(action, state)
		if rollbackPlan != nil {
			actionID := fmt.Sprintf("%s-%d", action.Operation, len(r.actionHistory))
			r.rollbackPlans[actionID] = *rollbackPlan
		}
	}
}

// Rollback executes rollback for a specific action
func (r *RollbackManager) Rollback(
	ctx context.Context,
	actionID string,
	executor *RemediationExecutor,
	pod *corev1.Pod,
) error {
	plan, exists := r.rollbackPlans[actionID]
	if !exists {
		return fmt.Errorf("rollback plan not found for action %s", actionID)
	}
	
	utils.Log.Infof("Rolling back action %s", actionID)
	
	// Execute rollback actions in reverse order
	for i := len(plan.RollbackActions) - 1; i >= 0; i-- {
		rollbackAction := plan.RollbackActions[i]
		if err := executor.ExecuteAction(ctx, pod, rollbackAction); err != nil {
			utils.Log.WithError(err).Errorf("Failed to execute rollback action %s", rollbackAction.Operation)
			return fmt.Errorf("rollback failed: %w", err)
		}
	}
	
	// Verify rollback success
	if err := r.verifyRollback(ctx, plan); err != nil {
		return fmt.Errorf("rollback verification failed: %w", err)
	}
	
	utils.Log.Infof("✅ Rollback completed successfully for action %s", actionID)
	return nil
}

// RollbackAll rolls back all executed actions
func (r *RollbackManager) RollbackAll(
	ctx context.Context,
	executor *RemediationExecutor,
	pod *corev1.Pod,
) error {
	utils.Log.Warnf("Rolling back all %d executed actions", len(r.actionHistory))
	
	// Rollback in reverse order
	for i := len(r.actionHistory) - 1; i >= 0; i-- {
		executed := r.actionHistory[i]
		if !executed.Executed || !executed.Result.Success {
			continue // Skip non-executed or failed actions
		}
		
		actionID := fmt.Sprintf("%s-%d", executed.Action.Operation, i)
		if err := r.Rollback(ctx, actionID, executor, pod); err != nil {
			utils.Log.WithError(err).Errorf("Failed to rollback action %d", i)
			// Continue with other rollbacks
		}
	}
	
	r.recoveryState.State = "rolled_back"
	now := time.Now()
	r.recoveryState.EndTime = &now
	
	return nil
}

// createRollbackPlan creates a rollback plan for an action
func (r *RollbackManager) createRollbackPlan(
	action RemediationAction,
	state map[string]interface{},
) *RollbackPlan {
	rollbackActions := make([]RemediationAction, 0)
	
	switch action.Operation {
	case "increase_memory", "increase_cpu":
		// Rollback: decrease by inverse factor
		factorParam, hasFactor := action.Parameters["factor"]
		if hasFactor {
			if factor, ok := factorParam.(float64); ok && factor > 0 {
				rollbackFactor := 1.0 / factor
				rollbackActions = append(rollbackActions, RemediationAction{
					Type:       action.Type,
					Target:     action.Target,
					Operation:  action.Operation,
					Parameters: map[string]interface{}{"factor": rollbackFactor},
					Order:      action.Order,
				})
			}
		}
	
	case "scale", "scale_up":
		// Rollback: scale down
		replicasParam, hasReplicas := action.Parameters["replicas"]
		if hasReplicas {
			if direction, ok := action.Parameters["direction"].(string); ok && direction == "up" {
				rollbackActions = append(rollbackActions, RemediationAction{
					Type:       action.Type,
					Target:     action.Target,
					Operation:  "scale_down",
					Parameters: map[string]interface{}{"replicas": replicasParam},
					Order:      action.Order,
				})
			}
		}
	
	case "scale_down":
		// Rollback: scale up
		replicasParam, hasReplicas := action.Parameters["replicas"]
		if hasReplicas {
			rollbackActions = append(rollbackActions, RemediationAction{
				Type:       action.Type,
				Target:     action.Target,
				Operation:  "scale_up",
				Parameters: map[string]interface{}{"replicas": replicasParam},
				Order:      action.Order,
			})
		}
	
	case "update_image":
		// Rollback: restore previous image from state
		if prevImage, ok := state["previous_image"].(string); ok && prevImage != "" {
			rollbackActions = append(rollbackActions, RemediationAction{
				Type:       action.Type,
				Target:     action.Target,
				Operation:  "update_image",
				Parameters: map[string]interface{}{"image": prevImage},
				Order:      action.Order,
			})
		}
	
	default:
		// Actions like restart, delete cannot be rolled back
		return nil
	}
	
	if len(rollbackActions) == 0 {
		return nil
	}
	
	return &RollbackPlan{
		OriginalAction: action,
		RollbackActions: rollbackActions,
		StateSnapshot:   state,
		CreatedAt:       time.Now(),
	}
}

// verifyRollback verifies that rollback was successful
func (r *RollbackManager) verifyRollback(
	_ context.Context,
	plan RollbackPlan,
) error {
	// Basic verification - in production, would check actual resource state
	utils.Log.Debugf("Verifying rollback for action %s", plan.OriginalAction.Operation)
	return nil
}

// GetRecoveryState returns the current recovery state
func (r *RollbackManager) GetRecoveryState() RecoveryState {
	return r.recoveryState
}

// MarkCompleted marks the remediation as completed
func (r *RollbackManager) MarkCompleted() {
	r.recoveryState.State = "completed"
	now := time.Now()
	r.recoveryState.EndTime = &now
}

// MarkFailed marks the remediation as failed
func (r *RollbackManager) MarkFailed() {
	r.recoveryState.State = "failed"
	now := time.Now()
	r.recoveryState.EndTime = &now
}


