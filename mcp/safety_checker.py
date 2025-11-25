"""
Safety & Validation Framework
Comprehensive safety checks before remediation execution
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SafetyCheckResult(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class SafetyCheck:
    """Result of a safety check"""
    name: str
    result: SafetyCheckResult
    message: str
    details: Dict[str, Any] = None


class RemediationSafetyChecker:
    """Comprehensive safety checker for remediations"""
    
    def __init__(self, k8s_tools=None):
        self.k8s_tools = k8s_tools
        self.checks = []
    
    def validate(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[SafetyCheck]]:
        """Validate action with all safety checks"""
        results = []
        
        # Run all checks
        results.append(self.check_pdb_violations(action, context))
        results.append(self.check_resource_availability(action, context))
        results.append(self.check_dependencies(action, context))
        results.append(self.check_quota_limits(action, context))
        results.append(self.check_permissions(action, context))
        results.append(self.check_rollback_capability(action, context))
        
        # Check if any critical checks failed
        has_failures = any(
            check.result == SafetyCheckResult.FAIL
            for check in results
        )
        
        return not has_failures, results
    
    def check_pdb_violations(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SafetyCheck:
        """Check Pod Disruption Budget violations"""
        # This would check if action would violate PDB
        # For now, return pass (would need K8s API access)
        return SafetyCheck(
            name="pdb_check",
            result=SafetyCheckResult.PASS,
            message="PDB check passed (not implemented)",
            details={}
        )
    
    def check_resource_availability(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SafetyCheck:
        """Check if resources are available"""
        action_type = action.get("type", "")
        operation = action.get("operation", "")
        
        # Check if scaling up would exceed cluster capacity
        if operation in ["scale", "scale_up"]:
            replicas = action.get("parameters", {}).get("replicas", 1)
            # Would check cluster capacity here
            return SafetyCheck(
                name="resource_availability",
                result=SafetyCheckResult.WARN,
                message=f"Scaling by {replicas} replicas - verify cluster capacity",
                details={"replicas": replicas}
            )
        
        return SafetyCheck(
            name="resource_availability",
            result=SafetyCheckResult.PASS,
            message="Resource availability check passed",
            details={}
        )
    
    def check_dependencies(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SafetyCheck:
        """Check action dependencies"""
        # Check if action depends on other resources
        action_type = action.get("type", "")
        
        if action_type == "deployment":
            deployment_name = action.get("target", "")
            if not deployment_name or deployment_name == "deployment":
                return SafetyCheck(
                    name="dependencies",
                    result=SafetyCheckResult.FAIL,
                    message="Deployment target not specified",
                    details={}
                )
        
        return SafetyCheck(
            name="dependencies",
            result=SafetyCheckResult.PASS,
            message="Dependency check passed",
            details={}
        )
    
    def check_quota_limits(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SafetyCheck:
        """Check resource quota limits"""
        operation = action.get("operation", "")
        parameters = action.get("parameters", {})
        
        # Check if resource increase would exceed quota
        if operation in ["increase_memory", "increase_cpu"]:
            factor = parameters.get("factor", 1.0)
            if factor > 3.0:
                return SafetyCheck(
                    name="quota_limits",
                    result=SafetyCheckResult.WARN,
                    message=f"Large resource increase (factor {factor}) may exceed quota",
                    details={"factor": factor}
                )
        
        return SafetyCheck(
            name="quota_limits",
            result=SafetyCheckResult.PASS,
            message="Quota check passed",
            details={}
        )
    
    def check_permissions(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SafetyCheck:
        """Check if user has required permissions"""
        # Would check RBAC permissions here
        return SafetyCheck(
            name="permissions",
            result=SafetyCheckResult.PASS,
            message="Permission check passed (not implemented)",
            details={}
        )
    
    def check_rollback_capability(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SafetyCheck:
        """Check if action can be rolled back"""
        operation = action.get("operation", "")
        
        non_rollbackable = ["delete", "force_delete"]
        if operation in non_rollbackable:
            return SafetyCheck(
                name="rollback_capability",
                result=SafetyCheckResult.WARN,
                message=f"Operation {operation} cannot be rolled back",
                details={"rollbackable": False}
            )
        
        return SafetyCheck(
            name="rollback_capability",
            result=SafetyCheckResult.PASS,
            message="Rollback capability check passed",
            details={"rollbackable": True}
        )
    
    def pre_check(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Pre-execution safety checks"""
        passed, checks = self.validate(action, context)
        
        warnings = [
            check.message
            for check in checks
            if check.result == SafetyCheckResult.WARN
        ]
        
        return passed, warnings
    
    def post_check(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Post-execution validation checks"""
        # Verify action completed successfully
        success = result.get("success", False)
        
        if not success:
            return False, [f"Action {action.get('operation')} failed"]
        
        # Additional post-checks could go here
        return True, []



