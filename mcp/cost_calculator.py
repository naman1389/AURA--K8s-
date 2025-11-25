"""
Cost-Aware Intelligent Remediation
Calculates costs of remediation actions and optimizes for cost efficiency
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Represents resource usage for cost calculation"""
    cpu_millicores: float = 0.0
    memory_mib: float = 0.0
    replicas: int = 1
    duration_seconds: int = 0


@dataclass
class CostEstimate:
    """Cost estimate for a remediation action"""
    base_cost: float = 0.0
    resource_cost: float = 0.0
    time_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_hour: float = 0.0
    savings: float = 0.0  # Potential savings from remediation


class CostCalculator:
    """Calculate costs of remediation actions"""
    
    # Cost per unit (in USD)
    CPU_COST_PER_MILLICORE_HOUR = 0.00001  # $0.01 per 1000 millicores per hour
    MEMORY_COST_PER_MIB_HOUR = 0.000001  # $0.001 per MiB per hour
    BASE_ACTION_COST = 0.10  # Base cost for any action
    
    # Cost multipliers by action type
    ACTION_COST_MULTIPLIERS = {
        "restart": 0.5,
        "delete": 0.3,
        "force_delete": 0.2,
        "increase_memory": 1.2,
        "increase_cpu": 1.2,
        "decrease_memory": -0.8,  # Negative = savings
        "decrease_cpu": -0.8,
        "scale": 1.5,
        "scale_up": 1.5,
        "scale_down": -0.5,
        "update_image": 0.8,
        "rollback": 0.6,
        "drain": 2.0,
        "uncordon": 0.3,
    }
    
    def __init__(self, custom_costs: Optional[Dict[str, float]] = None):
        """Initialize cost calculator with optional custom costs"""
        if custom_costs:
            if "cpu_cost" in custom_costs:
                self.CPU_COST_PER_MILLICORE_HOUR = custom_costs["cpu_cost"]
            if "memory_cost" in custom_costs:
                self.MEMORY_COST_PER_MIB_HOUR = custom_costs["memory_cost"]
            if "base_cost" in custom_costs:
                self.BASE_ACTION_COST = custom_costs["base_cost"]
    
    def calculate_action_cost(
        self,
        action_type: str,
        operation: str,
        parameters: Dict[str, Any],
        current_resources: Optional[ResourceUsage] = None,
        duration_seconds: int = 300
    ) -> float:
        """Calculate cost of a remediation action"""
        # Base cost
        base_cost = self.BASE_ACTION_COST
        
        # Operation multiplier
        multiplier = self.ACTION_COST_MULTIPLIERS.get(operation, 1.0)
        
        # Resource cost calculation
        resource_cost = 0.0
        
        if current_resources:
            # Calculate resource cost based on current usage
            cpu_cost = (
                current_resources.cpu_millicores *
                self.CPU_COST_PER_MILLICORE_HOUR *
                (duration_seconds / 3600.0)
            )
            memory_cost = (
                current_resources.memory_mib *
                self.MEMORY_COST_PER_MIB_HOUR *
                (duration_seconds / 3600.0)
            )
            resource_cost = cpu_cost + memory_cost
        
        # Calculate change in resource cost
        if operation in ["increase_memory", "increase_cpu"]:
            factor = parameters.get("factor", 1.5)
            increase = (factor - 1.0) * resource_cost
            resource_cost += increase
        elif operation in ["decrease_memory", "decrease_cpu"]:
            factor = parameters.get("factor", 0.8)
            decrease = (1.0 - factor) * resource_cost
            resource_cost -= decrease  # Savings
        
        elif operation in ["scale", "scale_up"]:
            replicas = parameters.get("replicas", 1)
            resource_cost *= (1 + replicas)  # Additional replicas
        
        elif operation in ["scale_down"]:
            replicas = parameters.get("replicas", 1)
            resource_cost *= (1 - replicas * 0.1)  # Reduction
        
        # Time cost (downtime cost)
        time_cost = 0.0
        if operation in ["restart", "delete", "force_delete", "drain"]:
            # Downtime cost (estimated at $1 per minute of downtime)
            downtime_cost_per_minute = 1.0
            downtime_minutes = duration_seconds / 60.0
            time_cost = downtime_minutes * downtime_cost_per_minute
        
        # Total cost
        total_cost = (base_cost * multiplier) + resource_cost + time_cost
        
        return max(0.0, total_cost)  # Ensure non-negative
    
    def calculate_savings(
        self,
        before_state: ResourceUsage,
        after_state: ResourceUsage,
        duration_hours: float = 24.0
    ) -> float:
        """Calculate cost savings from remediation"""
        before_cost = (
            before_state.cpu_millicores * self.CPU_COST_PER_MILLICORE_HOUR +
            before_state.memory_mib * self.MEMORY_COST_PER_MIB_HOUR
        ) * before_state.replicas * duration_hours
        
        after_cost = (
            after_state.cpu_millicores * self.CPU_COST_PER_MILLICORE_HOUR +
            after_state.memory_mib * self.MEMORY_COST_PER_MIB_HOUR
        ) * after_state.replicas * duration_hours
        
        savings = before_cost - after_cost
        return savings
    
    def estimate_monthly_cost(
        self,
        resource_usage: ResourceUsage,
        replicas: int = 1
    ) -> float:
        """Estimate monthly cost for resource usage"""
        hours_per_month = 730.0  # Average hours per month
        
        cpu_cost = (
            resource_usage.cpu_millicores *
            self.CPU_COST_PER_MILLICORE_HOUR *
            hours_per_month *
            replicas
        )
        
        memory_cost = (
            resource_usage.memory_mib *
            self.MEMORY_COST_PER_MIB_HOUR *
            hours_per_month *
            replicas
        )
        
        return cpu_cost + memory_cost
    
    def get_cost_breakdown(
        self,
        action_type: str,
        operation: str,
        parameters: Dict[str, Any],
        current_resources: Optional[ResourceUsage] = None
    ) -> CostEstimate:
        """Get detailed cost breakdown for an action"""
        duration = 300  # Default 5 minutes
        
        base_cost = self.BASE_ACTION_COST
        multiplier = self.ACTION_COST_MULTIPLIERS.get(operation, 1.0)
        base_cost *= multiplier
        
        resource_cost = 0.0
        if current_resources:
            resource_cost = self.calculate_action_cost(
                action_type, operation, parameters, current_resources, duration
            ) - base_cost
        
        time_cost = 0.0
        if operation in ["restart", "delete", "force_delete"]:
            downtime_minutes = duration / 60.0
            time_cost = downtime_minutes * 1.0  # $1 per minute
        
        total_cost = base_cost + resource_cost + time_cost
        
        # Calculate cost per hour
        cost_per_hour = total_cost / (duration / 3600.0)
        
        # Calculate potential savings (if optimization action)
        savings = 0.0
        if operation in ["decrease_memory", "decrease_cpu", "scale_down"]:
            if current_resources:
                # Estimate savings from resource reduction
                factor = parameters.get("factor", 0.8)
                savings = resource_cost * (1.0 - factor)
        
        return CostEstimate(
            base_cost=base_cost,
            resource_cost=resource_cost,
            time_cost=time_cost,
            total_cost=total_cost,
            cost_per_hour=cost_per_hour,
            savings=savings
        )


class CostOptimizedPlanner:
    """Generate cost-optimized remediation plans"""
    
    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calculator = cost_calculator
    
    def optimize_for_cost(
        self,
        strategies: list,
        budget_constraint: Optional[float] = None
    ) -> Any:
        """Select strategy with best cost efficiency"""
        if not strategies:
            return None
        
        # Filter by budget if provided
        if budget_constraint:
            strategies = [
                s for s in strategies
                if s.estimated_cost_impact <= budget_constraint
            ]
        
        if not strategies:
            logger.warning("No strategies within budget constraint")
            return None
        
        # Score strategies by cost efficiency (success rate / cost)
        scored = []
        for strategy in strategies:
            if strategy.estimated_cost_impact > 0:
                efficiency = strategy.expected_success_rate / strategy.estimated_cost_impact
            else:
                efficiency = strategy.expected_success_rate * 100  # High efficiency for free actions
            
            scored.append((strategy, efficiency))
        
        # Return most cost-efficient
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    def balance_cost_and_performance(
        self,
        strategies: list
    ) -> Any:
        """Balance cost vs performance"""
        if not strategies:
            return None
        
        # Score by weighted combination of cost and performance
        scored = []
        for strategy in strategies:
            # Normalize metrics
            cost_score = 1.0 / (1.0 + strategy.estimated_cost_impact / 100.0)
            performance_score = strategy.expected_success_rate
            time_score = 1.0 / (1.0 + strategy.estimated_recovery_time / 300.0)
            
            # Weighted combination
            combined_score = (
                0.3 * cost_score +
                0.5 * performance_score +
                0.2 * time_score
            )
            
            scored.append((strategy, combined_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]


class ResourceOptimizer:
    """Optimize resource allocation"""
    
    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calculator = cost_calculator
    
    def analyze_usage(
        self,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze resource usage patterns"""
        cpu_usage = metrics.get("cpu_usage_percent", 0.0)
        memory_usage = metrics.get("memory_usage_percent", 0.0)
        cpu_request = metrics.get("cpu_request_millicores", 0.0)
        memory_request = metrics.get("memory_request_mib", 0.0)
        
        analysis = {
            "cpu_utilization": cpu_usage,
            "memory_utilization": memory_usage,
            "cpu_efficiency": cpu_usage / 100.0 if cpu_request > 0 else 0.0,
            "memory_efficiency": memory_usage / 100.0 if memory_request > 0 else 0.0,
            "recommendations": []
        }
        
        # Generate recommendations
        if cpu_usage < 30 and cpu_request > 0:
            analysis["recommendations"].append({
                "type": "cpu",
                "action": "decrease",
                "factor": 0.7,
                "reason": "CPU usage is low, can reduce requests"
            })
        
        if memory_usage < 30 and memory_request > 0:
            analysis["recommendations"].append({
                "type": "memory",
                "action": "decrease",
                "factor": 0.7,
                "reason": "Memory usage is low, can reduce requests"
            })
        
        if cpu_usage > 80:
            analysis["recommendations"].append({
                "type": "cpu",
                "action": "increase",
                "factor": 1.5,
                "reason": "CPU usage is high, should increase limits"
            })
        
        if memory_usage > 80:
            analysis["recommendations"].append({
                "type": "memory",
                "action": "increase",
                "factor": 1.5,
                "reason": "Memory usage is high, should increase limits"
            })
        
        return analysis
    
    def recommend_rightsize(
        self,
        current_resources: Dict[str, Any],
        usage_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recommend optimal resource allocation"""
        cpu_usage = usage_patterns.get("cpu_usage_percent", 50.0)
        memory_usage = usage_patterns.get("memory_usage_percent", 50.0)
        
        current_cpu = current_resources.get("cpu_millicores", 1000.0)
        current_memory = current_resources.get("memory_mib", 512.0)
        
        # Calculate recommended resources based on usage
        # Target 70% utilization for headroom
        recommended_cpu = current_cpu * (cpu_usage / 70.0) if cpu_usage > 0 else current_cpu
        recommended_memory = current_memory * (memory_usage / 70.0) if memory_usage > 0 else current_memory
        
        # Round to reasonable values
        recommended_cpu = max(100.0, round(recommended_cpu / 100.0) * 100.0)  # Round to 100m
        recommended_memory = max(128.0, round(recommended_memory / 128.0) * 128.0)  # Round to 128Mi
        
        # Calculate savings
        current_usage = ResourceUsage(
            cpu_millicores=current_cpu,
            memory_mib=current_memory
        )
        recommended_usage = ResourceUsage(
            cpu_millicores=recommended_cpu,
            memory_mib=recommended_memory
        )
        
        savings = self.cost_calculator.calculate_savings(
            current_usage, recommended_usage
        )
        
        return {
            "current": {
                "cpu_millicores": current_cpu,
                "memory_mib": current_memory
            },
            "recommended": {
                "cpu_millicores": recommended_cpu,
                "memory_mib": recommended_memory
            },
            "savings_per_day": savings / 30.0,  # Approximate daily savings
            "savings_per_month": savings,
            "cpu_reduction_percent": ((current_cpu - recommended_cpu) / current_cpu * 100) if current_cpu > 0 else 0,
            "memory_reduction_percent": ((current_memory - recommended_memory) / current_memory * 100) if current_memory > 0 else 0
        }
    
    def calculate_savings(
        self,
        current: Dict[str, Any],
        recommended: Dict[str, Any]
    ) -> float:
        """Calculate cost savings from right-sizing"""
        current_usage = ResourceUsage(
            cpu_millicores=current.get("cpu_millicores", 0),
            memory_mib=current.get("memory_mib", 0)
        )
        recommended_usage = ResourceUsage(
            cpu_millicores=recommended.get("cpu_millicores", 0),
            memory_mib=recommended.get("memory_mib", 0)
        )
        
        return self.cost_calculator.calculate_savings(
            current_usage, recommended_usage
        )



