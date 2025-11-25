"""
Multi-Strategy Remediation Planner
Generates multiple remediation strategies and selects the best one based on:
- Success probability
- Cost impact
- Risk level
- Recovery time
- Historical success rate
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StrategyType(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"


@dataclass
class RemediationAction:
    """Represents a single remediation action"""
    type: str
    target: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    order: int = 0
    estimated_time: int = 30  # seconds
    estimated_cost: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM


@dataclass
class RemediationStrategy:
    """Represents a complete remediation strategy"""
    actions: List[RemediationAction]
    strategy_type: StrategyType
    expected_success_rate: float
    estimated_cost_impact: float
    estimated_recovery_time: int  # seconds
    risk_level: RiskLevel
    historical_success_rate: Optional[float] = None
    reasoning: str = ""
    confidence: float = 0.75
    
    def calculate_score(self, weights: Dict[str, float]) -> float:
        """Calculate strategy score based on weighted factors"""
        score = 0.0
        
        # Success rate (higher is better)
        score += weights.get("success_rate", 0.4) * self.expected_success_rate
        
        # Cost (lower is better, so invert)
        max_cost = 1000.0  # Normalize cost
        cost_score = 1.0 - min(self.estimated_cost_impact / max_cost, 1.0)
        score += weights.get("cost", 0.2) * cost_score
        
        # Recovery time (lower is better, so invert)
        max_time = 600  # 10 minutes
        time_score = 1.0 - min(self.estimated_recovery_time / max_time, 1.0)
        score += weights.get("recovery_time", 0.2) * time_score
        
        # Risk (lower is better, so invert)
        risk_scores = {RiskLevel.LOW: 1.0, RiskLevel.MEDIUM: 0.5, RiskLevel.HIGH: 0.0}
        score += weights.get("risk", 0.2) * risk_scores.get(self.risk_level, 0.5)
        
        return score


@dataclass
class MultiStrategyRemediationPlan:
    """Container for multiple remediation strategies"""
    strategies: List[RemediationStrategy]
    recommended_strategy: Optional[RemediationStrategy] = None
    strategy_comparison: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def select_best_strategy(
        self,
        weights: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> RemediationStrategy:
        """Select best strategy based on weights and constraints"""
        if not self.strategies:
            raise ValueError("No strategies available")
        
        if weights is None:
            weights = {
                "success_rate": 0.4,
                "cost": 0.2,
                "recovery_time": 0.2,
                "risk": 0.2
            }
        
        if constraints is None:
            constraints = {}
        
        # Filter strategies by constraints
        filtered_strategies = self.strategies
        if "max_cost" in constraints:
            filtered_strategies = [
                s for s in filtered_strategies
                if s.estimated_cost_impact <= constraints["max_cost"]
            ]
        if "max_time" in constraints:
            filtered_strategies = [
                s for s in filtered_strategies
                if s.estimated_recovery_time <= constraints["max_time"]
            ]
        if "max_risk" in constraints:
            max_risk = RiskLevel(constraints["max_risk"])
            risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
            filtered_strategies = [
                s for s in filtered_strategies
                if risk_order[s.risk_level] <= risk_order[max_risk]
            ]
        
        if not filtered_strategies:
            logger.warning("No strategies match constraints, using all strategies")
            filtered_strategies = self.strategies
        
        # Score and select best
        scored_strategies = [
            (s, s.calculate_score(weights))
            for s in filtered_strategies
        ]
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        
        best_strategy = scored_strategies[0][0]
        self.recommended_strategy = best_strategy
        
        # Store comparison
        self.strategy_comparison = {
            s.strategy_type.value: score
            for s, score in scored_strategies
        }
        
        return best_strategy


class RemediationPlanner:
    """Generates multiple remediation strategies for an issue"""
    
    def __init__(self, learner=None, cost_calculator=None):
        self.learner = learner
        self.cost_calculator = cost_calculator
    
    def generate_strategies(
        self,
        issue_type: str,
        severity: str,
        pod_context: Dict[str, Any],
        deployment: Optional[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> MultiStrategyRemediationPlan:
        """Generate multiple remediation strategies"""
        strategies = []
        
        # Conservative strategy (high success, slow, low risk)
        conservative = self._generate_conservative_strategy(
            issue_type, severity, pod_context, deployment
        )
        if conservative:
            strategies.append(conservative)
        
        # Balanced strategy (moderate everything)
        balanced = self._generate_balanced_strategy(
            issue_type, severity, pod_context, deployment
        )
        if balanced:
            strategies.append(balanced)
        
        # Aggressive strategy (fast, higher risk)
        aggressive = self._generate_aggressive_strategy(
            issue_type, severity, pod_context, deployment
        )
        if aggressive:
            strategies.append(aggressive)
        
        # Cost-optimized strategy
        cost_optimized = self._generate_cost_optimized_strategy(
            issue_type, severity, pod_context, deployment
        )
        if cost_optimized:
            strategies.append(cost_optimized)
        
        # Performance-optimized strategy
        performance_optimized = self._generate_performance_optimized_strategy(
            issue_type, severity, pod_context, deployment
        )
        if performance_optimized:
            strategies.append(performance_optimized)
        
        # Enhance strategies with historical data if available
        if self.learner and historical_data:
            for strategy in strategies:
                historical_rate = self.learner.get_success_rate(
                    issue_type, strategy.strategy_type.value
                )
                if historical_rate:
                    strategy.historical_success_rate = historical_rate
                    # Adjust expected success rate based on history
                    strategy.expected_success_rate = (
                        strategy.expected_success_rate * 0.7 +
                        historical_rate * 0.3
                    )
        
        # Calculate costs if cost calculator available
        if self.cost_calculator:
            for strategy in strategies:
                total_cost = 0.0
                for action in strategy.actions:
                    action_cost = self.cost_calculator.calculate_action_cost(
                        action.type, action.operation, action.parameters
                    )
                    action.estimated_cost = action_cost
                    total_cost += action_cost
                strategy.estimated_cost_impact = total_cost
        
        plan = MultiStrategyRemediationPlan(
            strategies=strategies,
            context={
                "issue_type": issue_type,
                "severity": severity,
                "pod_context": pod_context,
                "deployment": deployment
            }
        )
        
        return plan
    
    def _generate_conservative_strategy(
        self,
        issue_type: str,
        severity: str,
        pod_context: Dict[str, Any],
        deployment: Optional[Dict[str, Any]]
    ) -> Optional[RemediationStrategy]:
        """Generate conservative strategy (slow, safe, high success)"""
        actions = []
        
        if issue_type.lower() in ["oomkilled", "oom", "high_memory"]:
            if deployment:
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="increase_memory",
                    parameters={"factor": 1.3},  # Small increase
                    order=0,
                    estimated_time=90,
                    risk_level=RiskLevel.LOW
                ))
            else:
                actions.append(RemediationAction(
                    type="pod",
                    target=pod_context.get("name", "pod"),
                    operation="restart",
                    parameters={"grace_period_seconds": 60},
                    order=0,
                    estimated_time=120,
                    risk_level=RiskLevel.LOW
                ))
        
        elif issue_type.lower() in ["crashloopbackoff", "crash_loop"]:
            actions.append(RemediationAction(
                type="pod",
                target=pod_context.get("name", "pod"),
                operation="restart",
                parameters={"grace_period_seconds": 60},
                order=0,
                estimated_time=120,
                risk_level=RiskLevel.LOW
            ))
        
        elif issue_type.lower() in ["high_cpu", "cpu_spike"]:
            if deployment:
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="scale",
                    parameters={"replicas": 1, "direction": "up"},
                    order=0,
                    estimated_time=180,
                    risk_level=RiskLevel.LOW
                ))
        
        if not actions:
            return None
        
        return RemediationStrategy(
            actions=actions,
            strategy_type=StrategyType.CONSERVATIVE,
            expected_success_rate=0.95,
            estimated_cost_impact=50.0,
            estimated_recovery_time=300,  # 5 minutes
            risk_level=RiskLevel.LOW,
            reasoning="Conservative approach with gradual changes and high safety margins",
            confidence=0.90
        )
    
    def _generate_balanced_strategy(
        self,
        issue_type: str,
        severity: str,
        pod_context: Dict[str, Any],
        deployment: Optional[Dict[str, Any]]
    ) -> Optional[RemediationStrategy]:
        """Generate balanced strategy (moderate everything)"""
        actions = []
        
        if issue_type.lower() in ["oomkilled", "oom", "high_memory"]:
            if deployment:
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="increase_memory",
                    parameters={"factor": 1.5},
                    order=0,
                    estimated_time=90,
                    risk_level=RiskLevel.MEDIUM
                ))
            else:
                actions.append(RemediationAction(
                    type="pod",
                    target=pod_context.get("name", "pod"),
                    operation="restart",
                    parameters={"grace_period_seconds": 30},
                    order=0,
                    estimated_time=60,
                    risk_level=RiskLevel.MEDIUM
                ))
        
        elif issue_type.lower() in ["crashloopbackoff", "crash_loop"]:
            actions.append(RemediationAction(
                type="pod",
                target=pod_context.get("name", "pod"),
                operation="restart",
                parameters={"grace_period_seconds": 30},
                order=0,
                estimated_time=60,
                risk_level=RiskLevel.MEDIUM
            ))
        
        elif issue_type.lower() in ["high_cpu", "cpu_spike"]:
            if deployment:
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="increase_cpu",
                    parameters={"factor": 1.5},
                    order=0,
                    estimated_time=90,
                    risk_level=RiskLevel.MEDIUM
                ))
        
        if not actions:
            return None
        
        return RemediationStrategy(
            actions=actions,
            strategy_type=StrategyType.BALANCED,
            expected_success_rate=0.85,
            estimated_cost_impact=100.0,
            estimated_recovery_time=180,  # 3 minutes
            risk_level=RiskLevel.MEDIUM,
            reasoning="Balanced approach with moderate changes and acceptable risk",
            confidence=0.80
        )
    
    def _generate_aggressive_strategy(
        self,
        issue_type: str,
        severity: str,
        pod_context: Dict[str, Any],
        deployment: Optional[Dict[str, Any]]
    ) -> Optional[RemediationStrategy]:
        """Generate aggressive strategy (fast, higher risk)"""
        actions = []
        
        if issue_type.lower() in ["oomkilled", "oom", "high_memory"]:
            if deployment:
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="increase_memory",
                    parameters={"factor": 2.0},
                    order=0,
                    estimated_time=90,
                    risk_level=RiskLevel.HIGH
                ))
            else:
                actions.append(RemediationAction(
                    type="pod",
                    target=pod_context.get("name", "pod"),
                    operation="force_delete",
                    parameters={},
                    order=0,
                    estimated_time=30,
                    risk_level=RiskLevel.HIGH
                ))
        
        elif issue_type.lower() in ["crashloopbackoff", "crash_loop"]:
            actions.append(RemediationAction(
                type="pod",
                target=pod_context.get("name", "pod"),
                operation="force_delete",
                parameters={},
                order=0,
                estimated_time=30,
                risk_level=RiskLevel.HIGH
            ))
        
        elif issue_type.lower() in ["high_cpu", "cpu_spike"]:
            if deployment:
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="scale",
                    parameters={"replicas": 2, "direction": "up"},
                    order=0,
                    estimated_time=60,
                    risk_level=RiskLevel.HIGH
                ))
        
        if not actions:
            return None
        
        return RemediationStrategy(
            actions=actions,
            strategy_type=StrategyType.AGGRESSIVE,
            expected_success_rate=0.70,
            estimated_cost_impact=200.0,
            estimated_recovery_time=60,  # 1 minute
            risk_level=RiskLevel.HIGH,
            reasoning="Aggressive approach with rapid changes and higher risk",
            confidence=0.65
        )
    
    def _generate_cost_optimized_strategy(
        self,
        issue_type: str,
        severity: str,
        pod_context: Dict[str, Any],
        deployment: Optional[Dict[str, Any]]
    ) -> Optional[RemediationStrategy]:
        """Generate cost-optimized strategy (minimize cost)"""
        actions = []
        
        if issue_type.lower() in ["oomkilled", "oom", "high_memory"]:
            if deployment:
                # Small memory increase
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="increase_memory",
                    parameters={"factor": 1.2},
                    order=0,
                    estimated_time=90,
                    risk_level=RiskLevel.LOW
                ))
            else:
                actions.append(RemediationAction(
                    type="pod",
                    target=pod_context.get("name", "pod"),
                    operation="restart",
                    parameters={"grace_period_seconds": 30},
                    order=0,
                    estimated_time=60,
                    risk_level=RiskLevel.LOW
                ))
        
        elif issue_type.lower() in ["crashloopbackoff", "crash_loop"]:
            actions.append(RemediationAction(
                type="pod",
                target=pod_context.get("name", "pod"),
                operation="restart",
                parameters={"grace_period_seconds": 30},
                order=0,
                estimated_time=60,
                risk_level=RiskLevel.LOW
            ))
        
        if not actions:
            return None
        
        return RemediationStrategy(
            actions=actions,
            strategy_type=StrategyType.COST_OPTIMIZED,
            expected_success_rate=0.80,
            estimated_cost_impact=20.0,  # Very low cost
            estimated_recovery_time=300,  # 5 minutes
            risk_level=RiskLevel.LOW,
            reasoning="Cost-optimized approach minimizing resource usage",
            confidence=0.75
        )
    
    def _generate_performance_optimized_strategy(
        self,
        issue_type: str,
        severity: str,
        pod_context: Dict[str, Any],
        deployment: Optional[Dict[str, Any]]
    ) -> Optional[RemediationStrategy]:
        """Generate performance-optimized strategy (maximize performance)"""
        actions = []
        
        if issue_type.lower() in ["oomkilled", "oom", "high_memory"]:
            if deployment:
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="increase_memory",
                    parameters={"factor": 2.0},
                    order=0,
                    estimated_time=90,
                    risk_level=RiskLevel.MEDIUM
                ))
                # Also scale up for performance
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="scale",
                    parameters={"replicas": 1, "direction": "up"},
                    order=1,
                    estimated_time=60,
                    risk_level=RiskLevel.MEDIUM
                ))
        
        elif issue_type.lower() in ["high_cpu", "cpu_spike"]:
            if deployment:
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="increase_cpu",
                    parameters={"factor": 2.0},
                    order=0,
                    estimated_time=90,
                    risk_level=RiskLevel.MEDIUM
                ))
                actions.append(RemediationAction(
                    type="deployment",
                    target=deployment.get("name", "deployment"),
                    operation="scale",
                    parameters={"replicas": 2, "direction": "up"},
                    order=1,
                    estimated_time=60,
                    risk_level=RiskLevel.MEDIUM
                ))
        
        if not actions:
            return None
        
        return RemediationStrategy(
            actions=actions,
            strategy_type=StrategyType.PERFORMANCE_OPTIMIZED,
            expected_success_rate=0.90,
            estimated_cost_impact=300.0,  # Higher cost
            estimated_recovery_time=90,  # 1.5 minutes
            risk_level=RiskLevel.MEDIUM,
            reasoning="Performance-optimized approach maximizing throughput",
            confidence=0.85
        )



