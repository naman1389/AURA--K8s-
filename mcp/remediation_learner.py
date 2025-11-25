"""
Machine Learning from Remediations
Learn from past remediations to improve future decisions
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RemediationRecord:
    """Record of a past remediation"""
    remediation_id: str
    issue_type: str
    anomaly_type: str
    action: str
    strategy: str
    success: bool
    recovery_time: int  # seconds
    cost_impact: float
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendedAction:
    """Recommended action with reasoning"""
    action: str
    strategy: str
    confidence: float
    reasoning: str
    expected_success_rate: float
    expected_recovery_time: int
    expected_cost: float


@dataclass
class RankedAction:
    """Ranked action with score"""
    action: str
    strategy: str
    score: float
    success_rate: float
    cost: float
    recovery_time: int
    rank: int


class RemediationLearningEngine:
    """Learn from past remediations"""
    
    def __init__(self, storage_backend=None):
        self.storage_backend = storage_backend
        self.success_rates = defaultdict(lambda: defaultdict(float))
        self.action_counts = defaultdict(lambda: defaultdict(int))
        self.recovery_times = defaultdict(lambda: defaultdict(list))
        self.cost_impacts = defaultdict(lambda: defaultdict(list))
        self.patterns = {}
    
    def get_success_rate(
        self,
        anomaly_type: str,
        action: str
    ) -> float:
        """Get historical success rate for action on anomaly type"""
        key = f"{anomaly_type}:{action}"
        if key in self.success_rates:
            return self.success_rates[anomaly_type][action]
        return 0.0
    
    def get_best_action(
        self,
        anomaly_type: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Get best action based on historical data"""
        # Get all actions for this anomaly type
        actions = list(self.action_counts[anomaly_type].keys())
        if not actions:
            return None
        
        # Score each action
        scored_actions = []
        for action in actions:
            success_rate = self.get_success_rate(anomaly_type, action)
            avg_recovery_time = self.get_avg_recovery_time(anomaly_type, action)
            avg_cost = self.get_avg_cost(anomaly_type, action)
            
            # Calculate score (higher is better)
            # Weight: success rate (0.5), inverse recovery time (0.3), inverse cost (0.2)
            time_score = 1.0 / (1.0 + avg_recovery_time / 300.0)  # Normalize to 5 minutes
            cost_score = 1.0 / (1.0 + avg_cost / 100.0)  # Normalize to $100
            
            score = (
                0.5 * success_rate +
                0.3 * time_score +
                0.2 * cost_score
            )
            
            scored_actions.append((action, score))
        
        if not scored_actions:
            return None
        
        # Return best action
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        return scored_actions[0][0]
    
    def record_outcome(
        self,
        remediation_id: str,
        success: bool,
        metrics: Dict[str, Any]
    ):
        """Record remediation outcome for learning"""
        anomaly_type = metrics.get("anomaly_type", "unknown")
        action = metrics.get("action", "unknown")
        strategy = metrics.get("strategy", "unknown")
        recovery_time = metrics.get("recovery_time", 0)
        cost_impact = metrics.get("cost_impact", 0.0)
        
        # Update success rate
        key = f"{anomaly_type}:{action}"
        current_count = self.action_counts[anomaly_type][action]
        current_success_rate = self.success_rates[anomaly_type][action]
        
        if success:
            new_success_rate = (
                (current_success_rate * current_count + 1.0) / (current_count + 1)
            )
        else:
            new_success_rate = (
                (current_success_rate * current_count) / (current_count + 1)
            )
        
        self.success_rates[anomaly_type][action] = new_success_rate
        self.action_counts[anomaly_type][action] = current_count + 1
        
        # Track recovery times
        self.recovery_times[anomaly_type][action].append(recovery_time)
        # Keep only last 100 records
        if len(self.recovery_times[anomaly_type][action]) > 100:
            self.recovery_times[anomaly_type][action] = \
                self.recovery_times[anomaly_type][action][-100:]
        
        # Track costs
        self.cost_impacts[anomaly_type][action].append(cost_impact)
        if len(self.cost_impacts[anomaly_type][action]) > 100:
            self.cost_impacts[anomaly_type][action] = \
                self.cost_impacts[anomaly_type][action][-100:]
        
        logger.info(
            f"Recorded outcome: {anomaly_type}:{action} - "
            f"Success: {success}, Rate: {new_success_rate:.2f}"
        )
    
    def get_avg_recovery_time(
        self,
        anomaly_type: str,
        action: str
    ) -> float:
        """Get average recovery time"""
        times = self.recovery_times[anomaly_type][action]
        if not times:
            return 300.0  # Default 5 minutes
        return sum(times) / len(times)
    
    def get_avg_cost(
        self,
        anomaly_type: str,
        action: str
    ) -> float:
        """Get average cost impact"""
        costs = self.cost_impacts[anomaly_type][action]
        if not costs:
            return 50.0  # Default cost
        return sum(costs) / len(costs)
    
    def learn_patterns(
        self,
        remediation_history: List[RemediationRecord]
    ):
        """Extract patterns from remediation history"""
        # Group by anomaly type
        by_anomaly = defaultdict(list)
        for record in remediation_history:
            by_anomaly[record.anomaly_type].append(record)
        
        # Learn patterns for each anomaly type
        for anomaly_type, records in by_anomaly.items():
            # Find most successful action
            action_success = defaultdict(lambda: {"success": 0, "total": 0})
            for record in records:
                action_success[record.action]["total"] += 1
                if record.success:
                    action_success[record.action]["success"] += 1
            
            # Calculate success rates
            best_action = None
            best_rate = 0.0
            for action, stats in action_success.items():
                rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
                if rate > best_rate:
                    best_rate = rate
                    best_action = action
            
            # Store pattern
            if best_action:
                self.patterns[anomaly_type] = {
                    "best_action": best_action,
                    "success_rate": best_rate,
                    "sample_size": len(records)
                }
    
    def detect_pattern(
        self,
        current_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect if current state matches known pattern"""
        anomaly_type = current_state.get("anomaly_type", "unknown")
        if anomaly_type in self.patterns:
            return self.patterns[anomaly_type]
        return None
    
    def suggest_action_from_pattern(
        self,
        pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest action based on pattern"""
        return {
            "action": pattern.get("best_action", "restart"),
            "confidence": pattern.get("success_rate", 0.75),
            "reasoning": f"Based on historical pattern with {pattern.get('sample_size', 0)} samples"
        }


class RemediationSuccessPredictor:
    """Predict if remediation will succeed"""
    
    def __init__(self, learner: RemediationLearningEngine):
        self.learner = learner
    
    def predict_success(
        self,
        anomaly_type: str,
        remediation_plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Return probability of success (0.0-1.0)"""
        actions = remediation_plan.get("actions", [])
        if not actions:
            return 0.5  # Default uncertainty
        
        # Get success rates for each action
        success_rates = []
        for action in actions:
            action_type = action.get("operation", "unknown")
            rate = self.learner.get_success_rate(anomaly_type, action_type)
            if rate == 0.0:
                rate = 0.7  # Default if no history
            success_rates.append(rate)
        
        # Combined success rate (product for independent actions)
        combined_rate = 1.0
        for rate in success_rates:
            combined_rate *= rate
        
        # Adjust based on context
        severity = context.get("severity", "medium")
        if severity == "critical":
            combined_rate *= 0.9  # Slightly lower for critical
        elif severity == "low":
            combined_rate *= 1.1  # Slightly higher for low severity
        
        return min(1.0, max(0.0, combined_rate))
    
    def get_confidence_intervals(
        self,
        prediction: float
    ) -> Tuple[float, float]:
        """Get confidence intervals"""
        # Simple confidence interval based on prediction value
        # Higher predictions have tighter intervals
        if prediction > 0.8:
            margin = 0.05
        elif prediction > 0.5:
            margin = 0.10
        else:
            margin = 0.15
        
        lower = max(0.0, prediction - margin)
        upper = min(1.0, prediction + margin)
        
        return (lower, upper)


class BestActionRecommender:
    """Recommend best action based on history"""
    
    def __init__(self, learner: RemediationLearningEngine):
        self.learner = learner
    
    def recommend_action(
        self,
        anomaly_type: str,
        pod_context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> RecommendedAction:
        """Recommend best action with reasoning"""
        best_action = self.learner.get_best_action(anomaly_type, pod_context)
        
        if not best_action:
            # Fallback to default
            best_action = "restart"
            confidence = 0.6
            reasoning = "No historical data available, using default action"
        else:
            success_rate = self.learner.get_success_rate(anomaly_type, best_action)
            confidence = success_rate
            reasoning = (
                f"Based on {self.learner.action_counts[anomaly_type][best_action]} "
                f"past remediations with {success_rate:.1%} success rate"
            )
        
        recovery_time = self.learner.get_avg_recovery_time(anomaly_type, best_action)
        cost = self.learner.get_avg_cost(anomaly_type, best_action)
        
        return RecommendedAction(
            action=best_action,
            strategy="balanced",
            confidence=confidence,
            reasoning=reasoning,
            expected_success_rate=confidence,
            expected_recovery_time=int(recovery_time),
            expected_cost=cost
        )
    
    def get_action_ranking(
        self,
        anomaly_type: str,
        context: Dict[str, Any]
    ) -> List[RankedAction]:
        """Get ranked list of actions"""
        actions = list(self.learner.action_counts[anomaly_type].keys())
        if not actions:
            return []
        
        ranked = []
        for idx, action in enumerate(actions):
            success_rate = self.learner.get_success_rate(anomaly_type, action)
            recovery_time = self.learner.get_avg_recovery_time(anomaly_type, action)
            cost = self.learner.get_avg_cost(anomaly_type, action)
            
            # Calculate score
            time_score = 1.0 / (1.0 + recovery_time / 300.0)
            cost_score = 1.0 / (1.0 + cost / 100.0)
            score = 0.5 * success_rate + 0.3 * time_score + 0.2 * cost_score
            
            ranked.append(RankedAction(
                action=action,
                strategy="balanced",
                score=score,
                success_rate=success_rate,
                cost=cost,
                recovery_time=int(recovery_time),
                rank=idx + 1
            ))
        
        # Sort by score
        ranked.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for idx, action in enumerate(ranked):
            action.rank = idx + 1
        
        return ranked


class RemediationAnalytics:
    """Analyze remediation performance"""
    
    def __init__(self, learner: RemediationLearningEngine):
        self.learner = learner
    
    def get_success_rate_by_type(self) -> Dict[str, float]:
        """Success rate by anomaly type"""
        rates = {}
        for anomaly_type in self.learner.action_counts.keys():
            total_success = 0
            total_count = 0
            for action in self.learner.action_counts[anomaly_type].keys():
                count = self.learner.action_counts[anomaly_type][action]
                success_rate = self.learner.success_rates[anomaly_type][action]
                total_success += success_rate * count
                total_count += count
            
            if total_count > 0:
                rates[anomaly_type] = total_success / total_count
            else:
                rates[anomaly_type] = 0.0
        
        return rates
    
    def get_avg_recovery_time(
        self,
        anomaly_type: str
    ) -> float:
        """Average recovery time"""
        all_times = []
        for action in self.learner.recovery_times[anomaly_type].keys():
            all_times.extend(self.learner.recovery_times[anomaly_type][action])
        
        if not all_times:
            return 0.0
        
        return sum(all_times) / len(all_times)
    
    def get_cost_impact_analysis(self) -> Dict[str, Any]:
        """Analyze cost impact of remediations"""
        total_cost = 0.0
        cost_by_type = {}
        cost_by_action = {}
        
        for anomaly_type in self.learner.cost_impacts.keys():
            type_total = 0.0
            for action in self.learner.cost_impacts[anomaly_type].keys():
                costs = self.learner.cost_impacts[anomaly_type][action]
                if costs:
                    action_avg = sum(costs) / len(costs)
                    type_total += action_avg
                    cost_by_action[f"{anomaly_type}:{action}"] = action_avg
            
            cost_by_type[anomaly_type] = type_total
            total_cost += type_total
        
        return {
            "total_cost": total_cost,
            "cost_by_type": cost_by_type,
            "cost_by_action": cost_by_action,
            "avg_cost_per_remediation": total_cost / max(1, len(cost_by_action))
        }
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get improvement recommendations"""
        recommendations = []
        
        # Check for low success rates
        for anomaly_type in self.learner.success_rates.keys():
            for action in self.learner.success_rates[anomaly_type].keys():
                rate = self.learner.success_rates[anomaly_type][action]
                count = self.learner.action_counts[anomaly_type][action]
                
                if rate < 0.5 and count >= 5:
                    recommendations.append({
                        "type": "low_success_rate",
                        "anomaly_type": anomaly_type,
                        "action": action,
                        "current_rate": rate,
                        "sample_size": count,
                        "recommendation": f"Consider alternative actions for {anomaly_type}, {action} has only {rate:.1%} success rate"
                    })
        
        # Check for high costs
        cost_analysis = self.get_cost_impact_analysis()
        for key, cost in cost_analysis["cost_by_action"].items():
            if cost > 200.0:  # High cost threshold
                recommendations.append({
                    "type": "high_cost",
                    "action": key,
                    "current_cost": cost,
                    "recommendation": f"Action {key} has high cost (${cost:.2f}), consider cost-optimized alternatives"
                })
        
        return recommendations



