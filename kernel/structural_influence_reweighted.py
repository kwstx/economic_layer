from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Sequence
import math

@dataclass(frozen=True)
class InfluenceProfile:
    """
    Current reweighted influence state for an agent.
    """
    agent_id: str
    trust_coefficient: float
    base_influence: float
    reweighted_influence: float
    calibration_accuracy: float
    cooperative_stability: float
    last_updated: float = field(default_factory=lambda: 0.0) # Placeholder for timestamp

class StructuralInfluenceReweighter:
    """
    Engine for implementing Structural Influence Reweighting.
    
    Dynamically recalibrates agent influence based on trust coefficients derived 
    from predictive calibration accuracy and cooperative stability.
    This ensures influence is based on real-world impact and cooperative value
    rather than static authority.
    """
    
    def __init__(
        self, 
        accuracy_weight: float = 0.65, 
        stability_weight: float = 0.35,
        amplification_exponent: float = 1.8
    ):
        """
        Initialize the reweighter with specific weighting factors.
        
        :param accuracy_weight: How much weight to give to predictive accuracy (0.0 to 1.0).
        :param stability_weight: How much weight to give to cooperative stability (0.0 to 1.0).
        :param amplification_exponent: Power factor for influence scaling (higher = more meritocratic).
        """
        self.accuracy_weight = accuracy_weight
        self.stability_weight = stability_weight
        self.amplification_exponent = amplification_exponent

    def compute_trust_coefficient(
        self, 
        calibration_accuracy: float, 
        stability_coefficient: float
    ) -> float:
        """
        Derives a trust coefficient (0.0 to 1.0) from accuracy and stability.
        
        Uses a weighted geometric-arithmetic hybrid to ensure that zero in either 
        metric significantly penalizes the trust score.
        """
        # Ensure values are bound between 0 and 1
        acc = max(0.0, min(1.0, calibration_accuracy))
        stab = max(0.0, min(1.0, stability_coefficient))
        
        # Weighted arithmetic component
        arithmetic = (self.accuracy_weight * acc) + (self.stability_weight * stab)
        
        # Geometric component (penalizes low scores in either)
        geometric = math.sqrt(acc * stab) if acc > 0 and stab > 0 else 0.0
        
        # Hybrid trust score
        return (0.7 * arithmetic) + (0.3 * geometric)

    def calculate_reweighted_influence(
        self, 
        base_influence: float, 
        trust_coefficient: float
    ) -> float:
        """
        Scales base influence by the trust coefficient using a non-linear 
        amplification curve.
        """
        # We use a meritocratic amplification: agents with high trust get 
        # disproportionately more influence, while low trust agents are dampened.
        # multiplier = (trust / mean_trust)^exponent
        # For simplicity without global mean context here, we use power scaling.
        multiplier = math.pow(trust_coefficient, self.amplification_exponent) * 2.0
        return base_influence * multiplier

    def process_signals(
        self, 
        agent_id: str, 
        base_influence: float,
        calibration_delta: float,
        stability_coefficient: float,
        current_accuracy: float = 0.5
    ) -> InfluenceProfile:
        """
        Integrates new governance signals to produce an updated influence profile.
        """
        # Update calibration accuracy based on delta (EMA-like update)
        # In a real system, this would look up historical accuracy.
        updated_accuracy = max(0.0, min(1.0, current_accuracy + calibration_delta))
        
        trust_coeff = self.compute_trust_coefficient(updated_accuracy, stability_coefficient)
        reweighted = self.calculate_reweighted_influence(base_influence, trust_coeff)
        
        return InfluenceProfile(
            agent_id=agent_id,
            trust_coefficient=trust_coeff,
            base_influence=base_influence,
            reweighted_influence=reweighted,
            calibration_accuracy=updated_accuracy,
            cooperative_stability=stability_coefficient
        )

    def apply_to_forecasting(
        self, 
        propagation_nodes: List[Dict[str, Any]], 
        profiles: Dict[str, InfluenceProfile]
    ) -> List[Dict[str, Any]]:
        """
        Adjusts how much each agent's projections affect forecasting propagation.
        Modifies 'propagation_weight' in each node.
        """
        for node in propagation_nodes:
            agent_id = node.get("agent_id")
            if agent_id and agent_id in profiles:
                profile = profiles[agent_id]
                # Apply the influence ratio (normalized to base if needed, 
                # but here we just multiply by the reweighted factor)
                current_weight = node.get("propagation_weight", 1.0)
                node["propagation_weight"] = current_weight * profile.reweighted_influence
                
                # Update metadata for traceability
                node["influence_adjusted"] = True
                node["trust_scalar"] = profile.trust_coefficient
        
        return propagation_nodes

    def apply_to_negotiation(
        self, 
        negotiation_claims: List[Dict[str, Any]], 
        profiles: Dict[str, InfluenceProfile]
    ) -> List[Dict[str, Any]]:
        """
        Adjusts negotiation convergence weights based on influence profiles.
        Modifies 'convergence_force' or 'voting_weight'.
        """
        for claim in negotiation_claims:
            agent_id = claim.get("agent_id")
            if agent_id and agent_id in profiles:
                profile = profiles[agent_id]
                
                # Influence affects how much 'weight' an agent's claim carries 
                # during the convergence calculation.
                current_power = claim.get("negotiation_power", 1.0)
                claim["negotiation_power"] = current_power * profile.reweighted_influence
                
                # Add validation flags
                claim["reweighted_by_impact"] = True
                
        return negotiation_claims
