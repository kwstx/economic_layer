import pytest
import math
from kernel.structural_influence_reweighted import (
    StructuralInfluenceReweighter,
    InfluenceProfile
)

def test_trust_coefficient_derivation():
    reweighter = StructuralInfluenceReweighter(accuracy_weight=0.6, stability_weight=0.4)
    
    # High accuracy, high stability
    trust = reweighter.compute_trust_coefficient(0.9, 0.9)
    assert trust >= 0.85
    
    # Low accuracy, high stability
    trust_low_acc = reweighter.compute_trust_coefficient(0.2, 0.9)
    # Arithmetic: 0.6*0.2 + 0.4*0.9 = 0.12 + 0.36 = 0.48
    # Geometric: sqrt(0.18) = 0.42
    # Hybrid: 0.7*0.48 + 0.3*0.42 = 0.336 + 0.126 = 0.462
    assert 0.4 <= trust_low_acc <= 0.5
    
    # Zero case
    assert reweighter.compute_trust_coefficient(0, 0.9) < 0.4 # Geometric zero penalizes it

def test_influence_reweighting_curve():
    reweighter = StructuralInfluenceReweighter(amplification_exponent=2.0)
    base_influence = 0.5
    
    # High trust (0.9) -> Significant boost
    high_trust_influence = reweighter.calculate_reweighted_influence(base_influence, 0.9)
    # 0.5 * (0.9^2 * 2.0) = 0.5 * 0.81 * 2.0 = 0.81
    assert high_trust_influence > base_influence
    
    # Low trust (0.3) -> Significant dampening
    low_trust_influence = reweighter.calculate_reweighted_influence(base_influence, 0.3)
    # 0.5 * (0.3^2 * 2.0) = 0.5 * 0.09 * 2.0 = 0.09
    assert low_trust_influence < base_influence

def test_forecasting_adjustment():
    reweighter = StructuralInfluenceReweighter()
    profiles = {
        "agent_gold": InfluenceProfile("agent_gold", 0.9, 1.0, 1.5, 0.9, 0.9),
        "agent_unreliable": InfluenceProfile("agent_unreliable", 0.2, 1.0, 0.1, 0.2, 0.2)
    }
    
    nodes = [
        {"agent_id": "agent_gold", "propagation_weight": 1.0},
        {"agent_id": "agent_unreliable", "propagation_weight": 1.0},
        {"agent_id": "agent_unknown", "propagation_weight": 1.0}
    ]
    
    adjusted = reweighter.apply_to_forecasting(nodes, profiles)
    
    assert adjusted[0]["propagation_weight"] == 1.5
    assert adjusted[1]["propagation_weight"] == 0.1
    assert adjusted[2]["propagation_weight"] == 1.0 # Unchanged
    assert adjusted[0]["influence_adjusted"] is True

def test_negotiation_adjustment():
    reweighter = StructuralInfluenceReweighter()
    profiles = {
        "agent_a": InfluenceProfile("agent_a", 0.8, 0.5, 1.2, 0.8, 0.8)
    }
    
    claims = [
        {"agent_id": "agent_a", "negotiation_power": 10.0}
    ]
    
    reweighted_claims = reweighter.apply_to_negotiation(claims, profiles)
    
    assert reweighted_claims[0]["negotiation_power"] == 12.0
    assert reweighted_claims[0]["reweighted_by_impact"] is True
