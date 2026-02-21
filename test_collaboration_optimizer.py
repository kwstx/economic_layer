import pytest
from kernel.collaboration_optimizer import (
    CollaborationTopologyOptimizer, 
    SynergySignature, 
    CooperativeIntelligenceVector
)
from kernel.adaptive_synergy_amplifier import AdaptiveSynergyAmplifier

def test_topology_optimization_logic():
    amplifier = AdaptiveSynergyAmplifier()
    optimizer = CollaborationTopologyOptimizer(amplifier, min_diversity=0.2)
    
    # Setup Agents
    # Agent Alpha & Beta: High synergy with each other, low diversity (same tags)
    # Agent Gamma: Different tags, moderate synergy
    signatures = {
        "alpha": SynergySignature("alpha", 0.8, 0.7, {"tech", "optimization"}),
        "beta": SynergySignature("beta", 0.8, 0.7, {"tech", "optimization"}),
        "gamma": SynergySignature("gamma", 0.6, 0.5, {"creative", "ethics"}),
        "delta": SynergySignature("delta", 0.5, 0.4, {"legal", "compliance"})
    }
    
    intelligence = {
        "alpha": CooperativeIntelligenceVector("alpha", 0.9, 0.85, 0.9, 0.5),
        "beta": CooperativeIntelligenceVector("beta", 0.85, 0.8, 0.85, 0.4),
        "gamma": CooperativeIntelligenceVector("gamma", 0.7, 0.75, 0.8, 0.3),
        "delta": CooperativeIntelligenceVector("delta", 0.95, 0.9, 0.95, 0.2)
    }
    
    # Test Predicted Benefit
    # Alpha + Beta cluster
    syn_ab, ben_ab, div_ab = optimizer.compute_predicted_benefit(["alpha", "beta"], signatures, intelligence)
    
    # Alpha + Gamma cluster
    syn_ag, ben_ag, div_ag = optimizer.compute_predicted_benefit(["alpha", "gamma"], signatures, intelligence)
    
    # Gamma + Delta cluster
    syn_gd, ben_gd, div_gd = optimizer.compute_predicted_benefit(["gamma", "delta"], signatures, intelligence)
    
    # Diversity should be higher for Alpha+Gamma than Alpha+Beta
    assert div_ag > div_ab
    
    # Even if Alpha+Beta have higher raw synergy, Alpha+Gamma might have better 
    # structural benefit if diversity is valued enough.
    
    # Run optimization
    clusters = optimizer.optimize_topology(
        ["alpha", "beta", "gamma", "delta"],
        signatures,
        intelligence,
        target_cluster_size=2
    )
    
    assert len(clusters) == 2
    assert len(clusters[0].agent_ids) == 2
    assert len(clusters[1].agent_ids) == 2

def test_heuristic_refinement():
    amplifier = AdaptiveSynergyAmplifier()
    optimizer = CollaborationTopologyOptimizer(amplifier)
    
    initial_synergy_weight = optimizer.heuristic_weights["synergy_weight"]
    
    # Simulate a "Black Swan" or realization where synergy was much better than predicted
    optimizer.refine_heuristics(
        predicted_benefit=0.5,
        realized_benefit=1.0,
        diversity_maintenance=0.5
    )
    
    # synergy_weight should increase
    assert optimizer.heuristic_weights["synergy_weight"] > initial_synergy_weight
    
    # If diversity maintenance is low, diversity weight should increase
    initial_div_weight = optimizer.heuristic_weights["diversity_weight"]
    optimizer.refine_heuristics(
        predicted_benefit=0.5,
        realized_benefit=0.5,
        diversity_maintenance=0.1 # Below min_diversity (0.35)
    )
    assert optimizer.heuristic_weights["diversity_weight"] > initial_div_weight
