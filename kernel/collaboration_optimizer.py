from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Sequence, Tuple, Set
import math
from kernel.adaptive_synergy_amplifier import AdaptiveSynergyAmplifier

@dataclass(frozen=True)
class SynergySignature:
    """
    Captures the unique synergy profile of an agent or a pair.
    """
    agent_id: str
    synergy_density: float
    preferred_collaboration_density: float
    context_tags: Set[str] = field(default_factory=set)

@dataclass(frozen=True)
class CooperativeIntelligenceVector:
    """
    Multi-dimensional representation of an agent's cooperative performance.
    """
    agent_id: str
    trust_coefficient: float
    predictive_accuracy: float
    cooperative_stability: float
    structural_influence: float

@dataclass(frozen=True)
class CollaborationCluster:
    """
    A proposed team composition with predicted metrics.
    """
    agent_ids: List[str]
    predicted_synergy: float
    predicted_structural_benefit: float
    diversity_index: float
    cluster_metadata: Dict[str, Any] = field(default_factory=dict)

class CollaborationTopologyOptimizer:
    """
    Engine for optimizing multi-agent collaboration structures.
    
    Dynamically biases task formation toward high-density synergy clusters 
    while enforcing diversity constraints to prevent over-centralization.
    """
    
    def __init__(
        self,
        synergy_amplifier: AdaptiveSynergyAmplifier,
        min_diversity: float = 0.35,
        centralization_ceiling: float = 0.25,
        refinement_lr: float = 0.1
    ):
        self.synergy_amplifier = synergy_amplifier
        self.min_diversity = min_diversity
        self.centralization_ceiling = centralization_ceiling
        self.refinement_lr = refinement_lr
        
        # Internal heuristic weights that are refined over time
        self.heuristic_weights = {
            "synergy_weight": 0.7,
            "diversity_weight": 0.3,
            "stability_weight": 0.5
        }

    def compute_predicted_benefit(
        self,
        candidate_agents: List[str],
        signatures: Dict[str, SynergySignature],
        intelligence_vectors: Dict[str, CooperativeIntelligenceVector]
    ) -> Tuple[float, float, float]:
        """
        Computes predicted synergy, structural benefit, and diversity for a group.
        """
        if not candidate_agents:
            return 0.0, 0.0, 0.0
            
        # 1. Compute Predicted Synergy
        # Sum of individual synergy densities amplified by group connectivity
        base_synergy = sum(signatures[a].synergy_density for a in candidate_agents if a in signatures) / len(candidate_agents)
        
        # Use the amplifier for non-linear scale based on group size and intelligence
        avg_trust = sum(intelligence_vectors[a].trust_coefficient for a in candidate_agents if a in intelligence_vectors) / len(candidate_agents)
        
        predicted_synergy = self.synergy_amplifier.scale_synergy(
            base_synergy=base_synergy,
            structural_signal=avg_trust,
            pattern_signature={"group_size": len(candidate_agents)}
        )
        
        # 2. Compute Diversity Index (1 - Jaccard overlap of context tags)
        all_tags = []
        for a in candidate_agents:
            if a in signatures:
                all_tags.append(signatures[a].context_tags)
        
        if len(all_tags) < 2:
            diversity = 1.0
        else:
            # Simple diversity measure: ratio of unique tags to total potential tags
            unique_tags = set().union(*all_tags)
            total_tags_count = sum(len(t) for t in all_tags)
            diversity = len(unique_tags) / max(1, total_tags_count)
            
        # 3. Structural Benefit
        # Benefit = (Synergy * weights) + (Diversity * weights) - (Centralization Penalty)
        # Centralization is proxied by the variance of influence in the cluster
        influences = [intelligence_vectors[a].structural_influence for a in candidate_agents if a in intelligence_vectors]
        if len(influences) > 1:
            mean_inf = sum(influences) / len(influences)
            variance = sum((x - mean_inf) ** 2 for x in influences) / len(influences)
            centralization_penalty = variance * 2.0 # Scale factor
        else:
            centralization_penalty = 0.0
            
        benefit = (
            (predicted_synergy * self.heuristic_weights["synergy_weight"]) +
            (diversity * self.heuristic_weights["diversity_weight"]) -
            (centralization_penalty * self.centralization_ceiling)
        )
        
        return predicted_synergy, benefit, diversity

    def optimize_topology(
        self,
        available_agents: List[str],
        signatures: Dict[str, SynergySignature],
        intelligence_vectors: Dict[str, CooperativeIntelligenceVector],
        target_cluster_size: int = 3
    ) -> List[CollaborationCluster]:
        """
        Assembles agents into high-synergy clusters while respecting diversity.
        Uses a greedy-randomized search with bias toward high benefit.
        """
        clusters = []
        remaining = set(available_agents)
        
        while len(remaining) >= target_cluster_size:
            # Build a cluster
            best_cluster = None
            max_benefit = -float('inf')
            
            # Simple heuristic: pick a seed agent and find best partners
            seed = list(remaining)[0]
            current_cluster = [seed]
            
            potential_partners = list(remaining - {seed})
            
            # Iteratively add partners that maximize benefit
            while len(current_cluster) < target_cluster_size and potential_partners:
                best_partner = None
                best_addition_benefit = -float('inf')
                
                for p in potential_partners[:10]: # Check a subset for performance
                    test_group = current_cluster + [p]
                    _, benefit, div = self.compute_predicted_benefit(test_group, signatures, intelligence_vectors)
                    
                    # Apply diversity constraint
                    if div < self.min_diversity:
                        benefit -= 1.0 # Significant penalty for violating diversity
                        
                    if benefit > best_addition_benefit:
                        best_addition_benefit = benefit
                        best_partner = p
                
                if best_partner:
                    current_cluster.append(best_partner)
                    potential_partners.remove(best_partner)
                else:
                    break
            
            # Finalize cluster
            syn, ben, div = self.compute_predicted_benefit(current_cluster, signatures, intelligence_vectors)
            clusters.append(CollaborationCluster(
                agent_ids=current_cluster,
                predicted_synergy=syn,
                predicted_structural_benefit=ben,
                diversity_index=div
            ))
            
            # Remove from available
            for a in current_cluster:
                remaining.discard(a)
                
        return clusters

    def refine_heuristics(
        self,
        predicted_benefit: float,
        realized_benefit: float,
        diversity_maintenance: float
    ):
        """
        Continuously refines clustering weights using feedback from realized outcomes.
        """
        error = realized_benefit - predicted_benefit
        
        # Adjust synergy weight based on prediction error
        self.heuristic_weights["synergy_weight"] += error * self.refinement_lr
        
        # Adjust diversity weight based on how much it's being maintained
        # If diversity is consistently low, increase its weight
        if diversity_maintenance < self.min_diversity:
            self.heuristic_weights["diversity_weight"] += self.refinement_lr * 0.5
            
        # Keep weights within sane bounds
        for k in self.heuristic_weights:
            self.heuristic_weights[k] = max(0.1, min(2.0, self.heuristic_weights[k]))
