from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Sequence
import math
from models.governance_signal import GovernanceSignal

@dataclass(frozen=True)
class DriftAssessment:
    """
    Result of a behavioral drift analysis for a specific agent.
    """
    agent_id: str
    drift_score: float  # 0.0 to 1.0, where 1.0 is maximum gaming behavior detected
    is_gaming_detected: bool
    trust_damping: float  # Scalar (0.1 to 1.0) to reduce trust/influence
    inflation_ratio: float  # Ratio of projected vs realized impact magnitude
    synergy_distortion: float # Measure of outlier synergy relative to realized outcome
    penalized_metrics: List[str] # Specific metric dimensions that seem gamed

class BehavioralDriftDetector:
    """
    Analyzes whether agents begin optimizing toward short-term metric inflation 
    rather than genuine downstream impact.
    
    This module compares:
    1. Projected impact vectors vs realized outcome vectors over time.
    2. Synergy density reports vs realized amplification.
    
    If drift/gaming is detected, it computes dampening factors and identifies 
    which metrics require weight adjustments.
    """
    
    def __init__(
        self, 
        window_size: int = 15, 
        gaming_threshold: float = 0.45,
        min_samples: int = 5
    ):
        self.window_size = window_size
        self.gaming_threshold = gaming_threshold
        self.min_samples = min_samples
        
        # agent_id -> { "projections": deque(impact_vectors), "outcomes": deque(impact_vectors), "synergy": deque(float) }
        self._agent_histories: Dict[str, Dict[str, deque]] = {}

    def _get_history(self, agent_id: str) -> Dict[str, deque]:
        if agent_id not in self._agent_histories:
            self._agent_histories[agent_id] = {
                "projections": deque(maxlen=self.window_size),
                "outcomes": deque(maxlen=self.window_size),
                "synergy": deque(maxlen=self.window_size)
            }
        return self._agent_histories[agent_id]

    def ingest_signal(self, signal: GovernanceSignal):
        """
        Incorporates a new signal into the behavioral history of an agent.
        """
        history = self._get_history(signal.agent_id)
        
        if signal.source_operation == "retrieve_forecast":
            # Optimization: only store if it has significant impact data
            if signal.impact_vector:
                history["projections"].append(signal.impact_vector)
        
        elif signal.source_operation == "submit_outcome":
            if signal.impact_vector:
                history["outcomes"].append(signal.impact_vector)
        
        elif signal.source_operation == "query_synergy_density":
            history["synergy"].append(signal.synergy_density)

    def _vec_magnitude(self, vector: Dict[str, float]) -> float:
        if not vector:
            return 0.0
        return math.sqrt(sum(v**2 for v in vector.values()))

    def _get_trend_slope(self, values: List[float]) -> float:
        if len(values) < 3:
            return 0.0
        # Simple linear slope estimate
        x = list(range(len(values)))
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_xx = sum(xi * xi for xi in x)
        
        denominator = (n * sum_xx - sum_x * sum_x)
        if abs(denominator) < 1e-9:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denominator

    def analyze_drift(self, agent_id: str) -> DriftAssessment:
        """
        Performs comparative analysis on the agent's historical behavior to 
        detect systematic gaming or drift.
        """
        history = self._get_history(agent_id)
        
        projections = list(history["projections"])
        outcomes = list(history["outcomes"])
        synergy_scores = list(history["synergy"])
        
        if len(projections) < self.min_samples or len(outcomes) < self.min_samples:
            return DriftAssessment(agent_id, 0.0, False, 1.0, 1.0, 0.0, [])

        # 1. Analyze Metric Inflation (Magnitude)
        proj_mags = [self._vec_magnitude(v) for v in projections]
        out_mags = [self._vec_magnitude(v) for v in outcomes]
        
        avg_proj_mag = sum(proj_mags) / len(proj_mags)
        avg_out_mag = sum(out_mags) / len(out_mags)
        inflation_ratio = avg_proj_mag / (avg_out_mag + 1e-4)

        # 2. Analyze Trend Divergence
        # Gaming is often characterized by increasing claims while results stagnate or drop.
        proj_slope = self._get_trend_slope(proj_mags)
        out_slope = self._get_trend_slope(out_mags)
        
        # If projections are increasing but outcomes are not keeping pace (divergence)
        divergence = max(0.0, proj_slope - out_slope) if proj_slope > 0 else 0.0
        
        # 3. Analyze Synergy Distortion
        avg_reported_synergy = sum(synergy_scores) / len(synergy_scores) if synergy_scores else 1.0
        synergy_distortion = 0.0
        if avg_reported_synergy > 1.1:
            synergy_distortion = (avg_reported_synergy - 1.0) * max(0.0, inflation_ratio - 1.0)

        # 4. Compute Drift Score
        # Drift score combines inflation magnitude, trend divergence, and synergy distortion.
        inflation_component = min(1.0, max(0.0, (inflation_ratio - 1.2) / 1.5))
        divergence_component = min(1.0, divergence * 2.5) # Sensitive to slope differences
        distortion_component = min(1.0, synergy_distortion / 1.0)
        
        drift_score = (0.5 * inflation_component) + (0.3 * divergence_component) + (0.2 * distortion_component)
        
        is_gaming = drift_score >= self.gaming_threshold
        
        # 5. Corrective Damping
        if is_gaming:
            trust_damping = 1.0 / (1.0 + (drift_score * 6.0))
        else:
            trust_damping = 1.0 / (1.0 + (drift_score * 0.4))
            
        trust_damping = max(0.12, min(1.0, trust_damping))
        
        # 5. Identify Gamed Metrics
        # Look for specific dimensions where avg(projection) >> avg(outcome)
        penalized = []
        all_metric_keys = set()
        for v in projections: all_metric_keys.update(v.keys())
        
        for key in all_metric_keys:
            m_proj = sum(v.get(key, 0.0) for v in projections) / len(projections)
            m_out = sum(v.get(key, 0.0) for v in outcomes) / len(outcomes)
            # If projection is > 80% higher than reality, flag it
            if m_proj > (m_out * 1.8) + 0.05:
                penalized.append(key)

        return DriftAssessment(
            agent_id=agent_id,
            drift_score=drift_score,
            is_gaming_detected=is_gaming,
            trust_damping=trust_damping,
            inflation_ratio=inflation_ratio,
            synergy_distortion=synergy_distortion,
            penalized_metrics=penalized
        )

    def apply_corrections(self, assessment: DriftAssessment, base_trust: float) -> float:
        """
        Applies the trust damping to a base trust score.
        """
        return base_trust * assessment.trust_damping

    def apply_metric_penalties(self, assessment: DriftAssessment, impact_vector: Dict[str, float]) -> Dict[str, float]:
        """
        Reduces the weight of specific metrics identified as being gamed.
        """
        if not assessment.penalized_metrics:
            return impact_vector
            
        adjusted = impact_vector.copy()
        for metric in assessment.penalized_metrics:
            if metric in adjusted:
                # Apply a 70% penalty to the gamed metric
                adjusted[metric] *= 0.3
                
        return adjusted
