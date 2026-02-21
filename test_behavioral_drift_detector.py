import pytest
from kernel.behavioral_drift_detector import BehavioralDriftDetector, DriftAssessment
from models.governance_signal import GovernanceSignal
from datetime import datetime

def test_good_actor_drift():
    detector = BehavioralDriftDetector(window_size=10, gaming_threshold=0.4, min_samples=3)
    agent_id = "agent_trustworthy"
    
    # Simulate accurate projections and outcomes
    for i in range(5):
        # Forecast
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            impact_vector={"quality": 0.8, "throughput": 1.2},
            source_operation="retrieve_forecast"
        ))
        # Outcome (matched)
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            impact_vector={"quality": 0.85, "throughput": 1.15},
            source_operation="submit_outcome"
        ))
        # Synergy
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            synergy_density=1.05,
            source_operation="query_synergy_density"
        ))
    
    assessment = detector.analyze_drift(agent_id)
    assert not assessment.is_gaming_detected
    assert assessment.drift_score < 0.2
    assert assessment.trust_damping > 0.9

def test_gamer_inflation_drift():
    # Lower threshold for testing
    detector = BehavioralDriftDetector(window_size=10, gaming_threshold=0.4, min_samples=3)
    agent_id = "agent_gamer"
    
    # Simulate inflated projections vs mediocre outcomes
    for i in range(5):
        # High Forecast
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            impact_vector={"quality": 5.0, "throughput": 10.0},
            source_operation="retrieve_forecast"
        ))
        # Low Outcome
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            impact_vector={"quality": 0.5, "throughput": 1.0},
            source_operation="submit_outcome"
        ))
    
    assessment = detector.analyze_drift(agent_id)
    assert assessment.is_gaming_detected
    assert assessment.inflation_ratio > 5.0
    assert assessment.trust_damping < 0.5
    assert "quality" in assessment.penalized_metrics
    assert "throughput" in assessment.penalized_metrics

def test_synergy_gaming_drift():
    detector = BehavioralDriftDetector(window_size=10, gaming_threshold=0.3, min_samples=3)
    agent_id = "agent_synergy_gamer"
    
    # Reports high synergy but outcomes don't match projections
    for i in range(5):
        # Forecast
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            impact_vector={"collaboration_value": 2.0},
            source_operation="retrieve_forecast"
        ))
        # Outcome
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            impact_vector={"collaboration_value": 0.5},
            source_operation="submit_outcome"
        ))
        # High Synergy Density Report
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            synergy_density=3.5, # Massive synergy claimed
            source_operation="query_synergy_density"
        ))
    
    assessment = detector.analyze_drift(agent_id)
    assert assessment.is_gaming_detected
    assert assessment.synergy_distortion > 1.0
    assert assessment.trust_damping < 0.6

def test_insufficient_data():
    detector = BehavioralDriftDetector(min_samples=10)
    agent_id = "new_agent"
    
    detector.ingest_signal(GovernanceSignal(
        agent_id=agent_id,
        impact_vector={"x": 1.0},
        source_operation="retrieve_forecast"
    ))
    
    assessment = detector.analyze_drift(agent_id)
    assert not assessment.is_gaming_detected
    assert assessment.trust_damping == 1.0

def test_metric_penalties():
    detector = BehavioralDriftDetector()
    assessment = DriftAssessment(
        agent_id="test",
        drift_score=0.8,
        is_gaming_detected=True,
        trust_damping=0.2,
        inflation_ratio=5.0,
        synergy_distortion=2.0,
        penalized_metrics=["throughput"]
    )
    
    vector = {"throughput": 100.0, "quality": 1.0}
    adjusted = detector.apply_metric_penalties(assessment, vector)
    
    assert adjusted["throughput"] == 30.0
    assert adjusted["quality"] == 1.0

def test_trend_divergence_drift():
    detector = BehavioralDriftDetector(window_size=10, gaming_threshold=0.3, min_samples=3)
    agent_id = "agent_divergent"
    
    # Projections are increasing, outcomes are decreasing
    for i in range(5):
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            impact_vector={"value": 1.0 + i}, # Increasing: 1, 2, 3, 4, 5
            source_operation="retrieve_forecast"
        ))
        detector.ingest_signal(GovernanceSignal(
            agent_id=agent_id,
            impact_vector={"value": 1.0 - (i * 0.1)}, # Decreasing: 1.0, 0.9, 0.8, 0.7, 0.6
            source_operation="submit_outcome"
        ))
    
    assessment = detector.analyze_drift(agent_id)
    assert assessment.is_gaming_detected
    assert assessment.drift_score > 0.4
    assert assessment.trust_damping < 0.5
