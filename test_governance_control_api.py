import pytest
from datetime import datetime, UTC
from kernel.governance_control_api import GovernanceControlAPI
from kernel.policy_transformation_engine import PolicyTransformationEngine, PolicyMutation, PolicyParameters
from kernel.macro_counterfactual_simulator import MacroCounterfactualSimulator, HistoricalTaskCluster
from kernel.structural_influence_reweighted import StructuralInfluenceReweighter
from cooperative_state_model import AgentSnapshot

@pytest.fixture
def api():
    engine = PolicyTransformationEngine()
    simulator = MacroCounterfactualSimulator()
    reweighter = StructuralInfluenceReweighter()
    return GovernanceControlAPI(engine, simulator, reweighter)

def test_query_state_tensors(api):
    snapshots = [
        AgentSnapshot(
            agent_id="agent_1",
            trust=0.9,
            influence=0.33,
            collaboration_partners=["agent_2", "agent_3"],
            predictive_accuracy=(0.8, 0.85, 0.9),
            long_horizon_impact=(0.2, 0.3, 0.4),
            synergy_score=0.7
        ),
        AgentSnapshot(
            agent_id="agent_2",
            trust=0.85,
            influence=0.33,
            collaboration_partners=["agent_1", "agent_3"],
            predictive_accuracy=(0.75, 0.8, 0.82),
            long_horizon_impact=(0.15, 0.25, 0.35),
            synergy_score=0.65
        ),
        AgentSnapshot(
            agent_id="agent_3",
            trust=0.88,
            influence=0.34,
            collaboration_partners=["agent_1", "agent_2"],
            predictive_accuracy=(0.78, 0.82, 0.88),
            long_horizon_impact=(0.18, 0.28, 0.38),
            synergy_score=0.68
        )
    ]
    
    result = api.query_state_tensors(snapshots)
    
    assert "state_tensor" in result
    assert "regime_assessment" in result
    assert "interpretation" in result
    assert result["regime_assessment"]["current_regime"] == "healthy"
    assert result["state_tensor"]["global_synergy_distribution"] > 0

def test_mutation_history(api):
    mutation = PolicyMutation(
        event_type="PolicyMutation",
        rule_id="test_rule",
        rule_version="1.0.0",
        parameter="synergy_multiplier",
        previous_value=1.0,
        new_value=1.1,
        rationale="Increasing synergy for testing",
        indicators={"test_metric": 0.5},
        timestamp=datetime.now(UTC)
    )
    
    api.record_mutation(mutation)
    history = api.retrieve_mutation_history()
    
    assert len(history) == 1
    assert history[0]["parameter"] == "synergy_multiplier"
    assert history[0]["new_value"] == 1.1

def test_simulate_parameter_shift(api):
    snapshots = [
        AgentSnapshot("a1", 0.5, 1.0, [], (0.5,), (0.5,), 0.5),
        AgentSnapshot("a2", 0.5, 1.0, [], (0.5,), (0.5,), 0.5),
    ]
    clusters = [
        HistoricalTaskCluster(
            cluster_id="c1",
            snapshots=snapshots,
            surplus_allocation={"a1": 0.5, "a2": 0.5},
            predictive_calibration_curve=(0.5, 0.5)
        )
    ]
    baseline = PolicyParameters(synergy_multiplier=1.0)
    candidate = PolicyParameters(synergy_multiplier=1.2)
    
    result = api.simulate_parameter_shift(clusters, baseline, candidate)
    
    assert "evaluation_result" in result
    assert "recommendation" in result
    assert "causal_explanation" in result
    assert isinstance(result["causal_explanation"], str)

def test_inspect_structural_influence(api):
    agents_data = [
        {
            "agent_id": "agent_x",
            "base_influence": 1.0,
            "calibration_delta": 0.05,
            "stability_coefficient": 0.9,
            "current_accuracy": 0.8
        }
    ]
    
    profiles = api.inspect_structural_influence(agents_data)
    
    assert len(profiles) == 1
    assert profiles[0]["agent_id"] == "agent_x"
    assert "reweighted_influence" in profiles[0]
    assert "explanation" in profiles[0]
    assert "Agent agent_x" in profiles[0]["explanation"]
