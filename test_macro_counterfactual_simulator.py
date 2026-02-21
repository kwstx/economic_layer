from cooperative_state_model import AgentSnapshot
from kernel.macro_counterfactual_simulator import (
    HistoricalTaskCluster,
    MacroCounterfactualSimulator,
)
from kernel.policy_transformation_engine import (
    GovernanceIndicators,
    PolicyParameters,
    PolicyTransformationEngine,
)


def _clusters():
    cluster_a = HistoricalTaskCluster(
        cluster_id="cluster_a",
        snapshots=[
            AgentSnapshot(
                agent_id="a1",
                trust=0.62,
                influence=0.30,
                collaboration_partners=["a2", "a3"],
                predictive_accuracy=[0.56, 0.60, 0.64],
                long_horizon_impact=[0.12, 0.15, 0.18],
                synergy_score=0.58,
            ),
            AgentSnapshot(
                agent_id="a2",
                trust=0.66,
                influence=0.28,
                collaboration_partners=["a1", "a3"],
                predictive_accuracy=[0.58, 0.61, 0.65],
                long_horizon_impact=[0.13, 0.16, 0.20],
                synergy_score=0.60,
            ),
            AgentSnapshot(
                agent_id="a3",
                trust=0.59,
                influence=0.24,
                collaboration_partners=["a1", "a2"],
                predictive_accuracy=[0.54, 0.58, 0.61],
                long_horizon_impact=[0.10, 0.13, 0.17],
                synergy_score=0.57,
            ),
        ],
        surplus_allocation={"a1": 0.40, "a2": 0.35, "a3": 0.25},
        predictive_calibration_curve=[0.57, 0.60, 0.63],
        weight=1.0,
    )
    cluster_b = HistoricalTaskCluster(
        cluster_id="cluster_b",
        snapshots=[
            AgentSnapshot(
                agent_id="b1",
                trust=0.64,
                influence=0.34,
                collaboration_partners=["b2"],
                predictive_accuracy=[0.55, 0.59, 0.62],
                long_horizon_impact=[0.11, 0.15, 0.19],
                synergy_score=0.59,
            ),
            AgentSnapshot(
                agent_id="b2",
                trust=0.61,
                influence=0.22,
                collaboration_partners=["b1"],
                predictive_accuracy=[0.56, 0.58, 0.61],
                long_horizon_impact=[0.09, 0.13, 0.16],
                synergy_score=0.56,
            ),
        ],
        surplus_allocation={"b1": 0.55, "b2": 0.45},
        predictive_calibration_curve=[0.56, 0.59, 0.61],
        weight=0.8,
    )
    return [cluster_a, cluster_b]


def test_macro_counterfactual_allows_commit_on_long_term_gain_without_trust_instability():
    simulator = MacroCounterfactualSimulator(
        min_cooperative_intelligence_gain=0.0005,
        max_trust_variance_increase=0.01,
    )

    baseline = PolicyParameters()
    candidate = PolicyParameters(
        synergy_multiplier=1.10,
        trust_weight=0.92,
        temporal_weighting=1.15,
    )

    evaluation = simulator.evaluate_parameter_shift(
        clusters=_clusters(),
        baseline_parameters=baseline,
        candidate_parameters=candidate,
    )

    assert evaluation.improves_long_term_cooperation
    assert not evaluation.destabilizes_trust
    assert evaluation.should_commit
    assert evaluation.cooperative_intelligence_delta > 0


def test_macro_counterfactual_blocks_when_trust_variance_destabilizes():
    simulator = MacroCounterfactualSimulator(
        min_cooperative_intelligence_gain=0.0,
        max_trust_variance_increase=0.001,
    )

    baseline = PolicyParameters()
    candidate = PolicyParameters(
        synergy_multiplier=1.08,
        trust_weight=1.45,
        temporal_weighting=1.08,
    )

    evaluation = simulator.evaluate_parameter_shift(
        clusters=_clusters(),
        baseline_parameters=baseline,
        candidate_parameters=candidate,
    )

    assert evaluation.destabilizes_trust
    assert not evaluation.should_commit
    assert evaluation.trust_variance_delta > 0


def test_policy_transformation_engine_gates_major_mutations_with_macro_counterfactual():
    engine = PolicyTransformationEngine()
    indicators = GovernanceIndicators(
        cross_role_integration_depth=0.30,
        influence_concentration_entropy=0.70,
        long_term_impact_accumulation_rate=0.001,
    )
    simulator = MacroCounterfactualSimulator(
        min_cooperative_intelligence_gain=0.5,
        max_trust_variance_increase=0.01,
    )

    decision = engine.apply_with_counterfactual_gate(
        indicators,
        historical_clusters=_clusters(),
        simulator=simulator,
        parameters=PolicyParameters(),
    )

    assert decision.committed_events == ()
    assert len(decision.blocked_events) == 3
    assert decision.committed_parameters == PolicyParameters()
    assert decision.counterfactual is not None
