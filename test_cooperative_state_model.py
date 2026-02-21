from cooperative_state_model import (
    AgentSnapshot,
    EcosystemRegime,
    assess_ecosystem_state,
    build_cooperative_state_tensor,
)
from kernel.policy_transformation_engine import (
    GovernanceIndicators,
    PolicyParameters,
    PolicyTransformationEngine,
)
from kernel.adaptive_synergy_amplifier import AdaptiveSynergyAmplifier


def _sample_snapshots():
    return [
        AgentSnapshot(
            agent_id="a1",
            trust=0.82,
            influence=0.30,
            collaboration_partners=["a2", "a3"],
            predictive_accuracy=[0.62, 0.66, 0.70],
            long_horizon_impact=[0.10, 0.16, 0.22],
            synergy_score=0.72,
            stability_coefficient=0.85,
        ),
        AgentSnapshot(
            agent_id="a2",
            trust=0.79,
            influence=0.28,
            collaboration_partners=["a1", "a3"],
            predictive_accuracy=[0.60, 0.64, 0.67],
            long_horizon_impact=[0.09, 0.14, 0.20],
            synergy_score=0.69,
            stability_coefficient=0.82,
        ),
        AgentSnapshot(
            agent_id="a3",
            trust=0.81,
            influence=0.24,
            collaboration_partners=["a1", "a2", "a4"],
            predictive_accuracy=[0.58, 0.63, 0.68],
            long_horizon_impact=[0.08, 0.15, 0.21],
            synergy_score=0.71,
            stability_coefficient=0.88,
        ),
        AgentSnapshot(
            agent_id="a4",
            trust=0.80,
            influence=0.18,
            collaboration_partners=["a3"],
            predictive_accuracy=[0.57, 0.60, 0.64],
            long_horizon_impact=[0.06, 0.11, 0.18],
            synergy_score=0.68,
            stability_coefficient=0.75,
        ),
    ]


def test_tensor_dimensions_present():
    tensor = build_cooperative_state_tensor(_sample_snapshots())
    vector = tensor.as_vector()
    assert len(vector) == 6
    assert tensor.global_synergy_distribution > 0
    assert tensor.influence_concentration_entropy > 0
    assert tensor.collaboration_diversity_index >= 0


def test_healthy_regime_on_balanced_inputs():
    tensor = build_cooperative_state_tensor(_sample_snapshots())
    assessment = assess_ecosystem_state(
        tensor,
        min_synergy=0.65,
        max_trust_variance=0.03,
        min_entropy=1.0,
        min_diversity=0.25,
        min_accuracy_slope=0.0,
        min_long_horizon_rate=0.01,
    )
    assert assessment.regime == EcosystemRegime.HEALTHY


def test_short_term_bias_when_long_horizon_negative():
    snapshots = _sample_snapshots()
    modified = [
        AgentSnapshot(
            agent_id=s.agent_id,
            trust=s.trust,
            influence=s.influence,
            collaboration_partners=s.collaboration_partners,
            predictive_accuracy=s.predictive_accuracy,
            long_horizon_impact=[0.2, 0.15, 0.1],
            synergy_score=s.synergy_score,
        )
        for s in snapshots
    ]
    tensor = build_cooperative_state_tensor(modified)
    assessment = assess_ecosystem_state(tensor, min_long_horizon_rate=0.0)
    assert assessment.regime == EcosystemRegime.SHORT_TERM_BIAS


def test_policy_transformation_engine_applies_all_default_rules():
    engine = PolicyTransformationEngine()
    indicators = GovernanceIndicators(
        cross_role_integration_depth=0.30,
        influence_concentration_entropy=0.70,
        long_term_impact_accumulation_rate=0.001,
    )
    initial = PolicyParameters(
        synergy_multiplier=1.0,
        trust_weight=1.0,
        temporal_weighting=1.0,
    )

    updated, events = engine.apply(indicators, initial)

    assert updated.synergy_multiplier == 1.15
    assert updated.trust_weight == 0.9
    assert updated.temporal_weighting == 1.2
    assert len(events) == 3
    assert all(event.event_type == "PolicyMutation" for event in events)
    assert all(event.rule_version == "1.0.0" for event in events)


def test_policy_transformation_engine_emits_no_mutation_when_thresholds_not_crossed():
    engine = PolicyTransformationEngine()
    indicators = GovernanceIndicators(
        cross_role_integration_depth=0.65,
        influence_concentration_entropy=1.20,
        long_term_impact_accumulation_rate=0.04,
    )

    updated, events = engine.apply(indicators, PolicyParameters())

    assert updated == PolicyParameters()
    assert events == []


def test_policy_transformation_engine_can_map_from_tensor():
    snapshots = _sample_snapshots()
    tensor = build_cooperative_state_tensor(snapshots)
    indicators = GovernanceIndicators.from_tensor(tensor)

    assert indicators.cross_role_integration_depth == tensor.collaboration_diversity_index
    assert indicators.influence_concentration_entropy == tensor.influence_concentration_entropy
    assert (
        indicators.long_term_impact_accumulation_rate
        == tensor.long_horizon_impact_accumulation_rate
    )


def test_adaptive_synergy_amplifier_increases_exponent_when_underpredicting_persistently():
    amplifier = AdaptiveSynergyAmplifier(
        base_exponent=1.0,
        learning_rate=0.5,
        ema_alpha=1.0,
        min_observations=2,
        residual_tolerance=0.01,
    )
    pattern = {"cluster_size": 4, "topology": "mesh"}

    first = amplifier.adapt_exponent(
        predicted_amplification=1.1,
        observed_amplification=1.5,
        pattern_signature=pattern,
    )
    second = amplifier.adapt_exponent(
        predicted_amplification=1.2,
        observed_amplification=1.6,
        pattern_signature=pattern,
    )

    assert first.direction == "hold"
    assert second.direction == "increase"
    assert second.new_exponent > 1.0


def test_adaptive_synergy_amplifier_decreases_exponent_when_overpredicting_persistently():
    amplifier = AdaptiveSynergyAmplifier(
        base_exponent=1.2,
        learning_rate=0.5,
        ema_alpha=1.0,
        min_observations=2,
        residual_tolerance=0.01,
    )
    pattern = {"cluster_size": 4, "topology": "star"}

    amplifier.adapt_exponent(
        predicted_amplification=1.6,
        observed_amplification=1.2,
        pattern_signature=pattern,
    )
    adjustment = amplifier.adapt_exponent(
        predicted_amplification=1.5,
        observed_amplification=1.1,
        pattern_signature=pattern,
    )

    assert adjustment.direction == "decrease"
    assert adjustment.new_exponent < 1.2


def test_adaptive_synergy_scaling_uses_pattern_specific_exponent():
    amplifier = AdaptiveSynergyAmplifier(
        base_exponent=1.0,
        learning_rate=1.0,
        ema_alpha=1.0,
        min_observations=1,
        residual_tolerance=0.0,
    )
    pattern = {"cluster_size": 5, "topology": "ring"}

    baseline = amplifier.scale_synergy(
        base_synergy=0.8,
        structural_signal=0.5,
        pattern_signature=pattern,
    )
    amplifier.adapt_exponent(
        predicted_amplification=1.0,
        observed_amplification=1.4,
        pattern_signature=pattern,
    )
    adapted = amplifier.scale_synergy(
        base_synergy=0.8,
        structural_signal=0.5,
        pattern_signature=pattern,
    )

    assert adapted > baseline
