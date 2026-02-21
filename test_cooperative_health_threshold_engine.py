from cooperative_state_model import CooperativeStateTensor
from kernel.cooperative_health_threshold_engine import (
    CooperativeHealthThresholdEngine,
    HealthThresholdConstraints,
)
from kernel.policy_transformation_engine import PolicyParameters


def _healthy_tensor() -> CooperativeStateTensor:
    return CooperativeStateTensor(
        global_synergy_distribution=0.72,
        trust_variance=0.01,
        influence_concentration_entropy=0.80,
        collaboration_diversity_index=0.60,
        predictive_accuracy_curve_slope=0.02,
        long_horizon_impact_accumulation_rate=0.03,
    )


def test_engine_reports_healthy_when_all_constraints_satisfied():
    engine = CooperativeHealthThresholdEngine(
        constraints=HealthThresholdConstraints(
            min_collaboration_diversity_index=0.45,
            max_influence_concentration_ratio=0.50,
            max_predictive_deviation_abs=0.06,
            min_long_term_impact_slope=0.01,
        )
    )

    result = engine.evaluate(_healthy_tensor(), PolicyParameters())

    assert result.is_healthy
    assert result.violations == ()
    assert result.corrective_events == ()
    assert result.corrective_parameters == PolicyParameters()


def test_engine_flags_each_invariant_violation():
    engine = CooperativeHealthThresholdEngine(
        constraints=HealthThresholdConstraints(
            min_collaboration_diversity_index=0.45,
            max_influence_concentration_ratio=0.40,
            max_predictive_deviation_abs=0.03,
            min_long_term_impact_slope=0.02,
        )
    )
    tensor = CooperativeStateTensor(
        global_synergy_distribution=0.50,
        trust_variance=0.03,
        influence_concentration_entropy=0.30,
        collaboration_diversity_index=0.20,
        predictive_accuracy_curve_slope=-0.07,
        long_horizon_impact_accumulation_rate=-0.01,
    )

    result = engine.evaluate(tensor, PolicyParameters())
    violated_names = {violation.constraint for violation in result.violations}

    assert not result.is_healthy
    assert violated_names == {
        "min_collaboration_diversity_index",
        "max_influence_concentration_ratio",
        "max_predictive_deviation_abs",
        "min_long_term_impact_slope",
    }


def test_engine_triggers_policy_transformation_when_constraints_breach():
    engine = CooperativeHealthThresholdEngine()
    tensor = CooperativeStateTensor(
        global_synergy_distribution=0.55,
        trust_variance=0.02,
        influence_concentration_entropy=0.70,
        collaboration_diversity_index=0.30,
        predictive_accuracy_curve_slope=0.00,
        long_horizon_impact_accumulation_rate=0.001,
    )

    result = engine.evaluate(
        tensor,
        PolicyParameters(
            synergy_multiplier=1.0,
            trust_weight=1.0,
            temporal_weighting=1.0,
        ),
    )

    assert not result.is_healthy
    assert result.corrective_parameters.synergy_multiplier == 1.15
    assert result.corrective_parameters.trust_weight == 0.9
    assert result.corrective_parameters.temporal_weighting == 1.2
    assert len(result.corrective_events) == 3
