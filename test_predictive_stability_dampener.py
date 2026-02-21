import pytest
import math
import random
from kernel.predictive_stability_dampener import PredictiveStabilityDampener
from kernel.adaptive_synergy_amplifier import AdaptiveSynergyAmplifier
from kernel.economic_gradient_controller import EconomicGradientController, EconomicAgentSignal
from kernel.structural_influence_reweighted import StructuralInfluenceReweighter

def test_dampener_stable_state():
    dampener = PredictiveStabilityDampener(window_size=10)
    # Record flat metrics
    for _ in range(10):
        dampener.record_metrics(1.0, 0.5, 1.2)
    
    report = dampener.get_stability_report()
    for name, descriptor in report.items():
        assert descriptor.volatility < 0.1
        assert descriptor.oscillation_index == 0.0
        assert descriptor.dampening_factor == 1.0
        assert not descriptor.is_unstable

def test_dampener_detects_oscillation():
    dampener = PredictiveStabilityDampener(window_size=10, oscillation_threshold=0.4)
    # Record oscillatory reward
    for i in range(10):
        val = 1.0 if i % 2 == 0 else 0.5
        dampener.record_metrics(val, 0.5, 1.0)
    
    report = dampener.get_stability_report()
    reward_desc = report["reward_allocation"]
    # 9 deltas, sign changes at every step except the first few (8 sign changes)
    assert reward_desc.oscillation_index > 0.8
    assert reward_desc.is_unstable
    assert reward_desc.dampening_factor < 0.5

def test_dampener_detects_volatility():
    dampener = PredictiveStabilityDampener(window_size=10, volatility_threshold=0.3)
    # Record volatile synergy
    random.seed(42)
    for _ in range(10):
        # High random variance
        dampener.record_metrics(1.0, 0.5, random.uniform(0.1, 5.0))
    
    report = dampener.get_stability_report()
    synergy_desc = report["synergy_scaling"]
    assert synergy_desc.volatility > 0.3
    assert synergy_desc.is_unstable
    assert synergy_desc.dampening_factor < 1.0

def test_aggregate_dampening_takes_minimum():
    dampener = PredictiveStabilityDampener(window_size=10)
    # One is unstable, others are stable
    for i in range(10):
        reward = 1.0 if i % 2 == 0 else 0.2 # Oscillatory
        dampener.record_metrics(reward, 0.5, 1.0) # Others stable
    
    report = dampener.get_stability_report()
    factor = dampener.get_aggregate_dampening()
    
    assert factor == report["reward_allocation"].dampening_factor
    assert factor < 0.5
    assert report["trust_weighting"].dampening_factor == 1.0

def test_dampener_handles_near_zero_values():
    dampener = PredictiveStabilityDampener(window_size=5)
    for _ in range(5):
        dampener.record_metrics(0.0, 0.0, 0.0)
    
    report = dampener.get_stability_report()
    for d in report.values():
        assert d.volatility == 0.0
        assert d.dampening_factor == 1.0

def test_dampening_reduces_synergy_exponent_delta():
    amplifier = AdaptiveSynergyAmplifier(
        learning_rate=0.5, 
        min_observations=1, 
        residual_tolerance=0.0,
        ema_alpha=1.0 # Ensure immediate effect
    )
    pattern = {"test": 1}
    
    # Without dampening
    adj_no_damp = amplifier.adapt_exponent(
        predicted_amplification=1.0, 
        observed_amplification=1.4, 
        pattern_signature=pattern,
        stability_dampening=1.0
    )
    
    # New instance for comparison
    amplifier_damped = AdaptiveSynergyAmplifier(
        learning_rate=0.5, 
        min_observations=1, 
        residual_tolerance=0.0,
        ema_alpha=1.0
    )
    adj_damped = amplifier_damped.adapt_exponent(
        predicted_amplification=1.0, 
        observed_amplification=1.4, 
        pattern_signature=pattern,
        stability_dampening=0.2
    )
    
    delta_no_damp = adj_no_damp.new_exponent - adj_no_damp.previous_exponent
    delta_damped = adj_damped.new_exponent - adj_damped.previous_exponent
    
    assert delta_no_damp > 0
    assert delta_damped < delta_no_damp
    assert math.isclose(delta_damped, delta_no_damp * 0.2)

def test_dampening_reduces_economic_gradient_pressure():
    # Use a low threshold to ensure pressure is triggered
    controller = EconomicGradientController(
        diminishing_return_strength=0.5,
        concentration_ratio_threshold=1.5 
    )
    signals = [
        EconomicAgentSignal("a", marginal_influence=10.0, surplus_share=0.5, delayed_impact_score=0.5),
        EconomicAgentSignal("b", marginal_influence=0.1, surplus_share=0.5, delayed_impact_score=0.5)
    ]
    
    outcome_no_damp = controller.evaluate(signals, stability_dampening=1.0)
    outcome_damped = controller.evaluate(signals, stability_dampening=0.5)
    
    # Diminishing return modifier is 1 - delta. We want to see smaller delta when damped.
    reduction_no_damp = 1.0 - outcome_no_damp.modifiers["a"].diminishing_return_modifier
    reduction_damped = 1.0 - outcome_damped.modifiers["a"].diminishing_return_modifier
    
    assert reduction_no_damp > 0
    assert reduction_damped < reduction_no_damp
    assert math.isclose(reduction_damped, reduction_no_damp * 0.5)

def test_dampening_reduces_influence_recalibration():
    reweighter = StructuralInfluenceReweighter()
    
    profile_no_damp = reweighter.process_signals(
        agent_id="a",
        base_influence=1.0,
        calibration_delta=0.2,
        stability_coefficient=0.8,
        current_accuracy=0.5,
        stability_dampening=1.0
    )
    
    profile_damped = reweighter.process_signals(
        agent_id="a",
        base_influence=1.0,
        calibration_delta=0.2,
        stability_coefficient=0.8,
        current_accuracy=0.5,
        stability_dampening=0.5
    )
    
    # Accuracy should be less increased in damped version
    assert profile_damped.calibration_accuracy < profile_no_damp.calibration_accuracy
    assert math.isclose(profile_damped.calibration_accuracy, 0.5 + (0.2 * 0.5))
