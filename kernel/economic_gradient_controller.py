from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pvariance
from typing import Dict, Sequence


@dataclass(frozen=True)
class EconomicAgentSignal:
    agent_id: str
    marginal_influence: float
    surplus_share: float
    delayed_impact_score: float


@dataclass(frozen=True)
class AgentGradientModifier:
    diminishing_return_modifier: float
    temporal_compensation_weight: float
    adjusted_marginal_influence: float


@dataclass(frozen=True)
class EconomicGradientOutcome:
    modifiers: Dict[str, AgentGradientModifier]
    diagnostics: Dict[str, float]


class EconomicGradientController:
    """
    Applies fairness-preserving gradient controls to cooperative influence:
    1) dampen concentrated marginal influence with diminishing returns
    2) compensate temporally when delayed impact is undervalued
    """

    def __init__(
        self,
        *,
        surplus_variance_target: float = 0.04,
        concentration_ratio_threshold: float = 2.25,
        diminishing_return_strength: float = 0.35,
        minimum_diminishing_modifier: float = 0.55,
        temporal_compensation_rate: float = 0.5,
        max_temporal_compensation_boost: float = 0.6,
    ):
        if surplus_variance_target < 0:
            raise ValueError("surplus_variance_target must be >= 0.")
        if concentration_ratio_threshold <= 0:
            raise ValueError("concentration_ratio_threshold must be > 0.")
        if not 0 <= diminishing_return_strength <= 1:
            raise ValueError("diminishing_return_strength must be in [0, 1].")
        if not 0 < minimum_diminishing_modifier <= 1:
            raise ValueError("minimum_diminishing_modifier must be in (0, 1].")
        if temporal_compensation_rate < 0:
            raise ValueError("temporal_compensation_rate must be >= 0.")
        if max_temporal_compensation_boost < 0:
            raise ValueError("max_temporal_compensation_boost must be >= 0.")

        self._surplus_variance_target = surplus_variance_target
        self._concentration_ratio_threshold = concentration_ratio_threshold
        self._diminishing_return_strength = diminishing_return_strength
        self._minimum_diminishing_modifier = minimum_diminishing_modifier
        self._temporal_compensation_rate = temporal_compensation_rate
        self._max_temporal_compensation_boost = max_temporal_compensation_boost

    def evaluate(
        self, 
        agent_signals: Sequence[EconomicAgentSignal],
        stability_dampening: float = 1.0
    ) -> EconomicGradientOutcome:
        if not agent_signals:
            return EconomicGradientOutcome(modifiers={}, diagnostics={})

        influences = [max(0.0, s.marginal_influence) for s in agent_signals]
        surplus = [max(0.0, s.surplus_share) for s in agent_signals]
        delayed = [max(0.0, s.delayed_impact_score) for s in agent_signals]

        mean_influence = mean(influences)
        max_influence = max(influences)
        concentration_ratio = (
            (max_influence / mean_influence) if mean_influence > 0 else 0.0
        )
        surplus_variance = pvariance(surplus) if len(surplus) > 1 else 0.0

        concentration_pressure = max(
            0.0,
            (concentration_ratio - self._concentration_ratio_threshold)
            / self._concentration_ratio_threshold,
        )
        variance_pressure = (
            max(0.0, surplus_variance - self._surplus_variance_target)
            / (self._surplus_variance_target + 1e-9)
            if self._surplus_variance_target > 0
            else float(surplus_variance > 0)
        )
        fairness_pressure = min(1.0, (0.6 * concentration_pressure) + (0.4 * variance_pressure))
        mean_delayed_impact = mean(delayed)

        modifiers: Dict[str, AgentGradientModifier] = {}
        for signal in agent_signals:
            relative_influence = (
                max(0.0, signal.marginal_influence) / mean_influence
                if mean_influence > 0
                else 0.0
            )
            influence_excess = max(0.0, relative_influence - 1.0)
            diminishing = 1.0 - (
                fairness_pressure
                * (self._diminishing_return_strength * stability_dampening)
                * (influence_excess / (1.0 + influence_excess))
            )
            diminishing = max(self._minimum_diminishing_modifier, diminishing)

            undervaluation_gap = max(0.0, signal.delayed_impact_score - signal.surplus_share)
            high_impact = signal.delayed_impact_score >= mean_delayed_impact
            compensation_boost = (
                min(
                    self._max_temporal_compensation_boost,
                    self._temporal_compensation_rate * undervaluation_gap,
                )
                if high_impact
                else 0.0
            )
            temporal_weight = 1.0 + compensation_boost

            modifiers[signal.agent_id] = AgentGradientModifier(
                diminishing_return_modifier=diminishing,
                temporal_compensation_weight=temporal_weight,
                adjusted_marginal_influence=max(0.0, signal.marginal_influence) * diminishing,
            )

        diagnostics = {
            "surplus_variance": surplus_variance,
            "concentration_ratio": concentration_ratio,
            "fairness_pressure": fairness_pressure,
            "mean_delayed_impact": mean_delayed_impact,
        }
        return EconomicGradientOutcome(modifiers=modifiers, diagnostics=diagnostics)
