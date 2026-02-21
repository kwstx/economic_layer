from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, List, Sequence, Tuple

from cooperative_state_model import CooperativeStateTensor

from .policy_transformation_engine import (
    GovernanceIndicators,
    PolicyMutation,
    PolicyParameters,
    PolicyTransformationEngine,
)


@dataclass(frozen=True)
class HealthThresholdConstraints:
    min_collaboration_diversity_index: float = 0.45
    max_influence_concentration_ratio: float = 0.50
    max_predictive_deviation_abs: float = 0.06
    min_long_term_impact_slope: float = 0.01


@dataclass(frozen=True)
class ConstraintViolation:
    constraint: str
    observed_value: float
    threshold: float
    direction: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class CooperativeHealthAssessment:
    is_healthy: bool
    violations: Tuple[ConstraintViolation, ...]
    corrective_parameters: PolicyParameters
    corrective_events: Tuple[PolicyMutation, ...]
    diagnostics: Dict[str, float]


class CooperativeHealthThresholdEngine:
    """
    Enforces invariant cooperative health constraints and routes corrective
    structural adjustments through the PolicyTransformationEngine.
    """

    def __init__(
        self,
        *,
        constraints: HealthThresholdConstraints | None = None,
        policy_engine: PolicyTransformationEngine | None = None,
    ):
        self._constraints = constraints or HealthThresholdConstraints()
        self._policy_engine = policy_engine or PolicyTransformationEngine()

    @property
    def constraints(self) -> HealthThresholdConstraints:
        return self._constraints

    @staticmethod
    def _influence_concentration_ratio(entropy: float) -> float:
        bounded_entropy = max(0.0, min(1.0, entropy))
        return 1.0 - bounded_entropy

    @staticmethod
    def _predictive_deviation_abs(predictive_slope: float) -> float:
        return abs(predictive_slope)

    def evaluate(
        self,
        tensor: CooperativeStateTensor,
        parameters: PolicyParameters | None = None,
    ) -> CooperativeHealthAssessment:
        constraints = self._constraints
        influence_ratio = self._influence_concentration_ratio(
            tensor.influence_concentration_entropy
        )
        predictive_deviation_abs = self._predictive_deviation_abs(
            tensor.predictive_accuracy_curve_slope
        )
        diagnostics = {
            "collaboration_diversity_index": tensor.collaboration_diversity_index,
            "influence_concentration_ratio": influence_ratio,
            "predictive_deviation_abs": predictive_deviation_abs,
            "long_term_impact_slope": tensor.long_horizon_impact_accumulation_rate,
        }

        violations: List[ConstraintViolation] = []

        if tensor.collaboration_diversity_index < constraints.min_collaboration_diversity_index:
            violations.append(
                ConstraintViolation(
                    constraint="min_collaboration_diversity_index",
                    observed_value=tensor.collaboration_diversity_index,
                    threshold=constraints.min_collaboration_diversity_index,
                    direction="min",
                )
            )

        if influence_ratio > constraints.max_influence_concentration_ratio:
            violations.append(
                ConstraintViolation(
                    constraint="max_influence_concentration_ratio",
                    observed_value=influence_ratio,
                    threshold=constraints.max_influence_concentration_ratio,
                    direction="max",
                )
            )

        if predictive_deviation_abs > constraints.max_predictive_deviation_abs:
            violations.append(
                ConstraintViolation(
                    constraint="max_predictive_deviation_abs",
                    observed_value=predictive_deviation_abs,
                    threshold=constraints.max_predictive_deviation_abs,
                    direction="max",
                )
            )

        if tensor.long_horizon_impact_accumulation_rate < constraints.min_long_term_impact_slope:
            violations.append(
                ConstraintViolation(
                    constraint="min_long_term_impact_slope",
                    observed_value=tensor.long_horizon_impact_accumulation_rate,
                    threshold=constraints.min_long_term_impact_slope,
                    direction="min",
                )
            )

        if not violations:
            return CooperativeHealthAssessment(
                is_healthy=True,
                violations=tuple(),
                corrective_parameters=parameters or PolicyParameters(),
                corrective_events=tuple(),
                diagnostics=diagnostics,
            )

        indicators = GovernanceIndicators.from_tensor(tensor)
        corrective_parameters, corrective_events = self._policy_engine.apply(
            indicators=indicators,
            parameters=parameters,
        )
        return CooperativeHealthAssessment(
            is_healthy=False,
            violations=tuple(violations),
            corrective_parameters=corrective_parameters,
            corrective_events=tuple(corrective_events),
            diagnostics=diagnostics,
        )
