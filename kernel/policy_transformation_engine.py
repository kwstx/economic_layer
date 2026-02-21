from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence, Tuple

from cooperative_state_model import CooperativeStateTensor

if TYPE_CHECKING:
    from .macro_counterfactual_simulator import (
        HistoricalTaskCluster,
        MacroCounterfactualSimulator,
        MutationCommitDecision,
    )


@dataclass(frozen=True)
class PolicyParameters:
    synergy_multiplier: float = 1.0
    trust_weight: float = 1.0
    temporal_weighting: float = 1.0


@dataclass(frozen=True)
class GovernanceIndicators:
    cross_role_integration_depth: float
    influence_concentration_entropy: float
    long_term_impact_accumulation_rate: float

    @classmethod
    def from_tensor(cls, tensor: CooperativeStateTensor) -> "GovernanceIndicators":
        return cls(
            cross_role_integration_depth=tensor.collaboration_diversity_index,
            influence_concentration_entropy=tensor.influence_concentration_entropy,
            long_term_impact_accumulation_rate=tensor.long_horizon_impact_accumulation_rate,
        )


@dataclass(frozen=True)
class PolicyMutation:
    event_type: str
    rule_id: str
    rule_version: str
    parameter: str
    previous_value: float
    new_value: float
    rationale: str
    indicators: Dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class PolicyTransformationRule:
    rule_id: str
    version: str
    description: str
    condition: Callable[[GovernanceIndicators], bool]
    transform: Callable[[PolicyParameters], Tuple[str, float]]
    indicator_snapshot: Callable[[GovernanceIndicators], Dict[str, float]]


def _default_rules() -> List[PolicyTransformationRule]:
    return [
        PolicyTransformationRule(
            rule_id="cross_role_integration_decline_synergy_boost",
            version="1.0.0",
            description="Increase synergy multipliers when cross-role integration depth declines.",
            condition=lambda i: i.cross_role_integration_depth < 0.45,
            transform=lambda p: (
                "synergy_multiplier",
                round(p.synergy_multiplier * 1.15, 6),
            ),
            indicator_snapshot=lambda i: {
                "cross_role_integration_depth": i.cross_role_integration_depth,
            },
        ),
        PolicyTransformationRule(
            rule_id="influence_entropy_floor_trust_dampening",
            version="1.0.0",
            description="Dampen trust weights when influence concentration entropy is below threshold.",
            condition=lambda i: i.influence_concentration_entropy < 0.85,
            transform=lambda p: (
                "trust_weight",
                round(p.trust_weight * 0.9, 6),
            ),
            indicator_snapshot=lambda i: {
                "influence_concentration_entropy": i.influence_concentration_entropy,
            },
        ),
        PolicyTransformationRule(
            rule_id="long_term_slowdown_temporal_amplification",
            version="1.0.0",
            description="Amplify temporal weighting when long-term impact accumulation slows.",
            condition=lambda i: i.long_term_impact_accumulation_rate < 0.01,
            transform=lambda p: (
                "temporal_weighting",
                round(p.temporal_weighting * 1.2, 6),
            ),
            indicator_snapshot=lambda i: {
                "long_term_impact_accumulation_rate": i.long_term_impact_accumulation_rate,
            },
        ),
    ]


class PolicyTransformationEngine:
    """
    Rule-driven, versioned policy transformer for mapping governance indicators
    into parameter shifts.
    """

    def __init__(self, rules: Sequence[PolicyTransformationRule] | None = None):
        self._rules: List[PolicyTransformationRule] = list(rules or _default_rules())

    @property
    def rules(self) -> Sequence[PolicyTransformationRule]:
        return tuple(self._rules)

    def apply(
        self,
        indicators: GovernanceIndicators,
        parameters: PolicyParameters | None = None,
    ) -> Tuple[PolicyParameters, List[PolicyMutation]]:
        current = parameters or PolicyParameters()
        events: List[PolicyMutation] = []

        for rule in self._rules:
            if not rule.condition(indicators):
                continue

            parameter_name, new_value = rule.transform(current)
            previous_value = getattr(current, parameter_name)
            if previous_value == new_value:
                continue

            current = replace(current, **{parameter_name: new_value})
            events.append(
                PolicyMutation(
                    event_type="PolicyMutation",
                    rule_id=rule.rule_id,
                    rule_version=rule.version,
                    parameter=parameter_name,
                    previous_value=previous_value,
                    new_value=new_value,
                    rationale=rule.description,
                    indicators=rule.indicator_snapshot(indicators),
                )
            )

        return current, events

    def apply_with_counterfactual_gate(
        self,
        indicators: GovernanceIndicators,
        *,
        historical_clusters: Sequence["HistoricalTaskCluster"],
        simulator: "MacroCounterfactualSimulator",
        parameters: PolicyParameters | None = None,
        major_shift_threshold: float = 0.10,
    ) -> "MutationCommitDecision":
        from .macro_counterfactual_simulator import MutationCommitDecision

        baseline_parameters = parameters or PolicyParameters()
        candidate_parameters, events = self.apply(indicators, baseline_parameters)
        if not events:
            return MutationCommitDecision(
                committed_parameters=baseline_parameters,
                all_events=tuple(),
                committed_events=tuple(),
                blocked_events=tuple(),
                counterfactual=None,
                reason="No policy mutation generated.",
            )

        major_events = [
            event
            for event in events
            if abs(event.new_value - event.previous_value)
            / max(abs(event.previous_value), 1e-9)
            >= major_shift_threshold
        ]
        if not major_events:
            return MutationCommitDecision(
                committed_parameters=candidate_parameters,
                all_events=tuple(events),
                committed_events=tuple(events),
                blocked_events=tuple(),
                counterfactual=None,
                reason="No major mutation detected. Candidate committed.",
            )

        counterfactual = simulator.evaluate_parameter_shift(
            clusters=historical_clusters,
            baseline_parameters=baseline_parameters,
            candidate_parameters=candidate_parameters,
        )
        if counterfactual.should_commit:
            return MutationCommitDecision(
                committed_parameters=candidate_parameters,
                all_events=tuple(events),
                committed_events=tuple(events),
                blocked_events=tuple(),
                counterfactual=counterfactual,
                reason="Counterfactual replay improved long-term cooperation and maintained trust stability.",
            )

        return MutationCommitDecision(
            committed_parameters=baseline_parameters,
            all_events=tuple(events),
            committed_events=tuple(),
            blocked_events=tuple(events),
            counterfactual=counterfactual,
            reason="Major mutation blocked by macro counterfactual trust/cooperation gate.",
        )
