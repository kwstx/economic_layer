from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pvariance
from typing import Dict, List, Sequence, Tuple

from cooperative_state_model import AgentSnapshot, build_cooperative_state_tensor

from .policy_transformation_engine import PolicyMutation, PolicyParameters


@dataclass(frozen=True)
class HistoricalTaskCluster:
    cluster_id: str
    snapshots: Sequence[AgentSnapshot]
    surplus_allocation: Dict[str, float]
    predictive_calibration_curve: Sequence[float]
    weight: float = 1.0


@dataclass(frozen=True)
class ReplayAggregateMetrics:
    synergy_density: float
    surplus_distribution_variance: float
    predictive_calibration_curve_score: float
    trust_variance: float
    long_horizon_rate: float
    cooperative_intelligence_score: float


@dataclass(frozen=True)
class CounterfactualEvaluation:
    baseline: ReplayAggregateMetrics
    projected: ReplayAggregateMetrics
    cooperative_intelligence_delta: float
    trust_variance_delta: float
    improves_long_term_cooperation: bool
    destabilizes_trust: bool
    should_commit: bool


def _clamp(value: float, floor: float = 0.0, ceil: float = 1.0) -> float:
    return max(floor, min(ceil, value))


def _series_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _surplus_variance(distribution: Dict[str, float]) -> float:
    if not distribution:
        return 0.0
    raw = [max(0.0, v) for v in distribution.values()]
    total = sum(raw)
    if total <= 0:
        return 0.0
    normalized = [v / total for v in raw]
    return pvariance(normalized) if len(normalized) > 1 else 0.0


def _mutate_snapshot(snapshot: AgentSnapshot, params: PolicyParameters) -> AgentSnapshot:
    adjusted_trust = _clamp(0.5 + ((snapshot.trust - 0.5) * params.trust_weight))

    temporal_gain = max(0.0, params.temporal_weighting - 1.0)
    trust_gain = max(0.0, 1.0 - params.trust_weight) * 0.03
    predictive_shift = temporal_gain * 0.02 + trust_gain
    adjusted_accuracy = tuple(
        _clamp(v + predictive_shift) for v in snapshot.predictive_accuracy
    )

    adjusted_long_horizon = tuple(
        max(0.0, v * params.temporal_weighting) for v in snapshot.long_horizon_impact
    )
    adjusted_synergy = max(0.0, snapshot.synergy_score * params.synergy_multiplier)

    return AgentSnapshot(
        agent_id=snapshot.agent_id,
        trust=adjusted_trust,
        influence=snapshot.influence,
        collaboration_partners=snapshot.collaboration_partners,
        predictive_accuracy=adjusted_accuracy,
        long_horizon_impact=adjusted_long_horizon,
        synergy_score=adjusted_synergy,
        stability_coefficient=snapshot.stability_coefficient,
    )


def _mutate_surplus_distribution(
    cluster: HistoricalTaskCluster,
    params: PolicyParameters,
) -> Dict[str, float]:
    if not cluster.surplus_allocation:
        return {}

    snapshots_by_agent = {s.agent_id: s for s in cluster.snapshots}
    adjusted: Dict[str, float] = {}
    for agent_id, share in cluster.surplus_allocation.items():
        snapshot = snapshots_by_agent.get(agent_id)
        if snapshot is None:
            adjusted[agent_id] = max(0.0, share)
            continue
        long_horizon = _series_mean(snapshot.long_horizon_impact)
        weighted_share = max(0.0, share) * (
            (1.0 + (snapshot.influence * 0.1))
            * (1.0 + (long_horizon * 0.2 * params.temporal_weighting))
            * params.synergy_multiplier
        )
        adjusted[agent_id] = weighted_share

    total = sum(adjusted.values())
    if total <= 0:
        return {k: 0.0 for k in adjusted}
    return {k: v / total for k, v in adjusted.items()}


def _mutate_calibration_curve(
    curve: Sequence[float], params: PolicyParameters
) -> Tuple[float, ...]:
    if not curve:
        return tuple()
    temporal_gain = max(0.0, params.temporal_weighting - 1.0)
    trust_gain = max(0.0, 1.0 - params.trust_weight) * 0.03
    shift = temporal_gain * 0.02 + trust_gain
    return tuple(_clamp(v + shift) for v in curve)


def _aggregate(
    tensors: Sequence,
    surplus_distributions: Sequence[Dict[str, float]],
    calibration_curves: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> ReplayAggregateMetrics:
    if not tensors:
        zero = ReplayAggregateMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return zero

    total_weight = sum(max(0.0, w) for w in weights) or 1.0
    norm_weights = [max(0.0, w) / total_weight for w in weights]

    synergy_density = sum(
        t.global_synergy_distribution * w for t, w in zip(tensors, norm_weights)
    )
    trust_variance = sum(t.trust_variance * w for t, w in zip(tensors, norm_weights))
    long_horizon_rate = sum(
        t.long_horizon_impact_accumulation_rate * w for t, w in zip(tensors, norm_weights)
    )
    surplus_distribution_variance = sum(
        _surplus_variance(d) * w for d, w in zip(surplus_distributions, norm_weights)
    )
    predictive_curve_score = sum(
        _series_mean(c) * w for c, w in zip(calibration_curves, norm_weights)
    )

    cooperative_intelligence_score = (
        (synergy_density * 0.40)
        + ((1.0 - surplus_distribution_variance) * 0.20)
        + (predictive_curve_score * 0.20)
        + (long_horizon_rate * 0.20)
    )

    return ReplayAggregateMetrics(
        synergy_density=synergy_density,
        surplus_distribution_variance=surplus_distribution_variance,
        predictive_calibration_curve_score=predictive_curve_score,
        trust_variance=trust_variance,
        long_horizon_rate=long_horizon_rate,
        cooperative_intelligence_score=cooperative_intelligence_score,
    )


class MacroCounterfactualSimulator:
    """
    Replays historical task clusters with altered policy parameters to gate
    major policy mutations before they are committed.
    """

    def __init__(
        self,
        *,
        min_cooperative_intelligence_gain: float = 0.0,
        max_trust_variance_increase: float = 0.005,
    ):
        self._min_cooperative_intelligence_gain = min_cooperative_intelligence_gain
        self._max_trust_variance_increase = max_trust_variance_increase

    def evaluate_parameter_shift(
        self,
        *,
        clusters: Sequence[HistoricalTaskCluster],
        baseline_parameters: PolicyParameters,
        candidate_parameters: PolicyParameters,
    ) -> CounterfactualEvaluation:
        baseline_tensors: List = []
        projected_tensors: List = []
        baseline_surplus: List[Dict[str, float]] = []
        projected_surplus: List[Dict[str, float]] = []
        baseline_curves: List[Sequence[float]] = []
        projected_curves: List[Sequence[float]] = []
        weights: List[float] = []

        for cluster in clusters:
            baseline_snapshots = [
                _mutate_snapshot(s, baseline_parameters) for s in cluster.snapshots
            ]
            projected_snapshots = [
                _mutate_snapshot(s, candidate_parameters) for s in cluster.snapshots
            ]
            baseline_tensors.append(build_cooperative_state_tensor(baseline_snapshots))
            projected_tensors.append(build_cooperative_state_tensor(projected_snapshots))
            baseline_surplus.append(
                _mutate_surplus_distribution(cluster=cluster, params=baseline_parameters)
            )
            projected_surplus.append(
                _mutate_surplus_distribution(cluster=cluster, params=candidate_parameters)
            )
            baseline_curves.append(
                _mutate_calibration_curve(
                    curve=cluster.predictive_calibration_curve,
                    params=baseline_parameters,
                )
            )
            projected_curves.append(
                _mutate_calibration_curve(
                    curve=cluster.predictive_calibration_curve,
                    params=candidate_parameters,
                )
            )
            weights.append(cluster.weight)

        baseline = _aggregate(
            baseline_tensors, baseline_surplus, baseline_curves, weights
        )
        projected = _aggregate(
            projected_tensors, projected_surplus, projected_curves, weights
        )

        cooperative_intelligence_delta = (
            projected.cooperative_intelligence_score
            - baseline.cooperative_intelligence_score
        )
        trust_variance_delta = projected.trust_variance - baseline.trust_variance
        improves_long_term_cooperation = (
            cooperative_intelligence_delta >= self._min_cooperative_intelligence_gain
            and projected.long_horizon_rate >= baseline.long_horizon_rate
        )
        destabilizes_trust = trust_variance_delta > self._max_trust_variance_increase
        should_commit = improves_long_term_cooperation and not destabilizes_trust

        return CounterfactualEvaluation(
            baseline=baseline,
            projected=projected,
            cooperative_intelligence_delta=cooperative_intelligence_delta,
            trust_variance_delta=trust_variance_delta,
            improves_long_term_cooperation=improves_long_term_cooperation,
            destabilizes_trust=destabilizes_trust,
            should_commit=should_commit,
        )


@dataclass(frozen=True)
class MutationCommitDecision:
    committed_parameters: PolicyParameters
    all_events: Tuple[PolicyMutation, ...]
    committed_events: Tuple[PolicyMutation, ...]
    blocked_events: Tuple[PolicyMutation, ...]
    counterfactual: CounterfactualEvaluation | None
    reason: str
