from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import log
from statistics import mean, pvariance
from typing import Dict, List, Sequence


class EcosystemRegime(str, Enum):
    HEALTHY = "healthy"
    UNSTABLE = "unstable"
    CONCENTRATED = "concentrated"
    SHORT_TERM_BIAS = "drifting_short_term_bias"


@dataclass(frozen=True)
class AgentSnapshot:
    agent_id: str
    trust: float
    influence: float
    collaboration_partners: Sequence[str]
    predictive_accuracy: Sequence[float]
    long_horizon_impact: Sequence[float]
    synergy_score: float


@dataclass(frozen=True)
class CooperativeStateTensor:
    global_synergy_distribution: float
    trust_variance: float
    influence_concentration_entropy: float
    collaboration_diversity_index: float
    predictive_accuracy_curve_slope: float
    long_horizon_impact_accumulation_rate: float

    def as_vector(self) -> List[float]:
        return [
            self.global_synergy_distribution,
            self.trust_variance,
            self.influence_concentration_entropy,
            self.collaboration_diversity_index,
            self.predictive_accuracy_curve_slope,
            self.long_horizon_impact_accumulation_rate,
        ]


@dataclass(frozen=True)
class ControlAssessment:
    regime: EcosystemRegime
    tensor: CooperativeStateTensor
    diagnostics: Dict[str, float]


def _safe_entropy(weights: Sequence[float]) -> float:
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return 0.0
    probs = [max(0.0, w) / total for w in weights]
    values = [(-p * log(p)) for p in probs if p > 0]
    return sum(values)


def _average_pairwise_jaccard(items: Sequence[Sequence[str]]) -> float:
    sets = [set(x) for x in items]
    if len(sets) < 2:
        return 1.0
    pair_scores: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                pair_scores.append(1.0)
                continue
            pair_scores.append(len(sets[i] & sets[j]) / len(union))
    return mean(pair_scores) if pair_scores else 1.0


def _global_series_slope(series_bundle: Sequence[Sequence[float]]) -> float:
    slopes: List[float] = []
    for series in series_bundle:
        if len(series) < 2:
            slopes.append(0.0)
            continue
        slopes.append((series[-1] - series[0]) / (len(series) - 1))
    return mean(slopes) if slopes else 0.0


def build_cooperative_state_tensor(snapshots: Sequence[AgentSnapshot]) -> CooperativeStateTensor:
    if not snapshots:
        return CooperativeStateTensor(
            global_synergy_distribution=0.0,
            trust_variance=0.0,
            influence_concentration_entropy=0.0,
            collaboration_diversity_index=0.0,
            predictive_accuracy_curve_slope=0.0,
            long_horizon_impact_accumulation_rate=0.0,
        )

    synergy = mean(s.synergy_score for s in snapshots)
    trust_var = pvariance(s.trust for s in snapshots) if len(snapshots) > 1 else 0.0
    influence_entropy = _safe_entropy([s.influence for s in snapshots])

    avg_jaccard = _average_pairwise_jaccard([s.collaboration_partners for s in snapshots])
    diversity_index = 1.0 - avg_jaccard

    accuracy_slope = _global_series_slope([s.predictive_accuracy for s in snapshots])
    long_horizon_rate = _global_series_slope([s.long_horizon_impact for s in snapshots])

    return CooperativeStateTensor(
        global_synergy_distribution=synergy,
        trust_variance=trust_var,
        influence_concentration_entropy=influence_entropy,
        collaboration_diversity_index=diversity_index,
        predictive_accuracy_curve_slope=accuracy_slope,
        long_horizon_impact_accumulation_rate=long_horizon_rate,
    )


def assess_ecosystem_state(
    tensor: CooperativeStateTensor,
    *,
    min_synergy: float = 0.55,
    max_trust_variance: float = 0.08,
    min_entropy: float = 0.85,
    min_diversity: float = 0.45,
    min_accuracy_slope: float = 0.0,
    min_long_horizon_rate: float = 0.0,
) -> ControlAssessment:
    diagnostics = {
        "synergy": tensor.global_synergy_distribution,
        "trust_variance": tensor.trust_variance,
        "influence_entropy": tensor.influence_concentration_entropy,
        "collaboration_diversity": tensor.collaboration_diversity_index,
        "accuracy_slope": tensor.predictive_accuracy_curve_slope,
        "long_horizon_rate": tensor.long_horizon_impact_accumulation_rate,
    }

    if tensor.influence_concentration_entropy < min_entropy:
        return ControlAssessment(
            regime=EcosystemRegime.CONCENTRATED,
            tensor=tensor,
            diagnostics=diagnostics,
        )
    if tensor.long_horizon_impact_accumulation_rate < min_long_horizon_rate:
        return ControlAssessment(
            regime=EcosystemRegime.SHORT_TERM_BIAS,
            tensor=tensor,
            diagnostics=diagnostics,
        )
    unstable = (
        tensor.trust_variance > max_trust_variance
        or tensor.collaboration_diversity_index < min_diversity
        or tensor.predictive_accuracy_curve_slope < min_accuracy_slope
    )
    if unstable:
        return ControlAssessment(
            regime=EcosystemRegime.UNSTABLE,
            tensor=tensor,
            diagnostics=diagnostics,
        )
    healthy = tensor.global_synergy_distribution >= min_synergy
    return ControlAssessment(
        regime=EcosystemRegime.HEALTHY if healthy else EcosystemRegime.UNSTABLE,
        tensor=tensor,
        diagnostics=diagnostics,
    )
