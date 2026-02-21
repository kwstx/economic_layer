from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


PatternValue = str | int | float | bool
PatternKey = Tuple[Tuple[str, PatternValue], ...]


@dataclass(frozen=True)
class ExponentAdjustment:
    pattern_key: PatternKey
    previous_exponent: float
    new_exponent: float
    residual_ema: float
    observations: int
    direction: str


@dataclass
class _PatternState:
    exponent: float
    residual_ema: float = 0.0
    observations: int = 0


class AdaptiveSynergyAmplifier:
    """
    Learns a pattern-specific exponent for a non-linear synergy scaling function.
    """

    def __init__(
        self,
        *,
        base_exponent: float = 1.0,
        learning_rate: float = 0.15,
        ema_alpha: float = 0.35,
        min_observations: int = 3,
        residual_tolerance: float = 0.01,
        min_exponent: float = 0.5,
        max_exponent: float = 3.0,
    ):
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1].")
        if min_observations < 1:
            raise ValueError("min_observations must be >= 1.")
        if min_exponent > max_exponent:
            raise ValueError("min_exponent cannot exceed max_exponent.")
        if not (min_exponent <= base_exponent <= max_exponent):
            raise ValueError("base_exponent must be within exponent bounds.")

        self._base_exponent = base_exponent
        self._learning_rate = learning_rate
        self._ema_alpha = ema_alpha
        self._min_observations = min_observations
        self._residual_tolerance = residual_tolerance
        self._min_exponent = min_exponent
        self._max_exponent = max_exponent
        self._states: Dict[PatternKey, _PatternState] = {}

    @staticmethod
    def _to_pattern_key(pattern_signature: Mapping[str, PatternValue] | None) -> PatternKey:
        if not pattern_signature:
            return tuple()
        return tuple(sorted(pattern_signature.items(), key=lambda kv: kv[0]))

    def exponent_for(self, pattern_signature: Mapping[str, PatternValue] | None = None) -> float:
        key = self._to_pattern_key(pattern_signature)
        state = self._states.get(key)
        return state.exponent if state else self._base_exponent

    def scale_synergy(
        self,
        *,
        base_synergy: float,
        structural_signal: float,
        pattern_signature: Mapping[str, PatternValue] | None = None,
    ) -> float:
        signal = max(0.0, structural_signal)
        exponent = self.exponent_for(pattern_signature)
        return base_synergy * ((1.0 + signal) ** exponent)

    def adapt_exponent(
        self,
        *,
        predicted_amplification: float,
        observed_amplification: float,
        pattern_signature: Mapping[str, PatternValue] | None = None,
    ) -> ExponentAdjustment:
        key = self._to_pattern_key(pattern_signature)
        state = self._states.setdefault(key, _PatternState(exponent=self._base_exponent))

        residual = observed_amplification - predicted_amplification
        state.observations += 1
        state.residual_ema = (
            (1.0 - self._ema_alpha) * state.residual_ema + self._ema_alpha * residual
        )

        previous = state.exponent
        direction = "hold"

        if (
            state.observations >= self._min_observations
            and abs(state.residual_ema) > self._residual_tolerance
        ):
            delta = self._learning_rate * abs(state.residual_ema)
            if state.residual_ema > 0:
                state.exponent = min(self._max_exponent, state.exponent + delta)
                direction = "increase"
            else:
                state.exponent = max(self._min_exponent, state.exponent - delta)
                direction = "decrease"

        return ExponentAdjustment(
            pattern_key=key,
            previous_exponent=previous,
            new_exponent=state.exponent,
            residual_ema=state.residual_ema,
            observations=state.observations,
            direction=direction,
        )
