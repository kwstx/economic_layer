from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

@dataclass(frozen=True)
class StabilityDescriptor:
    volatility: float
    oscillation_index: float
    dampening_factor: float
    is_unstable: bool

class PredictiveStabilityDampener:
    """
    Subsystem that detects oscillatory behavior in reward allocation, 
    trust weighting, or synergy scaling by analyzing time-series volatility.
    
    If instability exceeds defined thresholds, it automatically computes 
    dampening factors to reduce adjustment amplitude and prevent runaway 
    feedback loops. This makes the system cybernetically stable.
    """
    
    def __init__(
        self,
        window_size: int = 15,
        volatility_threshold: float = 0.8,
        oscillation_threshold: float = 0.5
    ):
        self.window_size = window_size
        self.volatility_threshold = volatility_threshold
        self.oscillation_threshold = oscillation_threshold
        
        # History buffers for system-level metrics
        self._history: Dict[str, deque[float]] = {
            "reward_allocation": deque(maxlen=window_size),
            "trust_weighting": deque(maxlen=window_size),
            "synergy_scaling": deque(maxlen=window_size)
        }

    def record_metrics(
        self, 
        reward_metric: float, 
        trust_metric: float, 
        synergy_metric: float
    ):
        """
        Records the current system-wide values for monitored metrics.
        
        Typically:
        - reward_metric: variance of surplus distribution
        - trust_metric: mean trust coefficient or reliability recalibration delta
        - synergy_metric: mean synergy scaling exponent
        """
        self._history["reward_allocation"].append(reward_metric)
        self._history["trust_weighting"].append(trust_metric)
        self._history["synergy_scaling"].append(synergy_metric)

    def _analyze_series(self, series: deque[float]) -> StabilityDescriptor:
        if len(series) < 4:
            return StabilityDescriptor(0.0, 0.0, 1.0, False)
        
        data = list(series)
        deltas = [data[i] - data[i-1] for i in range(1, len(data))]
        
        # 1. Volatility: standard deviation of deltas normalized by absolute mean of data
        abs_mean = sum(abs(x) for x in data) / len(data)
        if abs_mean < 1e-9:
            volatility = 0.0
        else:
            delta_mean = sum(deltas) / len(deltas)
            # Use population variance for small windows
            variance = sum((d - delta_mean)**2 for d in deltas) / len(deltas)
            volatility = math.sqrt(variance) / abs_mean
            
        # 2. Oscillation Index: Frequency of sign changes in deltas
        # High frequency of sign changes indicates rapid back-and-forth oscillations.
        sign_changes = 0
        for i in range(1, len(deltas)):
            # Multiply consecutive deltas: negative result means sign changed
            if deltas[i] * deltas[i-1] < 0:
                sign_changes += 1
        
        oscillation_index = sign_changes / (len(deltas) - 1) if len(deltas) > 1 else 0.0
        
        # Determine dampening factor
        # We use a combined penalty for both volatility and oscillation.
        # If either exceeds threshold, penalty starts scaling.
        v_penalty = max(0.0, (volatility / self.volatility_threshold) - 1.0)
        o_penalty = max(0.0, (oscillation_index / self.oscillation_threshold) - 1.0)
        
        # Dampening factor scales from 1.0 (perfectly stable) down towards 0.1 (extremely unstable)
        # We use an inverse scaling to ensure it never hits zero.
        dampening = 1.0 / (1.0 + (v_penalty * 2.5) + (o_penalty * 4.0))
        dampening = max(0.1, min(1.0, dampening))
        
        is_unstable = volatility > self.volatility_threshold or oscillation_index > self.oscillation_threshold
        
        return StabilityDescriptor(
            volatility=volatility, 
            oscillation_index=oscillation_index, 
            dampening_factor=dampening,
            is_unstable=is_unstable
        )

    def get_stability_report(self) -> Dict[str, StabilityDescriptor]:
        """
        Returns stability descriptors for all monitored metrics.
        """
        return {
            name: self._analyze_series(series) 
            for name, series in self._history.items()
        }

    def get_aggregate_dampening(self) -> float:
        """
        Returns the most conservative (lowest) dampening factor across all metrics.
        This factor should be applied to adjustment amplitudes system-wide.
        """
        report = self.get_stability_report()
        return min(d.dampening_factor for d in report.values())
