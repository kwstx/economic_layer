from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Sequence

from cooperative_state_model import (
    AgentSnapshot,
    CooperativeStateTensor,
    assess_ecosystem_state,
    build_cooperative_state_tensor,
)
from kernel.macro_counterfactual_simulator import (
    CounterfactualEvaluation,
    HistoricalTaskCluster,
    MacroCounterfactualSimulator,
)
from kernel.policy_transformation_engine import (
    PolicyMutation,
    PolicyParameters,
    PolicyTransformationEngine,
)
from kernel.structural_influence_reweighted import (
    InfluenceProfile,
    StructuralInfluenceReweighter,
)


class GovernanceControlAPI:
    """
    Exposes a Governance Control API for querying system state, policy history,
    simulating shifts, and inspecting structural influence.
    
    This API provides structured representations and causal explanations to 
    ensure transparency and interpretability of the economic governance layer.
    """

    def __init__(
        self,
        transformation_engine: PolicyTransformationEngine,
        simulator: MacroCounterfactualSimulator,
        reweighter: StructuralInfluenceReweighter,
    ):
        self._transformation_engine = transformation_engine
        self._simulator = simulator
        self._reweighter = reweighter
        # In a production system, this would be backed by a persistent ledger.
        self._mutation_history: List[PolicyMutation] = []

    def record_mutation(self, mutation: PolicyMutation | Sequence[PolicyMutation]):
        """Records policy mutations into the history."""
        if isinstance(mutation, PolicyMutation):
            self._mutation_history.append(mutation)
        else:
            self._mutation_history.extend(mutation)

    def query_state_tensors(self, snapshots: Sequence[AgentSnapshot]) -> Dict[str, Any]:
        """
        Retrieves the current system state tensor along with a causal diagnostic assessment.
        """
        tensor = build_cooperative_state_tensor(snapshots)
        assessment = assess_ecosystem_state(tensor)
        
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "state_tensor": asdict(tensor),
            "regime_assessment": {
                "current_regime": assessment.regime.value,
                "causal_diagnostics": assessment.diagnostics,
                "is_healthy": assessment.regime.value == "healthy"
            },
            "interpretation": self._generate_state_interpretation(tensor, assessment.regime.value)
        }

    def retrieve_mutation_history(
        self, 
        limit: int = 50, 
        parameter_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns a history of policy mutations, preserving transparency into 
        how governance parameters have evolved.
        """
        history = self._mutation_history
        if parameter_filter:
            history = [m for m in history if m.parameter == parameter_filter]
            
        return [asdict(m) for m in sorted(history, key=lambda x: x.timestamp, reverse=True)[:limit]]

    def simulate_parameter_shift(
        self,
        clusters: Sequence[HistoricalTaskCluster],
        baseline_parameters: PolicyParameters,
        candidate_parameters: PolicyParameters,
    ) -> Dict[str, Any]:
        """
        Simulates a proposed parameter shift using historical data clusters to 
        predict downstream impact on trust and cooperation.
        """
        evaluation = self._simulator.evaluate_parameter_shift(
            clusters=clusters,
            baseline_parameters=baseline_parameters,
            candidate_parameters=candidate_parameters
        )
        
        return {
            "evaluation_result": asdict(evaluation),
            "recommendation": "COMMIT" if evaluation.should_commit else "BLOCK",
            "causal_explanation": self._explain_simulation_result(evaluation)
        }

    def inspect_structural_influence(
        self, 
        agents_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieves structural influence weights for agents, explaining how 
        trust and stability contribute to their system-wide impact.
        """
        profiles = []
        for data in agents_data:
            profile = self._reweighter.process_signals(
                agent_id=data["agent_id"],
                base_influence=data.get("base_influence", 1.0),
                calibration_delta=data.get("calibration_delta", 0.0),
                stability_coefficient=data.get("stability_coefficient", 1.0),
                current_accuracy=data.get("current_accuracy", 0.5)
            )
            
            profile_dict = asdict(profile)
            profile_dict["explanation"] = (
                f"Agent {profile.agent_id} has a reweighted influence of {profile.reweighted_influence:.4f} "
                f"(Base: {profile.base_influence:.2f}). This is derived from a trust coefficient of "
                f"{profile.trust_coefficient:.4f}, established through a predictive accuracy of "
                f"{profile.calibration_accuracy:.2f} and cooperative stability of {profile.cooperative_stability:.2f}."
            )
            profiles.append(profile_dict)
            
        return profiles

    def _generate_state_interpretation(self, tensor: CooperativeStateTensor, regime: str) -> str:
        """Generates a human-readable interpretation of the state tensor."""
        interpretations = []
        if tensor.influence_concentration_entropy < 0.85:
            interpretations.append("High influence concentration detected; power is becoming centralized.")
        if tensor.trust_variance > 0.08:
            interpretations.append("Trust variance is high, indicating growing disparity in agent reliability.")
        if tensor.long_horizon_impact_accumulation_rate < 0.01:
            interpretations.append("System is showing signs of short-term optimization bias.")
            
        if not interpretations:
            return f"System is currently in a {regime} state with balanced cooperative dynamics."
        return f"System is in a {regime} state. Observations: " + " ".join(interpretations)

    def _explain_simulation_result(self, eval: CounterfactualEvaluation) -> str:
        """Provides a causal explanation for the simulation outcome."""
        reasons = []
        if eval.improves_long_term_cooperation:
            reasons.append(f"Simulation shows a {eval.cooperative_intelligence_delta:+.4f} increase in cooperative intelligence.")
        else:
            reasons.append("Simulation indicates a decline or insufficient gain in long-term cooperation.")
            
        if eval.destabilizes_trust:
            reasons.append(f"Proposed shift would increase trust variance by {eval.trust_variance_delta:.4f}, exceeding stability thresholds.")
        else:
            reasons.append("Trust stability is maintained within acceptable variance bounds.")
            
        return " ".join(reasons)
