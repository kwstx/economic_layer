from __future__ import annotations
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator

# Import the standardized model
from models.governance_signal import GovernanceSignal

class MetricIngestionKernel:
    """
    Maintains real-time ingestion of structured outputs from the Metrics & Feedback Engine.
    Converts multi-dimensional impact data into normalized GovernanceSignals.
    """

    def __init__(self, logger_callback=None):
        self._logger = logger_callback

    def ingest_response(self, response_envelope: Dict[str, Any]) -> List[GovernanceSignal]:
        """
        Parses a single ApiResponse envelope into one or more GovernanceSignals.
        
        This handles the mapping between the raw Metrics Engine outputs and 
        the structural control inputs required by the Adaptive Economic Layer.
        """
        operation = response_envelope.get("operation")
        data = response_envelope.get("data", {})
        audit_id = response_envelope.get("audit_id")
        signals: List[GovernanceSignal] = []

        if not data:
            return []

        if operation == "retrieve_forecast":
            signals.append(self._parse_forecast(data, audit_id))
        
        elif operation == "query_synergy_density":
            signals.extend(self._parse_synergy(data, audit_id))
            
        elif operation == "submit_outcome":
            signals.append(self._parse_outcome(data, audit_id))
            
        elif operation == "run_counterfactual":
            signals.append(self._parse_counterfactual(data, audit_id))
            
        elif operation == "agent_impact_profile":
            signals.append(self._parse_profile(data, audit_id))

        return signals

    def _parse_forecast(self, data: Dict[str, Any], audit_id: Optional[str]) -> GovernanceSignal:
        # Extract agent_id from domain_context of the source if available, 
        # otherwise use a fallback or the one provided in the data.
        agent_id = self._extract_agent_id(data)
        
        return GovernanceSignal(
            agent_id=agent_id,
            impact_vector=data.get("predicted_impact_vector", {}),
            temporal_impact_weight=data.get("temporal_impact_weight", 1.0),
            source_operation="retrieve_forecast",
            audit_id=audit_id
        )

    def _parse_synergy(self, data: Dict[str, Any], audit_id: Optional[str]) -> List[GovernanceSignal]:
        # Synergy density is for a cluster. We emit a signal for each agent in the cluster.
        collaboration = data.get("collaboration_structure", [])
        synergy_ratio = data.get("synergy_density_ratio", 1.0)
        
        return [
            GovernanceSignal(
                agent_id=self._extract_agent_id({"id": node_id}), # Assume node_id might be agent_id or parsable
                synergy_density=synergy_ratio,
                source_operation="query_synergy_density",
                audit_id=audit_id
            ) for node_id in collaboration
        ]

    def _parse_outcome(self, data: Dict[str, Any], audit_id: Optional[str]) -> GovernanceSignal:
        calibration = data.get("calibration") or {}
        outcome = data.get("outcome") or {}
        
        agent_id = self._extract_agent_id(data)
        
        return GovernanceSignal(
            agent_id=agent_id,
            calibration_delta=calibration.get("reliability_delta", 0.0),
            impact_vector=outcome.get("realized_impact_vector", {}),
            source_operation="submit_outcome",
            audit_id=audit_id
        )

    def _parse_counterfactual(self, data: Dict[str, Any], audit_id: Optional[str]) -> GovernanceSignal:
        return GovernanceSignal(
            agent_id=data.get("removed_agent_id", "unknown_agent"),
            impact_vector=data.get("marginal_influence_vector", {}),
            source_operation="run_counterfactual",
            audit_id=audit_id
        )

    def _parse_profile(self, data: Dict[str, Any], audit_id: Optional[str]) -> GovernanceSignal:
        agent_id = data.get("agent_id", "unknown_agent")
        reliability = data.get("reliability", {})
        stability = data.get("stability", {})
        
        return GovernanceSignal(
            agent_id=agent_id,
            stability_coefficient=stability.get("aggregate_stability_coefficient", 1.0),
            source_operation="agent_impact_profile",
            audit_id=audit_id
        )

    def _extract_agent_id(self, data: Dict[str, Any]) -> str:
        """
        Helper to find agent_id in various formats of data.
        """
        # 1. Direct agent_id
        if "agent_id" in data:
            return data["agent_id"]
        
        # 2. Inside domain_context (common in nodes)
        context = data.get("domain_context", {})
        if isinstance(context, dict):
            if "agent_id" in context:
                return context["agent_id"]
            if "agent" in context:
                return context["agent"]
                
        # 3. Inside node/projection sub-dicts
        if "node" in data and isinstance(data["node"], dict):
            return self._extract_agent_id(data["node"])
        if "projection" in data and isinstance(data["projection"], dict):
            return self._extract_agent_id(data["projection"])
            
        # 4. Fallback to ID if it looks like an agent ID (optional heuristic)
        # For now, just return 'unknown'
        return "unknown_agent"

    def listen_and_emit(self, response_stream: Generator[Dict[str, Any], None, None]) -> Generator[GovernanceSignal, None, None]:
        """
        Continuously listens to a stream of API responses and yields standardized signals.
        """
        for envelope in response_stream:
            for signal in self.ingest_response(envelope):
                if self._logger:
                    self._logger(f"Ingested Governance Signal: {signal.agent_id} | Op: {signal.source_operation}")
                yield signal

    def normalize_signals(self, signals: List[GovernanceSignal]) -> List[GovernanceSignal]:
        """
        Applies cross-metric normalization to ensure weights are comparable 
        for control logic consumption.
        """
        # Placeholder for normalization logic (e.g., Z-score, Min-Max over a window)
        return signals
