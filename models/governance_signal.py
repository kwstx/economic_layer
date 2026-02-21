from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any

@dataclass(frozen=True)
class GovernanceSignal:
    """
    Standardized Governance Signal object consumed by the behavioral control logic.
    This represents a real-time structural input for decision modulation.
    """
    agent_id: str
    impact_vector: Dict[str, float] = field(default_factory=dict)
    synergy_density: float = 1.0
    calibration_delta: float = 0.0
    stability_coefficient: float = 1.0
    temporal_impact_weight: float = 1.0
    
    # Metadata for traceability
    source_operation: str = "unknown"
    audit_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "impact_vector": self.impact_vector,
            "synergy_density": self.synergy_density,
            "calibration_delta": self.calibration_delta,
            "stability_coefficient": self.stability_coefficient,
            "temporal_impact_weight": self.temporal_impact_weight,
            "source_operation": self.source_operation,
            "audit_id": self.audit_id,
            "timestamp": self.timestamp.isoformat()
        }
