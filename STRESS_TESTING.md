# Stress Testing the Economic Layer

This document outlines the strategy and methodology for stress testing the Cooperative Intelligence and Economic Control system. The goal is to identify performance bottlenecks, stability limits, and the efficacy of dampening mechanisms under extreme conditions.

## 1. Key Performance Dimensions

The system's performance is driven by five primary axes:

1.  **Agent Population Scale**: The number of concurrent agents and their historical snapshots.
2.  **Collaboration Density**: The number of edges in the collaboration graph (impacts Jaccard similarity and structural influence calculations).
3.  **Action Throughput**: The frequency of metric ingestion and policy mutation cycles.
4.  **Causal Graph Depth**: The complexity and depth of the Impact Graph during counterfactual simulations.
5.  **State Volatility**: The rate of change in trust, influence, and impact vectors.

## 2. Stress Testing Scenarios

### Scenario A: Population Scaling (Load Test)
**Objective**: Determine the upper bound of agent snapshots that the `CooperativeStateTensor` can process within acceptable latency.
-   **Target**: 10,000+ agents.
-   **Focus**: `_average_pairwise_jaccard` (O(N²) complexity) and `_safe_entropy`.
-   **Metrics**: Execution time for `build_cooperative_state_tensor`, memory footprint of the snaphot collection.

### Scenario B: High-Frequency Synergy Adaptation
**Objective**: Test the `AdaptiveSynergyAmplifier` under rapid, conflicting feedback loops.
-   **Inbound**: Concurrent streams of `predicted_amplification` vs `observed_amplification` with high variance.
-   **Focus**: EMA convergence stability and exponent adjustment frequency.
-   **Success Criteria**: No runaway feedback loops in the synergy multiplier; convergence on a stable exponent despite noisy input.

### Scenario C: Behavioral Drift & Trust Damping
**Objective**: Validate the `Behavioral Drift Detector` and `Predictive Stability Dampener` under "adversarial" volatility.
-   **Input**: Rapidly fluctuating agent reliability scores and simulated metric gaming (short-term spikes vs long-term decay).
-   **Focus**: Response time of corrective trust damping.
-   **Success Criteria**: Correct identification of "drifting" regimes and successful mitigation of trust variance before systemic instability occurs.

### Scenario D: Macro-Scaling Counterfactuals
**Objective**: Stress the `MacroCounterfactualSimulator` with deep, interconnected policy shifts.
-   **Input**: Proposing shifts in 10+ policy parameters simultaneously across a large state tensor.
-   **Focus**: Trace logic and causal propagation depth.
-   **Metrics**: Simulation latency and trace reproducibility.

### Scenario E: API Payload Scaling
**Objective**: Test the `GovernanceControlAPI` with large batches of agents and historical clusters.
-   **Input**: `inspect_structural_influence` with 5,000+ agent signals; `simulate_parameter_shift` with 100+ task clusters.
-   **Focus**: Serialization overhead and dict/asdict conversion performance.
-   **Metrics**: API method response time (latency) and peak memory usage during serialization.

## 3. Implementation of Stress Tests

Stress tests should be implemented as separate pytest modules or standalone scripts using the `pytest-benchmark` plugin.

### Example Stress Utility (Pseudo-code)

```python
import time
from cooperative_state_model import build_cooperative_state_tensor, AgentSnapshot

def run_scale_test(agent_count: int):
    # Generate synthetic snapshots
    snapshots = [
        AgentSnapshot(
            agent_id=f"agent_{i}",
            trust=0.5 + (0.4 * (i / agent_count)),
            influence=1.0 / agent_count,
            collaboration_partners=[f"agent_{j}" for j in range(i, i+10)],
            predictive_accuracy=[0.6, 0.7, 0.8],
            long_horizon_impact=[0.1, 0.2, 0.3],
            synergy_score=0.7
        ) for i in range(agent_count)
    ]
    
    start_time = time.perf_counter()
    tensor = build_cooperative_state_tensor(snapshots)
    end_time = time.perf_counter()
    
    print(f"Processed {agent_count} agents in {end_time - start_time:.4f}s")
```

## 4. Monitoring & Thresholds

During stress tests, monitor the following "Systemic Health" thresholds:

| Metric | Warning Threshold | Critical Threshold |
| :--- | :--- | :--- |
| Tensor Calculation Latency | > 500ms | > 2000ms |
| Trust Variance | > 0.15 | > 0.25 |
| Synergy Exponent Volatility | > 0.5 (std dev) | > 1.0 (std dev) |
| Memory Usage (Per 1k Agents) | > 50MB | > 200MB |

## 5. Mitigation Strategies for Bottlenecks

1.  **Jaccard Optimization**: If `_average_pairwise_jaccard` becomes a bottleneck, consider MinHash or sampling-based similarity estimation.
2.  **Snapshot Pruning**: Implement aging policies for agent historical data to keep the tensor calculation window fixed.
3.  **Parallel Propagation**: Use concurrent execution for independent branches of the Counterfactual Simulator's impact graph walkthrough.
