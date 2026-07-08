"""
EnergyMetrics  —  Step 33
---------------------------
Five neuromorphic energy metrics.

1. spike_count          : total + mean per step
2. latency              : mean steps from stimulus to action
3. synaptic_update_cost : total plasticity operations
4. memory_footprint     : weight matrix sparsity + size
5. neuromorphic_efficiency: accuracy / normalised_energy
"""

import numpy as np
from collections import deque


# Energy constants for Intel Loihi (pJ)
ENERGY_PER_SPIKE    = 23.0
ENERGY_PER_SYN      = 10.0
ENERGY_PER_UPDATE   = 5.0
ENERGY_PER_REASONING= 100.0


class EnergyMetrics:

    def __init__(self, dt: float = 0.1e-3):
        self.dt = dt

        self._spikes        = []
        self._syn_events    = []
        self._weight_updates= []
        self._reasoning_calls=[]
        self._latencies     = []
        self._action_steps  = []
        self._stimulus_steps= []
        self._weight_sizes  = []
        self._weight_sparse = []

    def reset(self) -> None:
        for a in [self._spikes, self._syn_events,
                  self._weight_updates, self._reasoning_calls,
                  self._latencies, self._action_steps,
                  self._stimulus_steps, self._weight_sizes,
                  self._weight_sparse]:
            a.clear()

    def record_step(self,
                     n_spikes         : int,
                     n_syn_events     : int,
                     n_weight_updates : int,
                     n_reasoning      : int,
                     step             : int,
                     gate_open        : bool = False,
                     new_stimulus     : bool = False) -> None:
        self._spikes.append(int(n_spikes))
        self._syn_events.append(int(n_syn_events))
        self._weight_updates.append(int(n_weight_updates))
        self._reasoning_calls.append(int(n_reasoning))
        if new_stimulus:
            self._stimulus_steps.append(int(step))
        if gate_open and self._stimulus_steps:
            lat = step - self._stimulus_steps[-1]
            self._latencies.append(max(0, int(lat)))

    def record_weights(self, weight_dict: dict) -> None:
        """Records weight matrix stats for memory footprint."""
        for W in weight_dict.values():
            W_arr = np.asarray(W)
            self._weight_sizes.append(float(W_arr.size))
            nz    = float(np.count_nonzero(W_arr))
            self._weight_sparse.append(
                float(1.0 - nz / max(W_arr.size, 1)))

    def spike_count_metrics(self) -> dict:
        arr = np.array(self._spikes)
        return {
            "total_spikes"   : int(arr.sum()),
            "mean_per_step"  : float(arr.mean()) if len(arr) else 0.0,
            "peak_per_step"  : int(arr.max())    if len(arr) else 0,
        }

    def latency(self) -> float:
        """Mean steps from stimulus to action release (ms)."""
        if not self._latencies:
            return 0.0
        return float(np.mean(self._latencies) * self.dt * 1000)

    def synaptic_update_cost_pJ(self) -> float:
        """Total energy in pJ from all weight updates."""
        return (float(np.sum(self._weight_updates))
                * ENERGY_PER_UPDATE)

    def total_energy_nJ(self) -> float:
        return (
            float(np.sum(self._spikes))         * ENERGY_PER_SPIKE
            + float(np.sum(self._syn_events))   * ENERGY_PER_SYN
            + float(np.sum(self._weight_updates))* ENERGY_PER_UPDATE
            + float(np.sum(self._reasoning_calls))* ENERGY_PER_REASONING
        ) / 1000.0

    def memory_footprint(self) -> dict:
        """Weight matrix sparsity and effective parameter count."""
        if not self._weight_sizes:
            return {"mean_sparsity": 0.0, "total_params": 0}
        mean_sp  = float(np.mean(self._weight_sparse))
        total_p  = int(np.sum(self._weight_sizes))
        active_p = int(total_p * (1.0 - mean_sp))
        return {
            "mean_sparsity" : mean_sp,
            "total_params"  : total_p,
            "active_params" : active_p,
        }

    def neuromorphic_efficiency(self,
                                  task_accuracy: float) -> float:
        """
        efficiency = accuracy / normalised_energy
        normalised_energy = total_nJ / baseline_nJ
        baseline = 2x observed (always-on estimate).
        """
        total_nJ   = self.total_energy_nJ()
        baseline   = max(total_nJ * 2.0, 1e-9)
        norm_energy = total_nJ / baseline
        return float(np.clip(
            task_accuracy / (norm_energy + 1e-8), 0.0, 10.0))

    def compute_all(self, task_accuracy: float = 0.5) -> dict:
        sc = self.spike_count_metrics()
        mf = self.memory_footprint()
        return {
            "total_spikes"         : sc["total_spikes"],
            "mean_spikes_per_step" : sc["mean_per_step"],
            "latency_ms"           : self.latency(),
            "synaptic_update_pJ"   : self.synaptic_update_cost_pJ(),
            "memory_sparsity"      : mf["mean_sparsity"],
            "active_params"        : mf["active_params"],
            "total_energy_nJ"      : self.total_energy_nJ(),
            "neuromorphic_efficiency": self.neuromorphic_efficiency(
                task_accuracy),
        }