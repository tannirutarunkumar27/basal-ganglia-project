"""
EnergyBudget  —  Phase 10 integration
---------------------------------------
Tracks total energy consumption across the full pipeline
and computes the neuromorphic efficiency score.

Metrics:
    spike_count       : total spikes across all populations
    synapse_events    : total synaptic transmission events
    plasticity_updates: number of weight update operations
    reasoning_calls   : number of symbolic reasoning invocations
    energy_nJ         : estimated energy in nanojoules

Neuromorphic hardware constants (Intel Loihi reference):
    Energy per spike                : 23 pJ
    Energy per synaptic event       : 10 pJ
    Energy per weight update        : 5 pJ
    Energy per reasoning call       : 100 pJ (approximate)
"""

import numpy as np
from collections import deque


ENERGY_PJ = {
    "spike"       : 23.0,
    "synapse_event": 10.0,
    "weight_update": 5.0,
    "reasoning"   : 100.0,
}


class EnergyBudget:

    def __init__(self,
                 episode_budget_nJ: float = 1000.0,
                 dt               : float = 0.1e-3):

        self.episode_budget_nJ = episode_budget_nJ
        self.dt                = dt

        # Cumulative counters
        self.total_spikes          = 0
        self.total_synapse_events  = 0
        self.total_weight_updates  = 0
        self.total_reasoning_calls = 0
        self.total_energy_pJ       = 0.0

        # Per-step history
        self.energy_history = deque(maxlen=5000)
        self.spike_history  = deque(maxlen=5000)
        self.step_count     = 0

        # Efficiency score (0-1, higher = more efficient)
        self.efficiency_score = 0.0

    def reset(self) -> None:
        self.total_spikes          = 0
        self.total_synapse_events  = 0
        self.total_weight_updates  = 0
        self.total_reasoning_calls = 0
        self.total_energy_pJ       = 0.0
        self.energy_history.clear()
        self.spike_history.clear()
        self.step_count            = 0
        self.efficiency_score      = 0.0

    def record_step(self,
                     n_spikes          : int,
                     n_synapse_events  : int,
                     n_weight_updates  : int,
                     n_reasoning_calls : int) -> float:
        """
        Records one timestep of activity and returns
        energy cost for that step in pJ.
        """
        self.step_count           += 1
        self.total_spikes         += n_spikes
        self.total_synapse_events += n_synapse_events
        self.total_weight_updates += n_weight_updates
        self.total_reasoning_calls+= n_reasoning_calls

        step_energy = (
            n_spikes           * ENERGY_PJ["spike"]
            + n_synapse_events * ENERGY_PJ["synapse_event"]
            + n_weight_updates * ENERGY_PJ["weight_update"]
            + n_reasoning_calls* ENERGY_PJ["reasoning"]
        )

        self.total_energy_pJ += step_energy
        self.energy_history.append(float(step_energy))
        self.spike_history.append(int(n_spikes))

        return float(step_energy)

    def total_energy_nJ(self) -> float:
        return self.total_energy_pJ / 1000.0

    def within_budget(self) -> bool:
        return self.total_energy_nJ() <= self.episode_budget_nJ

    def compute_efficiency_score(self,
                                  task_accuracy  : float,
                                  baseline_energy: float = None) -> float:
        """
        Efficiency score = accuracy / normalised_energy.
        Higher is better: accurate AND energy-efficient.

        baseline_energy: energy of unoptimized system (nJ)
                         defaults to 2x current energy.
        """
        if baseline_energy is None:
            baseline_energy = self.total_energy_nJ() * 2.0

        norm_energy = self.total_energy_nJ() / max(baseline_energy, 1e-9)
        self.efficiency_score = float(
            task_accuracy / (norm_energy + 1e-8))
        self.efficiency_score = float(
            np.clip(self.efficiency_score, 0.0, 10.0))
        return self.efficiency_score

    def mean_energy_per_step_pJ(self) -> float:
        hist = list(self.energy_history)
        return float(np.mean(hist)) if hist else 0.0

    def spike_rate_hz(self, n_neurons: int) -> float:
        """Mean population firing rate implied by spike counts."""
        time_s = self.step_count * self.dt
        return float(self.total_spikes
                     / max(n_neurons * time_s, 1e-9))

    def budget_summary(self) -> dict:
        return {
            "step_count"          : self.step_count,
            "total_spikes"        : self.total_spikes,
            "total_synapse_events": self.total_synapse_events,
            "total_weight_updates": self.total_weight_updates,
            "total_reasoning_calls": self.total_reasoning_calls,
            "total_energy_nJ"     : self.total_energy_nJ(),
            "mean_energy_pJ_step" : self.mean_energy_per_step_pJ(),
            "budget_nJ"           : self.episode_budget_nJ,
            "within_budget"       : self.within_budget(),
            "efficiency_score"    : float(self.efficiency_score),
        }

    def print_report(self):
        s = self.budget_summary()
        print("\n--- Energy Budget Report ---")
        print(f"  Total spikes           : {s['total_spikes']:>10d}")
        print(f"  Total synapse events   : {s['total_synapse_events']:>10d}")
        print(f"  Total weight updates   : {s['total_weight_updates']:>10d}")
        print(f"  Total reasoning calls  : {s['total_reasoning_calls']:>10d}")
        print(f"  Total energy           : {s['total_energy_nJ']:>10.3f} nJ")
        print(f"  Mean energy/step       : {s['mean_energy_pJ_step']:>10.3f} pJ")
        print(f"  Budget ({s['budget_nJ']:.0f} nJ)       : "
              f"{'WITHIN' if s['within_budget'] else 'EXCEEDED'}")
        print(f"  Efficiency score       : {s['efficiency_score']:>10.4f}")
        print("----------------------------")