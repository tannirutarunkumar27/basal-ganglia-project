"""
SpikeEvidenceExtractor
----------------------
Converts raw spike trains from the cortex / bayesian_layer
into per-action likelihood scores P(s|a).

For each action a, a dedicated neuron subpopulation
encodes evidence. The extractor:
  1. Maintains a sliding window of recent spikes
  2. Counts spikes per action subpopulation
  3. Normalises into a probability-like likelihood
"""

import numpy as np
from collections import deque


class SpikeEvidenceExtractor:

    def __init__(self,
                 n_actions      : int,
                 n_neurons_total: int,
                 window_steps   : int   = 100,
                 dt             : float = 0.1e-3):
        """
        n_actions       : number of possible actions (A)
        n_neurons_total : total neurons in the evidence population
        window_steps    : sliding window length in timesteps
        dt              : simulation timestep (s)
        """
        self.n_actions       = n_actions
        self.n_neurons_total = n_neurons_total
        self.window_steps    = window_steps
        self.dt              = dt

        # Assign contiguous neuron blocks to each action
        # e.g. 60 neurons, 4 actions -> 15 neurons per action
        self.neurons_per_action = n_neurons_total // n_actions
        self.action_slices = [
            slice(a * self.neurons_per_action,
                  (a + 1) * self.neurons_per_action)
            for a in range(n_actions)
        ]

        # Sliding spike count window: deque of (A,) arrays
        self.spike_window = deque(maxlen=window_steps)

        # Smoothed spike counts per action (A,)
        self.spike_counts = np.zeros(n_actions)

        # Small constant for numerical stability
        self._eps = 1e-8

    def reset(self):
        self.spike_window.clear()
        self.spike_counts = np.zeros(self.n_actions)

    def update(self, spike_vector: np.ndarray) -> np.ndarray:
        """
        Push one timestep of spikes, return updated spike counts.
        spike_vector: bool array (n_neurons_total,)
        Returns: spike_counts (n_actions,) — raw counts in window
        """
        spike_vector = np.asarray(spike_vector, dtype=float)

        # Pad or truncate to expected size
        sv = np.zeros(self.n_neurons_total)
        n  = min(len(spike_vector), self.n_neurons_total)
        sv[:n] = spike_vector[:n]

        # Per-action spike count for this timestep
        step_counts = np.array([
            sv[sl].sum() for sl in self.action_slices
        ])
        self.spike_window.append(step_counts)

        # Sum over window
        self.spike_counts = np.sum(self.spike_window, axis=0)
        return self.spike_counts.copy()

    def likelihood(self) -> np.ndarray:
        """
        P(s|a): normalised spike count -> likelihood.
        Returns array of shape (A,) summing to 1.
        """
        counts = self.spike_counts + self._eps
        return counts / counts.sum()

    def log_likelihood(self) -> np.ndarray:
        """log P(s|a) for each action."""
        return np.log(self.likelihood())

    def firing_rate_hz(self) -> np.ndarray:
        """Mean firing rate (Hz) per action subpopulation."""
        window_time = len(self.spike_window) * self.dt
        if window_time < self.dt:
            return np.zeros(self.n_actions)
        return (self.spike_counts /
                (self.neurons_per_action * window_time + self._eps))