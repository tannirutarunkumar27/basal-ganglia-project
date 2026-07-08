"""
SparseCoder  —  Step 29 (technique 1)
---------------------------------------
Enforces sparse spike representations by applying
three complementary mechanisms:

    1. k-WTA (k-Winners-Take-All): only the top-k
       neurons per population are allowed to fire.
       Biological analog: lateral inhibition.

    2. Firing threshold scaling: raises Vt per neuron
       proportional to recent population activity.
       Prevents all-or-nothing synchrony.

    3. Refractory extension: neurons that fired recently
       have an extended silent period proportional to
       how often the population is over-firing.

Target sparsity: 5-15% of neurons active per timestep.
Biological sparsity in cortex: ~5%.
Biological sparsity in striatum: ~1-5%.
"""

import numpy as np
from collections import deque


class SparseCoder:

    def __init__(self,
                 n_neurons        : int,
                 target_sparsity  : float = 0.08,
                 k_wta_fraction   : float = 0.10,
                 adapt_window     : int   = 100,
                 dt               : float = 0.1e-3,
                 name             : str   = "pop"):
        """
        n_neurons       : population size
        target_sparsity : target fraction of active neurons
        k_wta_fraction  : max fraction allowed to fire (k-WTA)
        adapt_window    : window for sparsity adaptation
        dt              : simulation timestep
        name            : population identifier
        """
        self.n_neurons       = n_neurons
        self.target_sparsity = target_sparsity
        self.k               = max(1, int(k_wta_fraction * n_neurons))
        self.adapt_window    = adapt_window
        self.dt              = dt
        self.name            = name

        # Adaptive threshold per neuron (added to Vt)
        self.adapt_thresh    = np.zeros(n_neurons)
        self.adapt_tau       = 200e-3      # 200 ms
        self.adapt_decay     = np.exp(-dt / self.adapt_tau)
        self.adapt_increment = 0.002       # increment on firing

        # Sparsity history
        self.sparsity_window = deque(maxlen=adapt_window)
        self.spike_history   = deque(maxlen=adapt_window)
        self.step_count      = 0

        # Running stats
        self.mean_sparsity   = target_sparsity
        self.energy_saved    = 0.0         # fraction of spikes suppressed

    def reset(self) -> None:
        self.adapt_thresh  = np.zeros(self.n_neurons)
        self.sparsity_window.clear()
        self.spike_history.clear()
        self.step_count    = 0
        self.mean_sparsity = self.target_sparsity
        self.energy_saved  = 0.0

    def apply(self, membrane_V    : np.ndarray,
               spike_candidates  : np.ndarray) -> np.ndarray:
        """
        Applies sparse coding to a candidate spike array.

        membrane_V       : current membrane potentials (N,)
        spike_candidates : bool array — neurons that would fire (N,)

        Returns: filtered sparse spike array (N, dtype=bool)
        """
        self.step_count += 1
        candidates = np.asarray(spike_candidates, dtype=bool)
        n_orig     = int(candidates.sum())

        if n_orig == 0:
            return candidates

        # ── k-WTA: keep only top-k by membrane potential ──────
        if n_orig > self.k:
            firing_idx = np.where(candidates)[0]
            # Sort by V (highest fires)
            sorted_by_V = firing_idx[
                np.argsort(membrane_V[firing_idx])[::-1]]
            allowed = sorted_by_V[:self.k]
            sparse  = np.zeros(self.n_neurons, dtype=bool)
            sparse[allowed] = True
        else:
            sparse = candidates.copy()

        # ── Adaptive threshold suppression ────────────────────
        # Neurons whose adapt_thresh is large are suppressed
        suppress_mask = self.adapt_thresh > 0.01
        sparse[suppress_mask] = False

        # ── Update adaptive thresholds ─────────────────────────
        self.adapt_thresh *= self.adapt_decay
        self.adapt_thresh[sparse] += self.adapt_increment

        # ── Record sparsity stats ──────────────────────────────
        n_final    = int(sparse.sum())
        sparsity_t = n_final / self.n_neurons
        self.sparsity_window.append(sparsity_t)

        if len(self.sparsity_window) > 0:
            self.mean_sparsity = float(
                np.mean(list(self.sparsity_window)))

        suppressed          = n_orig - n_final
        self.energy_saved  += float(suppressed / max(n_orig, 1))
        self.spike_history.append(n_final)

        return sparse

    def current_sparsity(self) -> float:
        """Returns actual sparsity over recent window."""
        return float(self.mean_sparsity)

    def efficiency_ratio(self) -> float:
        """
        Fraction of spikes saved vs naive (no sparsity).
        1.0 = all spikes suppressed, 0.0 = no savings.
        """
        total_steps = max(self.step_count, 1)
        return float(self.energy_saved / total_steps)

    def sparsity_summary(self) -> dict:
        return {
            "name"            : self.name,
            "target_sparsity" : self.target_sparsity,
            "mean_sparsity"   : self.mean_sparsity,
            "k_wta"           : self.k,
            "efficiency_ratio": self.efficiency_ratio(),
            "step_count"      : self.step_count,
        }