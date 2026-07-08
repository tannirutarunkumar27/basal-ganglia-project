"""
SynapsePruner  —  Step 29 (technique 3 + 5)
---------------------------------------------
Prunes weak synapses and applies selective plasticity.

Pruning strategy:
    1. Magnitude pruning   : remove synapses below w_min_abs
    2. Dormancy pruning    : remove synapses unused for tau_dormant
    3. Lottery ticket      : preserve top-k% by absolute weight

Selective plasticity:
    Only update synapses that are:
    - currently active (pre or post fired)
    - above a minimum weight threshold
    - not in the pruned mask

Biological analog:
    Synaptic elimination during development and learning.
    Approximately 50% of synapses are pruned in adolescence.
    Remaining synapses are strengthened (synaptic competition).
"""

import numpy as np
from collections import deque


class SynapsePruner:

    def __init__(self,
                 n_pre          : int,
                 n_post         : int,
                 w_min_abs      : float = 0.01,
                 prune_fraction : float = 0.30,
                 dormancy_steps : int   = 5000,
                 dt             : float = 0.1e-3,
                 name           : str   = "synapse"):
        """
        w_min_abs      : below this |weight|, synapse is prunable
        prune_fraction : fraction of weakest synapses to prune
        dormancy_steps : steps without use before dormancy pruning
        """
        self.n_pre          = n_pre
        self.n_post         = n_post
        self.w_min_abs      = w_min_abs
        self.prune_fraction = prune_fraction
        self.dormancy_steps = dormancy_steps
        self.dt             = dt
        self.name           = name

        # Pruning mask (True = alive, False = pruned)
        self.alive_mask  = np.ones((n_pre, n_post), dtype=bool)

        # Steps since each synapse last had activity
        self.dormant_ctr = np.zeros((n_pre, n_post), dtype=int)

        # Pruning stats
        self.prune_events    = 0
        self.total_pruned    = 0
        self.prune_history   = deque(maxlen=200)

    def reset(self) -> None:
        self.alive_mask  = np.ones(
            (self.n_pre, self.n_post), dtype=bool)
        self.dormant_ctr = np.zeros(
            (self.n_pre, self.n_post), dtype=int)
        self.prune_events  = 0
        self.total_pruned  = 0

    def update_activity(self,
                         pre_spikes : np.ndarray,
                         post_spikes: np.ndarray) -> None:
        """
        Increments dormancy counter for inactive synapses,
        resets for active ones.
        """
        pre  = np.asarray(pre_spikes,  dtype=bool)
        post = np.asarray(post_spikes, dtype=bool)

        n_pre  = min(len(pre),  self.n_pre)
        n_post = min(len(post), self.n_post)

        activity = np.zeros((self.n_pre, self.n_post), dtype=bool)
        for i in range(n_pre):
            if pre[i]:
                activity[i, :n_post] = True
        for j in range(n_post):
            if post[j]:
                activity[:n_pre, j] = True

        self.dormant_ctr[activity]  = 0
        self.dormant_ctr[~activity] += 1

    def prune(self, W: np.ndarray,
               force: bool = False) -> tuple:
        """
        Applies pruning to weight matrix W.

        Returns (W_pruned, pruned_mask) where:
          W_pruned    : weight matrix with pruned synapses set to 0
          pruned_mask : bool (N_pre, N_post), True = pruned this call
        """
        W_abs  = np.abs(W)
        n_orig = int(self.alive_mask.sum())

        # Magnitude pruning: find candidate weak synapses
        weak_mask   = (W_abs < self.w_min_abs) & self.alive_mask

        # Dormancy pruning: long-inactive synapses
        dormant_mask = (
            (self.dormant_ctr > self.dormancy_steps)
            & self.alive_mask)

        # Lottery pruning: prune bottom prune_fraction
        if self.alive_mask.any():
            alive_W   = W_abs[self.alive_mask]
            threshold = float(np.percentile(
                alive_W,
                self.prune_fraction * 100))
            lottery_mask = ((W_abs < threshold)
                            & self.alive_mask)
        else:
            lottery_mask = np.zeros_like(self.alive_mask)

        # Combined pruning decision
        to_prune = weak_mask | dormant_mask | lottery_mask

        # Apply
        self.alive_mask[to_prune] = False
        W_pruned          = W.copy()
        W_pruned[to_prune]= 0.0

        n_pruned = int(to_prune.sum())
        if n_pruned > 0:
            self.prune_events  += 1
            self.total_pruned  += n_pruned
            self.prune_history.append({
                "n_pruned"  : n_pruned,
                "n_alive"   : int(self.alive_mask.sum()),
                "n_orig"    : n_orig,
            })

        return W_pruned, to_prune

    def selective_plasticity_mask(self,
                                   pre_spikes : np.ndarray,
                                   post_spikes: np.ndarray
                                   ) -> np.ndarray:
        """
        Returns a bool mask (N_pre, N_post) indicating
        which synapses should receive plasticity updates.
        Only alive + currently active synapses update.
        """
        pre  = np.asarray(pre_spikes,  dtype=bool)
        post = np.asarray(post_spikes, dtype=bool)

        n_pre  = min(len(pre),  self.n_pre)
        n_post = min(len(post), self.n_post)

        active = np.zeros((self.n_pre, self.n_post), dtype=bool)
        for i in range(n_pre):
            if pre[i]:
                active[i, :n_post] = True
        for j in range(n_post):
            if post[j]:
                active[:n_pre, j] = True

        return active & self.alive_mask

    def sparsity(self) -> float:
        return float(1.0 - self.alive_mask.mean())

    def pruner_summary(self) -> dict:
        return {
            "name"          : self.name,
            "n_alive"       : int(self.alive_mask.sum()),
            "total_synapses": self.n_pre * self.n_post,
            "sparsity"      : self.sparsity(),
            "total_pruned"  : self.total_pruned,
            "prune_events"  : self.prune_events,
        }