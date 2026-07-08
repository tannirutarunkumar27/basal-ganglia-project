"""
HyperdirectPathway  —  Step 11
--------------------------------
Cortex → STN → GPi

Function:
  - Fast global stop signal
  - Conflict override — pauses all actions under ambiguity
  - Reasoning-based conflict controller (advanced innovation)

Conflict trigger:
  sum_{i,j} |Vi - Vj| < epsilon
  → ambiguity detected → STN raises GPi globally

Biological basis:
  The fastest cortex-to-BG route bypasses striatum entirely.
  STN receives direct cortical input and immediately excites GPi.
  This provides a rapid brake before the slower direct/indirect
  pathways have resolved the competition.
  
Advanced innovation:
  Conflict is computed from the Bayesian belief scores Va,
  making this a reasoning-driven conflict controller.
  The STN literally embodies "I am not sure — wait."
"""

import numpy as np


class HyperdirectPathway:

    def __init__(self,
                 n_actions      : int,
                 n_stn          : int   = 40,
                 w_ctx_stn      : float = 0.6e-9,
                 w_stn_gpi      : float = 1.5e-9,
                 conflict_eps   : float = 0.3,
                 stn_latency_ms : float = 6.0,
                 dt             : float = 0.1e-3):
        """
        n_actions      : number of competing actions
        n_stn          : STN population size
        w_ctx_stn      : cortex → STN weight
        w_stn_gpi      : STN → GPi weight (excitatory)
        conflict_eps   : threshold ε for conflict detection
                         lower ε → conflict detected more easily
        stn_latency_ms : biological STN response latency
        dt             : simulation timestep
        """
        self.n_actions      = n_actions
        self.n_stn          = n_stn
        self.w_ctx_stn      = w_ctx_stn
        self.w_stn_gpi      = w_stn_gpi
        self.conflict_eps   = conflict_eps
        self.dt             = dt
        self.latency_steps  = int(stn_latency_ms * 1e-3 / dt)

        # STN state
        self.stn_activity   = 0.0     # scalar global activity
        self.stn_burst      = False   # currently in burst mode

        # GPi excitation from hyperdirect pathway (global, not per-action)
        self.gpi_global_exc = 0.0

        # Conflict score (scalar) from belief scores
        self.conflict_score = 0.0
        self.conflict_flag  = False

        # STN latency buffer (circular)
        self.latency_buffer  = np.zeros(max(self.latency_steps, 1))
        self.latency_ptr     = 0

        # Suppression duration: how long STN stays elevated after trigger
        self.suppression_steps  = int(50e-3 / dt)   # 50 ms pause
        self.suppression_count  = 0

        # History
        self.stn_history        = []
        self.conflict_history   = []
        self.gpi_exc_history    = []

    def reset(self):
        self.stn_activity    = 0.0
        self.stn_burst       = False
        self.gpi_global_exc  = 0.0
        self.conflict_score  = 0.0
        self.conflict_flag   = False
        self.latency_buffer  = np.zeros(max(self.latency_steps, 1))
        self.latency_ptr     = 0
        self.suppression_count = 0
        self.stn_history.clear()
        self.conflict_history.clear()
        self.gpi_exc_history.clear()

    def compute_conflict(self, V: np.ndarray) -> tuple:
        """
        Conflict measure:  sum_{i≠j} |Vi - Vj|  /  N(N-1)

        This implements the reasoning-based conflict controller.
        Low conflict score = all beliefs similar = ambiguity.
        High conflict score = one belief dominates = clear winner.

        Returns (conflict_score, conflict_detected).
        """
        V = np.asarray(V, dtype=float)
        N = len(V)
        if N < 2:
            return 0.0, False

        # Pairwise absolute differences
        total = 0.0
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                total += abs(V[i] - V[j])
                count += 1

        # Normalise
        self.conflict_score = total / max(count, 1)

        # Conflict DETECTED when beliefs are TOO SIMILAR (ambiguous)
        # i.e. conflict_score < epsilon  → uncertain which action
        self.conflict_flag = self.conflict_score < self.conflict_eps

        return float(self.conflict_score), self.conflict_flag

    def step(self,
             cortex_spikes  : np.ndarray,
             belief_scores  : np.ndarray,
             stn_trigger    : float = 0.0) -> float:
        """
        One timestep of the hyperdirect pathway.

        cortex_spikes : spike array from cortex (N_ctx,)
        belief_scores : Va from Phase 3 Bayesian pipeline (A,)
        stn_trigger   : external STN drive (e.g. from uncertainty module)

        Returns gpi_global_excitation (scalar) — broadcast to all
        GPi action channels when conflict is detected.
        """
        # Compute conflict from belief scores (reasoning-based)
        conflict_score, conflict_flag = self.compute_conflict(belief_scores)

        # Cortical drive on STN
        ctx_rate = np.asarray(cortex_spikes, dtype=float).mean()
        ctx_drive = ctx_rate * self.w_ctx_stn * 1e9

        # Conflict-triggered STN burst
        if conflict_flag and not self.stn_burst:
            self.stn_burst         = True
            self.suppression_count = self.suppression_steps

        # Count down suppression
        if self.suppression_count > 0:
            self.suppression_count -= 1
        else:
            self.stn_burst = False

        # STN activation level
        burst_boost = 1.8 if self.stn_burst else 0.0
        target_stn  = ctx_drive + burst_boost + stn_trigger * 0.5
        self.stn_activity = (0.85 * self.stn_activity
                             + 0.15 * np.clip(target_stn, 0, 3.0))

        # Apply latency delay via circular buffer
        self.latency_buffer[self.latency_ptr] = self.stn_activity
        delayed_idx = (self.latency_ptr - self.latency_steps
                       ) % len(self.latency_buffer)
        delayed_stn = self.latency_buffer[delayed_idx]
        self.latency_ptr = (self.latency_ptr + 1) % len(self.latency_buffer)

        # GPi global excitation from hyperdirect
        self.gpi_global_exc = delayed_stn * self.w_stn_gpi * 1e9

        self.stn_history.append(float(self.stn_activity))
        self.conflict_history.append(float(conflict_score))
        self.gpi_exc_history.append(float(self.gpi_global_exc))

        return float(self.gpi_global_exc)

    def is_suppressing(self) -> bool:
        """True if STN is currently in global suppression burst."""
        return self.stn_burst

    def pathway_summary(self) -> dict:
        return {
            "stn_activity"   : float(self.stn_activity),
            "gpi_global_exc" : float(self.gpi_global_exc),
            "conflict_score" : float(self.conflict_score),
            "conflict_flag"  : bool(self.conflict_flag),
            "stn_burst"      : bool(self.stn_burst),
            "suppression_ms" : float(self.suppression_count * self.dt * 1000),
        }