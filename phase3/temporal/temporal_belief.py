"""
TemporalBeliefUpdater
---------------------
Implements Step 6:
    Va(t) = λ * Va(t-1) + (1-λ) * Va_hat(t)

where:
  Va_hat(t)  = instantaneous belief from PosteriorBeliefEncoder
  λ (lambda) = memory retention factor (0 = no memory, 1 = full)

Provides:
  - memory-based reasoning
  - temporal continuity across timesteps
  - belief persistence under noisy observations
  - recurrent cortical loop simulation
"""

import numpy as np
from collections import deque


class TemporalBeliefUpdater:

    def __init__(self,
                 n_actions     : int,
                 lam           : float = 0.85,
                 history_len   : int   = 500,
                 noise_floor   : float = 1e-6):
        """
        n_actions   : number of actions (A)
        lam         : memory factor λ ∈ [0, 1]
                      0.85 = strong recency bias with persistence
        history_len : how many past belief vectors to keep
        noise_floor : prevents beliefs from freezing at zero
        """
        self.n_actions   = n_actions
        self.lam         = lam
        self.history_len = history_len
        self.noise_floor = noise_floor

        # Temporal belief state Va(t), initialised to zero
        self.V_temporal  = np.zeros(n_actions)

        # Full history buffer
        self.history     = deque(maxlen=history_len)

        # Separate slow-timescale memory (λ_slow > λ)
        self.lam_slow    = min(lam + 0.1, 0.99)
        self.V_slow      = np.zeros(n_actions)   # long-term memory

        # Running statistics for adaptive λ
        self._update_count = 0
        self._volatility   = 0.0    # tracks how fast beliefs change

    def reset(self):
        self.V_temporal    = np.zeros(self.n_actions)
        self.V_slow        = np.zeros(self.n_actions)
        self.history.clear()
        self._update_count = 0
        self._volatility   = 0.0

    def update(self, V_hat: np.ndarray,
               adaptive_lam: bool = True) -> np.ndarray:
        """
        Core Step 6 update:
            Va(t) = λ * Va(t-1) + (1-λ) * Va_hat(t)

        V_hat        : instantaneous belief (A,) from encoder
        adaptive_lam : if True, λ adapts to belief volatility
        Returns      : smoothed temporal belief Va(t) of shape (A,)
        """
        V_hat = np.asarray(V_hat, dtype=float)

        # Adaptive lambda: reduce memory when beliefs change fast
        if adaptive_lam and self._update_count > 0:
            change = np.mean(np.abs(V_hat - self.V_temporal))
            # EMA of change magnitude = volatility
            self._volatility = (0.95 * self._volatility
                                 + 0.05 * change)
            # Decrease λ when volatile, increase when stable
            effective_lam = self.lam * np.exp(-2.0 * self._volatility)
            effective_lam = np.clip(effective_lam, 0.3, 0.98)
        else:
            effective_lam = self.lam

        # Main temporal update
        prev = self.V_temporal.copy()
        self.V_temporal = (effective_lam * self.V_temporal
                           + (1.0 - effective_lam) * V_hat)

        # Slow-timescale memory update (long-term habit)
        self.V_slow = (self.lam_slow * self.V_slow
                       + (1.0 - self.lam_slow) * V_hat)

        # Add noise floor to avoid freezing
        self.V_temporal += np.random.normal(
            0, self.noise_floor, self.n_actions)

        # Record in history
        self.history.append({
            "V_temporal"    : self.V_temporal.copy(),
            "V_hat"         : V_hat.copy(),
            "effective_lam" : float(effective_lam),
            "volatility"    : float(self._volatility),
        })

        self._update_count += 1
        return self.V_temporal.copy()

    def get_belief_trajectory(self, last_n: int = 50) -> np.ndarray:
        """
        Returns an array of shape (min(last_n, len(history)), A)
        containing the temporal belief history.
        """
        recent = list(self.history)[-last_n:]
        if not recent:
            return np.zeros((0, self.n_actions))
        return np.array([h["V_temporal"] for h in recent])

    def belief_change_rate(self, window: int = 20) -> float:
        """Mean absolute change in belief over the last `window` steps."""
        traj = self.get_belief_trajectory(window + 1)
        if traj.shape[0] < 2:
            return 0.0
        diffs = np.abs(np.diff(traj, axis=0))
        return float(diffs.mean())

    def temporal_summary(self) -> dict:
        return {
            "V_temporal"    : self.V_temporal.copy(),
            "V_slow"        : self.V_slow.copy(),
            "volatility"    : self._volatility,
            "update_count"  : self._update_count,
            "history_len"   : len(self.history),
        }