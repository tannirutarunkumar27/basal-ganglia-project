"""
MemoryTrace
-----------
Simulates the recurrent cortical loop that implements
temporal belief persistence. Maintains multiple timescale
traces that feed back into the belief encoder.

Three trace timescales:
  - fast  (τ = 20 ms)  : immediate context
  - mid   (τ = 200 ms) : working memory
  - slow  (τ = 2000 ms): episodic-like retention
"""

import numpy as np


class MemoryTrace:

    def __init__(self,
                 n_actions : int,
                 dt        : float = 0.1e-3):

        self.n_actions = n_actions
        self.dt        = dt

        # Three timescale decay constants
        self.tau_fast  = 20e-3      # 20 ms
        self.tau_mid   = 200e-3     # 200 ms
        self.tau_slow  = 2000e-3    # 2000 ms

        # Trace states — one per timescale per action
        self.trace_fast = np.zeros(n_actions)
        self.trace_mid  = np.zeros(n_actions)
        self.trace_slow = np.zeros(n_actions)

        # Weights for combining timescales
        self.w_fast = 0.5
        self.w_mid  = 0.3
        self.w_slow = 0.2

    def reset(self):
        self.trace_fast = np.zeros(self.n_actions)
        self.trace_mid  = np.zeros(self.n_actions)
        self.trace_slow = np.zeros(self.n_actions)

    def step(self, V_current: np.ndarray) -> np.ndarray:
        """
        Update all three timescale traces and return
        the combined memory-modulated belief signal.

        V_current: current belief Va shape (A,)
        Returns  : memory-modulated belief (A,)
        """
        V = np.asarray(V_current, dtype=float)

        # Exponential decay + input injection
        decay_f = np.exp(-self.dt / self.tau_fast)
        decay_m = np.exp(-self.dt / self.tau_mid)
        decay_s = np.exp(-self.dt / self.tau_slow)

        self.trace_fast = decay_f * self.trace_fast + (1 - decay_f) * V
        self.trace_mid  = decay_m * self.trace_mid  + (1 - decay_m) * V
        self.trace_slow = decay_s * self.trace_slow + (1 - decay_s) * V

        # Weighted combination
        memory_signal = (self.w_fast * self.trace_fast
                         + self.w_mid  * self.trace_mid
                         + self.w_slow * self.trace_slow)

        return memory_signal

    def dominant_timescale(self) -> str:
        """Returns the name of the most active trace."""
        magnitudes = {
            "fast" : np.abs(self.trace_fast).mean(),
            "mid"  : np.abs(self.trace_mid).mean(),
            "slow" : np.abs(self.trace_slow).mean(),
        }
        return max(magnitudes, key=magnitudes.get)

    def trace_summary(self) -> dict:
        return {
            "trace_fast" : self.trace_fast.copy(),
            "trace_mid"  : self.trace_mid.copy(),
            "trace_slow" : self.trace_slow.copy(),
            "dominant"   : self.dominant_timescale(),
        }