"""
FiringRateLimiter  —  Step 29 (technique 4)
---------------------------------------------
Limits unnecessary high-frequency firing by:

    1. Population homeostasis: if mean rate > target,
       globally raise spike threshold
    2. Individual neuron AHP: per-neuron after-hyperpolarisation
       that scales with recent firing frequency
    3. Energy penalty gate: neurons exceeding energy budget
       receive extra suppression

Biological analog:
    Homeostatic plasticity (Turrigiano 1999).
    Intrinsic excitability regulation.
    Metabolic cost constraints on firing.
"""

import numpy as np
from collections import deque


class FiringRateLimiter:

    def __init__(self,
                 n_neurons      : int,
                 target_rate_hz : float = 10.0,
                 max_rate_hz    : float = 80.0,
                 tau_homeo      : float = 1000e-3,
                 dt             : float = 0.1e-3,
                 name           : str   = "pop"):

        self.n_neurons     = n_neurons
        self.target_rate   = target_rate_hz
        self.max_rate      = max_rate_hz
        self.dt            = dt
        self.name          = name

        # Population-level homeostatic gain (scalar)
        self.homeo_gain    = 1.0
        self.tau_homeo     = tau_homeo
        self.decay_homeo   = np.exp(-dt / tau_homeo)

        # Per-neuron AHP (after-hyperpolarisation) current
        self.ahp           = np.zeros(n_neurons)
        self.tau_ahp       = 50e-3      # 50 ms
        self.decay_ahp     = np.exp(-dt / self.tau_ahp)
        self.ahp_increment = 0.3        # added on each spike

        # Spike rate estimation per neuron
        self.rate_est      = np.zeros(n_neurons)
        self.tau_rate      = 200e-3
        self.decay_rate    = np.exp(-dt / self.tau_rate)

        # Energy suppression counter
        self.suppressed_count = 0
        self.total_count      = 0
        self.rate_history     = deque(maxlen=2000)

    def reset(self) -> None:
        self.homeo_gain   = 1.0
        self.ahp          = np.zeros(self.n_neurons)
        self.rate_est     = np.zeros(self.n_neurons)
        self.suppressed_count = 0
        self.total_count      = 0

    def step(self, spikes: np.ndarray) -> tuple:
        """
        Applies rate limiting. Returns:
            (gated_spikes, I_ahp, suppress_mask)
        """
        sp = np.asarray(spikes, dtype=bool)
        n  = min(len(sp), self.n_neurons)
        sv = np.zeros(self.n_neurons, dtype=bool)
        sv[:n] = sp[:n]

        self.total_count += int(sv.sum())

        # Update per-neuron rate estimate
        self.rate_est = (self.decay_rate * self.rate_est
                         + (1 - self.decay_rate) * sv.astype(float) / self.dt)

        # Update AHP
        self.ahp = self.decay_ahp * self.ahp
        self.ahp[sv] += self.ahp_increment

        # Population mean rate
        pop_rate = float(self.rate_est.mean())
        self.rate_history.append(pop_rate)

        # Homeostatic adjustment: high rate → raise threshold (lower gain)
        if pop_rate > self.target_rate:
            excess = (pop_rate - self.target_rate) / self.target_rate
            self.homeo_gain = max(0.1,
                self.homeo_gain - 0.01 * excess)
        else:
            self.homeo_gain = min(1.0,
                self.homeo_gain + 0.001)

        # Suppress neurons with rate above max
        suppress = self.rate_est > (self.max_rate / self.dt)
        gated    = sv & ~suppress
        self.suppressed_count += int((sv & suppress).sum())

        # AHP current (inhibitory, added to neuron I)
        I_ahp = -self.ahp * self.homeo_gain

        return gated, I_ahp, suppress

    def suppression_rate(self) -> float:
        return float(self.suppressed_count
                     / max(self.total_count, 1))

    def mean_rate_hz(self) -> float:
        hist = list(self.rate_history)
        return float(np.mean(hist)) if hist else 0.0

    def limiter_summary(self) -> dict:
        return {
            "name"            : self.name,
            "target_rate_hz"  : self.target_rate,
            "mean_rate_hz"    : self.mean_rate_hz(),
            "homeo_gain"      : float(self.homeo_gain),
            "suppression_rate": self.suppression_rate(),
        }