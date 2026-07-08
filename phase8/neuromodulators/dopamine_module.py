"""
DopamineModule
--------------
Tracks DA level from:
  - Phase 6 delta_prime (predictive dopamine signal)
  - Phase 2 SNc spike rates
  - Tonic baseline modulated by reward history

DA regulates: learning rate, exploitation strength,
              direct pathway Go signal.
"""

import numpy as np
from collections import deque


class DopamineModule:

    def __init__(self,
                 tonic_baseline: float = 1.0,
                 burst_gain    : float = 2.0,
                 dip_gain      : float = 0.5,
                 tau_da        : float = 200e-3,
                 dt            : float = 0.1e-3):

        self.tonic_baseline = tonic_baseline
        self.burst_gain     = burst_gain
        self.dip_gain       = dip_gain
        self.tau_da         = tau_da
        self.dt             = dt

        self.DA_level       = tonic_baseline
        self.decay          = np.exp(-dt / tau_da)

        self.history        = deque(maxlen=5000)

    def reset(self) -> None:
        self.DA_level = self.tonic_baseline
        self.history.clear()

    def update(self, delta_prime: float,
                snc_rate_hz : float = 4.0) -> float:
        """
        Updates DA level from prediction error and SNc rate.

        delta_prime : enriched TD error from Phase 6
        snc_rate_hz : SNc population firing rate (Hz)

        Returns DA level (normalised around 1.0).
        """
        # SNc contribution: tonic + phasic
        snc_contrib = self.tonic_baseline * (snc_rate_hz / 4.0)

        # Prediction error contribution
        if delta_prime > 0:
            # Burst: positive RPE
            rpe_contrib = self.burst_gain * delta_prime
        else:
            # Dip: negative RPE
            rpe_contrib = self.dip_gain  * delta_prime

        target_DA = float(np.clip(
            snc_contrib + rpe_contrib, 0.1, 5.0))

        # Exponential decay toward baseline
        self.DA_level = (self.decay * self.DA_level
                         + (1.0 - self.decay) * target_DA)
        self.DA_level = float(np.clip(self.DA_level, 0.1, 5.0))

        self.history.append(self.DA_level)
        return self.DA_level

    def normalised(self) -> float:
        """DA level normalised to [0, 1]."""
        return float(np.clip(
            (self.DA_level - 0.1) / (5.0 - 0.1), 0.0, 1.0))