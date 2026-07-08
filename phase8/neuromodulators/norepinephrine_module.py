"""
NorepinephrineModule
--------------------
NE (noradrenaline) from the locus coeruleus regulates:
  - arousal and vigilance
  - response to surprising / unexpected events
  - gain modulation of neural responses
  - uncertainty-driven exploration

High NE:
  - high arousal -> faster responses
  - increases exploration temperature
  - boosts STN conflict sensitivity
  - raises uncertainty detection threshold

Low NE:
  - low arousal / habitual mode
  - reduced sensitivity to novelty
  - exploitative behaviour

Phasic NE bursts occur at unexpected stimuli.
Tonic NE tracks overall arousal / task engagement.
"""

import numpy as np
from collections import deque


class NorepinephrineModule:

    def __init__(self,
                 baseline     : float = 0.4,
                 tau_ne       : float = 300e-3,
                 surprise_gain: float = 2.0,
                 dt           : float = 0.1e-3):

        self.baseline      = baseline
        self.tau_ne        = tau_ne
        self.surprise_gain = surprise_gain
        self.dt            = dt

        self.NE_level      = baseline
        self.decay         = np.exp(-dt / tau_ne)

        # Running prediction error history for surprise estimation
        self.rpe_history   = deque(maxlen=100)
        self.history       = deque(maxlen=5000)

    def reset(self) -> None:
        self.NE_level = self.baseline
        self.rpe_history.clear()
        self.history.clear()

    def update(self, delta_prime: float,
                U: float,
                volatility: float = 0.0) -> float:
        """
        Updates NE from surprise (|delta|), uncertainty, volatility.

        delta_prime : prediction error magnitude = surprise signal
        U           : uncertainty Ut
        volatility  : reward volatility from MetaDopamine

        Returns NE level in [0, 1].
        """
        surprise = abs(float(delta_prime))
        self.rpe_history.append(surprise)

        # Mean surprise over recent history
        mean_surprise = float(np.mean(list(self.rpe_history)))

        # Phasic: current surprise burst
        phasic    = self.surprise_gain * surprise * 0.3

        # Tonic: driven by uncertainty and volatility
        tonic_U   = 0.3 * float(np.clip(U, 0.0, 1.0))
        tonic_vol = 0.2 * float(np.clip(volatility, 0.0, 1.0))

        target_NE = float(np.clip(
            self.baseline + phasic + tonic_U + tonic_vol,
            0.0, 1.0))

        self.NE_level = (self.decay * self.NE_level
                         + (1.0 - self.decay) * target_NE)
        self.NE_level = float(np.clip(self.NE_level, 0.0, 1.0))

        self.history.append(self.NE_level)
        return self.NE_level

    def arousal_level(self) -> str:
        """Classifies current arousal: low / moderate / high."""
        if self.NE_level > 0.65:
            return "high"
        elif self.NE_level > 0.35:
            return "moderate"
        else:
            return "low"

    def gain_modulation(self) -> float:
        """
        Returns neural gain multiplier [0.8, 2.0].
        High NE increases gain of all neural responses.
        """
        return float(0.8 + 1.2 * self.NE_level)