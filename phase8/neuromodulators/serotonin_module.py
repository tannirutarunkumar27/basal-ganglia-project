"""
SerotoninModule
---------------
5-HT regulates risk aversion and punishment avoidance.

High 5-HT:
  - increases patience (longer time horizons)
  - reduces risk-taking
  - suppresses impulsive action selection

Low 5-HT:
  - increases impulsivity
  - reduces punishment sensitivity
  - biases toward immediate rewards

Source: raphe nuclei.
Biological timescale: slow (minutes), but
phasic responses on seconds scale.
"""

import numpy as np
from collections import deque


class SerotoninModule:

    def __init__(self,
                 baseline  : float = 0.5,
                 tau_5ht   : float = 500e-3,
                 risk_gain : float = 1.5,
                 dt        : float = 0.1e-3):

        self.baseline  = baseline
        self.tau_5ht   = tau_5ht
        self.risk_gain = risk_gain
        self.dt        = dt

        self.ht5_level = baseline
        self.decay     = np.exp(-dt / tau_5ht)

        # Running loss history for punishment sensitivity
        self.loss_hist = deque(maxlen=200)
        self.history   = deque(maxlen=5000)

    def reset(self) -> None:
        self.ht5_level = self.baseline
        self.loss_hist.clear()
        self.history.clear()

    def update(self, reward: float,
                rho: float = 0.5,
                conflict_score: float = 0.1) -> float:
        """
        Updates 5-HT from reward sign, risk level, and conflict.

        reward         : observed reward (negative = punishment)
        rho            : risk aversion coefficient from Phase 6
        conflict_score : from Phase 4 hyperdirect pathway

        Returns 5-HT level in [0, 1].
        """
        self.loss_hist.append(min(reward, 0.0))

        # Recent punishment history
        mean_loss = abs(float(np.mean(list(self.loss_hist))))

        # High rho + recent punishment -> increase 5-HT
        # (more cautious, more risk-averse)
        rho_contrib  = self.risk_gain * rho * 0.3
        loss_contrib = self.risk_gain * mean_loss * 0.2
        conf_contrib = 0.1 * float(np.clip(
            conflict_score / 2.0, 0.0, 1.0))

        target_5ht = float(np.clip(
            self.baseline + rho_contrib
            + loss_contrib - conf_contrib,
            0.0, 1.0))

        self.ht5_level = (self.decay * self.ht5_level
                          + (1.0 - self.decay) * target_5ht)
        self.ht5_level = float(np.clip(self.ht5_level, 0.0, 1.0))

        self.history.append(self.ht5_level)
        return self.ht5_level

    def patience_factor(self) -> float:
        """
        Returns gamma scaling factor [0.9, 0.999].
        High 5-HT -> higher gamma (longer time horizon).
        """
        return float(0.9 + 0.099 * self.ht5_level)