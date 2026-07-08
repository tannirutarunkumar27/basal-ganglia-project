"""
LearningMetrics  —  Step 33
-----------------------------
Four learning-quality metrics derived from TD errors,
reward curves and weight-change history.

1. reward_convergence : smoothed reward stability at end
2. learning_speed     : mean abs(delta_total) early vs late
3. sample_efficiency  : reward per unit of weight update
4. success_rate       : fraction of episodes with reward > threshold
"""

import numpy as np
from collections import deque


class LearningMetrics:

    def __init__(self,
                 success_threshold: float = 0.7,
                 window           : int   = 200):

        self.success_threshold = success_threshold
        self.window            = window

        self._rewards      = []
        self._deltas       = []
        self._weight_mags  = []
        self._episode_accs = []

    def reset(self) -> None:
        self._rewards     = []
        self._deltas      = []
        self._weight_mags = []
        self._episode_accs = []

    def record_step(self, reward      : float,
                     delta_total      : float,
                     mean_weight_mag  : float) -> None:
        self._rewards.append(float(reward))
        self._deltas.append(float(abs(delta_total)))
        self._weight_mags.append(float(mean_weight_mag))

    def record_episode(self, accuracy: float) -> None:
        self._episode_accs.append(float(accuracy))

    def reward_convergence(self) -> float:
        """
        Coefficient of variation (std/mean) of reward in the
        last `window` steps. Lower = more stable = better.
        Returned as 1 - CV (higher is better convergence).
        """
        arr = np.array(self._rewards[-self.window:])
        if len(arr) < 2:
            return 0.0
        mean = float(arr.mean())
        std  = float(arr.std())
        cv   = std / (abs(mean) + 1e-8)
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def learning_speed(self) -> float:
        """
        Ratio of mean |delta| in early steps vs late steps.
        High early delta / low late delta = rapid convergence.
        Returns a score in [0, 1]; higher = faster learning.
        """
        d = np.array(self._deltas)
        if len(d) < 20:
            return 0.0
        mid      = len(d) // 2
        early    = float(d[:mid].mean()) + 1e-8
        late     = float(d[mid:].mean()) + 1e-8
        ratio    = early / late
        return float(np.clip((ratio - 1.0) / (ratio + 1.0), 0.0, 1.0))

    def sample_efficiency(self) -> float:
        """
        Cumulative reward divided by total weight-update magnitude.
        Higher = more reward extracted per unit of plasticity.
        Normalised to [0, 1] via sigmoid.
        """
        total_reward  = float(np.sum(self._rewards))
        total_updates = float(np.sum(self._weight_mags)) + 1e-8
        raw_eff = total_reward / total_updates
        return float(1.0 / (1.0 + np.exp(-raw_eff * 0.5)))

    def success_rate(self) -> float:
        """
        Fraction of episodes where accuracy exceeded
        success_threshold.
        """
        if not self._episode_accs:
            return 0.0
        return float(
            np.mean(np.array(self._episode_accs)
                    >= self.success_threshold))

    def compute_all(self) -> dict:
        return {
            "reward_convergence": self.reward_convergence(),
            "learning_speed"    : self.learning_speed(),
            "sample_efficiency" : self.sample_efficiency(),
            "success_rate"      : self.success_rate(),
        }