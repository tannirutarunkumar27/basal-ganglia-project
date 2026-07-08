"""
BehavioralMetrics  —  Step 33
-------------------------------
Computes all five behavioral metrics from a step log.

1. accuracy          : % optimal actions
2. reward_rate       : mean reward per step
3. convergence_speed : step reaching stable performance
4. regret            : cumulative gap vs optimal
5. error_recovery    : mean steps to recover after mistake
"""

import numpy as np
from collections import deque


class BehavioralMetrics:

    def __init__(self,
                 n_actions         : int,
                 optimal_reward    : float = 1.0,
                 stability_window  : int   = 200,
                 stability_threshold: float = 0.65,
                 recovery_window   : int   = 50):

        self.n_actions          = n_actions
        self.optimal_reward     = optimal_reward
        self.stability_window   = stability_window
        self.stability_threshold = stability_threshold
        self.recovery_window    = recovery_window

        # Running buffers
        self._rewards    = []
        self._correct    = []
        self._actions    = []

    def reset(self) -> None:
        self._rewards = []
        self._correct = []
        self._actions = []

    def record(self, action: int, reward: float,
                correct_action: int) -> None:
        self._rewards.append(float(reward))
        self._correct.append(int(action == correct_action))
        self._actions.append(int(action))

    def accuracy(self) -> float:
        """Percentage of correct / optimal action selections."""
        if not self._correct:
            return 0.0
        return float(np.mean(self._correct))

    def reward_rate(self) -> float:
        """Mean reward per step over the evaluation period."""
        if not self._rewards:
            return 0.0
        return float(np.mean(self._rewards))

    def convergence_speed(self) -> int:
        """
        First step at which accuracy exceeded stability_threshold
        for stability_window consecutive steps.
        Returns -1 if never converged.
        """
        arr = np.array(self._correct, dtype=float)
        w   = self.stability_window
        for i in range(len(arr) - w):
            if arr[i:i + w].mean() >= self.stability_threshold:
                return int(i)
        return -1

    def regret(self) -> float:
        """
        Cumulative regret = sum(optimal_reward - reward_t).
        """
        arr = np.array(self._rewards)
        return float(np.sum(
            np.maximum(self.optimal_reward - arr, 0.0)))

    def error_recovery(self) -> float:
        """
        Mean number of steps to regain correct action after
        a mistake. Returns nan if no mistakes found.
        """
        arr     = np.array(self._correct)
        recoveries = []
        i = 0
        while i < len(arr) - 1:
            if arr[i] == 0:        # mistake
                steps = 0
                j = i + 1
                while j < min(i + self.recovery_window, len(arr)):
                    steps += 1
                    if arr[j] == 1:
                        recoveries.append(steps)
                        break
                    j += 1
                i = j
            else:
                i += 1
        return float(np.mean(recoveries)) if recoveries else float("nan")

    def compute_all(self) -> dict:
        return {
            "accuracy"         : self.accuracy(),
            "reward_rate"      : self.reward_rate(),
            "convergence_speed": self.convergence_speed(),
            "regret"           : self.regret(),
            "error_recovery"   : self.error_recovery(),
        }