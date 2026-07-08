"""
VolatileRewardTask — reward probabilities shift every ~30 steps.
Tests adaptation to environmental volatility.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "../../phase1"))

from tasks.base_task import BaseTask


class VolatileRewardTask(BaseTask):

    def __init__(self, k=4, max_steps=3000,
                 shift_interval=30, shift_std=0.2):
        super().__init__("volatile_reward", k, max_steps)
        self.k             = k
        self.shift_interval = shift_interval
        self.shift_std     = shift_std
        self.true_probs    = None
        self.steps_since_shift = 0

    def reset(self):
        self.true_probs    = np.random.dirichlet(np.ones(self.k))
        self.step_count    = 0
        self.done          = False
        self.steps_since_shift = 0
        return np.zeros(self.k)

    def step(self, action: int):
        # Probabilistic shift
        if self.steps_since_shift >= self.shift_interval:
            noise = np.random.randn(self.k) * self.shift_std
            self.true_probs = np.clip(
                self.true_probs + noise, 0.01, None)
            self.true_probs /= self.true_probs.sum()
            self.steps_since_shift = 0

        reward = float(np.random.rand() < self.true_probs[action])
        self.step_count        += 1
        self.steps_since_shift += 1
        self.done = self.step_count >= self.max_steps
        info = {
            "true_probs"  : self.true_probs.copy(),
            "optimal"     : int(np.argmax(self.true_probs)),
        }
        return np.zeros(self.k), reward, self.done, info