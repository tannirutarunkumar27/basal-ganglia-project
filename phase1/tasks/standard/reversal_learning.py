import numpy as np
from tasks.base_task import BaseTask

class ReversalLearning(BaseTask):
    """
    Two-choice task where the rewarded action flips
    at a random reversal point.
    Tests learning rate adaptation and flexibility.
    """

    def __init__(self, max_steps=600, reversal_prob=0.01):
        super().__init__("reversal_learning", n_actions=2, max_steps=max_steps)
        self.reversal_prob = reversal_prob
        self.correct_action = None

    def reset(self):
        self.correct_action = np.random.randint(0, 2)
        self.step_count = 0
        self.done = False
        return np.array([float(self.correct_action)])

    def step(self, action: int):
        # random reversal
        if np.random.rand() < self.reversal_prob:
            self.correct_action = 1 - self.correct_action

        reward = 1.0 if action == self.correct_action else -0.5
        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        info = {"correct_action": self.correct_action,
                "reversal_prob"  : self.reversal_prob}
        return np.array([float(self.correct_action)]), reward, self.done, info