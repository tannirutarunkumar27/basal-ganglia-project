import numpy as np
from tasks.base_task import BaseTask

class ProbabilisticBandit(BaseTask):
    """
    k-armed bandit with stochastic reward probabilities.
    Tests basic RL convergence and reward learning.
    """

    def __init__(self, k=4, max_steps=500, noise_std=0.1):
        super().__init__("probabilistic_bandit", n_actions=k, max_steps=max_steps)
        self.k         = k
        self.noise_std = noise_std
        self.true_probs = None

    def reset(self):
        self.true_probs = np.random.dirichlet(np.ones(self.k))
        self.step_count = 0
        self.done       = False
        return np.zeros(self.k)     # no state in bandit

    def step(self, action: int):
        reward = float(np.random.rand() < self.true_probs[action])
        reward += np.random.normal(0, self.noise_std)
        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        info = {"true_prob": self.true_probs[action],
                "optimal_action": int(np.argmax(self.true_probs))}
        return np.zeros(self.k), reward, self.done, info