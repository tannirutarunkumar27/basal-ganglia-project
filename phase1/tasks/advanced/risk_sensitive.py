import numpy as np
from tasks.base_task import BaseTask

class RiskSensitiveChoice(BaseTask):
    """
    Two-option gamble: safe (moderate fixed reward) vs
    risky (high variance reward). Tests risk-sensitive RL.
    The optimal strategy changes with variance penalty rho.
    """

    def __init__(self, max_steps=500, rho=0.5):
        super().__init__("risk_sensitive_choice", n_actions=2, max_steps=max_steps)
        self.rho = rho          # risk aversion coefficient

    def reset(self):
        self.step_count = 0
        self.done       = False
        return np.array([self.rho])

    def step(self, action: int):
        if action == 0:         # safe option
            reward   = 1.0
            variance = 0.01
        else:                   # risky option
            reward   = np.random.choice([3.0, -1.0], p=[0.5, 0.5])
            variance = 4.0

        # risk-adjusted utility: Q^risk = E[R] - rho * Var(R)
        utility = reward - self.rho * variance

        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        info = {"raw_reward": reward, "variance": variance,
                "risk_utility": utility, "rho": self.rho}
        return np.array([self.rho]), utility, self.done, info