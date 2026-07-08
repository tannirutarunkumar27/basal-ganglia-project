import numpy as np
from tasks.base_task import BaseTask

class CounterfactualTask(BaseTask):
    """
    Records what reward would have been obtained for every
    un-chosen action. Enables counterfactual reasoning training.
    """

    def __init__(self, k=4, max_steps=500):
        super().__init__("counterfactual_evaluation", k, max_steps)
        self.k          = k
        self.true_probs = None

    def reset(self):
        self.true_probs = np.random.dirichlet(np.ones(self.k))
        self.step_count = 0
        self.done       = False
        return np.zeros(self.k)

    def step(self, action: int):
        # sample ALL action rewards (counterfactuals included)
        all_rewards = np.array([
            float(np.random.rand() < p)
            for p in self.true_probs
        ])
        reward = all_rewards[action]

        self.step_count += 1
        self.done = self.step_count >= self.max_steps

        info = {
            "all_rewards"        : all_rewards,
            "counterfactuals"    : {i: all_rewards[i]
                                    for i in range(self.k) if i != action},
            "optimal_action"     : int(np.argmax(self.true_probs)),
            "regret"             : float(np.max(all_rewards) - reward),
        }
        return np.zeros(self.k), reward, self.done, info