import numpy as np
from tasks.base_task import BaseTask

class MultiObjectiveTask(BaseTask):
    """
    Actions yield multiple reward signals (speed, safety, energy).
    Agent must balance competing objectives via scalarisation.
    """

    def __init__(self, n_actions=4, max_steps=500,
                 weights=(0.5, 0.3, 0.2)):
        super().__init__("multi_objective", n_actions, max_steps)
        self.weights = np.array(weights)
        # reward table per objective per action
        self.obj_rewards = np.random.rand(n_actions, len(weights))

    def reset(self):
        self.obj_rewards = np.random.rand(self.n_actions, len(self.weights))
        self.step_count  = 0
        self.done        = False
        return self.obj_rewards.flatten()

    def step(self, action: int):
        rewards_vec = self.obj_rewards[action] + \
                      np.random.normal(0, 0.05, len(self.weights))
        scalar_reward = float(np.dot(self.weights, rewards_vec))

        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        info = {"objective_rewards": rewards_vec,
                "scalar_reward"    : scalar_reward,
                "weights"          : self.weights}
        return self.obj_rewards.flatten(), scalar_reward, self.done, info