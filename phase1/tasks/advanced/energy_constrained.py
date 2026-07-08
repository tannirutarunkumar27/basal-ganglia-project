import numpy as np
from tasks.base_task import BaseTask

class EnergyConstrainedTask(BaseTask):
    """
    Agent has a finite energy budget per episode.
    High-reward actions cost more energy.
    Tests energy-efficient neuromorphic optimisation.
    """

    def __init__(self, n_actions=4, max_steps=400,
                 energy_budget=100.0):
        super().__init__("energy_constrained", n_actions, max_steps)
        self.energy_budget   = energy_budget
        self.energy_left     = energy_budget
        # cost and reward per action
        self.action_costs    = np.linspace(1.0, 10.0, n_actions)
        self.action_rewards  = np.linspace(0.5,  5.0, n_actions)

    def reset(self):
        self.energy_left = self.energy_budget
        self.step_count  = 0
        self.done        = False
        return np.array([self.energy_left / self.energy_budget])

    def step(self, action: int):
        cost   = self.action_costs[action]
        reward = self.action_rewards[action]

        if self.energy_left >= cost:
            self.energy_left -= cost
        else:
            reward = -1.0           # penalty for exceeding budget

        efficiency = reward / max(cost, 1e-9)

        self.step_count += 1
        self.done = (self.step_count >= self.max_steps or
                     self.energy_left <= 0)
        info = {"cost": cost, "energy_left": self.energy_left,
                "efficiency": efficiency}
        return np.array([self.energy_left / self.energy_budget]), \
               reward, self.done, info