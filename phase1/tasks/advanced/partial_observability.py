import numpy as np
from tasks.base_task import BaseTask

class PartialObservabilityTask(BaseTask):
    """
    Hidden-state bandit: state is latent; observation is noisy.
    Forces exploration and Bayesian belief updating.
    """

    def __init__(self, n_states=3, n_actions=3,
                 max_steps=400, obs_noise=0.3):
        super().__init__("partial_observability", n_actions, max_steps)
        self.n_states  = n_states
        self.obs_noise = obs_noise
        self.true_state = None
        # reward table: reward[state][action]
        self.reward_table = np.random.rand(n_states, n_actions)

    def reset(self):
        self.true_state = np.random.randint(0, self.n_states)
        self.step_count = 0
        self.done       = False
        return self._get_obs()

    def _get_obs(self):
        one_hot = np.eye(self.n_states)[self.true_state]
        noisy   = one_hot + np.random.normal(0, self.obs_noise, self.n_states)
        return np.clip(noisy, 0, 1)     # noisy partial observation

    def step(self, action: int):
        # randomly transition hidden state
        if np.random.rand() < 0.1:
            self.true_state = np.random.randint(0, self.n_states)

        reward = self.reward_table[self.true_state, action]
        reward += np.random.normal(0, 0.2)

        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        info = {"true_state": self.true_state}
        return self._get_obs(), reward, self.done, info