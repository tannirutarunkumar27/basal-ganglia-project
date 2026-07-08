import numpy as np
from tasks.base_task import BaseTask

class DelayedRewardTask(BaseTask):
    """
    Agent chooses at step 0; reward only arrives after a
    random delay of D steps. Tests STDE credit assignment.
    """

    def __init__(self, max_steps=500, max_delay=20, n_actions=3):
        super().__init__("delayed_reward", n_actions, max_steps)
        self.max_delay      = max_delay
        self.pending_rewards = []   # (deliver_at_step, value)
        self.true_probs      = None

    def reset(self):
        self.true_probs      = np.random.dirichlet(np.ones(self.n_actions))
        self.pending_rewards = []
        self.step_count      = 0
        self.done            = False
        return np.zeros(self.n_actions)

    def step(self, action: int):
        # schedule reward delivery after random delay
        delay      = np.random.randint(1, self.max_delay + 1)
        value      = float(np.random.rand() < self.true_probs[action])
        deliver_at = self.step_count + delay
        self.pending_rewards.append((deliver_at, value))

        # deliver any matured rewards
        reward = 0.0
        remaining = []
        for (d_at, v) in self.pending_rewards:
            if d_at <= self.step_count:
                reward += v
            else:
                remaining.append((d_at, v))
        self.pending_rewards = remaining

        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        info = {"delay": delay, "pending": len(self.pending_rewards)}
        return np.zeros(self.n_actions), reward, self.done, info