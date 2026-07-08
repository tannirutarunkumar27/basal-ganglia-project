import numpy as np
from tasks.base_task import BaseTask

class SequentialDecisionTask(BaseTask):
    """
    Multi-step chain task: correct sequence of actions
    yields a large final reward. Wrong step resets.
    Tests temporal credit assignment and planning.
    """

    def __init__(self, sequence_length=4, n_actions=3, max_steps=500):
        super().__init__("sequential_decision", n_actions, max_steps)
        self.seq_len        = sequence_length
        self.target_seq     = None
        self.current_pos    = 0

    def reset(self):
        self.target_seq  = np.random.randint(0, self.n_actions, self.seq_len)
        self.current_pos = 0
        self.step_count  = 0
        self.done        = False
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros(self.n_actions + 1)
        obs[-1] = self.current_pos / self.seq_len
        if self.current_pos < self.seq_len:
            obs[self.target_seq[self.current_pos]] = 1.0
        return obs

    def step(self, action: int):
        correct = (action == self.target_seq[self.current_pos])

        if correct:
            self.current_pos += 1
            if self.current_pos == self.seq_len:
                reward   = 5.0       # large terminal reward
                self.done = True
            else:
                reward = 0.1         # small step reward
        else:
            reward           = -1.0
            self.current_pos = 0     # reset to start

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        info = {"position": self.current_pos, "target": int(self.target_seq[min(self.current_pos, self.seq_len-1)])}
        return self._get_obs(), reward, self.done, info