import numpy as np
from tasks.base_task import BaseTask

class StopSignalTask(BaseTask):
    """
    Go/No-Go task with a stop signal on a fraction of trials.
    Tests hyperdirect pathway (STN conflict control).
    """

    def __init__(self, max_steps=400, stop_prob=0.25):
        super().__init__("stop_signal", n_actions=2, max_steps=max_steps)
        self.stop_prob   = stop_prob    # probability of stop signal
        self.is_stop     = False

    def reset(self):
        self.step_count = 0
        self.done       = False
        self.is_stop    = False
        return self._get_obs()

    def _get_obs(self):
        self.is_stop = np.random.rand() < self.stop_prob
        return np.array([1.0, float(self.is_stop)])   # [go_cue, stop_signal]

    def step(self, action: int):
        # action 0 = Go, action 1 = Stop
        if self.is_stop:
            reward = 1.0 if action == 1 else -1.0     # must inhibit on stop trial
        else:
            reward = 1.0 if action == 0 else -0.5     # must respond on go trial

        self.step_count += 1
        self.done = self.step_count >= self.max_steps
        obs = self._get_obs()
        info = {"stop_signal": self.is_stop}
        return obs, reward, self.done, info