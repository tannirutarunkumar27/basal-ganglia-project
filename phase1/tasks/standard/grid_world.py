import numpy as np
from tasks.base_task import BaseTask

class GridWorld(BaseTask):
    """
    NxN grid. Agent navigates from start to goal.
    Optional obstacles. Tests spatial planning.
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    """

    def __init__(self, grid_size=5, max_steps=200,
                 obstacle_prob=0.15):
        super().__init__("grid_world", n_actions=4, max_steps=max_steps)
        self.grid_size     = grid_size
        self.obstacle_prob = obstacle_prob
        self.grid          = None
        self.agent_pos     = None
        self.goal_pos      = None

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # place obstacles randomly (not at corners)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in [(0,0),(self.grid_size-1,self.grid_size-1)]:
                    if np.random.rand() < self.obstacle_prob:
                        self.grid[r, c] = -1

        self.agent_pos = [0, 0]
        self.goal_pos  = [self.grid_size-1, self.grid_size-1]
        self.step_count = 0
        self.done       = False
        return self._get_obs()

    def _get_obs(self):
        obs = self.grid.flatten().copy()
        obs[self.agent_pos[0]*self.grid_size + self.agent_pos[1]] = 2.0
        obs[self.goal_pos[0] *self.grid_size + self.goal_pos[1]]  = 3.0
        return obs

    def step(self, action: int):
        moves = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
        dr, dc = moves[action]
        nr = np.clip(self.agent_pos[0] + dr, 0, self.grid_size-1)
        nc = np.clip(self.agent_pos[1] + dc, 0, self.grid_size-1)

        if self.grid[nr, nc] != -1:      # not an obstacle
            self.agent_pos = [nr, nc]

        if self.agent_pos == self.goal_pos:
            reward    = 10.0
            self.done = True
        else:
            reward = -0.1                # step penalty

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        info = {"agent": self.agent_pos, "goal": self.goal_pos}
        return self._get_obs(), reward, self.done, info