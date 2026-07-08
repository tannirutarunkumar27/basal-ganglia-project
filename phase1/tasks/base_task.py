import numpy as np
from abc import ABC, abstractmethod

class BaseTask(ABC):
    """
    Abstract base for all benchmark tasks.
    Every task exposes: reset(), step(), and task_info().
    """

    def __init__(self, name: str, n_actions: int, max_steps: int):
        self.name      = name
        self.n_actions = n_actions
        self.max_steps = max_steps
        self.step_count = 0
        self.done = False

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Returns initial state as a numpy array."""
        ...

    @abstractmethod
    def step(self, action: int):
        """
        Applies action, returns (next_state, reward, done, info).
        info is a dict with any task-specific diagnostics.
        """
        ...

    def task_info(self) -> dict:
        return {
            "name"      : self.name,
            "n_actions" : self.n_actions,
            "max_steps" : self.max_steps,
        }