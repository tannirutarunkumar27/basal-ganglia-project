"""
PriorTracker
------------
Maintains and updates P(a) — the learned prior over actions.
Updated by reward history using a running exponential average.
This encodes the agent's long-term preference / habit.
"""

import numpy as np


class PriorTracker:

    def __init__(self, n_actions: int,
                 alpha_prior: float = 0.05):
        """
        n_actions   : number of actions
        alpha_prior : learning rate for prior update (0 < α ≤ 1)
        """
        self.n_actions   = n_actions
        self.alpha_prior = alpha_prior

        # Uniform initialisation — no preference
        self.prior = np.ones(n_actions) / n_actions

        # Count of how many times each action has been rewarded
        self.reward_counts  = np.zeros(n_actions)
        self.selection_counts = np.zeros(n_actions)

    def reset(self):
        self.prior            = np.ones(self.n_actions) / self.n_actions
        self.reward_counts    = np.zeros(self.n_actions)
        self.selection_counts = np.zeros(self.n_actions)

    def update(self, action: int, reward: float) -> np.ndarray:
        """
        Update prior after observing (action, reward).
        Uses exponential moving average towards empirical frequencies.
        Returns updated prior (A,).
        """
        self.selection_counts[action] += 1
        self.reward_counts[action]    += max(reward, 0.0)

        # Empirical reward-weighted frequency
        total = self.reward_counts.sum() + 1e-8
        empirical = self.reward_counts / total

        # EMA update
        self.prior = ((1 - self.alpha_prior) * self.prior
                      + self.alpha_prior * empirical)

        # Renormalise to valid probability distribution
        self.prior = np.clip(self.prior, 1e-8, None)
        self.prior /= self.prior.sum()

        return self.prior.copy()

    def log_prior(self) -> np.ndarray:
        """log P(a) for each action."""
        return np.log(self.prior + 1e-8)

    def entropy(self) -> float:
        """Shannon entropy of the prior (nats)."""
        p = self.prior + 1e-8
        return float(-np.sum(p * np.log(p)))