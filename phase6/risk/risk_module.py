"""
RiskModule  —  Step 18
-----------------------
Q_a^risk = E[R_a] - rho * Var(R_a)

where:
  E[R_a]   = expected reward for action a  (exploitation)
  Var(R_a) = reward variance for action a  (risk measure)
  rho      = risk aversion coefficient (from uncertainty Ut)

Advanced innovation:
  rho is not fixed — it is dynamically adapted from:
    - uncertainty Ut   : high Ut → more risk averse
    - conflict score   : high conflict → more risk averse
    - episode history  : after a bad outcome → more risk averse

The risk-adjusted utility modifies action selection by
biasing the agent away from high-variance outcomes even
when their expected return is similar to safe alternatives.
"""

import numpy as np
from collections import deque


class RiskModule:

    def __init__(self,
                 n_actions   : int,
                 rho_base    : float = 0.5,
                 rho_min     : float = 0.05,
                 rho_max     : float = 2.0,
                 window      : int   = 50,
                 dt          : float = 0.1e-3):
        """
        n_actions  : number of action channels
        rho_base   : base risk aversion coefficient
        rho_min    : minimum rho (pure exploitation mode)
        rho_max    : maximum rho (extreme risk aversion)
        window     : rolling window for variance estimation
        dt         : simulation timestep
        """
        self.n_actions = n_actions
        self.rho_base  = rho_base
        self.rho_min   = rho_min
        self.rho_max   = rho_max
        self.window    = window
        self.dt        = dt

        # Current rho
        self.rho       = rho_base

        # Rolling reward history per action
        self.r_history = {a: deque(maxlen=window)
                          for a in range(n_actions)}

        # Running statistics per action
        self.E_r    = np.zeros(n_actions)    # E[R_a]
        self.Var_r  = np.zeros(n_actions)    # Var(R_a)

        # Risk-adjusted Q values
        self.Q_risk = np.zeros(n_actions)

        # History
        self.rho_history    = []
        self.q_risk_history = []

    def reset(self):
        self.rho     = self.rho_base
        for a in range(self.n_actions):
            self.r_history[a].clear()
        self.E_r         = np.zeros(self.n_actions)
        self.Var_r       = np.zeros(self.n_actions)
        self.Q_risk      = np.zeros(self.n_actions)
        self.rho_history.clear()
        self.q_risk_history.clear()

    def update_statistics(self, action: int,
                           reward: float) -> None:
        """
        Updates rolling E[R_a] and Var(R_a) for the given action.
        Uses Welford's online algorithm for numerical stability.
        """
        self.r_history[action].append(reward)
        hist = list(self.r_history[action])

        if len(hist) > 0:
            self.E_r[action]   = float(np.mean(hist))
            self.Var_r[action] = (float(np.var(hist))
                                  if len(hist) > 1 else 0.0)

    def adapt_rho(self, U: float,
                   conflict_score: float,
                   recent_loss: float = 0.0) -> float:
        """
        Dynamically adjusts rho based on:
          - uncertainty U    : high U → more cautious
          - conflict_score   : high conflict → more cautious
          - recent_loss      : negative reward history → more cautious

        rho(t) = rho_base * (1 + w_U*U + w_c*conflict + w_l*loss)
        """
        conflict_norm = float(np.clip(conflict_score / 2.0, 0.0, 1.0))
        loss_penalty  = float(np.clip(-recent_loss, 0.0, 1.0))

        rho_target = self.rho_base * (
            1.0 + 1.0 * U
                + 0.5 * conflict_norm
                + 0.3 * loss_penalty)

        # Smooth rho update
        self.rho = float(np.clip(
            0.9 * self.rho + 0.1 * rho_target,
            self.rho_min, self.rho_max))

        self.rho_history.append(self.rho)
        return self.rho

    def compute_q_risk(self) -> np.ndarray:
        """
        Core Step 18:
            Q_a^risk = E[R_a] - rho * Var(R_a)

        Returns risk-adjusted utility per action (A,).
        """
        self.Q_risk = self.E_r - self.rho * self.Var_r
        self.q_risk_history.append(self.Q_risk.copy())
        return self.Q_risk.copy()

    def risk_adjusted_action_probs(self,
                                    base_probs: np.ndarray,
                                    blend: float = 0.4) -> np.ndarray:
        """
        Blends base action probabilities with risk-adjusted scores.
        blend=0.4 means 40% risk adjustment, 60% base policy.

        base_probs : (A,) from actor or Bayesian pipeline
        blend      : how much risk adjustment influences selection
        Returns    : (A,) modified probability distribution
        """
        bp = np.asarray(base_probs, dtype=float)

        # Softmax of Q_risk
        q  = self.Q_risk - self.Q_risk.max()
        eq = np.exp(np.clip(q, -20, 20))
        risk_probs = eq / (eq.sum() + 1e-12)

        # Blend
        mixed = (1 - blend) * bp + blend * risk_probs
        mixed = np.clip(mixed, 1e-8, None)
        return mixed / mixed.sum()

    def variance_ranking(self) -> np.ndarray:
        """Returns actions sorted by variance (lowest first = safest)."""
        return np.argsort(self.Var_r)

    def risk_summary(self) -> dict:
        return {
            "rho"         : float(self.rho),
            "E_r"         : self.E_r.copy(),
            "Var_r"       : self.Var_r.copy(),
            "Q_risk"      : self.Q_risk.copy(),
            "safest_action": int(np.argmin(self.Var_r)),
            "best_q_action": int(np.argmax(self.Q_risk)),
        }