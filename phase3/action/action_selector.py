"""
ActionSelector
--------------
Implements Step 7:
    P(a|s) = softmax(Va) = exp(Va) / sum_j exp(Vj)

Converts temporal belief scores Va(t) into:
  1. A probability distribution P(a|s) over actions
  2. A sampled action (probabilistic or greedy)
  3. A ranked list of action preferences
  4. Soft competition metric (how peaked the distribution is)
"""

import numpy as np


class ActionSelector:

    def __init__(self,
                 n_actions  : int,
                 temperature: float = 1.0,
                 min_temp   : float = 0.1,
                 max_temp   : float = 5.0):
        """
        n_actions   : number of possible actions
        temperature : softmax temperature τ
                      τ → 0  = greedy (argmax)
                      τ → ∞  = uniform (maximum exploration)
        """
        self.n_actions   = n_actions
        self.temperature = temperature
        self.min_temp    = min_temp
        self.max_temp    = max_temp

        # Current probability distribution (A,)
        self.prob = np.ones(n_actions) / n_actions

        # Selection history
        self.action_history     = []
        self.prob_history       = []
        self.selection_counts   = np.zeros(n_actions, dtype=int)

    def reset(self):
        self.prob             = np.ones(self.n_actions) / self.n_actions
        self.action_history   = []
        self.prob_history     = []
        self.selection_counts = np.zeros(self.n_actions, dtype=int)

    def softmax(self, V: np.ndarray,
                temperature: float = None) -> np.ndarray:
        """
        Numerically stable softmax.
        P(a|s) = exp((Va - max(V)) / τ) / Z
        """
        tau = temperature if temperature is not None else self.temperature
        tau = np.clip(tau, self.min_temp, self.max_temp)

        V_shifted = (np.asarray(V, dtype=float) - np.max(V)) / tau
        exp_V     = np.exp(np.clip(V_shifted, -50, 50))
        return exp_V / (exp_V.sum() + 1e-12)

    def select(self, V: np.ndarray,
               mode    : str   = "probabilistic",
               temp_override: float = None) -> tuple:
        """
        Main selection method.

        V    : belief scores Va shape (A,)
        mode : "probabilistic" — sample from P(a|s)
               "greedy"        — argmax
               "epsilon_greedy"— greedy with ε-noise
        Returns: (action: int, prob: np.ndarray, info: dict)
        """
        tau       = temp_override if temp_override else self.temperature
        self.prob = self.softmax(V, tau)

        if mode == "greedy":
            action = int(np.argmax(self.prob))

        elif mode == "epsilon_greedy":
            eps = 0.1
            if np.random.rand() < eps:
                action = np.random.randint(self.n_actions)
            else:
                action = int(np.argmax(self.prob))

        else:   # probabilistic (default)
            action = int(np.random.choice(self.n_actions, p=self.prob))

        self.action_history.append(action)
        self.prob_history.append(self.prob.copy())
        self.selection_counts[action] += 1

        info = {
            "prob"          : self.prob.copy(),
            "action"        : action,
            "action_prob"   : float(self.prob[action]),
            "entropy"       : self._entropy(),
            "confidence"    : float(np.max(self.prob)),
            "ranked_actions": np.argsort(self.prob)[::-1].tolist(),
            "temperature"   : tau,
        }
        return action, self.prob.copy(), info

    def adapt_temperature(self, uncertainty: float) -> float:
        """
        Adjust temperature based on uncertainty:
          high uncertainty -> higher τ (more exploration)
          low uncertainty  -> lower τ (more exploitation)
        """
        # Linear mapping: U ∈ [0,1] → τ ∈ [min_temp, max_temp]
        self.temperature = (self.min_temp
                            + uncertainty * (self.max_temp - self.min_temp))
        return self.temperature

    def _entropy(self) -> float:
        p = self.prob + 1e-12
        return float(-np.sum(p * np.log(p)))

    def soft_competition_score(self) -> float:
        """
        Measures how peaked the distribution is.
        1.0 = all mass on one action (maximum competition won).
        0.0 = uniform distribution (no competition resolved).
        """
        p_max  = np.max(self.prob)
        p_unif = 1.0 / self.n_actions
        return float((p_max - p_unif) / (1.0 - p_unif + 1e-8))

    def selection_summary(self) -> dict:
        total = max(self.selection_counts.sum(), 1)
        return {
            "selection_freq"        : self.selection_counts / total,
            "current_prob"          : self.prob.copy(),
            "temperature"           : self.temperature,
            "soft_competition_score": self.soft_competition_score(),
            "total_selections"      : int(total),
        }