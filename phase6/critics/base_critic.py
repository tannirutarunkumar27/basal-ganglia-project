"""
BaseCritic — shared TD-learning logic for all five critics.

Each critic k maintains:
  - Value function  V_k(s)  encoded as a linear function of state
  - TD error        delta_k = r_k + gamma * V_k(s') - V_k(s)
  - Eligibility trace for STDE credit assignment
"""

import numpy as np
from collections import deque


class BaseCritic:

    def __init__(self,
                 name          : str,
                 state_dim     : int,
                 n_actions     : int,
                 gamma         : float = 0.99,
                 alpha         : float = 0.05,
                 lam_weight    : float = 0.2,
                 tau_elig      : float = 100e-3,
                 dt            : float = 0.1e-3):
        """
        name       : critic identifier
        state_dim  : dimensionality of state representation
        n_actions  : number of actions (for Q-value critics)
        gamma      : discount factor
        alpha      : critic learning rate
        lam_weight : combination weight lambda_k in delta_total
        tau_elig   : eligibility trace decay (s)
        dt         : simulation timestep
        """
        self.name        = name
        self.state_dim   = state_dim
        self.n_actions   = n_actions
        self.gamma       = gamma
        self.alpha       = alpha
        self.lam_weight  = lam_weight
        self.tau_elig    = tau_elig
        self.dt          = dt

        # Value function weights V(s) = W_v . state
        self.W_v         = np.zeros(state_dim)

        # Q-value weights Q(s,a) = W_q[a] . state
        self.W_q         = np.zeros((n_actions, state_dim))

        # Eligibility traces
        self.e_v         = np.zeros(state_dim)
        self.e_q         = np.zeros((n_actions, state_dim))

        # Last TD error
        self.delta       = 0.0

        # Running reward stats for normalisation
        self._r_mean     = 0.0
        self._r_var      = 1.0
        self._r_count    = 0

        self.delta_history = deque(maxlen=2000)
        self.value_history = deque(maxlen=2000)

    def reset(self):
        self.W_v     = np.zeros(self.state_dim)
        self.W_q     = np.zeros((self.n_actions, self.state_dim))
        self.e_v     = np.zeros(self.state_dim)
        self.e_q     = np.zeros((self.n_actions, self.state_dim))
        self.delta   = 0.0
        self.delta_history.clear()
        self.value_history.clear()

    def value(self, state: np.ndarray) -> float:
        """V(s) = W_v . state"""
        s = self._prep_state(state)
        return float(np.dot(self.W_v, s))

    def q_value(self, state: np.ndarray,
                 action: int) -> float:
        """Q(s,a) = W_q[a] . state"""
        s = self._prep_state(state)
        return float(np.dot(self.W_q[action], s))

    def compute_delta(self, reward: float,
                       state: np.ndarray,
                       next_state: np.ndarray,
                       done: bool = False) -> float:
        """
        delta_k = r_k + gamma * V_k(s') - V_k(s)
        """
        s  = self._prep_state(state)
        sp = self._prep_state(next_state)

        V_s  = float(np.dot(self.W_v, s))
        V_sp = 0.0 if done else float(np.dot(self.W_v, sp))

        self.delta = reward + self.gamma * V_sp - V_s
        self.delta_history.append(self.delta)
        self.value_history.append(V_s)
        return self.delta

    def update(self, state: np.ndarray,
                action: int,
                alpha_override: float = None) -> None:
        """
        Update value weights using eligibility-trace TD:
            e_v <- decay * e_v + state
            W_v += alpha * delta * e_v
        """
        lr = alpha_override if alpha_override else self.alpha
        s  = self._prep_state(state)

        # Update eligibility
        decay    = np.exp(-self.dt / self.tau_elig)
        self.e_v = decay * self.e_v + s
        self.e_q[action] = decay * self.e_q[action] + s

        # Weight updates
        self.W_v        += lr * self.delta * self.e_v
        self.W_q[action]+= lr * self.delta * self.e_q[action]

        # Clip weights
        self.W_v         = np.clip(self.W_v, -10.0, 10.0)
        self.W_q         = np.clip(self.W_q, -10.0, 10.0)

    def _prep_state(self, state: np.ndarray) -> np.ndarray:
        """Normalise and pad/truncate state to state_dim."""
        s = np.asarray(state, dtype=float).flatten()
        if len(s) < self.state_dim:
            s = np.pad(s, (0, self.state_dim - len(s)))
        else:
            s = s[:self.state_dim]
        norm = np.linalg.norm(s)
        return s / (norm + 1e-8)

    def normalise_reward(self, r: float) -> float:
        """Running mean/var normalisation of reward."""
        self._r_count += 1
        self._r_mean   = (0.99 * self._r_mean + 0.01 * r)
        self._r_var    = (0.99 * self._r_var
                          + 0.01 * (r - self._r_mean) ** 2)
        return float((r - self._r_mean)
                     / (np.sqrt(self._r_var) + 1e-8))

    def critic_summary(self) -> dict:
        hist = list(self.delta_history)
        return {
            "name"        : self.name,
            "lam_weight"  : self.lam_weight,
            "last_delta"  : float(self.delta),
            "mean_delta"  : float(np.mean(hist)) if hist else 0.0,
            "std_delta"   : float(np.std(hist))  if hist else 0.0,
        }