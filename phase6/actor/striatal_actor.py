"""
StriatalActor  —  Step 16
--------------------------
pi(a|s) = softmax(f_D1,a)

The actor encodes action preference through:
  - D1 MSN firing rates per action channel
  - Direct-pathway corticostriatal weight strength
  - Dopamine-modulated D1 excitability

Biological basis:
  D1 MSNs in the striatum form the biological substrate
  of the actor in actor-critic RL. Their synaptic weights
  (corticostriatal) are strengthened by dopamine bursts
  (reward) and weakened by dopamine dips (punishment).
  The policy is implicit in the weight distribution —
  the most strongly weighted action channel fires most
  in response to a given cortical input state.
"""

"""
StriatalActor  —  Step 16
pi(a|s) = softmax(f_D1,a)
"""

import numpy as np
from collections import deque


class StriatalActor:

    def __init__(self,
                 n_actions      : int,
                 n_d1_per_action: int   = 20,
                 temperature    : float = 1.0,
                 alpha_actor    : float = 0.05,
                 dt             : float = 0.1e-3):

        self.n_actions       = n_actions
        self.n_d1_per_action = n_d1_per_action
        self.n_d1_total      = n_actions * n_d1_per_action
        self.temperature     = temperature
        self.alpha_actor     = alpha_actor
        self.dt              = dt

        self.action_preference = np.zeros(n_actions)
        self.d1_rate           = np.zeros(n_actions)
        self.pi                = np.ones(n_actions) / n_actions
        self.eligibility       = np.zeros(n_actions)
        self.tau_elig          = 50e-3

        self.pi_history        = deque(maxlen=2000)
        self.pref_history      = deque(maxlen=2000)
        self.update_count      = 0

    def reset(self):
        self.action_preference = np.zeros(self.n_actions)
        self.d1_rate           = np.zeros(self.n_actions)
        self.pi                = np.ones(self.n_actions) / self.n_actions
        self.eligibility       = np.zeros(self.n_actions)
        self.pi_history.clear()
        self.pref_history.clear()
        self.update_count      = 0

    def encode_d1_activity(self, d1_spikes: np.ndarray,
                            dopamine_level: float = 1.0) -> np.ndarray:
        """
        Converts raw D1 spike vector into per-action firing rates.
        Dopamine multiplicatively scales D1 excitability.
        """
        d1_spikes = np.asarray(d1_spikes, dtype=float)
        sv        = np.zeros(self.n_d1_total)
        n         = min(len(d1_spikes), self.n_d1_total)
        sv[:n]    = d1_spikes[:n]

        da_gain = float(np.clip(dopamine_level, 0.1, 5.0))

        for a in range(self.n_actions):
            sl = slice(a * self.n_d1_per_action,
                       (a + 1) * self.n_d1_per_action)
            raw = sv[sl].mean()
            self.d1_rate[a] = (0.9 * self.d1_rate[a]
                               + 0.1 * raw * da_gain)

        return self.d1_rate.copy()

    def compute_policy(self,
                        d1_rate       : np.ndarray = None,
                        belief_scores : np.ndarray = None,
                        blend_alpha   : float = 0.6) -> np.ndarray:
        """
        Step 16 core:
            pi(a|s) = softmax(f_D1,a)

        Optionally blends D1 rates with Bayesian belief scores.
        """
        if d1_rate is not None:
            d1_norm = np.asarray(d1_rate, dtype=float)
        else:
            d1_norm = self.d1_rate.copy()

        combined = d1_norm + self.action_preference

        if belief_scores is not None:
            V      = np.asarray(belief_scores, dtype=float)
            v_min  = V.min()
            v_range = V.max() - v_min         # replaces deprecated ptp()
            V_norm = (V - v_min) / (v_range + 1e-8)
            combined = blend_alpha * combined + (1.0 - blend_alpha) * V_norm

        tau     = max(self.temperature, 1e-3)
        shifted = (combined - combined.max()) / tau
        exp_c   = np.exp(np.clip(shifted, -50, 50))
        self.pi = exp_c / (exp_c.sum() + 1e-12)

        decay            = np.exp(-self.dt / self.tau_elig)
        self.eligibility = decay * self.eligibility + self.pi

        self.pi_history.append(self.pi.copy())
        self.pref_history.append(self.action_preference.copy())

        return self.pi.copy()

    def select_action(self, mode: str = "sample") -> int:
        if mode == "greedy":
            return int(np.argmax(self.pi))
        return int(np.random.choice(self.n_actions, p=self.pi))

    def update(self, delta_total: float,
                selected_action: int) -> None:
        """
        Actor update:
            preference[a] += alpha * delta * grad_log_pi[a]
        """
        grad = np.zeros(self.n_actions)
        grad[selected_action] = 1.0 - self.pi[selected_action]
        for a in range(self.n_actions):
            if a != selected_action:
                grad[a] = -self.pi[a]

        self.action_preference += (self.alpha_actor
                                    * delta_total
                                    * grad
                                    * self.eligibility)
        self.action_preference = np.clip(
            self.action_preference, -5.0, 5.0)

        self.update_count += 1

    def adapt_temperature(self, U: float,
                           min_tau: float = 0.3,
                           max_tau: float = 3.0) -> float:
        self.temperature = min_tau + (max_tau - min_tau) * U
        return float(self.temperature)

    def actor_summary(self) -> dict:
        return {
            "pi"               : self.pi.copy(),
            "action_preference": self.action_preference.copy(),
            "d1_rate"          : self.d1_rate.copy(),
            "temperature"      : float(self.temperature),
            "eligibility"      : self.eligibility.copy(),
            "update_count"     : self.update_count,
            "entropy"          : float(-np.sum(
                (self.pi + 1e-12) * np.log(self.pi + 1e-12))),
        }