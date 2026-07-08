"""
DirectPathway  —  Step 9
------------------------
Cortex → D1 MSNs → GPi↓ → Thalamus → Action

Function:
  - Promotes the selected action (Go signal)
  - Encodes dopamine-dependent facilitation of D1 neurons
  - Disinhibits thalamic relay when D1 fires strongly
  - Implements reward-strengthened corticostriatal weights

Biological basis:
  D1 MSNs are excited by dopamine (D1 receptor).
  When they fire, they inhibit GPi neurons.
  Reduced GPi activity releases thalamus from inhibition.
  Thalamus activates motor cortex → action executed.
"""

import numpy as np


class DirectPathway:

    def __init__(self,
                 n_actions      : int,
                 n_d1_per_action: int   = 20,
                 w_ctx_d1       : float = 0.8e-9,
                 w_d1_gpi       : float = 1.2e-9,
                 dopamine_gain  : float = 1.5,
                 dt             : float = 0.1e-3):

        self.n_actions       = n_actions
        self.n_d1            = n_actions * n_d1_per_action
        self.n_d1_per_action = n_d1_per_action
        self.w_ctx_d1        = w_ctx_d1
        self.w_d1_gpi        = w_d1_gpi
        self.dopamine_gain   = dopamine_gain
        self.dt              = dt

        self.d1_activity     = np.zeros(n_actions)
        self.gpi_inhibition  = np.zeros(n_actions)
        self.ctx_d1_weights  = np.ones(n_actions) * w_ctx_d1
        self.dopamine_level  = 1.0
        self.eligibility     = np.zeros(n_actions)
        self.tau_elig        = 100e-3

        self.d1_history      = []
        self.gpi_inh_history = []

    def reset(self):
        self.d1_activity    = np.zeros(self.n_actions)
        self.gpi_inhibition = np.zeros(self.n_actions)
        self.eligibility    = np.zeros(self.n_actions)
        self.d1_history.clear()
        self.gpi_inh_history.clear()

    def step(self,
             cortex_spikes  : np.ndarray,
             d1_spikes      : np.ndarray,
             dopamine_level : float = 1.0) -> np.ndarray:

        self.dopamine_level = dopamine_level

        d1_spikes = np.asarray(d1_spikes, dtype=float)
        n  = min(len(d1_spikes), self.n_d1)
        sv = np.zeros(self.n_d1)
        sv[:n] = d1_spikes[:n]

        for a in range(self.n_actions):
            sl  = slice(a * self.n_d1_per_action,
                        (a + 1) * self.n_d1_per_action)
            raw = sv[sl].mean()
            da_factor = 1.0 + (dopamine_level - 1.0) * self.dopamine_gain
            da_factor = np.clip(da_factor, 0.1, 5.0)
            self.d1_activity[a] = (0.9 * self.d1_activity[a]
                                   + 0.1 * raw * da_factor)

        # Raw GPi inhibition in physical units
        self.gpi_inhibition = (self.d1_activity * self.w_d1_gpi)

        decay = np.exp(-self.dt / self.tau_elig)
        self.eligibility = decay * self.eligibility + self.d1_activity

        self.d1_history.append(self.d1_activity.copy())
        self.gpi_inh_history.append(self.gpi_inhibition.copy())

        return self.gpi_inhibition.copy()

    def normalised_go_signal(self, scale: float = 2.0) -> np.ndarray:
        """
        Returns GPi inhibition normalised to [0, 1].
        scale: expected maximum raw inhibition value for clipping.
        This is what GPiGateEngine expects as direct_inh.
        """
        return np.clip(self.gpi_inhibition * 1e9 / scale, 0.0, 1.0)

    def apply_reward_update(self, delta: float,
                             selected_action: int,
                             alpha: float = 0.01) -> None:
        dW = alpha * delta * self.eligibility
        self.ctx_d1_weights += dW
        self.ctx_d1_weights  = np.clip(self.ctx_d1_weights,
                                        0.1e-9, 5e-9)

    def go_signal(self) -> np.ndarray:
        if self.gpi_inhibition.max() < 1e-18:
            return np.zeros(self.n_actions)
        return (self.gpi_inhibition
                / (self.gpi_inhibition.max() + 1e-18))

    def pathway_summary(self) -> dict:
        return {
            "d1_activity"    : self.d1_activity.copy(),
            "gpi_inhibition" : self.gpi_inhibition.copy(),
            "go_signal"      : self.go_signal(),
            "go_normalised"  : self.normalised_go_signal(),
            "dopamine_level" : self.dopamine_level,
        }