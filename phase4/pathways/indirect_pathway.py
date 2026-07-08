"""
IndirectPathway  —  Step 10
----------------------------
Cortex → D2 MSNs → GPe → STN → GPi↑

Function:
  - Suppresses competing actions (No-Go signal)
  - D2 activation inhibits GPe
  - Disinhibited STN drives GPi excitation
  - Improves action selectivity and error suppression

Biological basis:
  D2 MSNs are INHIBITED by dopamine (D2 receptor).
  Low dopamine → D2 fires more → suppresses GPe.
  Less GPe inhibition on STN → STN fires more.
  STN excites GPi → stronger thalamic suppression.
  Net effect: competing actions are suppressed.
"""

import numpy as np


class IndirectPathway:

    def __init__(self,
                 n_actions      : int,
                 n_d2_per_action: int   = 20,
                 w_d2_gpe       : float = 1.2e-9,
                 w_gpe_stn      : float = 1.0e-9,
                 w_stn_gpi      : float = 1.5e-9,
                 dopamine_gain  : float = 1.5,
                 dt             : float = 0.1e-3):

        self.n_actions       = n_actions
        self.n_d2_per_action = n_d2_per_action
        self.n_d2            = n_actions * n_d2_per_action
        self.w_d2_gpe        = w_d2_gpe
        self.w_gpe_stn       = w_gpe_stn
        self.w_stn_gpi       = w_stn_gpi
        self.dopamine_gain   = dopamine_gain
        self.dt              = dt

        self.d2_activity     = np.zeros(n_actions)
        self.gpe_activity    = np.zeros(n_actions)
        self.stn_drive       = np.zeros(n_actions)
        self.gpi_excitation  = np.zeros(n_actions)
        self.gpe_tonic       = np.ones(n_actions) * 0.8
        self.eligibility     = np.zeros(n_actions)
        self.tau_elig        = 100e-3

        self.gpi_exc_history = []

    def reset(self):
        self.d2_activity     = np.zeros(self.n_actions)
        self.gpe_activity    = np.zeros(self.n_actions)
        self.stn_drive       = np.zeros(self.n_actions)
        self.gpi_excitation  = np.zeros(self.n_actions)
        self.eligibility     = np.zeros(self.n_actions)
        self.gpi_exc_history = []

    def step(self,
             d2_spikes     : np.ndarray,
             dopamine_level: float = 1.0) -> np.ndarray:

        d2_spikes = np.asarray(d2_spikes, dtype=float)
        n  = min(len(d2_spikes), self.n_d2)
        sv = np.zeros(self.n_d2)
        sv[:n] = d2_spikes[:n]

        for a in range(self.n_actions):
            sl  = slice(a * self.n_d2_per_action,
                        (a + 1) * self.n_d2_per_action)
            raw = sv[sl].mean()
            da_sup = 1.0 - (dopamine_level - 1.0) * 0.3 * self.dopamine_gain
            da_sup = np.clip(da_sup, 0.1, 3.0)
            self.d2_activity[a] = (0.9 * self.d2_activity[a]
                                   + 0.1 * raw * da_sup)

        d2_inh            = self.d2_activity * self.w_d2_gpe * 1e9
        self.gpe_activity = np.clip(self.gpe_tonic - d2_inh, 0.0, None)
        gpe_inh           = self.gpe_activity * self.w_gpe_stn * 1e9
        self.stn_drive    = np.clip(0.6 - gpe_inh + 0.1, 0.0, None)
        self.gpi_excitation = self.stn_drive * self.w_stn_gpi * 1e9

        decay = np.exp(-self.dt / self.tau_elig)
        self.eligibility = decay * self.eligibility + self.d2_activity

        self.gpi_exc_history.append(self.gpi_excitation.copy())
        return self.gpi_excitation.copy()

    def normalised_nogo_signal(self, scale: float = 2.0) -> np.ndarray:
        """
        Returns GPi excitation normalised to [0, 1].
        scale: expected maximum raw excitation value.
        """
        return np.clip(self.gpi_excitation / (scale + 1e-9), 0.0, 1.0)

    def apply_reward_update(self, delta: float,
                             alpha: float = 0.01) -> None:
        dW = -alpha * delta * self.eligibility
        self.w_d2_gpe += float(dW.mean())
        self.w_d2_gpe  = np.clip(self.w_d2_gpe, 0.1e-9, 5e-9)

    def nogo_signal(self) -> np.ndarray:
        if self.gpi_excitation.max() < 1e-9:
            return np.zeros(self.n_actions)
        return self.gpi_excitation / (self.gpi_excitation.max() + 1e-9)

    def pathway_summary(self) -> dict:
        return {
            "d2_activity"    : self.d2_activity.copy(),
            "gpe_activity"   : self.gpe_activity.copy(),
            "stn_drive"      : self.stn_drive.copy(),
            "gpi_excitation" : self.gpi_excitation.copy(),
            "nogo_signal"    : self.nogo_signal(),
            "nogo_normalised": self.normalised_nogo_signal(),
        }