"""
GPiGate — combines all three pathways into the final gate signal.

GPi_a = Base
      - wGo   * D1_inhibition_a       (direct pathway)
      + wNoGo * Indirect_excitation_a (indirect pathway)
      + wSTN  * STN_global_exc        (hyperdirect pathway)

Action released when: GPi_a < threshold θ_t
θ_t = θ_0 + β*Ut - κ*Ct   (confidence-based gate from Phase 5)
"""

import numpy as np


class GPiGate:

    def __init__(self,
                 n_actions     : int,
                 gpi_base      : float = 1.0,
                 theta_0       : float = 0.5,
                 beta          : float = 0.4,
                 kappa         : float = 0.3):

        self.n_actions = n_actions
        self.gpi_base  = gpi_base
        self.theta_0   = theta_0
        self.beta      = beta
        self.kappa     = kappa

        self.gpi_activity    = np.full(n_actions, gpi_base)
        self.threshold       = theta_0
        self.released_action = None

        self.gpi_history     = []
        self.threshold_history = []

    def reset(self):
        self.gpi_activity    = np.full(self.n_actions, self.gpi_base)
        self.threshold       = self.theta_0
        self.released_action = None
        self.gpi_history.clear()
        self.threshold_history.clear()

    def step(self,
             direct_inh  : np.ndarray,
             indirect_exc: np.ndarray,
             stn_global  : float,
             w_go        : float,
             w_nogo      : float,
             w_stn       : float,
             U           : float,
             C           : float) -> dict:
        """
        Compute GPi gate signal and determine action release.

        direct_inh   : (A,) GPi inhibition from direct pathway
        indirect_exc : (A,) GPi excitation from indirect pathway
        stn_global   : scalar GPi excitation from hyperdirect
        w_go/nogo/stn: dynamic pathway weights from Step 12
        U, C         : uncertainty and confidence from Phase 3

        Returns dict with gate state and released action.
        """
        # Confidence-adaptive threshold: θ_t = θ_0 + β*Ut - κ*Ct
        self.threshold = (self.theta_0
                          + self.beta  * U
                          - self.kappa * C)
        self.threshold = float(np.clip(self.threshold, 0.1, 1.5))

        # GPi = Base - wGo*D1 + wNoGo*Indirect + wSTN*STN
        self.gpi_activity = (self.gpi_base
                             - w_go   * direct_inh
                             + w_nogo * indirect_exc
                             + w_stn  * stn_global)
        self.gpi_activity = np.clip(self.gpi_activity, 0.0, 3.0)

        # Action release: lowest GPi below threshold wins
        below = self.gpi_activity < self.threshold
        if below.any():
            # Release the action with lowest GPi (most disinhibited)
            self.released_action = int(np.argmin(self.gpi_activity))
        else:
            self.released_action = None   # all actions suppressed

        self.gpi_history.append(self.gpi_activity.copy())
        self.threshold_history.append(float(self.threshold))

        return {
            "gpi_activity"   : self.gpi_activity.copy(),
            "threshold"      : float(self.threshold),
            "released_action": self.released_action,
            "gate_open"      : bool(self.released_action is not None),
            "gate_margin"    : float(self.threshold
                                     - self.gpi_activity.min()),
            "below_threshold": below.copy(),
        }

    def gate_summary(self) -> dict:
        releases = [h for h in self.gpi_history
                    if (np.array(h) < self.threshold).any()]
        return {
            "n_releases"         : len(releases),
            "release_rate"       : len(releases) / max(len(self.gpi_history), 1),
            "mean_threshold"     : float(np.mean(self.threshold_history)),
            "current_threshold"  : float(self.threshold),
        }