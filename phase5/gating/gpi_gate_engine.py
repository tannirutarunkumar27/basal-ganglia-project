"""
GPiGateEngine  —  Step 13
--------------------------
Final action gating computation:

  GPi_a = Base
        - wGo   * D1_a        (direct  — inhibitory)
        + wNoGo * D2_a        (indirect — excitatory via D2→GPe→STN)
        + wSTN  * STN_global  (hyperdirect — global excitatory)

Decision rule:
  GPi_a < θ_t  →  thalamic disinhibition  →  action execution

This is the convergence point of all three BG pathways.
The gate is the ONLY output of the BG circuit — everything
computed in Phases 2–4 flows here.

Biological basis:
  GPi neurons fire tonically at ~60 Hz, keeping thalamus silent.
  When the direct pathway fires strongly on action channel a,
  D1 inhibition reduces GPi_a below baseline.
  If GPi_a < threshold, thalamic relay neurons for action a
  are disinhibited and drive motor cortex output.
"""
"""
GPiGateEngine  —  Step 13
--------------------------
GPi_a = Base - wGo*D1_a + wNoGo*D2_a + wSTN*STN_global

All inputs are expected in normalised units [0, 1].
The BGPathwayController normalises before calling this.
"""

import numpy as np
from collections import deque


class GPiGateEngine:

    def __init__(self,
                 n_actions  : int,
                 gpi_base   : float = 1.0,
                 min_gpi    : float = 0.0,
                 max_gpi    : float = 3.0,
                 smooth_tau : float = 0.8,
                 dt         : float = 0.1e-3):

        self.n_actions  = n_actions
        self.gpi_base   = gpi_base
        self.min_gpi    = min_gpi
        self.max_gpi    = max_gpi
        self.smooth_tau = smooth_tau
        self.dt         = dt

        self.gpi          = np.full(n_actions, gpi_base)
        self.contrib_go   = np.zeros(n_actions)
        self.contrib_nogo = np.zeros(n_actions)
        self.contrib_stn  = np.zeros(n_actions)
        self.contrib_base = np.full(n_actions, gpi_base)

        self.gpi_history = deque(maxlen=5000)
        self.step_count  = 0

    def reset(self):
        self.gpi          = np.full(self.n_actions, self.gpi_base)
        self.contrib_go   = np.zeros(self.n_actions)
        self.contrib_nogo = np.zeros(self.n_actions)
        self.contrib_stn  = np.zeros(self.n_actions)
        self.gpi_history.clear()
        self.step_count   = 0

    def compute(self,
                direct_inh  : np.ndarray,
                indirect_exc: np.ndarray,
                stn_global  : float,
                w_go        : float,
                w_nogo      : float,
                w_stn       : float) -> np.ndarray:
        """
        GPi_a = Base - wGo*D1_a + wNoGo*D2_a + wSTN*STN_global

        All inputs MUST be normalised to [0, 1].
        direct_inh   : (A,) normalised Go inhibition per action
        indirect_exc : (A,) normalised No-Go excitation per action
        stn_global   : scalar normalised STN broadcast
        """
        di = np.zeros(self.n_actions)
        ie = np.zeros(self.n_actions)
        n  = min(len(direct_inh),   self.n_actions)
        m  = min(len(indirect_exc), self.n_actions)
        di[:n] = np.asarray(direct_inh[:n],   dtype=float)
        ie[:m] = np.asarray(indirect_exc[:m], dtype=float)

        # Store contributions for explainability
        self.contrib_go   = w_go   * di
        self.contrib_nogo = w_nogo * ie
        self.contrib_stn  = np.full(self.n_actions,
                                     float(w_stn * stn_global))
        self.contrib_base = np.full(self.n_actions, self.gpi_base)

        # GPi computation — inputs are already normalised
        raw_gpi = (self.gpi_base
                   - self.contrib_go
                   + self.contrib_nogo
                   + self.contrib_stn)

        # EMA smoothing
        alpha    = 1.0 - self.smooth_tau
        self.gpi = np.clip(
            self.smooth_tau * self.gpi + alpha * raw_gpi,
            self.min_gpi, self.max_gpi
        )

        self.gpi_history.append(self.gpi.copy())
        self.step_count += 1
        return self.gpi.copy()

    def gate_open_mask(self, threshold: float) -> np.ndarray:
        return self.gpi < threshold

    def winning_action(self, threshold: float):
        mask = self.gate_open_mask(threshold)
        if not mask.any():
            return None
        candidates = np.where(mask)[0]
        return int(candidates[np.argmin(self.gpi[candidates])])

    def gate_margins(self, threshold: float) -> np.ndarray:
        return threshold - self.gpi

    def pathway_contributions(self) -> dict:
        return {
            "base" : self.contrib_base.copy(),
            "go"   : self.contrib_go.copy(),
            "nogo" : self.contrib_nogo.copy(),
            "stn"  : self.contrib_stn.copy(),
            "net"  : self.gpi.copy(),
        }

    def dominant_pathway(self) -> str:
        vals = {
            "go"  : self.contrib_go.mean(),
            "nogo": self.contrib_nogo.mean(),
            "stn" : self.contrib_stn.mean(),
        }
        return max(vals, key=vals.get)

    def gate_summary(self, threshold: float) -> dict:
        return {
            "gpi_activity"    : self.gpi.copy(),
            "threshold"       : float(threshold),
            "gate_open_mask"  : self.gate_open_mask(threshold),
            "winning_action"  : self.winning_action(threshold),
            "gate_margins"    : self.gate_margins(threshold),
            "dominant_pathway": self.dominant_pathway(),
            "pathway_contribs": self.pathway_contributions(),
        }