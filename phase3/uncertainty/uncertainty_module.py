"""
UncertaintyModule
-----------------
Implements Step 8:
    Ut = (1/N) * sum_a Var(Va)
    Ct = 1 - Ut

Uncertainty Ut and confidence Ct are active control variables
that regulate ALL downstream components:

  Component              | Regulated by
  ───────────────────────┼────────────────────────────────────
  Learning rate α        | high Ut  → larger α (more plastic)
  Pathway weights        | high Ut  → stronger STN / No-Go
  Action gate threshold  | high Ut  → higher GPi threshold
  Exploration            | high Ut  → higher softmax τ
  Risk handling          | high Ut  → stronger risk penalty ρ
  STN conflict trigger   | high Ut  → earlier STN activation
  Meta-dopamine          | high Ut  → boosted DA modulation
"""

import numpy as np
from collections import deque


class UncertaintyModule:

    def __init__(self,
                 n_actions       : int,
                 variance_window : int   = 50,
                 smooth_factor   : float = 0.9,
                 clip_min        : float = 0.0,
                 clip_max        : float = 1.0):
        """
        n_actions       : number of actions A
        variance_window : window length for rolling variance estimate
        smooth_factor   : EMA smoothing factor for Ut
        clip_min/max    : bounds for reported Ut and Ct
        """
        self.n_actions       = n_actions
        self.variance_window = variance_window
        self.smooth_factor   = smooth_factor
        self.clip_min        = clip_min
        self.clip_max        = clip_max

        # Rolling belief history for variance computation
        self.V_window = deque(maxlen=variance_window)

        # Smoothed uncertainty and confidence
        self.U_smooth = 0.5   # start at maximum uncertainty
        self.C_smooth = 0.5

        # Raw (unsmoothed) values
        self.U_raw = 0.5
        self.C_raw = 0.5

        # Per-action variance (A,)
        self.per_action_var = np.ones(n_actions) * 0.5

        # History
        self.U_history = []
        self.C_history = []

        # Derived control signals (updated each step)
        self.learning_rate_factor  = 1.0
        self.pathway_weight_factor = 0.5
        self.gate_threshold_offset = 0.0
        self.exploration_temp      = 1.0
        self.risk_aversion         = 0.5
        self.stn_trigger_factor    = 0.5

    def reset(self):
        self.V_window.clear()
        self.U_smooth = 0.5
        self.C_smooth = 0.5
        self.U_raw    = 0.5
        self.C_raw    = 0.5
        self.per_action_var = np.ones(self.n_actions) * 0.5
        self.U_history.clear()
        self.C_history.clear()

    def update(self, V: np.ndarray) -> dict:
        """
        Core Step 8 computation:
            Ut = (1/N) * sum_a Var(Va)
            Ct = 1 - Ut

        V: temporal belief scores (A,)
        Returns: dict with Ut, Ct, and all derived control signals
        """
        V = np.asarray(V, dtype=float)
        self.V_window.append(V.copy())

        if len(self.V_window) < 2:
            return self._package_output()

        # Rolling variance per action over window
        V_matrix = np.array(self.V_window)   # (window, A)
        self.per_action_var = np.var(V_matrix, axis=0)

        # Ut = mean variance across actions — normalised to [0,1]
        raw_U = float(np.mean(self.per_action_var))

        # Normalise using sigmoid squashing
        self.U_raw = self._normalise(raw_U)
        self.C_raw = np.clip(1.0 - self.U_raw, 0.0, 1.0)

        # EMA smoothing
        α = 1.0 - self.smooth_factor
        self.U_smooth = (self.smooth_factor * self.U_smooth
                         + α * self.U_raw)
        self.C_smooth = np.clip(1.0 - self.U_smooth, 0.0, 1.0)

        # Clip to [0, 1]
        self.U_smooth = np.clip(self.U_smooth, self.clip_min, self.clip_max)
        self.C_smooth = np.clip(self.C_smooth, self.clip_min, self.clip_max)

        # Update all derived control signals
        self._update_control_signals()

        self.U_history.append(float(self.U_smooth))
        self.C_history.append(float(self.C_smooth))

        return self._package_output()

    def _normalise(self, raw_U: float,
                   scale: float = 2.0) -> float:
        """
        Maps raw variance (unbounded positive) to [0,1]
        using a sigmoid: U = 2*sigmoid(scale * raw_U) - 1
        clipped to [0,1].
        """
        sig = 1.0 / (1.0 + np.exp(-scale * raw_U))
        return float(np.clip(2 * sig - 1, 0.0, 1.0))

    def _update_control_signals(self):
        """
        Derives all downstream control variables from Ut and Ct.
        These are the 'active control' aspect of the advanced innovation.
        """
        U = self.U_smooth
        C = self.C_smooth

        # Learning rate: high U -> stronger plasticity
        # α_t = α_0 * (1 + η * Ut)   [from Phase 8 meta-dopamine]
        self.learning_rate_factor = 1.0 + 2.0 * U

        # BG pathway weighting: high U -> more STN/No-Go influence
        # w_Go decreases, w_STN increases with uncertainty
        self.pathway_weight_factor = U          # 0=full Go, 1=full STN

        # Action gate threshold: high U raises the bar for release
        # θ_t = θ_0 + β * Ut - κ * Ct   [from Phase 5]
        beta, kappa = 0.4, 0.3
        self.gate_threshold_offset = beta * U - kappa * C

        # Exploration temperature: high U -> more exploratory
        self.exploration_temp = 0.5 + 3.0 * U

        # Risk aversion: high U -> stronger risk penalty ρ
        self.risk_aversion = 0.2 + 0.8 * U

        # STN conflict trigger sensitivity: high U -> easier trigger
        self.stn_trigger_factor = 0.3 + 0.7 * U

    def _package_output(self) -> dict:
        return {
            "U"                     : float(self.U_smooth),
            "C"                     : float(self.C_smooth),
            "U_raw"                 : float(self.U_raw),
            "C_raw"                 : float(self.C_raw),
            "per_action_var"        : self.per_action_var.copy(),
            "learning_rate_factor"  : float(self.learning_rate_factor),
            "pathway_weight_factor" : float(self.pathway_weight_factor),
            "gate_threshold_offset" : float(self.gate_threshold_offset),
            "exploration_temp"      : float(self.exploration_temp),
            "risk_aversion"         : float(self.risk_aversion),
            "stn_trigger_factor"    : float(self.stn_trigger_factor),
        }

    def is_high_uncertainty(self, threshold: float = 0.6) -> bool:
        return self.U_smooth > threshold

    def is_confident(self, threshold: float = 0.6) -> bool:
        return self.C_smooth > threshold

    def uncertainty_summary(self) -> dict:
        out = self._package_output()
        out["history_len"] = len(self.U_history)
        out["mean_U_last_100"] = (float(np.mean(self.U_history[-100:]))
                                   if self.U_history else 0.5)
        return out