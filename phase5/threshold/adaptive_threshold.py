"""
AdaptiveThreshold  —  Step 14
-------------------------------
Confidence-based action gating:

    θ_t = θ₀ + β·Ut − κ·Ct

where:
  θ₀ : baseline threshold (biological GPi release point)
  β  : uncertainty sensitivity — high Ut raises bar
  κ  : confidence sensitivity — high Ct lowers bar
  Ut : uncertainty from Phase 3 UncertaintyModule
  Ct : confidence = 1 - Ut

Interpretation:
  High uncertainty → θ_t rises → gate requires stronger Go signal
  High confidence  → θ_t falls → gate opens more easily
  Action release becomes evidence-sensitive, not fixed

Advanced innovation:
  Unlike fixed-threshold BG models, this threshold is a
  continuously updated cognitive signal. The agent literally
  becomes more cautious when unsure and more decisive when
  confident — mirroring human speed-accuracy tradeoffs.

Additional features:
  - Hysteresis band: prevents rapid threshold oscillations
  - Per-action thresholds: different urgency per action channel
  - Volatility-adaptive β and κ: auto-tune sensitivity
"""

import numpy as np
from collections import deque


class AdaptiveThreshold:

    def __init__(self,
                 n_actions  : int,
                 theta_0    : float = 0.5,
                 beta       : float = 0.4,
                 kappa      : float = 0.3,
                 theta_min  : float = 0.05,
                 theta_max  : float = 1.4,
                 hysteresis : float = 0.02,
                 smooth_tau : float = 0.9,
                 dt         : float = 0.1e-3):
        """
        n_actions   : number of action channels (for per-action thresholds)
        theta_0     : base threshold θ₀
        beta        : uncertainty weight β
        kappa       : confidence weight κ
        theta_min   : minimum threshold (always allows very strong signals)
        theta_max   : maximum threshold (prevents total gate lockout)
        hysteresis  : minimum change needed to update threshold
        smooth_tau  : EMA smoothing on threshold
        dt          : simulation timestep
        """
        self.n_actions  = n_actions
        self.theta_0    = theta_0
        self.beta       = beta
        self.kappa      = kappa
        self.theta_min  = theta_min
        self.theta_max  = theta_max
        self.hysteresis = hysteresis
        self.smooth_tau = smooth_tau
        self.dt         = dt

        # Global threshold (scalar)
        self.theta       = theta_0

        # Per-action thresholds (A,) — can differ by action priority
        self.theta_per_action = np.full(n_actions, theta_0)

        # Action priority biases — set externally if needed
        self.action_bias      = np.zeros(n_actions)

        # History
        self.theta_history    = deque(maxlen=5000)
        self.U_history        = deque(maxlen=5000)
        self.C_history        = deque(maxlen=5000)

        # Adaptive beta/kappa (tracks how well they predict releases)
        self._beta_adaptive   = beta
        self._kappa_adaptive  = kappa
        self._update_count    = 0

        # Volatility tracking (for adaptive sensitivity)
        self._U_window        = deque(maxlen=50)
        self._volatility      = 0.0

    def reset(self):
        self.theta            = self.theta_0
        self.theta_per_action = np.full(self.n_actions, self.theta_0)
        self.theta_history.clear()
        self.U_history.clear()
        self.C_history.clear()
        self._update_count    = 0
        self._U_window.clear()
        self._volatility      = 0.0

    def update(self, U: float, C: float = None) -> dict:
        """
        Step 14 core:
            θ_t = θ₀ + β·Ut − κ·Ct

        U : uncertainty ∈ [0,1]   from Phase 3
        C : confidence ∈ [0,1]   (= 1-U if not given)

        Returns dict with global and per-action thresholds.
        """
        U = float(np.clip(U, 0.0, 1.0))
        C = float(np.clip(1.0 - U if C is None else C, 0.0, 1.0))

        # Adaptive beta/kappa based on volatility
        self._U_window.append(U)
        if len(self._U_window) > 1:
            self._volatility = float(np.std(list(self._U_window)))
        # High volatility → increase beta (more cautious under change)
        self._beta_adaptive  = self.beta  * (1.0 + 0.5 * self._volatility)
        self._kappa_adaptive = self.kappa * (1.0 + 0.3 * self._volatility)

        # Core formula: θ_t = θ₀ + β·U - κ·C
        target_theta = (self.theta_0
                        + self._beta_adaptive  * U
                        - self._kappa_adaptive * C)
        target_theta = float(np.clip(target_theta,
                                      self.theta_min, self.theta_max))

        # Hysteresis: only update if change exceeds band
        if abs(target_theta - self.theta) > self.hysteresis:
            α = 1.0 - self.smooth_tau
            self.theta = (self.smooth_tau * self.theta
                          + α * target_theta)

        # Per-action thresholds (global + action-specific bias)
        self.theta_per_action = np.clip(
            self.theta + self.action_bias,
            self.theta_min, self.theta_max
        )

        self.theta_history.append(float(self.theta))
        self.U_history.append(U)
        self.C_history.append(C)
        self._update_count += 1

        return {
            "theta"            : float(self.theta),
            "theta_per_action" : self.theta_per_action.copy(),
            "theta_target"     : float(target_theta),
            "U"                : U,
            "C"                : C,
            "beta_adaptive"    : float(self._beta_adaptive),
            "kappa_adaptive"   : float(self._kappa_adaptive),
            "volatility"       : float(self._volatility),
            "regime"           : self._classify_regime(U),
        }

    def set_action_priority(self, action: int,
                             bias: float) -> None:
        """
        Lowers threshold for urgent actions (negative bias)
        or raises it for risky ones (positive bias).
        """
        self.action_bias[action] = float(bias)
        self.theta_per_action    = np.clip(
            self.theta + self.action_bias,
            self.theta_min, self.theta_max
        )

    def _classify_regime(self, U: float) -> str:
        if U > 0.65:
            return "cautious"     # high uncertainty — hard to release
        elif U < 0.35:
            return "decisive"     # low uncertainty — easy release
        else:
            return "balanced"

    def speed_accuracy_tradeoff(self) -> float:
        """
        Returns SAT score: how much accuracy is being traded for speed.
        High θ = accuracy focus. Low θ = speed focus.
        Scaled to [-1 (speed), +1 (accuracy)].
        """
        mid   = (self.theta_min + self.theta_max) / 2.0
        range_= (self.theta_max - self.theta_min) / 2.0
        return float((self.theta - mid) / range_)

    def threshold_summary(self) -> dict:
        hist  = list(self.theta_history)
        U_hist = list(self.U_history)
        return {
            "current_theta"    : float(self.theta),
            "theta_per_action" : self.theta_per_action.copy(),
            "mean_theta"       : float(np.mean(hist)) if hist else self.theta_0,
            "std_theta"        : float(np.std(hist))  if hist else 0.0,
            "beta"             : float(self._beta_adaptive),
            "kappa"            : float(self._kappa_adaptive),
            "sat_score"        : self.speed_accuracy_tradeoff(),
            "mean_U"           : float(np.mean(U_hist)) if U_hist else 0.5,
        }