"""
MetaDopamine  —  Step 23
-------------------------
Uncertainty-controlled learning rate:

    alpha_t = alpha_0 + eta * Ut

where:
    alpha_0 : baseline learning rate
    eta     : meta-dopamine sensitivity
    Ut      : uncertainty from Phase 3

Interpretation:
    high Ut  ->  high alpha_t  ->  stronger plasticity
    low  Ut  ->  low  alpha_t  ->  conservative updating

Core innovation:
    The learning rate itself is a dynamic variable driven
    by uncertainty, not a fixed hyperparameter. This means
    the agent automatically learns faster when it is
    uncertain (needs to update beliefs) and slower when it
    is confident (preserve learned policy).

Biological basis:
    Dopamine neurons exhibit uncertainty-dependent
    firing rate modulation. Under high uncertainty,
    dopamine bursts are larger and broader, producing
    stronger LTP/LTD at corticostriatal synapses.
    This is captured here as the meta-dopamine mechanism.

Additional features:
    - Volatility adaptation: eta increases during volatile
      reward environments
    - Momentum: smoothed alpha_t avoids abrupt jumps
    - Regime detection: classifies plasticity as
      exploratory / balanced / consolidating
    - Adaptation history for diagnostic analysis
"""

import numpy as np
from collections import deque


class MetaDopamine:

    def __init__(self,
                 alpha_0       : float = 0.05,
                 eta           : float = 0.10,
                 alpha_min     : float = 0.001,
                 alpha_max     : float = 0.30,
                 smooth_tau    : float = 0.90,
                 volatility_win: int   = 50,
                 dt            : float = 0.1e-3):
        """
        alpha_0        : base learning rate
        eta            : meta-dopamine gain (sensitivity to Ut)
        alpha_min      : minimum allowed alpha_t
        alpha_max      : maximum allowed alpha_t
        smooth_tau     : EMA smoothing factor for alpha_t
        volatility_win : window for reward volatility estimation
        dt             : simulation timestep
        """
        self.alpha_0        = alpha_0
        self.eta            = eta
        self.alpha_min      = alpha_min
        self.alpha_max      = alpha_max
        self.smooth_tau     = smooth_tau
        self.dt             = dt

        # Current smoothed learning rate
        self.alpha_t        = alpha_0

        # Volatility estimation
        self.volatility_win = volatility_win
        self.reward_window  = deque(maxlen=volatility_win)
        self.volatility     = 0.0
        self._eta_effective = eta

        # History
        self.alpha_history  = deque(maxlen=5000)
        self.U_history      = deque(maxlen=5000)
        self.vol_history    = deque(maxlen=5000)
        self.step_count     = 0

    def reset(self) -> None:
        self.alpha_t    = self.alpha_0
        self.volatility = 0.0
        self.reward_window.clear()
        self.alpha_history.clear()
        self.U_history.clear()
        self.vol_history.clear()
        self.step_count = 0

    def update(self, U: float,
                reward: float = None) -> float:
        """
        Step 23 core:
            alpha_t = alpha_0 + eta * Ut

        U      : uncertainty Ut from Phase 3 (in [0,1])
        reward : optional reward for volatility estimation

        Returns alpha_t (smoothed, clipped).
        """
        U = float(np.clip(U, 0.0, 1.0))
        self.step_count += 1

        # Update reward volatility
        if reward is not None:
            self.reward_window.append(float(reward))
            if len(self.reward_window) > 1:
                self.volatility = float(
                    np.std(list(self.reward_window)))
            # Volatility increases effective eta
            vol_scale         = 1.0 + 2.0 * self.volatility
            self._eta_effective = float(
                np.clip(self.eta * vol_scale, self.eta, 5.0 * self.eta))

        # Core formula: alpha_t = alpha_0 + eta_eff * Ut
        target_alpha = self.alpha_0 + self._eta_effective * U
        target_alpha = float(np.clip(
            target_alpha, self.alpha_min, self.alpha_max))

        # EMA smoothing — prevents abrupt jumps
        a = 1.0 - self.smooth_tau
        self.alpha_t = float(
            self.smooth_tau * self.alpha_t + a * target_alpha)

        self.alpha_history.append(self.alpha_t)
        self.U_history.append(U)
        self.vol_history.append(self.volatility)

        return self.alpha_t

    def plasticity_regime(self) -> str:
        """
        Classifies the current plasticity mode:
            exploratory   : alpha_t > 0.75 * alpha_max
            balanced      : 0.25-0.75 * alpha_max
            consolidating : alpha_t < 0.25 * alpha_max
        """
        ratio = self.alpha_t / (self.alpha_max + 1e-8)
        if ratio > 0.75:
            return "exploratory"
        elif ratio > 0.25:
            return "balanced"
        else:
            return "consolidating"

    def adaptation_efficiency(self, window: int = 100) -> float:
        """
        Measures how responsive alpha_t is to uncertainty.
        High efficiency = alpha correlates well with U.
        """
        a_hist = list(self.alpha_history)[-window:]
        u_hist = list(self.U_history)[-window:]
        if len(a_hist) < 10:
            return 0.0
        corr = np.corrcoef(a_hist, u_hist)
        return float(np.clip(corr[0, 1], 0.0, 1.0))

    def meta_summary(self) -> dict:
        a_hist = list(self.alpha_history)
        return {
            "alpha_t"          : float(self.alpha_t),
            "alpha_0"          : self.alpha_0,
            "eta_effective"    : float(self._eta_effective),
            "volatility"       : float(self.volatility),
            "regime"           : self.plasticity_regime(),
            "mean_alpha"       : float(np.mean(a_hist))
                                 if a_hist else self.alpha_0,
            "adaptation_eff"   : self.adaptation_efficiency(),
            "step_count"       : self.step_count,
        }