"""
NeuromodulatorFusion  —  Step 24
----------------------------------
Fused neuromodulator signal:

    Mt = omega_d * DA_t + omega_s * 5HT_t + omega_n * NE_t

Then uses Mt to regulate five control variables:
    1. learning_rate      : alpha_t * Mt_scaled
    2. exploration_temp   : tau softmax temperature
    3. confidence_gate    : theta_t adjustment
    4. pathway_balance    : wGo vs wNoGo weighting
    5. risk_sensitivity   : rho adjustment

Advanced innovation:
    The three neuromodulators are not just parallel signals —
    they interact non-linearly. Their joint effect is
    richer than any single one alone:

    DA dominance  : exploitation, fast plasticity, Go bias
    5-HT dominance: patience, risk aversion, No-Go bias
    NE dominance  : exploration, arousal, conflict sensitivity

    The fusion weights omega_d/s/n are adaptive:
    they track which neuromodulator is most predictive
    of positive outcomes and shift toward it.
"""

import numpy as np
from collections import deque


class NeuromodulatorFusion:

    def __init__(self,
                 omega_d     : float = 0.5,
                 omega_s     : float = 0.3,
                 omega_n     : float = 0.2,
                 alpha_omega : float = 0.02,
                 dt          : float = 0.1e-3):
        """
        omega_d/s/n  : initial fusion weights (must sum to 1)
        alpha_omega  : learning rate for adaptive weights
        dt           : simulation timestep
        """
        # Normalise initial weights
        total = omega_d + omega_s + omega_n
        self.omega_d    = omega_d / total
        self.omega_s    = omega_s / total
        self.omega_n    = omega_n / total
        self.alpha_omega = alpha_omega
        self.dt         = dt

        # Fused signal
        self.Mt         = 0.5

        # Control output dict (updated each step)
        self._ctrl      = {}

        # Omega prediction accuracy tracking
        self._da_acc    = 0.5
        self._ht5_acc   = 0.5
        self._ne_acc    = 0.5

        # History
        self.Mt_history     = deque(maxlen=5000)
        self.omega_history  = deque(maxlen=5000)
        self.ctrl_history   = deque(maxlen=5000)
        self.step_count     = 0

    def reset(self) -> None:
        self.Mt         = 0.5
        self._ctrl      = {}
        self._da_acc    = self._ht5_acc = self._ne_acc = 0.5
        self.Mt_history.clear()
        self.omega_history.clear()
        self.ctrl_history.clear()
        self.step_count = 0

    def fuse(self,
             DA_t  : float,
             ht5_t : float,
             NE_t  : float) -> float:
        """
        Core Step 24 fusion:
            Mt = omega_d * DA_t + omega_s * 5HT_t + omega_n * NE_t

        DA_t  : dopamine level (normalised [0,1])
        ht5_t : serotonin level [0,1]
        NE_t  : norepinephrine level [0,1]

        Returns Mt in [0, 1].
        """
        self.Mt = float(np.clip(
            self.omega_d * DA_t
            + self.omega_s * ht5_t
            + self.omega_n * NE_t,
            0.0, 1.0))

        self.Mt_history.append(self.Mt)
        self.omega_history.append(
            (self.omega_d, self.omega_s, self.omega_n))
        self.step_count += 1

        return self.Mt

    def compute_control_signals(self,
                                  alpha_t_base : float,
                                  U            : float,
                                  DA_t         : float,
                                  ht5_t        : float,
                                  NE_t         : float) -> dict:
        """
        Derives all five control variables from Mt and
        individual neuromodulator levels.

        Returns dict with all control signals.
        """
        # Update fusion
        self.fuse(DA_t, ht5_t, NE_t)
        Mt = self.Mt

        # ── 1. Learning rate ──────────────────────────────────────
        # alpha_t modulated by fused signal
        # High Mt -> stronger plasticity
        learning_rate = float(np.clip(
            alpha_t_base * (0.5 + Mt), 0.001, 0.50))

        # ── 2. Exploration temperature ────────────────────────────
        # NE drives arousal -> exploration
        # High NE -> higher temperature (more exploratory)
        explore_temp = float(np.clip(
            0.3 + 3.0 * NE_t, 0.1, 5.0))

        # ── 3. Confidence gate adjustment ─────────────────────────
        # High DA (confident of reward) lowers gate threshold
        # High 5-HT (risk-averse) raises gate threshold
        gate_adjustment = float(
            -0.2 * DA_t + 0.3 * ht5_t)

        # ── 4. Pathway balance ────────────────────────────────────
        # DA biases toward Go (direct pathway)
        # 5-HT biases toward No-Go (indirect pathway)
        # NE modulates STN hyperdirect
        w_go_scale  = float(np.clip(0.6 + 0.8 * DA_t, 0.2, 2.0))
        w_nogo_scale = float(np.clip(0.6 + 0.8 * ht5_t, 0.2, 2.0))
        w_stn_scale  = float(np.clip(0.4 + 1.2 * NE_t, 0.1, 2.0))

        # ── 5. Risk sensitivity ────────────────────────────────────
        # 5-HT increases risk aversion rho
        # NE surprise can temporarily lower rho (bold under arousal)
        rho_adjustment = float(
            0.4 * ht5_t - 0.1 * NE_t)

        self._ctrl = {
            "Mt"             : float(Mt),
            "omega_d"        : float(self.omega_d),
            "omega_s"        : float(self.omega_s),
            "omega_n"        : float(self.omega_n),

            # Five regulated outputs
            "learning_rate"  : learning_rate,
            "explore_temp"   : explore_temp,
            "gate_adjustment": gate_adjustment,
            "w_go_scale"     : w_go_scale,
            "w_nogo_scale"   : w_nogo_scale,
            "w_stn_scale"    : w_stn_scale,
            "rho_adjustment" : rho_adjustment,

            # Per-neuromodulator levels for logging
            "DA"             : float(DA_t),
            "5HT"            : float(ht5_t),
            "NE"             : float(NE_t),
        }
        self.ctrl_history.append(self._ctrl.copy())
        return self._ctrl

    def adapt_weights(self, reward: float,
                       DA_t : float,
                       ht5_t: float,
                       NE_t : float) -> None:
        """
        Adapts fusion weights toward the neuromodulator
        that best predicted the observed reward.

        Neuromodulator with highest correlation to reward
        over recent history receives higher omega.
        """
        reward = float(reward)

        # Proxy: neuromodulator level * sign of reward
        da_pred  = DA_t  * np.sign(reward)
        ht5_pred = ht5_t * np.sign(reward)
        ne_pred  = NE_t  * np.sign(reward)

        alpha = self.alpha_omega
        self._da_acc  = (1-alpha)*self._da_acc  + alpha*float(np.clip(da_pred,  0,1))
        self._ht5_acc = (1-alpha)*self._ht5_acc + alpha*float(np.clip(ht5_pred, 0,1))
        self._ne_acc  = (1-alpha)*self._ne_acc  + alpha*float(np.clip(ne_pred,  0,1))

        # Softmax over accuracies
        accs    = np.array([self._da_acc, self._ht5_acc, self._ne_acc])
        exp_acc = np.exp(accs * 3.0)
        weights = exp_acc / (exp_acc.sum() + 1e-8)

        # Smooth update
        a = 0.05
        self.omega_d = (1-a)*self.omega_d + a*float(weights[0])
        self.omega_s = (1-a)*self.omega_s + a*float(weights[1])
        self.omega_n = (1-a)*self.omega_n + a*float(weights[2])

        # Renormalise
        total = self.omega_d + self.omega_s + self.omega_n + 1e-8
        self.omega_d /= total
        self.omega_s /= total
        self.omega_n /= total

    def dominant_neuromodulator(self) -> str:
        omegas = {"DA": self.omega_d,
                  "5HT": self.omega_s,
                  "NE": self.omega_n}
        return max(omegas, key=omegas.get)

    def fusion_summary(self) -> dict:
        hist = list(self.Mt_history)
        return {
            "Mt"          : float(self.Mt),
            "omega_d"     : float(self.omega_d),
            "omega_s"     : float(self.omega_s),
            "omega_n"     : float(self.omega_n),
            "dominant"    : self.dominant_neuromodulator(),
            "mean_Mt"     : float(np.mean(hist)) if hist else 0.5,
            "last_ctrl"   : self._ctrl.copy(),
            "step_count"  : self.step_count,
        }