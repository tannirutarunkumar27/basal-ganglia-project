"""
NeuromodulatorController — Phase 8 integration.
Combines Steps 23 and 24 into one call per timestep.

Inputs (from previous phases):
    delta_prime  : Phase 6 predictive dopamine signal
    U, C         : Phase 3 uncertainty and confidence
    reward       : observed reward
    rho          : Phase 6 risk aversion coefficient
    conflict     : Phase 4 hyperdirect conflict score
    snc_rate     : Phase 2 SNc spike rate
    volatility   : Phase 8 MetaDopamine volatility tracker

Outputs (to subsequent phases):
    alpha_t      : meta learning rate -> Phase 7 PlasticityManager
    Mt           : fused signal       -> all phases
    learning_rate: modulated alpha    -> Phase 7
    explore_temp : softmax tau        -> Phase 3 ActionSelector
    gate_adjust  : threshold offset   -> Phase 5
    w_go_scale   : Go pathway scale   -> Phase 4
    w_nogo_scale : No-Go pathway scale -> Phase 4
    w_stn_scale  : STN pathway scale  -> Phase 4
    rho_adj      : risk aversion adj  -> Phase 6
    DA, 5HT, NE  : individual levels  -> logging
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from meta_dopamine.meta_dopamine          import MetaDopamine
from neuromodulators.dopamine_module       import DopamineModule
from neuromodulators.serotonin_module      import SerotoninModule
from neuromodulators.norepinephrine_module import NorepinephrineModule
from fusion.neuromodulator_fusion          import NeuromodulatorFusion


class NeuromodulatorController:

    def __init__(self,
                 alpha_0  : float = 0.05,
                 eta      : float = 0.10,
                 omega_d  : float = 0.5,
                 omega_s  : float = 0.3,
                 omega_n  : float = 0.2,
                 dt       : float = 0.1e-3):

        self.dt = dt

        # Step 23: meta-dopamine
        self.meta_da = MetaDopamine(
            alpha_0=alpha_0, eta=eta, dt=dt)

        # Step 24: three neuromodulator modules
        self.da_mod  = DopamineModule(dt=dt)
        self.ht5_mod = SerotoninModule(dt=dt)
        self.ne_mod  = NorepinephrineModule(dt=dt)

        # Step 24: fusion engine
        self.fusion  = NeuromodulatorFusion(
            omega_d=omega_d, omega_s=omega_s,
            omega_n=omega_n, dt=dt)

        self.step_count = 0
        self._last_out  = {}

    def reset(self) -> None:
        self.meta_da.reset()
        self.da_mod.reset()
        self.ht5_mod.reset()
        self.ne_mod.reset()
        self.fusion.reset()
        self.step_count = 0
        self._last_out  = {}

    def step(self,
             delta_prime   : float,
             U             : float,
             C             : float,
             reward        : float,
             rho           : float,
             conflict_score: float,
             snc_rate_hz   : float = 4.0) -> dict:
        """
        Full Phase 8 step — Steps 23 and 24 combined.

        Returns complete neuromodulator control dict.
        """
        self.step_count += 1

        # Step 23: meta-dopamine learning rate
        alpha_t = self.meta_da.update(U, reward)

        # Step 24a: update individual neuromodulators
        DA      = self.da_mod.update(delta_prime, snc_rate_hz)
        DA_norm = self.da_mod.normalised()
        ht5     = self.ht5_mod.update(reward, rho, conflict_score)
        NE      = self.ne_mod.update(
            delta_prime, U, self.meta_da.volatility)

        # Step 24b: fuse and compute control signals
        ctrl = self.fusion.compute_control_signals(
            alpha_t_base = alpha_t,
            U            = U,
            DA_t         = DA_norm,
            ht5_t        = ht5,
            NE_t         = NE)

        # Adapt fusion weights
        self.fusion.adapt_weights(reward, DA_norm, ht5, NE)

        self._last_out = {
            # Step 23 output
            "alpha_t"          : float(alpha_t),
            "meta_regime"      : self.meta_da.plasticity_regime(),
            "volatility"       : float(self.meta_da.volatility),

            # Neuromodulator levels
            "DA"               : float(DA_norm),
            "5HT"              : float(ht5),
            "NE"               : float(NE),

            # Step 24 outputs — all five regulated signals
            "Mt"               : ctrl["Mt"],
            "omega_d"          : ctrl["omega_d"],
            "omega_s"          : ctrl["omega_s"],
            "omega_n"          : ctrl["omega_n"],
            "dominant_nm"      : self.fusion.dominant_neuromodulator(),

            # Five control variables consumed by other phases
            "learning_rate"    : ctrl["learning_rate"],
            "explore_temp"     : ctrl["explore_temp"],
            "gate_adjustment"  : ctrl["gate_adjustment"],
            "w_go_scale"       : ctrl["w_go_scale"],
            "w_nogo_scale"     : ctrl["w_nogo_scale"],
            "w_stn_scale"      : ctrl["w_stn_scale"],
            "rho_adjustment"   : ctrl["rho_adjustment"],
        }

        return self._last_out

    def controller_summary(self) -> dict:
        return {
            "step_count" : self.step_count,
            "meta_da"    : self.meta_da.meta_summary(),
            "fusion"     : self.fusion.fusion_summary(),
            "NE_arousal" : self.ne_mod.arousal_level(),
            "5HT_patience": self.ht5_mod.patience_factor(),
            "NE_gain"    : self.ne_mod.gain_modulation(),
        }