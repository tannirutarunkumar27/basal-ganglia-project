"""
BGPathwayController — Phase 4 integration.

Combines Steps 9-12:
  Step  9: DirectPathway   — Go signal
  Step 10: IndirectPathway — No-Go signal
  Step 11: HyperdirectPathway — conflict Stop signal
  Step 12: DynamicPathwayWeights — uncertainty-driven weighting
  Gate:    GPiGate — combined decision output

Receives inputs from Phase 2 (BGNetwork spikes) and
Phase 3 (BayesianReasoningPipeline outputs).
Produces: selected action, gate margin, pathway states.
"""
"""
BGPathwayController — Phase 4 integration.
Steps 9-12 combined. Outputs normalised signals for Phase 5.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathways.direct_pathway         import DirectPathway
from pathways.indirect_pathway       import IndirectPathway
from pathways.hyperdirect_pathway    import HyperdirectPathway
from weighting.dynamic_pathway_weights import DynamicPathwayWeights
from gating.gpi_gate                 import GPiGate


class BGPathwayController:

    def __init__(self,
                 n_actions       : int,
                 n_d1_per_action : int   = 20,
                 n_d2_per_action : int   = 20,
                 conflict_eps    : float = 0.3,
                 dt              : float = 0.1e-3):

        self.n_actions = n_actions
        self.dt        = dt

        self.direct      = DirectPathway(
            n_actions, n_d1_per_action, dt=dt)
        self.indirect    = IndirectPathway(
            n_actions, n_d2_per_action, dt=dt)
        self.hyperdirect = HyperdirectPathway(
            n_actions, conflict_eps=conflict_eps, dt=dt)
        self.weights     = DynamicPathwayWeights()
        self.gate        = GPiGate(n_actions)

        self.step_count     = 0
        self.action_history = []

    def reset(self):
        self.direct.reset()
        self.indirect.reset()
        self.hyperdirect.reset()
        self.weights.reset()
        self.gate.reset()
        self.step_count     = 0
        self.action_history = []

    def step(self,
             cortex_spikes  : np.ndarray,
             d1_spikes      : np.ndarray,
             d2_spikes      : np.ndarray,
             belief_scores  : np.ndarray,
             U              : float,
             C              : float,
             dopamine_level : float = 1.0) -> dict:

        self.step_count += 1

        # Step 9: direct pathway (raw physical units)
        _ = self.direct.step(cortex_spikes, d1_spikes, dopamine_level)

        # Step 10: indirect pathway (raw physical units)
        _ = self.indirect.step(d2_spikes, dopamine_level)

        # Step 11: hyperdirect pathway
        stn_raw = self.hyperdirect.step(
            cortex_spikes, belief_scores, stn_trigger=U)

        # Step 12: dynamic weights
        w      = self.weights.update(U, C)
        w_go   = w["w_go"]
        w_nogo = w["w_nogo"]
        w_stn  = w["w_stn"]

        # ── Normalise pathway outputs for GPiGateEngine ──────────
        # direct_inh:   normalised Go inhibition [0,1]
        # indirect_exc: normalised No-Go excitation [0,1]
        # stn_norm:     normalised STN broadcast [0,1]
        direct_inh_norm   = self.direct.normalised_go_signal(scale=2.0)
        indirect_exc_norm = self.indirect.normalised_nogo_signal(scale=2.0)
        stn_norm          = float(np.clip(stn_raw / 2.0, 0.0, 1.0))

        # Internal GPi gate (for Phase 4 own use)
        gate_out = self.gate.step(
            direct_inh_norm, indirect_exc_norm, stn_norm,
            w_go, w_nogo, w_stn, U, C)

        action = gate_out["released_action"]
        self.action_history.append(action)

        return {
            # Normalised signals — consumed by Phase 5 GPiGateEngine
            "direct_inh"     : direct_inh_norm,
            "indirect_exc"   : indirect_exc_norm,
            "stn_global"     : stn_norm,

            # Weights and regime
            "w_go"           : w_go,
            "w_nogo"         : w_nogo,
            "w_stn"          : w_stn,
            "regime"         : w["regime"],

            # Gate output
            "gpi_activity"   : gate_out["gpi_activity"],
            "threshold"      : gate_out["threshold"],
            "released_action": action,
            "gate_open"      : gate_out["gate_open"],
            "gate_margin"    : gate_out["gate_margin"],

            # Conflict
            "conflict_score" : self.hyperdirect.conflict_score,
            "stn_burst"      : self.hyperdirect.stn_burst,

            "U": U, "C": C, "dopamine": dopamine_level,
        }

    def apply_reward(self, delta: float, action: int):
        self.direct.apply_reward_update(delta, action)
        self.indirect.apply_reward_update(delta)