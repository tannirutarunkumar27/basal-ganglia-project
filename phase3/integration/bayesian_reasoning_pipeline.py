"""
BayesianReasoningPipeline
-------------------------
Integrates Steps 5-8 into a single call:

  spikes_in
      → PosteriorBeliefEncoder  (Step 5: Va = logP(s|a) + logP(a))
      → TemporalBeliefUpdater   (Step 6: Va(t) = λVa(t-1) + (1-λ)Va_hat)
      → MemoryTrace             (Step 6: multi-timescale recurrent loop)
      → ActionSelector          (Step 7: P(a|s) = softmax(Va))
      → UncertaintyModule       (Step 8: Ut = Var(Va)/N, Ct = 1-Ut)
      → control signals out

This pipeline feeds directly into:
  - Phase 4 (BG pathway weighting via pathway_weight_factor)
  - Phase 5 (action gate via gate_threshold_offset)
  - Phase 6 (multi-critic RL via exploration_temp)
  - Phase 8 (meta-dopamine via learning_rate_factor)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from belief.posterior_encoder       import PosteriorBeliefEncoder
from temporal.temporal_belief       import TemporalBeliefUpdater
from temporal.memory_trace          import MemoryTrace
from action.action_selector         import ActionSelector
from uncertainty.uncertainty_module import UncertaintyModule


class BayesianReasoningPipeline:

    def __init__(self,
                 n_actions      : int,
                 n_neurons_total: int,
                 window_steps   : int   = 100,
                 lam            : float = 0.85,
                 temperature    : float = 1.0,
                 alpha_prior    : float = 0.05,
                 dt             : float = 0.1e-3):

        self.n_actions = n_actions
        self.dt        = dt

        # Step 5: posterior belief encoder
        self.encoder = PosteriorBeliefEncoder(
            n_actions       = n_actions,
            n_neurons_total = n_neurons_total,
            window_steps    = window_steps,
            alpha_prior     = alpha_prior,
            dt              = dt,
        )

        # Step 6a: temporal belief updater
        self.updater = TemporalBeliefUpdater(
            n_actions   = n_actions,
            lam         = lam,
        )

        # Step 6b: multi-timescale memory trace
        self.memory = MemoryTrace(n_actions, dt=dt)

        # Step 7: probabilistic action selector
        self.selector = ActionSelector(
            n_actions   = n_actions,
            temperature = temperature,
        )

        # Step 8: uncertainty and confidence module
        self.unc = UncertaintyModule(
            n_actions       = n_actions,
            variance_window = 50,
        )

        # Output state (last step)
        self.last_action     = 0
        self.last_prob       = np.ones(n_actions) / n_actions
        self.last_V_temporal = np.zeros(n_actions)
        self.last_uc         = {}

    def reset(self):
        self.encoder.reset()
        self.updater.reset()
        self.memory.reset()
        self.selector.reset()
        self.unc.reset()
        self.last_action     = 0
        self.last_prob       = np.ones(self.n_actions) / self.n_actions
        self.last_V_temporal = np.zeros(self.n_actions)
        self.last_uc         = {}

    def step(self, spike_vector : np.ndarray,
             selection_mode     : str   = "probabilistic",
             reward             : float = None,
             prev_action        : int   = None) -> dict:
        """
        Full pipeline step.

        spike_vector   : cortex/bayesian_layer spikes (N_neurons,)
        selection_mode : "probabilistic" | "greedy" | "epsilon_greedy"
        reward         : if provided, updates the prior P(a)
        prev_action    : action taken at previous step (for prior update)

        Returns a single output dict with all signals.
        """
        # ── Step 5: posterior belief ──────────────────────────────
        V_hat = self.encoder.encode(spike_vector)

        # ── Step 6a: temporal belief update ──────────────────────
        V_temporal = self.updater.update(V_hat, adaptive_lam=True)

        # ── Step 6b: memory trace ─────────────────────────────────
        V_memory = self.memory.step(V_temporal)

        # Blend temporal + memory (70 / 30 mix)
        V_combined = 0.7 * V_temporal + 0.3 * V_memory

        # ── Step 8: uncertainty BEFORE action selection ───────────
        uc = self.unc.update(V_combined)

        # ── Step 7: action selection using uncertainty-adapted τ ──
        action, prob, sel_info = self.selector.select(
            V_combined,
            mode          = selection_mode,
            temp_override = uc["exploration_temp"],
        )

        # Update prior if reward provided
        if reward is not None and prev_action is not None:
            self.encoder.update_prior(prev_action, reward)

        # Cache
        self.last_action     = action
        self.last_prob       = prob
        self.last_V_temporal = V_temporal
        self.last_uc         = uc

        return {
            # Step 5 outputs
            "V_hat"                 : V_hat,
            "log_likelihood"        : self.encoder.evidence_extractor.log_likelihood(),
            "log_prior"             : self.encoder.prior_tracker.log_prior(),

            # Step 6 outputs
            "V_temporal"            : V_temporal,
            "V_memory"              : V_memory,
            "V_combined"            : V_combined,
            "temporal_volatility"   : self.updater._volatility,
            "memory_dominant"       : self.memory.dominant_timescale(),

            # Step 7 outputs
            "action"                : action,
            "action_prob"           : float(prob[action]),
            "prob"                  : prob,
            "ranked_actions"        : sel_info["ranked_actions"],
            "selector_entropy"      : sel_info["entropy"],
            "soft_competition"      : self.selector.soft_competition_score(),

            # Step 8 outputs — active control variables
            "U"                     : uc["U"],
            "C"                     : uc["C"],
            "learning_rate_factor"  : uc["learning_rate_factor"],
            "pathway_weight_factor" : uc["pathway_weight_factor"],
            "gate_threshold_offset" : uc["gate_threshold_offset"],
            "exploration_temp"      : uc["exploration_temp"],
            "risk_aversion"         : uc["risk_aversion"],
            "stn_trigger_factor"    : uc["stn_trigger_factor"],
        }

    def inject_dopamine_signal(self, delta: float):
        """
        External dopamine/reward signal modulates the prior.
        Called from Phase 6 (multi-critic RL) after reward receipt.
        """
        if self.last_action is not None and abs(delta) > 0:
            self.encoder.update_prior(self.last_action, delta)

    def pipeline_summary(self) -> dict:
        return {
            "n_actions"         : self.n_actions,
            "last_action"       : self.last_action,
            "last_U"            : self.last_uc.get("U", 0.5),
            "last_C"            : self.last_uc.get("C", 0.5),
            "total_selections"  : self.selector.selection_summary()["total_selections"],
            "prior"             : self.encoder.prior_tracker.prior.copy(),
            "V_temporal"        : self.last_V_temporal.copy(),
        }