"""
ActionGatingController — Phase 5 integration.

Combines Steps 13–15:
  Step 13: GPiGateEngine      — gate computation
  Step 14: AdaptiveThreshold  — confidence-based θ_t
  Step 15: ThalamocorticalRelay + ExplainabilityLogger

Consumes outputs from Phase 4 (BGPathwayController)
and Phase 3 (BayesianReasoningPipeline).
Produces: final action decision + complete explainability record.
"""

"""
ActionGatingController — Phase 5 integration.
Steps 13 + 14 + 15 combined.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gating.gpi_gate_engine             import GPiGateEngine
from threshold.adaptive_threshold       import AdaptiveThreshold
from relay.thalamocortical_relay        import ThalamocorticalRelay
from xai_logging.explainability_logger  import ExplainabilityLogger  # FIXED


class ActionGatingController:

    def __init__(self,
                 n_actions     : int,
                 gpi_base      : float = 1.0,
                 theta_0       : float = 0.5,
                 beta          : float = 0.4,
                 kappa         : float = 0.3,
                 refractory_ms : float = 150.0,
                 log_dir       : str   = "results",
                 dt            : float = 0.1e-3):

        self.n_actions = n_actions
        self.dt        = dt

        self.gate = GPiGateEngine(
            n_actions=n_actions, gpi_base=gpi_base, dt=dt)

        self.threshold = AdaptiveThreshold(
            n_actions=n_actions, theta_0=theta_0,
            beta=beta, kappa=kappa, dt=dt)

        self.relay = ThalamocorticalRelay(
            n_actions=n_actions, refractory_ms=refractory_ms, dt=dt)

        self.logger = ExplainabilityLogger(
            n_actions=n_actions, log_dir=log_dir)

        self.step_count = 0

    def reset(self):
        self.gate.reset()
        self.threshold.reset()
        self.relay.reset()
        self.logger.reset()
        self.step_count = 0

    def step(self,
             direct_inh    : np.ndarray,
             indirect_exc  : np.ndarray,
             stn_global    : float,
             w_go          : float,
             w_nogo        : float,
             w_stn         : float,
             U             : float,
             C             : float,
             action_probs  : np.ndarray,
             belief_scores : np.ndarray,
             conflict_score: float,
             t_ms          : float = 0.0) -> dict:

        self.step_count += 1

        # Step 13
        gpi = self.gate.compute(
            direct_inh, indirect_exc, stn_global,
            w_go, w_nogo, w_stn)
        pc = self.gate.pathway_contributions()

        # Step 14
        t_out = self.threshold.update(U, C)
        theta = t_out["theta"]

        # Step 15
        record = self.relay.step(
            gpi_activity     = gpi,
            threshold        = theta,
            action_probs     = action_probs,
            U                = U,
            C                = C,
            conflict_score   = conflict_score,
            pathway_contribs = pc,
            t_ms             = t_ms)

        # Enrich record
        record["belief_scores"]    = np.asarray(belief_scores).tolist()
        record["w_go"]             = float(w_go)
        record["w_nogo"]           = float(w_nogo)
        record["w_stn"]            = float(w_stn)
        record["threshold_regime"] = t_out["regime"]
        record["sat_score"]        = float(
            self.threshold.speed_accuracy_tradeoff())
        record["dominant_pathway"] = self.gate.dominant_pathway()

        self.logger.log(record)
        return record

    def get_explanation(self, action_names: list = None) -> str:
        records = self.logger.get_last_n(1)
        if not records:
            return "No decisions logged yet."
        return self.logger.generate_text_explanation(
            records[-1], action_names)

    def save_log(self, filename: str = "phase5_decisions.json") -> str:
        return self.logger.save_log(filename)

    def controller_summary(self) -> dict:
        return {
            "gate"     : self.gate.gate_summary(self.threshold.theta),
            "threshold": self.threshold.threshold_summary(),
            "relay"    : self.relay.relay_summary(),
            "logger"   : {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.logger.stats.items()
            },
        }