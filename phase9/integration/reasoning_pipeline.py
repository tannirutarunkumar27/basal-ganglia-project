"""
ReasoningPipeline — Phase 9 integration.
Steps 25-28 combined into one call per timestep.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from symbolic.neuro_symbolic_reasoner      import NeuroSymbolicReasoner
from attention.attention_module            import AttentionModule
from counterfactual.counterfactual_engine  import CounterfactualEngine
from explanation.explanation_composer      import ExplanationComposer

from collections import deque


class ReasoningPipeline:

    def __init__(self,
                 n_actions   : int,
                 action_names: list = None,
                 log_dir     : str  = "results",
                 dt          : float = 0.1e-3):

        self.n_actions = n_actions
        self.dt        = dt

        names = (action_names if action_names
                 else [f"a{i}" for i in range(n_actions)])

        # Steps 25-28
        self.reasoner = NeuroSymbolicReasoner(n_actions, dt=dt)
        self.attention = AttentionModule(n_actions, dt=dt)
        self.cf_engine = CounterfactualEngine(n_actions,
                                               action_names=names)
        self.composer  = ExplanationComposer(n_actions,
                                              action_names=names,
                                              log_dir=log_dir)

        # Reward history buffer
        self.reward_history = deque(maxlen=100)
        self.step_count     = 0

    def reset(self) -> None:
        self.reasoner.reset()
        self.attention.reset()
        self.cf_engine.reset()
        self.composer.reset()
        self.reward_history.clear()
        self.step_count = 0

    def step(self,
             V_combined    : np.ndarray,
             U             : float,
             C             : float,
             Q_risk        : np.ndarray,
             conflict_score: float,
             stn_burst     : bool,
             direct_inh    : np.ndarray,
             indirect_exc  : np.ndarray,
             stn_global    : float,
             gate_margins  : np.ndarray,
             DA            : float,
             ht5           : float,
             NE            : float,
             alpha_t       : float,
             Mt            : float,
             rho           : float,
             reward        : float = None,
             gate_action   : int   = None,
             t_ms          : float = 0.0,
             explain       : bool  = False) -> dict:
        """
        Full Phase 9 step (Steps 25-28).

        explain=True  : generate full text explanation (every step is slow)
        explain=False : compute all signals but skip text generation
        """
        self.step_count += 1

        if reward is not None:
            self.reward_history.append(float(reward))

        rw_list = list(self.reward_history)

        # Step 25: symbolic reasoning
        r_out = self.reasoner.reason(
            V_combined     = V_combined,
            U              = U,
            C              = C,
            Q_risk         = Q_risk,
            conflict_score = conflict_score,
            stn_burst      = stn_burst,
            direct_inh     = direct_inh,
            indirect_exc   = indirect_exc,
            DA             = DA,
            ht5            = ht5,
            NE             = NE,
            reward_history = rw_list,
            gate_margins   = gate_margins,
            gate_action    = gate_action)

        # Step 26: attention
        V_hist = self.reasoner.belief_trajectory(last_n=10)
        a_out  = self.attention.compute(
            V_combined     = V_combined,
            Q_risk         = Q_risk,
            direct_inh     = direct_inh,
            indirect_exc   = indirect_exc,
            stn_global     = stn_global,
            reward_history = rw_list,
            V_history      = V_hist,
            DA             = DA,
            ht5            = ht5,
            NE             = NE)

        if reward is not None:
            self.attention.update_weights(reward,
                                          r_out["selected_action"])

        # Step 27: counterfactuals
        cfs = self.cf_engine.generate(
            selected_action = r_out["selected_action"],
            V_combined      = V_combined,
            Q_risk          = Q_risk,
            direct_inh      = direct_inh,
            U               = U,
            C               = C,
            conflict_score  = conflict_score,
            stn_burst       = stn_burst,
            gate_margins    = gate_margins,
            rho             = rho,
            DA              = DA,
            ht5             = ht5,
            NE              = NE)

        output = {
            "selected_action"  : r_out["selected_action"],
            "explanation_conf" : r_out["explanation_conf"],
            "n_rules_fired"    : r_out["n_rules_fired"],
            "blocked_actions"  : r_out["blocked_actions"],
            "dominant_signal"  : a_out.get("dominant_signal", ""),
            "attention_weights": a_out.get("attention_weights", []),
            "n_counterfactuals": len(cfs),
        }

        # Step 28: full text explanation (generated selectively)
        if explain:
            gm = np.asarray(gate_margins, dtype=float)
            gm_sel = (float(gm[r_out["selected_action"]])
                      if len(gm) > r_out["selected_action"] else 0.0)
            e_out = self.composer.compose(
                reasoning_out  = r_out,
                attention_out  = a_out,
                counterfactuals= cfs,
                U=U, C=C, DA=DA, ht5=ht5, NE=NE,
                alpha_t=alpha_t, Mt=Mt, rho=rho,
                conflict_score=conflict_score,
                gate_margin=gm_sel, t_ms=t_ms)
            output["full_explanation"] = e_out["full_explanation"]
            output["machine_record"]   = e_out["machine_record"]
        else:
            output["full_explanation"] = ""
            output["machine_record"]   = {}

        return output

    def save_explanations(self,
                           filename: str = "phase9_log.json") -> str:
        return self.composer.save_log(filename)