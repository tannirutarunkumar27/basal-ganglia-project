"""
NeuroSymbolicReasoner  —  Step 25
-----------------------------------
Top-level neuro-symbolic layer.

Inputs  (neural quantities from Phases 2-8):
    posterior belief  Va(t)       Phase 3
    temporal history  V_history   Phase 3
    uncertainty/conf  U, C        Phase 3
    risk estimate     Q_risk      Phase 6
    conflict score    conflict    Phase 4
    pathway signals   direct/ind  Phase 4
    reward history                Phase 6
    neuromodulators   DA/5HT/NE   Phase 8

Outputs (symbolic + numeric):
    selected action            int
    rationale                  str
    alternative ranking        list[(action, score)]
    explanation confidence     float
    counterfactual_placeholder list (filled by Step 27)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from symbolic.symbolic_rules import SymbolicRuleEngine
from collections              import deque


class NeuroSymbolicReasoner:

    def __init__(self, n_actions: int, dt: float = 0.1e-3):
        self.n_actions = n_actions
        self.dt        = dt
        self.rules     = SymbolicRuleEngine(n_actions)

        self.V_history      = deque(maxlen=500)
        self.output_history = deque(maxlen=2000)
        self.step_count     = 0

    def reset(self) -> None:
        self.rules.reset()
        self.V_history.clear()
        self.output_history.clear()
        self.step_count = 0

    def reason(self,
               V_combined    : np.ndarray,
               U             : float,
               C             : float,
               Q_risk        : np.ndarray,
               conflict_score: float,
               stn_burst     : bool,
               direct_inh    : np.ndarray,
               indirect_exc  : np.ndarray,
               DA            : float,
               ht5           : float,
               NE            : float,
               reward_history: list,
               gate_margins  : np.ndarray,
               gate_action   : int = None) -> dict:
        """
        Step 25 core reasoning pass.
        Returns structured reasoning output dict.
        """
        self.step_count += 1
        V = np.asarray(V_combined, dtype=float)

        # Store belief history
        self.V_history.append(V.copy())

        # Evaluate all symbolic rules
        conclusions = self.rules.evaluate(
            V_combined     = V,
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
            reward_history = reward_history,
            gate_margins   = gate_margins)

        # Aggregate into per-action scores
        agg    = self.rules.aggregate_verdicts(conclusions)
        scores = np.array(agg["scores"])

        # Blend symbolic scores with neural belief
        v_min   = V.min()
        v_range = V.max() - v_min          # replaces deprecated ptp()
        V_norm  = (V - v_min) / (v_range + 1e-8)

        s_min   = scores.min()
        s_range = scores.max() - s_min     # replaces deprecated ptp()
        s_norm  = np.clip(
            (scores - s_min) / (s_range + 1e-8),
            0.0, 1.0)

        combined_score = 0.6 * V_norm + 0.4 * s_norm

        # Block any explicitly blocked actions
        for a in agg["blocked_actions"]:
            combined_score[a] = -999.0

        # Select action
        selected = (gate_action if gate_action is not None
                    else int(np.argmax(combined_score)))

        # Alternative ranking (excluding selected and blocked)
        ranked = [
            (a, float(combined_score[a]))
            for a in np.argsort(combined_score)[::-1]
            if a != selected
        ][:self.n_actions - 1]

        # Primary rationale from highest-strength conclusions
        rationale_parts = []
        for c in sorted(conclusions,
                         key=lambda x: x.strength, reverse=True)[:3]:
            if c.action == selected or c.action is None:
                rationale_parts.append(c.rationale)
        primary_rationale = (
            " | ".join(rationale_parts)
            if rationale_parts else "No dominant rule fired.")

        # Explanation confidence
        expl_conf = float(np.clip(
            C * 0.5
            + (1.0 - float(stn_burst)) * 0.3
            + (1.0 - U) * 0.2,
            0.0, 1.0))

        output = {
            "selected_action"    : selected,
            "rationale"          : primary_rationale,
            "alternative_ranking": ranked,
            "explanation_conf"   : expl_conf,
            "symbolic_scores"    : scores.tolist(),
            "combined_scores"    : combined_score.tolist(),
            "blocked_actions"    : agg["blocked_actions"],
            "conclusions"        : [
                {
                    "rule"     : c.rule_name,
                    "action"   : c.action,
                    "verdict"  : c.verdict,
                    "strength" : round(c.strength, 4),
                    "rationale": c.rationale,
                }
                for c in conclusions
            ],
            "n_rules_fired"      : len(conclusions),
            "V_combined"         : V.tolist(),
            "U"                  : float(U),
            "C"                  : float(C),
        }

        self.output_history.append(output.copy())
        return output

    def belief_trajectory(self, last_n: int = 20) -> np.ndarray:
        """Returns (last_n, A) array of recent belief states."""
        hist = list(self.V_history)[-last_n:]
        if not hist:
            return np.zeros((0, self.n_actions))
        return np.array(hist)