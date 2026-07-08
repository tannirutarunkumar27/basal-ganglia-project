"""
CounterfactualEngine  —  Step 27
----------------------------------
Generates "what-if" alternative decision explanations:

    For each rejected action a_j (j != selected):
        - Would reward be higher?
        - Would STN still block it?
        - Would uncertainty change?
        - What pathway contributions would differ?

Advanced innovation:
    Counterfactual reasoning transforms the system from a
    black box into an interpretable agent that can explain
    WHY it rejected alternatives — not just which one it chose.

Example output:
    "Action 3 was rejected because under current risk (rho=0.6)
     and uncertainty (U=0.45), action 2 had higher expected
     utility Q_risk=0.79 vs Q_risk=-0.07 for action 3.
     Additionally, the direct pathway Go signal for action 3
     was insufficient (di=0.04) to open the GPi gate."
"""

import numpy as np
from dataclasses import dataclass, field
from typing      import List


@dataclass
class Counterfactual:
    """A single counterfactual explanation for one rejected action."""
    rejected_action  : int
    selected_action  : int
    reason_codes     : List[str]
    explanation      : str
    delta_reward     : float    # reward(selected) - reward(rejected)
    delta_Q_risk     : float    # Q_risk(selected) - Q_risk(rejected)
    stn_would_block  : bool
    uncertainty_delta: float    # U change if rejected was taken
    pathway_gap      : float    # Go(selected) - Go(rejected)
    confidence       : float    # confidence in this counterfactual


class CounterfactualEngine:

    def __init__(self, n_actions: int,
                 action_names: list = None):
        self.n_actions    = n_actions
        self.action_names = (action_names if action_names
                             else [f"action_{a}"
                                   for a in range(n_actions)])
        self.cf_history   = []
        self.step_count   = 0

    def reset(self) -> None:
        self.cf_history.clear()
        self.step_count = 0

    def generate(self,
                  selected_action: int,
                  V_combined     : np.ndarray,
                  Q_risk         : np.ndarray,
                  direct_inh     : np.ndarray,
                  U              : float,
                  C              : float,
                  conflict_score : float,
                  stn_burst      : bool,
                  gate_margins   : np.ndarray,
                  rho            : float,
                  DA             : float,
                  ht5            : float,
                  NE             : float) -> List[Counterfactual]:
        """
        Step 27 core: generates one Counterfactual per rejected action.

        Returns list of Counterfactual objects (length n_actions - 1).
        """
        self.step_count += 1
        V  = np.asarray(V_combined, dtype=float)
        Q  = np.asarray(Q_risk,     dtype=float)
        di = np.asarray(direct_inh, dtype=float)
        gm = np.asarray(gate_margins, dtype=float)

        counterfactuals = []

        for a in range(self.n_actions):
            if a == selected_action:
                continue

            reason_codes = []
            explanation_parts = []

            sel = selected_action
            sel_name = self.action_names[sel]
            rej_name = self.action_names[a]

            # ── 1. Q_risk comparison ───────────────────────────
            dQ = (float(Q[sel]) - float(Q[a])
                  if len(Q) > max(sel, a) else 0.0)
            if dQ > 0.05:
                reason_codes.append("LOWER_Q_RISK")
                explanation_parts.append(
                    f"{rej_name} had lower risk-adjusted utility "
                    f"Q_risk={Q[a]:.3f} vs Q_risk={Q[sel]:.3f} "
                    f"for {sel_name} (gap={dQ:+.3f}, rho={rho:.2f})")
            elif dQ < -0.05:
                reason_codes.append("HIGHER_Q_RISK_BUT_BLOCKED")
                explanation_parts.append(
                    f"{rej_name} had higher Q_risk={Q[a]:.3f} "
                    f"but was blocked by other constraints")

            # ── 2. Belief comparison ───────────────────────────
            dV = (float(V[sel]) - float(V[a])
                  if len(V) > max(sel, a) else 0.0)
            if dV > 0.1:
                reason_codes.append("LOWER_BELIEF")
                explanation_parts.append(
                    f"Bayesian belief lower for {rej_name}: "
                    f"Va={V[a]:.3f} vs Va={V[sel]:.3f}")

            # ── 3. STN / conflict block ────────────────────────
            stn_would_block = stn_burst
            if stn_burst:
                reason_codes.append("STN_BLOCKED")
                explanation_parts.append(
                    f"STN burst active (conflict={conflict_score:.3f}) "
                    f"— {rej_name} would also be suppressed")

            # ── 4. Gate margin (GPi) comparison ───────────────
            gate_gap = (float(gm[sel]) - float(gm[a])
                        if len(gm) > max(sel, a) else 0.0)
            if gate_gap > 0.05:
                reason_codes.append("INSUFFICIENT_GO_SIGNAL")
                explanation_parts.append(
                    f"GPi gate margin for {rej_name}={gm[a]:.4f} "
                    f"vs {sel_name}={gm[sel]:.4f} — "
                    f"direct pathway insufficient")

            # ── 5. Neuromodulator context effect ──────────────
            nm_effect = ""
            if DA > 0.6 and Q[sel] > Q[a]:
                nm_effect = (
                    f"DA-dominant exploitation mode "
                    f"(DA={DA:.2f}) favours known high-value action")
                reason_codes.append("DA_EXPLOITATION")
                explanation_parts.append(nm_effect)
            elif ht5 > 0.6:
                explanation_parts.append(
                    f"5-HT risk-aversion (5HT={ht5:.2f}) penalises "
                    f"high-variance alternatives")
                reason_codes.append("5HT_RISK_AVERSION")

            # ── 6. Uncertainty impact ─────────────────────────
            # Hypothetical: if rejected action were taken,
            # belief variance might increase (less evidence)
            if len(V) > a:
                hyp_var_increase = float(np.abs(V[a] - V.mean()))
                unc_delta = float(U + 0.1 * hyp_var_increase)
            else:
                unc_delta = float(U)
            uncertainty_delta = unc_delta - float(U)

            if uncertainty_delta > 0.05:
                reason_codes.append("HIGHER_UNCERTAINTY")
                explanation_parts.append(
                    f"Taking {rej_name} would increase uncertainty "
                    f"by ~{uncertainty_delta:.3f}")

            # ── Build full explanation string ─────────────────
            if not explanation_parts:
                reason_codes.append("SUBOPTIMAL")
                explanation_parts.append(
                    f"{rej_name} was suboptimal across all criteria")

            full_explanation = (
                f"{rej_name} was rejected: "
                + "; ".join(explanation_parts) + ".")

            # Confidence in this counterfactual
            cf_conf = float(np.clip(
                C * 0.4 + abs(dQ) * 0.3 + abs(dV) * 0.3,
                0.0, 1.0))

            go_gap = (float(di[sel]) - float(di[a])
                      if len(di) > max(sel, a) else 0.0)

            counterfactuals.append(Counterfactual(
                rejected_action   = a,
                selected_action   = sel,
                reason_codes      = reason_codes,
                explanation       = full_explanation,
                delta_reward      = float(dQ),
                delta_Q_risk      = float(dQ),
                stn_would_block   = stn_would_block,
                uncertainty_delta = float(uncertainty_delta),
                pathway_gap       = float(go_gap),
                confidence        = cf_conf,
            ))

        self.cf_history.append(counterfactuals)
        return counterfactuals

    def format_counterfactuals(self,
                                counterfactuals: List[Counterfactual],
                                max_per_output: int = 3) -> str:
        """
        Formats counterfactuals as a structured text block.
        Returns human-readable multi-line string.
        """
        if not counterfactuals:
            return "No alternative actions to evaluate."

        lines = ["Counterfactual analysis:"]
        for cf in sorted(counterfactuals,
                          key=lambda x: abs(x.delta_Q_risk),
                          reverse=True)[:max_per_output]:
            lines.append(
                f"  [{cf.rejected_action}] {cf.explanation}")
            lines.append(
                f"      Reward gap: {cf.delta_reward:+.3f}  "
                f"Pathway gap: {cf.pathway_gap:+.4f}  "
                f"CF confidence: {cf.confidence:.2f}")
        return "\n".join(lines)