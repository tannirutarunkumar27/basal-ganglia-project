"""
SymbolicRuleEngine  —  Step 25 (part 1)
-----------------------------------------
Encodes biologically grounded decision rules as symbolic
predicates that operate on top of neural evidence.

Rules are if-then logical statements whose antecedents
are neural quantities (belief, uncertainty, conflict, risk)
and whose consequents are symbolic conclusions:
    "action a is PREFERRED / BLOCKED / UNCERTAIN"

Rule categories:
    1. Conflict rules   : STN burst -> block all actions
    2. Uncertainty rules: high Ut   -> defer or explore
    3. Risk rules       : high Var  -> avoid high-variance actions
    4. Reward rules     : positive  -> strengthen selected
    5. Pathway rules    : Go > NoGo -> direct release permitted
    6. Neuromod rules   : DA/5HT/NE -> modulate all decisions

Advanced innovation:
    Rules are not hand-designed — they are derived from
    neural quantities that emerge during learning, bridging
    the gap between neural computation and symbolic cognition.
"""

import numpy as np
from dataclasses import dataclass, field
from typing      import List, Optional


@dataclass
class SymbolicConclusion:
    """A single symbolic conclusion from one rule evaluation."""
    rule_name    : str
    action       : Optional[int]       # None = applies globally
    verdict      : str                 # PREFERRED / BLOCKED / UNCERTAIN / NEUTRAL
    strength     : float               # 0-1, how strongly the rule fires
    rationale    : str                 # human-readable reason
    evidence     : dict = field(default_factory=dict)


class SymbolicRuleEngine:

    def __init__(self,
                 n_actions        : int,
                 conflict_eps     : float = 0.3,
                 high_U_threshold : float = 0.6,
                 high_risk_sigma  : float = 0.5,
                 go_dominance_thr : float = 0.6):

        self.n_actions        = n_actions
        self.conflict_eps     = conflict_eps
        self.high_U_threshold = high_U_threshold
        self.high_risk_sigma  = high_risk_sigma
        self.go_dominance_thr = go_dominance_thr

        self.conclusions_history = []

    def reset(self) -> None:
        self.conclusions_history.clear()

    def evaluate(self,
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
                  gate_margins  : np.ndarray) -> List[SymbolicConclusion]:
        """
        Evaluates all symbolic rules against current neural state.
        Returns list of SymbolicConclusion objects.
        """
        conclusions = []
        V = np.asarray(V_combined, dtype=float)
        Q = np.asarray(Q_risk,     dtype=float)
        di = np.asarray(direct_inh,  dtype=float)
        ie = np.asarray(indirect_exc, dtype=float)
        gm = np.asarray(gate_margins, dtype=float)

        # ── Rule 1: STN global stop ─────────────────────────────
        if stn_burst:
            conclusions.append(SymbolicConclusion(
                rule_name = "STN_global_stop",
                action    = None,
                verdict   = "BLOCKED",
                strength  = 1.0,
                rationale = (
                    "STN burst detected — global action suppression active. "
                    "Conflict between alternatives prevents release."),
                evidence  = {
                    "conflict_score": float(conflict_score),
                    "stn_burst"     : True}
            ))

        # ── Rule 2: High uncertainty deference ─────────────────
        if U > self.high_U_threshold:
            conclusions.append(SymbolicConclusion(
                rule_name = "high_uncertainty_defer",
                action    = None,
                verdict   = "UNCERTAIN",
                strength  = float(U),
                rationale = (
                    f"Uncertainty Ut={U:.3f} exceeds threshold "
                    f"{self.high_U_threshold}. System in exploratory mode — "
                    f"confidence gate raised to require stronger evidence."),
                evidence  = {"U": float(U), "C": float(C)}
            ))

        # ── Rule 3: Belief-dominant action preferred ────────────
        if len(V) > 0:
            best_a   = int(np.argmax(V))
            v_sorted = np.sort(V)[::-1]
            margin   = float(v_sorted[0] - v_sorted[1]) \
                       if len(v_sorted) > 1 else 0.0

            if margin > 0.2 * abs(v_sorted[0]):
                conclusions.append(SymbolicConclusion(
                    rule_name = "belief_dominant",
                    action    = best_a,
                    verdict   = "PREFERRED",
                    strength  = float(np.clip(margin, 0, 1)),
                    rationale = (
                        f"Action {best_a} has highest posterior belief "
                        f"Va={V[best_a]:.3f} with margin {margin:.3f} "
                        f"over next-best alternative."),
                    evidence  = {
                        "V"      : V.tolist(),
                        "best_a" : best_a,
                        "margin" : margin}
                ))

        # ── Rule 4: Risk-dominated block ────────────────────────
        if len(Q) > 0:
            worst_q = int(np.argmin(Q))
            if Q[worst_q] < -self.high_risk_sigma:
                conclusions.append(SymbolicConclusion(
                    rule_name = "risk_block",
                    action    = worst_q,
                    verdict   = "BLOCKED",
                    strength  = float(np.clip(-Q[worst_q], 0, 1)),
                    rationale = (
                        f"Action {worst_q} has negative risk-adjusted utility "
                        f"Q_risk={Q[worst_q]:.3f}. High reward variance makes "
                        f"this choice unsafe under current risk aversion."),
                    evidence  = {"Q_risk": Q.tolist(), "worst": worst_q}
                ))

        # ── Rule 5: Direct pathway dominance ────────────────────
        if len(di) > 0 and len(ie) > 0:
            go_strength   = float(np.max(di))
            nogo_strength = float(np.mean(ie))
            ratio = go_strength / (nogo_strength + 1e-8)

            if ratio > self.go_dominance_thr / (1.0 - self.go_dominance_thr):
                best_go = int(np.argmax(di))
                conclusions.append(SymbolicConclusion(
                    rule_name = "direct_pathway_dominance",
                    action    = best_go,
                    verdict   = "PREFERRED",
                    strength  = float(np.clip(ratio / 3.0, 0, 1)),
                    rationale = (
                        f"Direct pathway Go signal for action {best_go} "
                        f"dominates indirect No-Go (ratio={ratio:.2f}). "
                        f"Thalamic disinhibition pathway is active."),
                    evidence  = {
                        "go_strength"  : go_strength,
                        "nogo_strength": nogo_strength,
                        "ratio"        : ratio}
                ))

        # ── Rule 6: Gate open — imminent release ────────────────
        if len(gm) > 0:
            open_channels = [a for a in range(self.n_actions)
                             if a < len(gm) and gm[a] > 0]
            if open_channels:
                winner = open_channels[
                    int(np.argmax([gm[a] for a in open_channels]))]
                conclusions.append(SymbolicConclusion(
                    rule_name = "gate_open",
                    action    = winner,
                    verdict   = "PREFERRED",
                    strength  = float(np.clip(
                        gm[winner] / 0.5, 0, 1)),
                    rationale = (
                        f"GPi gate open for action {winner} — "
                        f"margin={gm[winner]:.4f}. "
                        f"Thalamus disinhibited, motor cortex ready."),
                    evidence  = {
                        "gate_margins" : gm.tolist(),
                        "winner"       : winner}
                ))

        # ── Rule 7: Neuromodulator context ──────────────────────
        dominant = ("DA" if DA >= ht5 and DA >= NE
                    else "5HT" if ht5 >= NE else "NE")
        nm_rationale_map = {
            "DA" : ("Dopamine dominant — exploitation mode active. "
                    "System favours high-value known actions."),
            "5HT": ("Serotonin dominant — risk-averse mode. "
                    "System avoids high-variance outcomes."),
            "NE" : ("Norepinephrine dominant — arousal/exploration mode. "
                    "System is sensitive to novel or surprising options."),
        }
        conclusions.append(SymbolicConclusion(
            rule_name = "neuromodulator_context",
            action    = None,
            verdict   = "NEUTRAL",
            strength  = float(max(DA, ht5, NE)),
            rationale = nm_rationale_map[dominant],
            evidence  = {"DA": float(DA),
                         "5HT": float(ht5),
                         "NE": float(NE),
                         "dominant": dominant}
        ))

        # ── Rule 8: Reward history trend ────────────────────────
        if len(reward_history) >= 5:
            recent_mean = float(np.mean(reward_history[-5:]))
            trend = float(np.mean(np.diff(reward_history[-5:])))
            if trend > 0.1:
                verdict = "PREFERRED"
                rationale = (
                    f"Positive reward trend (slope={trend:+.3f}). "
                    f"Recent mean={recent_mean:.3f}. Current policy improving.")
            elif trend < -0.1:
                verdict = "UNCERTAIN"
                rationale = (
                    f"Negative reward trend (slope={trend:+.3f}). "
                    f"Recent mean={recent_mean:.3f}. Policy may need revision.")
            else:
                verdict = "NEUTRAL"
                rationale = (
                    f"Stable reward history. Mean={recent_mean:.3f}.")
            conclusions.append(SymbolicConclusion(
                rule_name = "reward_trend",
                action    = None,
                verdict   = verdict,
                strength  = float(abs(trend)),
                rationale = rationale,
                evidence  = {
                    "trend"      : trend,
                    "recent_mean": recent_mean}
            ))

        self.conclusions_history.append(conclusions)
        return conclusions

    def aggregate_verdicts(self,
                            conclusions: List[SymbolicConclusion]
                            ) -> dict:
        """
        Aggregates all conclusions into per-action verdict scores.
        Returns {action_id: net_score} and dominant verdict per action.
        """
        scores = np.zeros(self.n_actions)

        for c in conclusions:
            sign = (+1.0 if c.verdict == "PREFERRED"
                    else -1.0 if c.verdict == "BLOCKED"
                    else 0.0)
            if c.action is not None and c.action < self.n_actions:
                scores[c.action] += sign * c.strength
            else:
                # Global rule: applies to all actions
                # blocked globals suppress all; preferred globals boost all
                scores += sign * c.strength * 0.3

        return {
            "scores"         : scores.tolist(),
            "best_action"    : int(np.argmax(scores)),
            "worst_action"   : int(np.argmin(scores)),
            "blocked_actions": [a for a in range(self.n_actions)
                                 if scores[a] < -0.2],
            "n_conclusions"  : len(conclusions),
        }