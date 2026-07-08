"""
ExplanationComposer  —  Step 28
---------------------------------
Assembles all Phase 9 outputs into a single, structured,
human-readable decision explanation record.

Output fields:
    chosen_action            int
    confidence               float
    uncertainty              float
    selected_pathway_rationale str
    rejected_alternatives    list[str]
    counterfactual_comparison str
    neuromodulatory_summary  str
    attention_summary        str
    full_explanation         str
    machine_readable         dict

This is the final output of the entire 9-phase pipeline:
a fully explainable neuromorphic decision record.
"""

import numpy as np
import json
import os
from collections import deque
from typing      import List


class ExplanationComposer:

    def __init__(self, n_actions: int,
                 action_names : list = None,
                 log_dir      : str  = "results"):

        self.n_actions    = n_actions
        self.action_names = (action_names if action_names
                             else [f"action_{a}"
                                   for a in range(n_actions)])
        self.log_dir      = log_dir

        self.explanation_history = deque(maxlen=1000)
        self.step_count          = 0

    def reset(self) -> None:
        self.explanation_history.clear()
        self.step_count = 0

    def compose(self,
                # Step 25 outputs
                reasoning_out    : dict,
                # Step 26 outputs
                attention_out    : dict,
                # Step 27 outputs
                counterfactuals  : list,
                # Raw signals
                U                : float,
                C                : float,
                DA               : float,
                ht5              : float,
                NE               : float,
                alpha_t          : float,
                Mt               : float,
                rho              : float,
                conflict_score   : float,
                gate_margin      : float,
                t_ms             : float = 0.0) -> dict:
        """
        Step 28 core: assembles the complete explanation record.
        """
        self.step_count += 1

        action   = reasoning_out["selected_action"]
        name     = self.action_names[action]
        conf     = reasoning_out["explanation_conf"]
        rationale = reasoning_out["rationale"]

        # ── Pathway rationale ─────────────────────────────────
        conclusions = reasoning_out.get("conclusions", [])
        pathway_rules = [c for c in conclusions
                          if "pathway" in c["rule"].lower()
                          or "gate" in c["rule"].lower()]
        if pathway_rules:
            pathway_rationale = pathway_rules[0]["rationale"]
        else:
            pathway_rationale = (
                f"No dominant pathway rule. "
                f"Gate margin = {gate_margin:.4f}. "
                f"Confidence = {C:.3f}.")

        # ── Rejected alternatives ─────────────────────────────
        alt_ranking = reasoning_out.get("alternative_ranking", [])
        rejected_texts = []
        for a, score in alt_ranking[:3]:
            a_name  = self.action_names[a]
            blocked = a in reasoning_out.get("blocked_actions", [])
            status  = "BLOCKED" if blocked else "SUBOPTIMAL"
            rejected_texts.append(
                f"{a_name} [{status}] (score={score:.3f})")

        # ── Counterfactual text ───────────────────────────────
        cf_lines = []
        for cf in sorted(counterfactuals,
                          key=lambda x: abs(x.delta_Q_risk),
                          reverse=True)[:3]:
            cf_lines.append(
                f"  If {self.action_names[cf.rejected_action]} "
                f"were chosen: reward gap = {cf.delta_Q_risk:+.3f}, "
                f"confidence = {cf.confidence:.2f}. "
                + cf.explanation)
        cf_text = ("\n".join(cf_lines)
                   if cf_lines else "No counterfactuals available.")

        # ── Neuromodulatory summary ───────────────────────────
        dominant_nm = ("DA" if DA >= ht5 and DA >= NE
                       else "5HT" if ht5 >= NE else "NE")
        nm_mode_map = {
            "DA" : "exploitation",
            "5HT": "risk-avoidance",
            "NE" : "exploration",
        }
        nm_summary = (
            f"Neuromodulator state: DA={DA:.3f}, 5HT={ht5:.3f}, "
            f"NE={NE:.3f}. Mt={Mt:.3f}. "
            f"Dominant={dominant_nm} "
            f"({nm_mode_map[dominant_nm]} mode). "
            f"alpha_t={alpha_t:.5f}, rho={rho:.3f}.")

        # ── Attention summary ─────────────────────────────────
        top_sigs = attention_out.get("top_signals", [])
        if top_sigs:
            attn_text = (
                "Decision primarily driven by: "
                + ", ".join(
                    f"{s['name']} ({s['weight']:.2f})"
                    for s in top_sigs[:3]) + ".")
        else:
            attn_text = "Attention not computed."

        # ── Full explanation (structured prose) ───────────────
        full_lines = [
            f"DECISION  : {name} (action {action})",
            f"CONFIDENCE: {conf:.3f}  |  UNCERTAINTY: {U:.3f}  "
            f"|  GATE MARGIN: {gate_margin:.4f}",
            f"",
            f"RATIONALE : {rationale}",
            f"",
            f"PATHWAY   : {pathway_rationale}",
            f"",
            f"ATTENTION : {attn_text}",
            f"",
            f"REJECTED ALTERNATIVES:",
        ]
        for r in rejected_texts:
            full_lines.append(f"  - {r}")
        full_lines += [
            f"",
            f"COUNTERFACTUALS:",
            cf_text,
            f"",
            f"NEUROMODULATORS: {nm_summary}",
        ]
        full_explanation = "\n".join(full_lines)

        # ── Machine-readable record ───────────────────────────
        machine_record = {
            "t_ms"                      : float(t_ms),
            "step"                      : self.step_count,
            "chosen_action"             : int(action),
            "action_name"               : name,
            "confidence"                : float(conf),
            "uncertainty"               : float(U),
            "gate_margin"               : float(gate_margin),
            "rationale"                 : rationale,
            "pathway_rationale"         : pathway_rationale,
            "rejected_alternatives"     : rejected_texts,
            "counterfactual_comparison" : cf_text,
            "neuromodulatory_summary"   : nm_summary,
            "attention_summary"         : attn_text,
            "dominant_nm"               : dominant_nm,
            "alpha_t"                   : float(alpha_t),
            "Mt"                        : float(Mt),
            "rho"                       : float(rho),
            "conflict_score"            : float(conflict_score),
            "n_rules_fired"             : reasoning_out.get(
                "n_rules_fired", 0),
            "attention_weights"         : attention_out.get(
                "attention_weights", []),
            "counterfactuals"           : [
                {"rejected": cf.rejected_action,
                 "delta_Q" : cf.delta_Q_risk,
                 "conf"    : cf.confidence,
                 "text"    : cf.explanation}
                for cf in counterfactuals
            ],
        }

        output = {
            "full_explanation"    : full_explanation,
            "machine_record"      : machine_record,
            "chosen_action"       : int(action),
            "confidence"          : float(conf),
            "uncertainty"         : float(U),
            "pathway_rationale"   : pathway_rationale,
            "rejected_texts"      : rejected_texts,
            "cf_text"             : cf_text,
            "nm_summary"          : nm_summary,
            "attn_text"           : attn_text,
        }

        self.explanation_history.append(machine_record)
        return output

    def save_log(self, filename: str = "phase9_explanations.json",
                  last_n: int = 200) -> str:
        """Saves last_n explanation records to JSON."""
        os.makedirs(self.log_dir, exist_ok=True)
        path    = os.path.join(self.log_dir, filename)
        records = list(self.explanation_history)[-last_n:]
        with open(path, "w") as f:
            json.dump({"explanations": records,
                       "n_total"     : self.step_count}, f, indent=2)
        return path

    def confidence_stats(self) -> dict:
        hist = list(self.explanation_history)
        if not hist:
            return {}
        confs = [r["confidence"] for r in hist]
        return {
            "mean_confidence": float(np.mean(confs)),
            "std_confidence" : float(np.std(confs)),
            "min_confidence" : float(np.min(confs)),
            "max_confidence" : float(np.max(confs)),
        }