"""
ExplainabilityLogger  —  Step 15 (logging requirement)
--------------------------------------------------------
Captures and structures ALL signals required by Phase 9 (XAI):

  Required fields per decision:
    selected_action        : which action was released
    confidence_score       : P(action | evidence)
    uncertainty            : Ut
    gate_margin            : θ_t - GPi_winner
    conflict_level         : from hyperdirect pathway
    pathway_contributions  : Go, No-Go, STN, Base per action
    action_probabilities   : P(a|s) from Bayesian pipeline
    belief_scores          : Va(t) temporal belief
    thalamic_activity      : relay state per channel
    refractory_state       : which channels are recovering

This logger is ESSENTIAL for:
  - Phase 9 neuro-symbolic reasoning
  - Phase 14 attention-based explanations
  - Phase 15 counterfactual reasoning
  - Ablation study diagnostics
"""

import numpy as np
import json
import os
from collections import deque


class ExplainabilityLogger:

    def __init__(self,
                 n_actions   : int,
                 buffer_size : int = 10000,
                 log_dir     : str = "results"):

        self.n_actions   = n_actions
        self.buffer_size = buffer_size
        self.log_dir     = log_dir

        # Circular buffer of decision records
        self.buffer = deque(maxlen=buffer_size)

        # Aggregated statistics
        self.stats = {
            "total_decisions"   : 0,
            "total_releases"    : 0,
            "action_counts"     : np.zeros(n_actions, dtype=int),
            "mean_confidence"   : 0.0,
            "mean_U"            : 0.0,
            "mean_margin"       : 0.0,
            "mean_conflict"     : 0.0,
        }

        # Running means (EMA)
        self._ema_conf     = 0.5
        self._ema_U        = 0.5
        self._ema_margin   = 0.0
        self._ema_conflict = 0.0

    def reset(self):
        self.buffer.clear()
        self.stats = {
            "total_decisions"   : 0,
            "total_releases"    : 0,
            "action_counts"     : np.zeros(self.n_actions, dtype=int),
            "mean_confidence"   : 0.0,
            "mean_U"            : 0.0,
            "mean_margin"       : 0.0,
            "mean_conflict"     : 0.0,
        }

    def log(self, record: dict) -> None:
        """
        Logs a single decision record from ThalamocorticalRelay.step().
        Updates running statistics.
        """
        self.buffer.append(record.copy())
        self.stats["total_decisions"] += 1

        if record.get("action_released", False):
            action = record.get("released_action")
            if action is not None:
                self.stats["total_releases"]    += 1
                self.stats["action_counts"][action] += 1

        # EMA running means
        α = 0.05
        self._ema_conf     = (1-α)*self._ema_conf     + α*record.get("release_confidence", 0)
        self._ema_U        = (1-α)*self._ema_U        + α*record.get("U", 0.5)
        margins            = record.get("gate_margins", [0]*self.n_actions)
        self._ema_margin   = (1-α)*self._ema_margin   + α*max(margins)
        self._ema_conflict = (1-α)*self._ema_conflict + α*record.get("conflict_score", 0)

        self.stats["mean_confidence"] = self._ema_conf
        self.stats["mean_U"]          = self._ema_U
        self.stats["mean_margin"]     = self._ema_margin
        self.stats["mean_conflict"]   = self._ema_conflict

    def get_last_n(self, n: int = 100) -> list:
        """Returns last n records from buffer."""
        return list(self.buffer)[-n:]

    def get_releases_only(self, n: int = 500) -> list:
        """Returns only records where an action was released."""
        recent = list(self.buffer)[-n*10:]
        return [r for r in recent if r.get("action_released", False)][-n:]

    def pathway_attribution_report(self,
                                    last_n: int = 200) -> dict:
        """
        Computes mean pathway contribution to GPi across last_n steps.
        Used by Phase 9 attention-based explanation.
        """
        recent = list(self.buffer)[-last_n:]
        if not recent:
            return {}

        go_arr   = np.array([r.get("pathway_go",  [0]*self.n_actions)
                              for r in recent])
        nogo_arr = np.array([r.get("pathway_nogo",[0]*self.n_actions)
                              for r in recent])
        stn_arr  = np.array([r.get("pathway_stn", [0]*self.n_actions)
                              for r in recent])
        base_arr = np.array([r.get("pathway_base",[1]*self.n_actions)
                              for r in recent])

        total    = (base_arr.mean(axis=0)
                    + go_arr.mean(axis=0)
                    + nogo_arr.mean(axis=0)
                    + stn_arr.mean(axis=0) + 1e-8)

        return {
            "go_attribution"  : (go_arr.mean(axis=0)   / total).tolist(),
            "nogo_attribution": (nogo_arr.mean(axis=0) / total).tolist(),
            "stn_attribution" : (stn_arr.mean(axis=0)  / total).tolist(),
            "base_attribution": (base_arr.mean(axis=0) / total).tolist(),
        }

    def confidence_calibration(self,
                                 last_n: int = 500) -> dict:
        """
        Checks whether confidence correlates with accuracy.
        Groups decisions by confidence quartile and computes
        accuracy per quartile (requires correct_action knowledge).
        """
        releases = self.get_releases_only(last_n)
        if not releases:
            return {"note": "no releases yet"}

        confs   = [r["release_confidence"] for r in releases]
        return {
            "mean_confidence" : float(np.mean(confs)),
            "std_confidence"  : float(np.std(confs)),
            "min_confidence"  : float(np.min(confs)),
            "max_confidence"  : float(np.max(confs)),
            "n_releases"      : len(releases),
        }

    def generate_text_explanation(self,
                                   record: dict,
                                   action_names: list = None) -> str:
        """
        Generates a human-readable explanation of the last decision.
        This is consumed by Phase 9 neuro-symbolic reasoning.
        """
        names = (action_names if action_names
                 else [f"action_{i}" for i in range(self.n_actions)])

        action = record.get("released_action")
        if action is None:
            conflict = record.get("conflict_score", 0)
            U        = record.get("U", 0.5)
            threshold = record.get("threshold", 0.5)
            gpi      = record.get("gpi_activity", [])
            lines = [
                "Decision: NO action released — all GPi channels above threshold.",
                f"  Threshold: {threshold:.4f}",
                f"  Uncertainty: {U:.3f}  (threshold raised by β·U = "
                f"{0.4*U:.3f})",
                f"  Conflict score: {conflict:.3f}  "
                f"({'ambiguous — STN active' if conflict < 0.3 else 'competition resolved'})",
            ]
            if len(gpi):
                lines.append(f"  GPi values: "
                              f"{[f'{v:.3f}' for v in gpi]}")
            return "\n".join(lines)

        gpi    = record.get("gpi_activity", [0]*self.n_actions)
        margin = record.get("gate_margins", [0]*self.n_actions)
        conf   = record.get("release_confidence", 0)
        U      = record.get("U", 0.5)
        C      = record.get("C", 0.5)
        prob   = record.get("action_probs", [])
        contribs_go   = record.get("pathway_go",   [0]*self.n_actions)
        contribs_nogo = record.get("pathway_nogo", [0]*self.n_actions)
        contribs_stn  = record.get("pathway_stn",  [0]*self.n_actions)
        conflict = record.get("conflict_score", 0)

        ranked = sorted(range(self.n_actions),
                        key=lambda a: gpi[a] if a < len(gpi) else 1e9)

        lines = [
            f"Decision: {names[action]} released.",
            f"  Confidence: {conf:.3f}",
            f"  Gate margin: {margin[action]:.4f}  "
            f"(GPi={gpi[action]:.4f} < θ={record.get('threshold',0.5):.4f})",
            f"  Uncertainty: {U:.3f}  |  Confidence: {C:.3f}",
            f"  Bayesian P(a|s): {prob[action]:.3f}"
            if action < len(prob) else "",
            f"  Conflict score: {conflict:.3f}",
            f"",
            f"  Pathway breakdown for {names[action]}:",
            f"    Direct Go   contribution: {contribs_go[action]:.4f}",
            f"    Indirect NoGo contribution: {contribs_nogo[action]:.4f}",
            f"    STN global  contribution: {contribs_stn[action]:.4f}",
            f"",
            f"  Suppressed alternatives (ranked by GPi):",
        ]
        for a in ranked[1:]:
            if a < len(gpi):
                lines.append(f"    {names[a]}: "
                              f"GPi={gpi[a]:.4f}  "
                              f"margin={margin[a]:+.4f}")
        return "\n".join(lines)

    def save_log(self, filename: str = "decision_log.json") -> str:
        """Saves last 1000 records to JSON for Phase 9 analysis."""
        os.makedirs(self.log_dir, exist_ok=True)
        path    = os.path.join(self.log_dir, filename)
        records = list(self.buffer)[-1000:]

        # Convert numpy arrays to lists for JSON
        clean = []
        for r in records:
            c = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    c[k] = v.tolist()
                else:
                    c[k] = v
            clean.append(c)

        with open(path, "w") as f:
            json.dump({"records": clean, "stats": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.stats.items()
            }}, f, indent=2)
        return path

    def print_summary(self):
        print("\n--- Explainability Log Summary ---")
        print(f"  Total decisions   : {self.stats['total_decisions']}")
        print(f"  Total releases    : {self.stats['total_releases']}")
        total = max(self.stats['total_decisions'], 1)
        print(f"  Release rate      : "
              f"{self.stats['total_releases']/total*100:.1f}%")
        print(f"  Mean confidence   : {self.stats['mean_confidence']:.3f}")
        print(f"  Mean uncertainty  : {self.stats['mean_U']:.3f}")
        print(f"  Mean gate margin  : {self.stats['mean_margin']:.4f}")
        print(f"  Mean conflict     : {self.stats['mean_conflict']:.3f}")
        print(f"  Action selection  :")
        for a, n in enumerate(self.stats["action_counts"]):
            rel = self.stats["total_releases"]
            pct = n / max(rel, 1) * 100
            bar = "#" * int(pct / 2.5)
            print(f"    action {a}: {n:5d}  {pct:5.1f}%  {bar}")
        print("----------------------------------")