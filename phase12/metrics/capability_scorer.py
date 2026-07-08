"""
CapabilityScorer  —  Step 32
------------------------------
Maps raw task results to the six capability scores defined
in the methodology:

    1. learn_rewards     : reward rate + convergence speed
    2. resolve_conflict  : stop-signal + STN activation accuracy
    3. adapt_volatility  : reversal learning + volatile task
    4. reason_uncertainty: mean confidence + explanation quality
    5. stable_selection  : action selection stability + low regret
    6. explain_choices   : explanation confidence + rules fired

Each score is normalised to [0, 1]. A score of 1.0 means the
system performed at or above expected biological performance.
"""

import numpy as np
from typing import Dict


class CapabilityScorer:

    def __init__(self, n_actions: int = 4):
        self.n_actions = n_actions

    def score_all(self, results: dict) -> dict:
        """
        Computes capability scores for all tasks.
        Returns dict {capability_name: score} in [0,1].
        """
        scores = {}

        scores["learn_rewards"]      = self._learn_rewards(results)
        scores["resolve_conflict"]   = self._resolve_conflict(results)
        scores["adapt_volatility"]   = self._adapt_volatility(results)
        scores["reason_uncertainty"] = self._reason_uncertainty(results)
        scores["stable_selection"]   = self._stable_selection(results)
        scores["explain_choices"]    = self._explain_choices(results)

        scores["overall"] = float(np.mean(list(scores.values())))
        return scores

    def _get(self, results: dict, name: str,
              field: str, default: float = 0.0) -> float:
        """Safe accessor for result fields."""
        r = results.get(name)
        if r is None:
            return default
        return float(getattr(r, field, default))

    def _learn_rewards(self, r: dict) -> float:
        """
        Based on probabilistic bandit + sequential planning.
        Score = 0.5*accuracy + 0.5*(1 - convergence_speed_norm)
        """
        bandit_acc   = self._get(r, "probabilistic_bandit", "accuracy", 0.5)
        seq_acc      = self._get(r, "sequential_decision",  "accuracy", 0.5)

        # Convergence: lower step = better; normalize by 3000
        bandit_conv  = self._get(r, "probabilistic_bandit", "convergence_step", -1)
        conv_score   = 0.0 if bandit_conv < 0 else \
                       float(1.0 - bandit_conv / 3000.0)

        return float(np.clip(
            0.4*bandit_acc + 0.3*seq_acc + 0.3*conv_score,
            0.0, 1.0))

    def _resolve_conflict(self, r: dict) -> float:
        """
        Based on stop-signal task.
        Measures STN activation rate during stop trials.
        """
        stop_acc = self._get(r, "stop_signal", "accuracy", 0.5)

        # STN activity should be high (frequent activation = good)
        step_log = getattr(r.get("stop_signal"), "step_log", [])
        if step_log:
            stn_rate = float(np.mean([
                s.get("stn_active", False) for s in step_log]))
        else:
            stn_rate = 0.0

        return float(np.clip(
            0.6 * stop_acc + 0.4 * stn_rate,
            0.0, 1.0))

    def _adapt_volatility(self, r: dict) -> float:
        """
        Based on reversal learning + volatile reward.
        Score penalises slow adaptation after reversals.
        """
        rev_acc   = self._get(r, "reversal_learning", "accuracy", 0.5)
        vol_acc   = self._get(r, "volatile_reward",   "accuracy", 0.5)
        rev_rew   = self._get(r, "reversal_learning", "mean_reward", 0.0)

        return float(np.clip(
            0.4 * rev_acc + 0.4 * vol_acc + 0.2 * (rev_rew + 1) / 2,
            0.0, 1.0))

    def _reason_uncertainty(self, r: dict) -> float:
        """
        Based on hidden-rule inference + counterfactual tasks.
        Measures mean_U calibration and explanation confidence.
        """
        hr_acc    = self._get(r, "hidden_rule",   "accuracy",        0.5)
        cf_acc    = self._get(r, "counterfactual","accuracy",        0.5)
        mean_U    = self._get(r, "hidden_rule",   "mean_U",          0.5)
        # Good uncertainty calibration: moderate U (not always 0 or 1)
        U_calib   = float(1.0 - abs(mean_U - 0.4))   # best near 0.4

        return float(np.clip(
            0.35 * hr_acc + 0.35 * cf_acc + 0.3 * U_calib,
            0.0, 1.0))

    def _stable_selection(self, r: dict) -> float:
        """
        Based on grid-world + risk-sensitive choice.
        Measures regret (lower = better) and accuracy.
        """
        gw_acc   = self._get(r, "grid_world",   "accuracy",    0.5)
        rs_acc   = self._get(r, "risk_sensitive","accuracy",    0.5)
        gw_reg   = self._get(r, "grid_world",   "regret",      1000.0)
        # Normalise regret (lower is better, cap at 3000)
        reg_norm = float(1.0 - min(gw_reg, 3000.0) / 3000.0)

        return float(np.clip(
            0.35 * gw_acc + 0.35 * rs_acc + 0.3 * reg_norm,
            0.0, 1.0))

    def _explain_choices(self, r: dict) -> float:
        """
        Aggregated over all tasks.
        Measures explanation confidence and rule diversity.
        """
        all_ec   = [getattr(v, "mean_expl_conf", 0.5)
                    for v in r.values() if v is not None]
        all_nr   = [getattr(v, "mean_n_rules", 0.0)
                    for v in r.values() if v is not None]

        mean_ec  = float(np.mean(all_ec)) if all_ec else 0.5
        # Scale rules: target ~4 rules/step; normalise to [0,1]
        mean_nr  = float(np.mean(all_nr)) if all_nr else 0.0
        nr_score = float(np.clip(mean_nr / 6.0, 0.0, 1.0))

        return float(np.clip(
            0.6 * mean_ec + 0.4 * nr_score,
            0.0, 1.0))

    def print_scores(self, scores: dict) -> None:
        print("\n  Capability scores:")
        bar_max = 30
        for cap, score in scores.items():
            if cap == "overall":
                continue
            bar = "#" * int(score * bar_max)
            print(f"    {cap:<25s}: {score:.3f}  {bar}")
        print(f"    {'overall':<25s}: "
              f"{scores['overall']:.3f}  "
              f"{'#'*int(scores['overall']*bar_max)}")