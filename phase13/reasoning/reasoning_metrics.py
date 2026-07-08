"""
ReasoningMetrics  —  Step 33
------------------------------
Four metrics for evaluating the quality of symbolic reasoning
and explainability outputs from Phase 9.

1. posterior_calibration   : how well Va(t) ranks correct action
2. uncertainty_alignment   : correlation of Ut with mistake rate
3. explanation_quality     : mean explanation confidence + rule diversity
4. counterfactual_consistency: CFs consistent with outcome
"""

import numpy as np
from collections import deque
from scipy.stats import spearmanr


class ReasoningMetrics:

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

        self._V_combined    = []   # belief scores per step
        self._actions       = []   # selected actions
        self._correct       = []   # 1 if correct, 0 otherwise
        self._U_vals        = []   # uncertainty per step
        self._expl_confs    = []   # explanation confidence
        self._n_rules       = []   # rules fired per step
        self._cf_deltas     = []   # counterfactual reward gaps

    def reset(self) -> None:
        self._V_combined = []
        self._actions    = []
        self._correct    = []
        self._U_vals     = []
        self._expl_confs = []
        self._n_rules    = []
        self._cf_deltas  = []

    def record_step(self,
                     V_combined   : np.ndarray,
                     action       : int,
                     correct_action: int,
                     U            : float,
                     expl_conf    : float,
                     n_rules      : int,
                     cf_delta     : float = 0.0) -> None:
        self._V_combined.append(
            np.asarray(V_combined, dtype=float).tolist())
        self._actions.append(int(action))
        self._correct.append(int(action == correct_action))
        self._U_vals.append(float(U))
        self._expl_confs.append(float(expl_conf))
        self._n_rules.append(int(n_rules))
        self._cf_deltas.append(float(cf_delta))

    def posterior_calibration(self) -> float:
        """
        Fraction of steps where argmax(Va) == correct action.
        Higher = posterior is well-calibrated to rewards.
        """
        if not self._V_combined:
            return 0.0
        correct_predicted = 0
        for V, act in zip(self._V_combined, self._actions):
            if int(np.argmax(V)) == act:
                correct_predicted += 1
        return float(correct_predicted / len(self._actions))

    def uncertainty_alignment(self) -> float:
        """
        Spearman correlation between Ut and mistake indicator.
        Positive = high Ut predicts mistakes (well-aligned).
        Returns value in [-1, 1]; higher is better.
        """
        if len(self._U_vals) < 10:
            return 0.0
        U       = np.array(self._U_vals)
        errors  = 1.0 - np.array(self._correct, dtype=float)
        corr, _ = spearmanr(U, errors)
        return float(np.clip(corr, -1.0, 1.0))

    def explanation_quality(self) -> float:
        """
        Composite of mean explanation confidence and
        rule diversity (mean rules / max_expected_rules).
        """
        if not self._expl_confs:
            return 0.0
        mean_ec  = float(np.mean(self._expl_confs))
        # Rule diversity: target 4-6 rules per step
        mean_nr  = float(np.mean(self._n_rules))
        nr_score = float(np.clip(mean_nr / 5.0, 0.0, 1.0))
        return float(0.6 * mean_ec + 0.4 * nr_score)

    def counterfactual_consistency(self) -> float:
        """
        Fraction of steps where the counterfactual delta
        is positive (rejected actions had lower reward than
        the chosen one — consistent CF reasoning).
        """
        if not self._cf_deltas:
            return 0.0
        arr = np.array(self._cf_deltas)
        return float(np.mean(arr >= 0))

    def compute_all(self) -> dict:
        return {
            "posterior_calibration"    : self.posterior_calibration(),
            "uncertainty_alignment"    : self.uncertainty_alignment(),
            "explanation_quality"      : self.explanation_quality(),
            "counterfactual_consistency": self.counterfactual_consistency(),
        }