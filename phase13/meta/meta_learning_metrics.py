"""
MetaLearningMetrics  —  Step 33
---------------------------------
Three metrics for the meta-dopamine and adaptation quality.

1. U_alpha_correlation   : Spearman r(Ut, alpha_t) — should be ~1
2. volatility_adaptability: accuracy delta between stable and
                            volatile reward periods
3. reversal_robustness   : accuracy retained after a reversal
"""

import numpy as np
from scipy.stats import spearmanr


class MetaLearningMetrics:

    def __init__(self):
        self._U_vals    = []
        self._alpha_vals = []
        self._rewards   = []
        self._correct   = []
        self._reversal_steps = []

    def reset(self) -> None:
        self._U_vals        = []
        self._alpha_vals    = []
        self._rewards       = []
        self._correct       = []
        self._reversal_steps = []

    def record_step(self, U       : float,
                     alpha_t     : float,
                     reward      : float,
                     correct     : int,
                     is_reversal : bool = False,
                     step        : int  = 0) -> None:
        self._U_vals.append(float(U))
        self._alpha_vals.append(float(alpha_t))
        self._rewards.append(float(reward))
        self._correct.append(int(correct))
        if is_reversal:
            self._reversal_steps.append(int(step))

    def U_alpha_correlation(self) -> float:
        """
        Spearman correlation between Ut and alpha_t.
        Should be close to 1.0 (meta-dopamine working).
        """
        if len(self._U_vals) < 10:
            return 0.0
        corr, _ = spearmanr(self._U_vals, self._alpha_vals)
        return float(np.clip(corr, -1.0, 1.0))

    def volatility_adaptability(self,
                                  half_split: bool = True) -> float:
        """
        Accuracy in the second half minus the first half.
        Positive = system improved accuracy during volatile period.
        Normalised to [0, 1].
        """
        arr = np.array(self._correct, dtype=float)
        if len(arr) < 20:
            return 0.0
        mid   = len(arr) // 2
        early = float(arr[:mid].mean())
        late  = float(arr[mid:].mean())
        delta = late - early
        return float(np.clip((delta + 1.0) / 2.0, 0.0, 1.0))

    def reversal_robustness(self,
                             recovery_window: int = 100) -> float:
        """
        Mean accuracy in the `recovery_window` steps after
        each recorded reversal. Higher = faster recovery.
        """
        if not self._reversal_steps:
            return float(np.mean(self._correct)) \
                   if self._correct else 0.0
        arr      = np.array(self._correct, dtype=float)
        post_accs = []
        for s in self._reversal_steps:
            end   = min(s + recovery_window, len(arr))
            if end > s:
                post_accs.append(float(arr[s:end].mean()))
        return float(np.mean(post_accs)) if post_accs else 0.0

    def compute_all(self) -> dict:
        return {
            "U_alpha_correlation"    : self.U_alpha_correlation(),
            "volatility_adaptability": self.volatility_adaptability(),
            "reversal_robustness"    : self.reversal_robustness(),
        }