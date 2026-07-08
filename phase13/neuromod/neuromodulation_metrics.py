"""
NeuromodulationMetrics  —  Step 33
------------------------------------
Four metrics for the multi-neuromodulator system.

1. DA_5HT_NE_interaction : mutual information proxy between the three
2. risk_sensitivity       : correlation of 5-HT with correct avoidance
3. arousal_control        : NE follows uncertainty (Spearman r)
4. confidence_modulation  : DA tracks explanation confidence
"""

import numpy as np
from scipy.stats import spearmanr


class NeuromodulationMetrics:

    def __init__(self):
        self._DA  = []
        self._ht5 = []
        self._NE  = []
        self._U   = []
        self._rho = []
        self._expl_conf = []
        self._risky_correct = []   # 1 when high-risk action was avoided

    def reset(self) -> None:
        self._DA  = []
        self._ht5 = []
        self._NE  = []
        self._U   = []
        self._rho = []
        self._expl_conf = []
        self._risky_correct = []

    def record_step(self,
                     DA         : float,
                     ht5        : float,
                     NE         : float,
                     U          : float,
                     rho        : float,
                     expl_conf  : float,
                     risky_avoided: int = 0) -> None:
        self._DA.append(float(DA))
        self._ht5.append(float(ht5))
        self._NE.append(float(NE))
        self._U.append(float(U))
        self._rho.append(float(rho))
        self._expl_conf.append(float(expl_conf))
        self._risky_correct.append(int(risky_avoided))

    def DA_5HT_NE_interaction(self) -> float:
        """
        Proxy for tri-modulator coherence: mean absolute
        pairwise Spearman correlation. Higher = signals
        interact coherently (they modulate each other).
        """
        da  = np.array(self._DA)
        ht5 = np.array(self._ht5)
        ne  = np.array(self._NE)
        if len(da) < 10:
            return 0.0
        r1, _ = spearmanr(da,  ht5)
        r2, _ = spearmanr(da,  ne)
        r3, _ = spearmanr(ht5, ne)
        return float(np.mean(np.abs([r1, r2, r3])))

    def risk_sensitivity(self) -> float:
        """
        Correlation between 5-HT level and successful
        avoidance of high-risk choices.
        """
        if len(self._ht5) < 10:
            return 0.0
        ht5  = np.array(self._ht5)
        risky = np.array(self._risky_correct, dtype=float)
        corr, _ = spearmanr(ht5, risky)
        return float(np.clip(corr, -1.0, 1.0))

    def arousal_control(self) -> float:
        """
        NE should track uncertainty (arousal during uncertain states).
        Returns Spearman r(NE, Ut) normalised to [0, 1].
        """
        if len(self._NE) < 10:
            return 0.0
        corr, _ = spearmanr(self._NE, self._U)
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def confidence_modulation(self) -> float:
        """
        DA should track explanation confidence (high reward →
        high DA → high confidence). Spearman r(DA, expl_conf).
        """
        if len(self._DA) < 10:
            return 0.0
        corr, _ = spearmanr(self._DA, self._expl_conf)
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def compute_all(self) -> dict:
        return {
            "DA_5HT_NE_interaction": self.DA_5HT_NE_interaction(),
            "risk_sensitivity"      : self.risk_sensitivity(),
            "arousal_control"       : self.arousal_control(),
            "confidence_modulation" : self.confidence_modulation(),
        }