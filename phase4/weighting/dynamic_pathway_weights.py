"""
DynamicPathwayWeights  —  Step 12
-----------------------------------
wGo(t), wNoGo(t), wSTN(t) = f(Ut)

Replaces fixed pathway gains with uncertainty-driven weights.

Behaviour:
  High uncertainty  → stronger STN and No-Go influence
                      (system pauses, explores, avoids commitment)
  Low uncertainty   → stronger Go influence
                      (system exploits, commits, executes)
  Intermediate Ut   → balanced competition between all pathways

Advanced innovation:
  The weight function is differentiable and smooth, allowing
  gradual transitions rather than binary switches.
  Three regimes are implemented:
    Explore regime  (Ut > 0.65): STN/No-Go dominant
    Balanced regime (0.35 < Ut ≤ 0.65): equal weighting
    Exploit regime  (Ut ≤ 0.35): Go dominant
"""

import numpy as np


class DynamicPathwayWeights:

    def __init__(self,
                 w_go_min   : float = 0.2,
                 w_go_max   : float = 1.2,
                 w_nogo_min : float = 0.2,
                 w_nogo_max : float = 1.2,
                 w_stn_min  : float = 0.1,
                 w_stn_max  : float = 1.5,
                 smooth_tau : float = 0.95):
        """
        w_*_min / w_*_max : weight bounds for each pathway
        smooth_tau        : EMA smoothing factor for weight transitions
                            (prevents abrupt switches)
        """
        self.w_go_min   = w_go_min
        self.w_go_max   = w_go_max
        self.w_nogo_min = w_nogo_min
        self.w_nogo_max = w_nogo_max
        self.w_stn_min  = w_stn_min
        self.w_stn_max  = w_stn_max
        self.smooth_tau = smooth_tau

        # Current smoothed weights
        self.w_go   = (w_go_min  + w_go_max)  / 2.0
        self.w_nogo = (w_nogo_min + w_nogo_max) / 2.0
        self.w_stn  = (w_stn_min + w_stn_max)  / 2.0

        # History
        self.weight_history = []

    def reset(self):
        self.w_go   = (self.w_go_min   + self.w_go_max)   / 2.0
        self.w_nogo = (self.w_nogo_min + self.w_nogo_max) / 2.0
        self.w_stn  = (self.w_stn_min  + self.w_stn_max)  / 2.0
        self.weight_history = []

    def update(self, U: float,
               C: float = None) -> dict:
        """
        Core Step 12 computation:
            wGo(t)   = f_go(Ut)
            wNoGo(t) = f_nogo(Ut)
            wSTN(t)  = f_stn(Ut)

        U: uncertainty Ut ∈ [0,1]  from Phase 3
        C: confidence Ct ∈ [0,1]   (optional, = 1 - U if not given)

        Returns dict with current weights and regime label.
        """
        U = float(np.clip(U, 0.0, 1.0))
        C = float(np.clip(1.0 - U if C is None else C, 0.0, 1.0))

        # Target weights using smooth sigmoid-based mapping
        # wGo   decreases with uncertainty (Go weakens under doubt)
        # wNoGo increases with uncertainty (No-Go strengthens under doubt)
        # wSTN  peaks at intermediate uncertainty (conflict brake)

        t_go   = self._map_go(U)
        t_nogo = self._map_nogo(U)
        t_stn  = self._map_stn(U)

        # EMA smoothing — prevents abrupt weight jumps
        α = 1.0 - self.smooth_tau
        self.w_go   = self.smooth_tau * self.w_go   + α * t_go
        self.w_nogo = self.smooth_tau * self.w_nogo + α * t_nogo
        self.w_stn  = self.smooth_tau * self.w_stn  + α * t_stn

        regime = self._classify_regime(U)

        result = {
            "w_go"   : float(self.w_go),
            "w_nogo" : float(self.w_nogo),
            "w_stn"  : float(self.w_stn),
            "U"      : U,
            "C"      : C,
            "regime" : regime,
            "target_go"   : t_go,
            "target_nogo" : t_nogo,
            "target_stn"  : t_stn,
        }
        self.weight_history.append(result.copy())
        return result

    def _map_go(self, U: float) -> float:
        """wGo = w_go_max * (1 - U) + w_go_min * U"""
        return self.w_go_max * (1.0 - U) + self.w_go_min * U

    def _map_nogo(self, U: float) -> float:
        """wNoGo = w_nogo_min * (1-U) + w_nogo_max * U"""
        return self.w_nogo_min * (1.0 - U) + self.w_nogo_max * U

    def _map_stn(self, U: float) -> float:
        """
        wSTN peaks at intermediate uncertainty:
        Uses an inverted-U shaped function —
        maximum at U=0.5, lower at both extremes.
        """
        # Bell curve centred at U=0.5, width σ=0.25
        sigma  = 0.25
        peak   = self.w_stn_max
        base   = self.w_stn_min
        bell   = np.exp(-0.5 * ((U - 0.5) / sigma) ** 2)
        return base + (peak - base) * bell

    def _classify_regime(self, U: float) -> str:
        if U > 0.65:
            return "explore"      # STN/No-Go dominant
        elif U < 0.35:
            return "exploit"      # Go dominant
        else:
            return "balanced"     # competition

    def get_weights(self) -> tuple:
        """Returns (w_go, w_nogo, w_stn) as a tuple."""
        return (float(self.w_go),
                float(self.w_nogo),
                float(self.w_stn))

    def weight_summary(self) -> dict:
        if not self.weight_history:
            return {}
        regimes = [h["regime"] for h in self.weight_history]
        return {
            "current"         : self.get_weights(),
            "mean_w_go"       : float(np.mean([h["w_go"]   for h in self.weight_history])),
            "mean_w_nogo"     : float(np.mean([h["w_nogo"] for h in self.weight_history])),
            "mean_w_stn"      : float(np.mean([h["w_stn"]  for h in self.weight_history])),
            "explore_fraction": regimes.count("explore")  / len(regimes),
            "exploit_fraction": regimes.count("exploit")  / len(regimes),
            "balanced_fraction":regimes.count("balanced") / len(regimes),
        }