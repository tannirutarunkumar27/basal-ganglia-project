"""
NeuralMetrics  —  Step 33
---------------------------
Five metrics characterising the spiking neural dynamics.

1. spike_sparsity          : mean fraction of neurons silent
2. adaptation_dynamics     : AHP adaptation current stability
3. dopamine_burst_timing   : latency of DA burst after reward
4. pathway_balance         : Go vs No-Go pathway ratio
5. eligibility_trace_evol  : trace magnitude at reward time
"""

import numpy as np
from collections import deque


class NeuralMetrics:

    def __init__(self, dt: float = 0.1e-3):
        self.dt = dt

        self._spike_fractions  = []   # fraction active per step
        self._da_levels        = []   # DA level per step
        self._reward_steps     = []   # steps where reward > 0
        self._go_signals       = []   # direct pathway Go per step
        self._nogo_signals     = []   # indirect pathway NoGo per step
        self._trace_mags       = []   # e_total magnitude per step
        self._ahp_mags         = []   # AHP current magnitude

    def reset(self) -> None:
        self._spike_fractions = []
        self._da_levels       = []
        self._reward_steps    = []
        self._go_signals      = []
        self._nogo_signals    = []
        self._trace_mags      = []
        self._ahp_mags        = []

    def record_step(self,
                     n_spikes        : int,
                     n_neurons_total : int,
                     DA              : float,
                     reward          : float,
                     direct_inh      : np.ndarray,
                     indirect_exc    : np.ndarray,
                     trace_mag       : float,
                     ahp_mag         : float,
                     step            : int) -> None:

        frac = float(n_spikes / max(n_neurons_total, 1))
        self._spike_fractions.append(frac)
        self._da_levels.append(float(DA))
        if reward > 0.1:
            self._reward_steps.append(int(step))
        self._go_signals.append(
            float(np.asarray(direct_inh).mean()))
        self._nogo_signals.append(
            float(np.asarray(indirect_exc).mean()))
        self._trace_mags.append(float(trace_mag))
        self._ahp_mags.append(float(ahp_mag))

    def spike_sparsity(self) -> float:
        """Mean fraction of neurons silent per step (1 - active)."""
        if not self._spike_fractions:
            return 0.0
        return float(1.0 - np.mean(self._spike_fractions))

    def adaptation_dynamics(self) -> float:
        """
        Stability of AHP magnitudes over time.
        Low coefficient of variation = stable adaptation.
        Returned as 1 - CV, higher = more stable.
        """
        arr = np.array(self._ahp_mags)
        if len(arr) < 2:
            return 0.0
        cv = float(arr.std() / (arr.mean() + 1e-8))
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def dopamine_burst_timing(self) -> float:
        """
        Mean DA level in the 10 steps following each reward.
        Higher = larger phasic DA burst after reward.
        Normalised to [0, 1].
        """
        da  = np.array(self._da_levels)
        if not self._reward_steps:
            return 0.0
        bursts = []
        for s in self._reward_steps:
            window = da[s: min(s + 10, len(da))]
            if len(window) > 0:
                bursts.append(float(window.mean()))
        if not bursts:
            return 0.0
        return float(np.clip(np.mean(bursts), 0.0, 1.0))

    def pathway_balance(self) -> float:
        """
        Ratio of mean Go signal to mean NoGo signal.
        Ideal ~1.5-2.0 (Go slightly dominant for action selection).
        Returns score in [0, 1] based on proximity to ideal.
        """
        go   = float(np.mean(self._go_signals))   if self._go_signals   else 0.0
        nogo = float(np.mean(self._nogo_signals)) if self._nogo_signals else 0.0
        if nogo < 1e-9:
            return 0.5
        ratio = go / (nogo + 1e-8)
        # Ideal ratio ~1.5; score = 1 at ideal, 0 far from ideal
        ideal = 1.5
        score = float(np.exp(-0.5 * ((ratio - ideal) / ideal) ** 2))
        return float(np.clip(score, 0.0, 1.0))

    def eligibility_trace_evolution(self) -> float:
        """
        Mean trace magnitude at reward steps (vs all steps).
        Higher ratio = trace holds useful credit longer.
        """
        tr = np.array(self._trace_mags)
        if len(tr) == 0:
            return 0.0
        mean_all = float(tr.mean()) + 1e-9
        if self._reward_steps:
            reward_tr = np.array(
                [tr[s] for s in self._reward_steps if s < len(tr)])
            if len(reward_tr):
                mean_rew = float(reward_tr.mean())
                return float(np.clip(mean_rew / mean_all, 0.0, 2.0) / 2.0)
        return 0.0

    def compute_all(self) -> dict:
        return {
            "spike_sparsity"          : self.spike_sparsity(),
            "adaptation_dynamics"     : self.adaptation_dynamics(),
            "dopamine_burst_timing"   : self.dopamine_burst_timing(),
            "pathway_balance"         : self.pathway_balance(),
            "eligibility_trace_evol"  : self.eligibility_trace_evolution(),
        }