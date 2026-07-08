"""
MultiTimescaleTraces  —  Step 21
----------------------------------
Uses M eligibility traces with different decay constants:

    e_dot_ij(m) = -e_ij(m) / tau_m + STDP(pre, post)

Three timescales:
    m=0  fast  tau = 20 ms    short-term: immediate context
    m=1  mid   tau = 200 ms   medium-term: working memory
    m=2  slow  tau = 2000 ms  long-term: episodic retention

This supports:
  - Short-term learning  : fast trace captures recent spikes
  - Medium-term retention: mid trace bridges reward delays
  - Long-term credit     : slow trace handles delayed outcomes
  - Hierarchical learning: each timescale provides different signal

Advanced innovation:
  Unlike single-tau models, this allows the agent to assign
  credit for outcomes that occurred up to seconds earlier —
  consistent with biological dopamine neuromodulation profiles
  and the known temporal extent of corticostriatal plasticity.

Combined trace (used in Step 22):
    e_total_ij = sum_m w_m * e_ij(m)

where w_m are combination weights that can be fixed or
learned based on which timescale predicted reward best.
"""

import numpy as np
from traces.eligibility_trace import EligibilityTrace


class MultiTimescaleTraces:

    # Default timescale configuration
    TIMESCALES = [
        {"name": "fast", "tau_e": 20e-3,   "weight": 0.5},
        {"name": "mid",  "tau_e": 200e-3,  "weight": 0.3},
        {"name": "slow", "tau_e": 2000e-3, "weight": 0.2},
    ]

    def __init__(self,
                 n_pre      : int,
                 n_post     : int,
                 timescales : list  = None,
                 dt         : float = 0.1e-3,
                 adaptive_weights: bool = True):
        """
        n_pre           : pre-synaptic population size
        n_post          : post-synaptic population size
        timescales      : list of {name, tau_e, weight} dicts
                          (defaults to fast/mid/slow)
        dt              : simulation timestep
        adaptive_weights: if True, adjust timescale weights
                          based on reward prediction accuracy
        """
        self.n_pre   = n_pre
        self.n_post  = n_post
        self.dt      = dt
        self.adaptive_weights = adaptive_weights

        cfg = timescales if timescales else self.TIMESCALES
        self.M = len(cfg)

        # Build one EligibilityTrace per timescale
        self.traces  = []
        self.names   = []
        self.weights = np.array([c["weight"] for c in cfg],
                                 dtype=float)
        self.weights /= self.weights.sum()   # normalise

        for c in cfg:
            t = EligibilityTrace(
                n_pre  = n_pre,
                n_post = n_post,
                tau_e  = c["tau_e"],
                dt     = dt,
                name   = c["name"])
            self.traces.append(t)
            self.names.append(c["name"])

        # Combined total trace (N_pre, N_post)
        self.e_total  = np.zeros((n_pre, n_post))

        # Reward prediction accuracy per timescale (for weight adapt)
        self._pred_acc = np.ones(self.M) / self.M
        self._step_count = 0

        # History
        self.mag_history    = {n: [] for n in self.names}
        self.total_history  = []
        self.weight_history = []

    def reset(self) -> None:
        for t in self.traces:
            t.reset()
        self.e_total[:] = 0.0
        self._pred_acc  = np.ones(self.M) / self.M
        self._step_count = 0
        for name in self.names:
            self.mag_history[name].clear()
        self.total_history.clear()
        self.weight_history.clear()

    def step(self, stdp_update: np.ndarray) -> np.ndarray:
        """
        Step 21 core:
            e_ij(m) <- decay_m * e_ij(m) + STDP(pre, post)
            e_total  = sum_m w_m * e_ij(m)

        stdp_update: (N_pre, N_post) from STDPKernel

        Returns e_total (N_pre, N_post) — the combined trace
        that will be used in the STDE weight update.
        """
        self._step_count += 1

        # Update each timescale trace
        individual_traces = []
        for trace in self.traces:
            e_m = trace.step(stdp_update)
            individual_traces.append(e_m)
            self.mag_history[trace.name].append(
                trace.mean_magnitude())

        # Weighted combination
        self.e_total = sum(
            self.weights[m] * individual_traces[m]
            for m in range(self.M)
        )

        self.total_history.append(
            float(np.abs(self.e_total).mean()))
        self.weight_history.append(self.weights.copy())

        return self.e_total.copy()

    def update_weights(self, delta: float,
                        reward: float) -> np.ndarray:
        """
        Adaptive timescale weights:
        Traces that better predicted the current reward receive
        higher weight in the combined trace next time.

        delta  : dopamine prediction error (from Phase 6)
        reward : actual reward received

        This implements a meta-learning loop over timescales.
        """
        if not self.adaptive_weights:
            return self.weights.copy()

        # Each trace's "prediction quality" = how much its
        # trace magnitude correlates with the reward sign
        for m, trace in enumerate(self.traces):
            trace_mag = trace.mean_magnitude()
            # Proxy: if trace is large when reward is positive,
            # it predicted well
            signal    = trace_mag * np.sign(reward)
            accuracy  = float(np.clip(signal / (abs(delta) + 1e-8),
                                       0.0, 1.0))
            self._pred_acc[m] = (0.95 * self._pred_acc[m]
                                  + 0.05 * accuracy)

        # Renormalise weights toward best predictors
        softmax_acc  = np.exp(self._pred_acc * 5.0)
        self.weights = softmax_acc / softmax_acc.sum()

        return self.weights.copy()

    def get_individual_traces(self) -> dict:
        """Returns {name: trace_matrix} for all timescales."""
        return {t.name: t.e.copy() for t in self.traces}

    def dominant_timescale(self) -> str:
        """Returns the name of the timescale with largest weight."""
        return self.names[int(np.argmax(self.weights))]

    def timescale_summary(self) -> dict:
        return {
            "M"              : self.M,
            "weights"        : self.weights.tolist(),
            "dominant"       : self.dominant_timescale(),
            "e_total_mean"   : float(np.abs(self.e_total).mean()),
            "e_total_max"    : float(np.abs(self.e_total).max()),
            "per_trace"      : [t.trace_summary()
                                 for t in self.traces],
            "step_count"     : self._step_count,
        }