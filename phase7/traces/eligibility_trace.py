"""
EligibilityTrace  —  Step 20
------------------------------
For each synapse (i, j), maintains:
    e_dot_ij = -e_ij / tau_e + STDP(pre, post)

This is the core biological mechanism:
  - STDP updates inject evidence of correlated pre-post firing
  - The trace decays exponentially until dopamine arrives
  - When dopamine (delta) arrives, the trace is read out
    and used to update the synaptic weight

The trace is LOCAL — it only depends on:
  - pre-synaptic activity (which came before?)
  - post-synaptic activity (which fired as a consequence?)
  - local dopamine concentration (was the outcome good?)

No global error signal, no backpropagation required.
"""

import numpy as np


class EligibilityTrace:

    def __init__(self,
                 n_pre  : int,
                 n_post : int,
                 tau_e  : float = 100e-3,
                 dt     : float = 0.1e-3,
                 name   : str   = "e"):

        self.n_pre  = n_pre
        self.n_post = n_post
        self.tau_e  = tau_e
        self.dt     = dt
        self.name   = name

        self.e      = np.zeros((n_pre, n_post))
        self.decay  = np.exp(-dt / tau_e)

        self.update_count = 0
        self.total_stdp   = 0.0

    def reset(self) -> None:
        self.e[:]         = 0.0
        self.update_count = 0
        self.total_stdp   = 0.0

    def step(self, stdp_update: np.ndarray) -> np.ndarray:
        """
        e_ij <- decay * e_ij + STDP(pre, post)
        """
        dW = np.asarray(stdp_update, dtype=float)

        # Shape safety
        if dW.shape != self.e.shape:
            padded = np.zeros(self.e.shape)
            r = min(dW.shape[0], self.n_pre)
            c = min(dW.shape[1], self.n_post)
            padded[:r, :c] = dW[:r, :c]
            dW = padded

        # Guard against NaN from extreme STDP values
        dW = np.nan_to_num(dW, nan=0.0, posinf=0.0, neginf=0.0)

        self.e = self.decay * self.e + dW

        # Soft clip to prevent runaway trace
        self.e = np.clip(self.e, -5.0, 5.0)

        self.update_count += 1
        self.total_stdp   += float(np.abs(dW).sum())

        return self.e.copy()

    def read_out(self) -> np.ndarray:
        return self.e.copy()

    def mean_magnitude(self) -> float:
        return float(np.abs(self.e).mean())

    def max_synapse(self) -> tuple:
        idx = np.unravel_index(
            np.abs(self.e).argmax(), self.e.shape)
        return (int(idx[0]), int(idx[1]))

    def trace_summary(self) -> dict:
        return {
            "name"          : self.name,
            "tau_e_ms"      : self.tau_e * 1000,
            "mean_magnitude": self.mean_magnitude(),
            "max_value"     : float(np.abs(self.e).max()),
            "update_count"  : self.update_count,
            "sparsity"      : float((self.e == 0).mean()),
        }