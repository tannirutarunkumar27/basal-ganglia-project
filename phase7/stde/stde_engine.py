"""
STDEEngine  —  Step 22
-----------------------
Spike-Timing Dependent Eligibility (STDE) learning rule:

    dW_ij = alpha_t * delta_total * sum_m e_ij(m)

where:
    alpha_t     = meta learning rate (from Phase 8 / uncertainty)
    delta_total = combined TD error (from Phase 6 multi-critic)
    e_ij(m)     = multi-timescale eligibility trace (Step 21)

Core innovation: STDE delayed reward learning.

Key properties:
  1. LOCAL: only uses pre/post spike times and local dopamine
  2. DELAYED: credit assigned long after the action
  3. BIOLOGICALLY PLAUSIBLE: matches three-factor plasticity
  4. NO BACKPROPAGATION: pure spike-based local update

Three-factor plasticity rule:
    Factor 1: Pre-synaptic activity (spike timing)
    Factor 2: Post-synaptic activity (spike timing)
    Factor 3: Neuromodulator (dopamine = delta_total)

Applied to these plastic synaptic connections:
    cortex  → D1 MSNs  (direct pathway, Go learning)
    cortex  → D2 MSNs  (indirect pathway, No-Go learning)
    D1 MSNs → GPi      (direct pathway output)
    D2 MSNs → GPe      (indirect pathway output)
"""

import numpy as np
from collections import deque


class STDEEngine:

    def __init__(self,
                 n_pre          : int,
                 n_post         : int,
                 sign           : int   = +1,
                 w_init_mean    : float = 0.5,
                 w_init_std     : float = 0.1,
                 w_min          : float = 0.0,
                 w_max          : float = 2.0,
                 conn_prob      : float = 0.4,
                 enforce_dale   : bool  = True,
                 dt             : float = 0.1e-3,
                 name           : str   = "synapse"):
        """
        n_pre          : pre-synaptic population size
        n_post         : post-synaptic population size
        sign           : +1 excitatory, -1 inhibitory
        w_init_mean    : initial weight mean (normalised)
        w_init_std     : initial weight std
        w_min / w_max  : weight bounds
        conn_prob      : connection probability (sparsity)
        enforce_dale   : maintain weight signs (Dale's law)
        dt             : simulation timestep
        name           : synapse identifier
        """
        self.n_pre        = n_pre
        self.n_post       = n_post
        self.sign         = sign
        self.w_min        = w_min
        self.w_max        = w_max
        self.enforce_dale = enforce_dale
        self.dt           = dt
        self.name         = name

        # Sparse random weight matrix (N_pre, N_post)
        mask        = np.random.rand(n_pre, n_post) < conn_prob
        W_abs       = np.abs(np.random.normal(
            w_init_mean, w_init_std, (n_pre, n_post)))
        self.W      = W_abs * mask * sign

        # Sign constraint mask — weights never change sign
        self.sign_mask = np.sign(self.W + 1e-30)

        # Cumulative weight change
        self.dW_total   = np.zeros((n_pre, n_post))

        # Update statistics
        self.update_count   = 0
        self.dW_mag_history = deque(maxlen=2000)
        self.W_mean_history = deque(maxlen=2000)

    def reset(self) -> None:
        self.dW_total     = np.zeros((self.n_pre, self.n_post))
        self.update_count = 0

    def update(self,
               e_total    : np.ndarray,
               delta_total: float,
               alpha_t    : float,
               mask       : np.ndarray = None) -> np.ndarray:
        """
        Step 22 core:
            dW_ij = alpha_t * delta_total * sum_m e_ij(m)

        e_total    : combined eligibility trace (N_pre, N_post)
        delta_total: combined TD error from Phase 6
        alpha_t    : meta learning rate (uncertainty-adapted)
        mask       : optional boolean mask for selective updates

        Returns dW (N_pre, N_post) — weight change matrix.
        """
        e = np.asarray(e_total, dtype=float)

        if e.shape != self.W.shape:
            padded = np.zeros(self.W.shape)
            r = min(e.shape[0], self.n_pre)
            c = min(e.shape[1], self.n_post)
            padded[:r, :c] = e[:r, :c]
            e = padded

        # Core STDE update
        dW = float(alpha_t) * float(delta_total) * e

        # Apply optional mask (only update certain synapses)
        if mask is not None:
            dW = dW * np.asarray(mask, dtype=float)

        # Apply weight update
        self.W         += dW
        self.dW_total  += dW
        self.update_count += 1

        # Enforce Dale's law — weights never change sign
        if self.enforce_dale:
            self.W = self.sign_mask * np.abs(self.W)

        # Clip to weight bounds
        abs_max = self.w_max
        self.W  = np.clip(self.W, -abs_max, abs_max)

        # Normalise: prevent runaway weights
        # Soft homeostatic scaling
        w_mean = np.abs(self.W[self.W != 0]).mean() \
            if (self.W != 0).any() else 0.0
        if w_mean > 1.5:
            scale  = 1.5 / w_mean
            self.W = self.W * scale

        # Record statistics
        dW_mag = float(np.abs(dW).mean())
        self.dW_mag_history.append(dW_mag)
        self.W_mean_history.append(
            float(np.abs(self.W[self.W != 0]).mean())
            if (self.W != 0).any() else 0.0)

        return dW

    def effective_weights(self) -> np.ndarray:
        """Returns |W| for analysis (sign is in sign_mask)."""
        return np.abs(self.W)

    def sparsity(self) -> float:
        return float((self.W == 0).mean())

    def weight_stats(self) -> dict:
        nz = self.W[self.W != 0]
        return {
            "mean"    : float(nz.mean())        if len(nz) else 0.0,
            "std"     : float(nz.std())         if len(nz) else 0.0,
            "max_abs" : float(np.abs(nz).max()) if len(nz) else 0.0,
            "sparsity": self.sparsity(),
        }

    def stde_summary(self) -> dict:
        dW_hist = list(self.dW_mag_history)
        W_hist  = list(self.W_mean_history)
        return {
            "name"          : self.name,
            "update_count"  : self.update_count,
            "mean_dW"       : float(np.mean(dW_hist)) if dW_hist else 0.0,
            "mean_W"        : float(np.mean(W_hist))  if W_hist  else 0.0,
            "weight_stats"  : self.weight_stats(),
        }