"""
STDPKernel  —  Step 20
-----------------------
Spike-Timing Dependent Plasticity kernel.

For each pre-post spike pair, computes:
    STDP(pre, post) = A+ * exp(-|dt| / tau+)  if post fires after pre
                    = A- * exp(-|dt| / tau-)  if pre fires after post

The eligibility trace stores this value until dopamine arrives:
    e_dot_ij = -e_ij / tau_e + STDP(pre, post)

Three trace variants supported:
  - standard  : classic exponential STDP window
  - symmetric : LTP for any coincident activity
  - asymmetric: strict causal (pre before post = LTP only)

Biological basis:
  STDP was discovered in cortical and hippocampal synapses.
  The causal window (pre before post = LTP) is dominant in
  corticostriatal synapses — exactly the pathway that needs
  to learn reward-contingent actions.
"""

import numpy as np


class STDPKernel:

    def __init__(self,
                 A_plus     : float = 0.01,
                 A_minus    : float = 0.0105,
                 tau_plus   : float = 20e-3,
                 tau_minus  : float = 20e-3,
                 mode       : str   = "asymmetric",
                 dt         : float = 0.1e-3):
        """
        A_plus    : LTP amplitude (pre then post)
        A_minus   : LTD amplitude (post then pre)
        tau_plus  : LTP time window (s)
        tau_minus : LTD time window (s)
        mode      : "asymmetric" | "symmetric" | "standard"
        dt        : simulation timestep
        """
        self.A_plus    = A_plus
        self.A_minus   = A_minus
        self.tau_plus  = tau_plus
        self.tau_minus = tau_minus
        self.mode      = mode
        self.dt        = dt

        # Pre and post spike traces for STDP computation
        # x_pre[i]  : decaying trace of pre-synaptic neuron i
        # x_post[j] : decaying trace of post-synaptic neuron j
        self.x_pre  = None    # initialised on first call
        self.x_post = None

    def initialise(self, n_pre: int, n_post: int) -> None:
        """Allocate pre/post spike traces."""
        self.x_pre  = np.zeros(n_pre)
        self.x_post = np.zeros(n_post)

    def update_traces(self, pre_spikes: np.ndarray,
                       post_spikes: np.ndarray) -> None:
        """
        Decay existing traces and inject new spikes.
            x_pre  <- x_pre  * decay_pre  + pre_spikes
            x_post <- x_post * decay_post + post_spikes
        """
        if self.x_pre is None:
            self.initialise(len(pre_spikes), len(post_spikes))

        decay_pre  = np.exp(-self.dt / self.tau_plus)
        decay_post = np.exp(-self.dt / self.tau_minus)

        self.x_pre  = (decay_pre  * self.x_pre
                       + np.asarray(pre_spikes,  dtype=float))
        self.x_post = (decay_post * self.x_post
                       + np.asarray(post_spikes, dtype=float))

    def compute_stdp(self,
                      pre_spikes : np.ndarray,
                      post_spikes: np.ndarray) -> np.ndarray:
        """
        Computes the STDP update matrix of shape (N_pre, N_post).

        For each (i, j) pair:
          LTP : post fires → reward pre-to-post causality
                dW[i,j] += A+ * x_pre[i]  (if post_j fired)
          LTD : pre fires  → penalise post-before-pre
                dW[i,j] -= A- * x_post[j] (if pre_i fired)

        Returns STDP matrix (N_pre, N_post).
        """
        pre  = np.asarray(pre_spikes,  dtype=float)
        post = np.asarray(post_spikes, dtype=float)

        if self.x_pre is None:
            self.initialise(len(pre), len(post))

        # LTP: post spike * pre trace  -> (N_pre, N_post)
        # When post fires, LTP proportional to recent pre activity
        LTP = self.A_plus  * np.outer(self.x_pre,  post)

        # LTD: pre spike * post trace -> (N_pre, N_post)
        # When pre fires, LTD proportional to recent post activity
        LTD = self.A_minus * np.outer(pre, self.x_post)

        if self.mode == "asymmetric":
            # Only LTP (causal: pre before post)
            dW = LTP - LTD
        elif self.mode == "symmetric":
            # Both LTP and LTD, symmetric windows
            dW = LTP + LTD
        else:
            # Standard: full STDP
            dW = LTP - LTD

        return dW

    def reset(self) -> None:
        if self.x_pre  is not None: self.x_pre[:]  = 0.0
        if self.x_post is not None: self.x_post[:] = 0.0

    def stdp_summary(self) -> dict:
        return {
            "mode"      : self.mode,
            "A_plus"    : self.A_plus,
            "A_minus"   : self.A_minus,
            "tau_plus_ms" : self.tau_plus  * 1000,
            "tau_minus_ms": self.tau_minus * 1000,
            "x_pre_mean": float(self.x_pre.mean())
                          if self.x_pre  is not None else 0.0,
            "x_post_mean": float(self.x_post.mean())
                           if self.x_post is not None else 0.0,
        }