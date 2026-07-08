"""
Conductance-based synapse with optional STDP eligibility trace.
Supports excitatory (AMPA/NMDA-like) and inhibitory (GABA-like) dynamics.
"""

import numpy as np


class SynapseGroup:
    """
    Connects a pre-synaptic population (N_pre) to a
    post-synaptic population (N_post).
    Weight matrix shape: (N_pre, N_post).
    """

    def __init__(self,
                 name       : str,
                 N_pre      : int,
                 N_post     : int,
                 sign       : int,
                 weight_mean: float = 0.5e-9,
                 weight_std : float = 0.1e-9,
                 conn_prob  : float = 0.3,
                 tau_syn    : float = 5e-3,
                 E_rev      : float = None,
                 plastic    : bool  = False,
                 dt         : float = 0.1e-3):

        self.name    = name
        self.N_pre   = N_pre
        self.N_post  = N_post
        self.sign    = sign
        self.tau_syn = tau_syn
        self.dt      = dt
        self.plastic = plastic

        # Reversal potentials
        if E_rev is None:
            self.E_rev = 0.0 if sign > 0 else -70e-3
        else:
            self.E_rev = E_rev

        # Sparse random weight matrix, biologically signed
        mask = (np.random.rand(N_pre, N_post) < conn_prob).astype(float)
        W    = np.abs(np.random.normal(weight_mean, weight_std,
                                        (N_pre, N_post)))
        self.W         = W * mask * sign
        self.sign_mask = np.sign(self.W + 1e-30)

        # Conductance state (N_post,)
        self.g = np.zeros(N_post)

        # STDP eligibility trace (N_pre, N_post) — only when plastic
        self.e_trace = np.zeros((N_pre, N_post)) if plastic else None

        self.n_events = 0

    def step(self, pre_spikes: np.ndarray,
             post_V: np.ndarray) -> np.ndarray:
        """
        One timestep: decay conductance, add spike contributions,
        return post-synaptic current array of shape (N_post,).
        """
        # Safety: ensure correct shapes
        pre_spikes = np.asarray(pre_spikes, dtype=bool)
        post_V     = np.asarray(post_V,     dtype=float)

        if pre_spikes.shape[0] != self.N_pre:
            # Pad or truncate silently
            tmp = np.zeros(self.N_pre, dtype=bool)
            n   = min(pre_spikes.shape[0], self.N_pre)
            tmp[:n] = pre_spikes[:n]
            pre_spikes = tmp

        if post_V.shape[0] != self.N_post:
            tmp = np.zeros(self.N_post, dtype=float) - 65e-3
            n   = min(post_V.shape[0], self.N_post)
            tmp[:n] = post_V[:n]
            post_V = tmp

        # Exponential conductance decay
        self.g *= np.exp(-self.dt / self.tau_syn)

        # Add weighted spike contributions
        if pre_spikes.any():
            self.g += pre_spikes.astype(float) @ self.W
            self.n_events += int(pre_spikes.sum())

        # I = g * (E_rev - V_post)
        return self.g * (self.E_rev - post_V)

    def update_eligibility(self, pre_spikes: np.ndarray,
                            post_spikes: np.ndarray,
                            tau_e: float = 20e-3) -> None:
        """Update STDP eligibility trace (only when plastic=True)."""
        if not self.plastic or self.e_trace is None:
            return

        pre_spikes  = np.asarray(pre_spikes,  dtype=bool)
        post_spikes = np.asarray(post_spikes, dtype=bool)

        # Pad/truncate to match weight matrix dimensions
        if pre_spikes.shape[0] != self.N_pre:
            tmp = np.zeros(self.N_pre, dtype=bool)
            n   = min(pre_spikes.shape[0], self.N_pre)
            tmp[:n] = pre_spikes[:n]
            pre_spikes = tmp

        if post_spikes.shape[0] != self.N_post:
            tmp = np.zeros(self.N_post, dtype=bool)
            n   = min(post_spikes.shape[0], self.N_post)
            tmp[:n] = post_spikes[:n]
            post_spikes = tmp

        # Decay
        self.e_trace *= np.exp(-self.dt / tau_e)
        # Pre fires -> increment pre rows
        if pre_spikes.any():
            self.e_trace[pre_spikes, :] += 0.01
        # Post fires -> Hebbian LTP
        if post_spikes.any():
            self.e_trace[:, post_spikes] += 0.01

    def apply_reward_modulated_update(self, delta: float,
                                       alpha: float = 1e-9) -> None:
        """Reward-modulated STDP: dW = alpha * delta * e_trace."""
        if not self.plastic or self.e_trace is None:
            return
        dW        = alpha * delta * self.e_trace
        self.W   += dW
        # Re-apply Dale's law: weights never change sign
        self.W    = self.sign_mask * np.abs(self.W)
        # Hard clip
        self.W    = np.clip(self.W, 0, 5e-9 * abs(self.sign))

    def enforce_dale(self):
        self.W = self.sign_mask * np.abs(self.W)

    def sparsity(self) -> float:
        return float((self.W == 0).mean())

    def weight_stats(self) -> dict:
        nz = self.W[self.W != 0]
        return {
            "mean"    : float(nz.mean())        if len(nz) else 0.0,
            "std"     : float(nz.std())         if len(nz) else 0.0,
            "max"     : float(np.abs(nz).max()) if len(nz) else 0.0,
            "sparsity": self.sparsity(),
        }