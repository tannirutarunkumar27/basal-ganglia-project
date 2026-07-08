"""
PlasticityManager — Phase 7 integration.
Manages STDE for all four plastic synaptic connections:

    ctx  → d1_msn  (direct pathway  Go)
    ctx  → d2_msn  (indirect pathway No-Go)
    d1   → gpi     (direct pathway output)
    d2   → gpe     (indirect pathway output)

Each connection has its own:
  - STDPKernel
  - MultiTimescaleTraces
  - STDEEngine

On every timestep:
  1. STDP traces are updated from spike data
  2. Eligibility traces accumulate evidence
  3. When delta_prime > threshold, STDE weight update fires

Supports:
  - Delayed reward assignment (up to 2000 ms)
  - Per-connection learning rates
  - Dopamine-gate: only update when |delta| > min_delta
  - Weight export/import for checkpointing
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stdp.stdp_kernel              import STDPKernel
from traces.multi_timescale_traces import MultiTimescaleTraces
from stde.stde_engine              import STDEEngine


class PlasticityManager:

    # Define all plastic connections
    CONNECTION_DEFS = [
        {"name": "ctx_d1",  "sign": +1, "alpha_scale": 1.0},
        {"name": "ctx_d2",  "sign": +1, "alpha_scale": 0.8},
        {"name": "d1_gpi",  "sign": -1, "alpha_scale": 0.6},
        {"name": "d2_gpe",  "sign": -1, "alpha_scale": 0.6},
    ]

    def __init__(self,
                 pop_sizes   : dict,
                 n_actions   : int,
                 base_alpha  : float = 0.05,
                 min_delta   : float = 0.05,
                 dt          : float = 0.1e-3):
        """
        pop_sizes  : dict with keys 'cortex','d1_msn','d2_msn',
                     'gpi','gpe' giving population sizes
        n_actions  : number of competing actions
        base_alpha : base learning rate (scaled by alpha_t)
        min_delta  : minimum |delta| to trigger weight update
        dt         : simulation timestep
        """
        self.n_actions  = n_actions
        self.base_alpha = base_alpha
        self.min_delta  = min_delta
        self.dt         = dt

        # Map connection name to (N_pre, N_post)
        size_map = {
            "ctx_d1": (pop_sizes["cortex"],  pop_sizes["d1_msn"]),
            "ctx_d2": (pop_sizes["cortex"],  pop_sizes["d2_msn"]),
            "d1_gpi": (pop_sizes["d1_msn"],  pop_sizes["gpi"]),
            "d2_gpe": (pop_sizes["d2_msn"],  pop_sizes["gpe"]),
        }

        # Build kernel, trace, and STDE engine for each connection
        self.kernels  = {}
        self.traces   = {}
        self.engines  = {}
        self.alphas   = {}

        for cdef in self.CONNECTION_DEFS:
            name  = cdef["name"]
            n_pre, n_post = size_map[name]

            k = STDPKernel(mode="asymmetric", dt=dt)
            k.initialise(n_pre, n_post)
            self.kernels[name] = k

            self.traces[name]  = MultiTimescaleTraces(
                n_pre, n_post, dt=dt, adaptive_weights=True)

            self.engines[name] = STDEEngine(
                n_pre, n_post,
                sign  = cdef["sign"],
                name  = name,
                dt    = dt)

            self.alphas[name]  = (base_alpha
                                   * cdef["alpha_scale"])

        # Update statistics
        self.step_count   = 0
        self.update_count = 0
        self.weight_change_history = {
            name: [] for name in self.engines}

    def reset(self) -> None:
        for name in self.kernels:
            self.kernels[name].reset()
            self.traces[name].reset()
            self.engines[name].reset()
        self.step_count   = 0
        self.update_count = 0

    def step(self,
             spike_dict    : dict,
             delta_prime   : float,
             alpha_t       : float,
             force_update  : bool = False) -> dict:
        """
        One plasticity timestep.

        spike_dict : {pop_name: bool_spike_array}
                     must contain 'cortex','d1_msn','d2_msn',
                     'gpi','gpe'
        delta_prime: enriched dopamine signal from Phase 6
        alpha_t    : meta learning rate from uncertainty
        force_update: bypass min_delta gate (for testing)

        Returns dict of per-connection dW magnitudes.
        """
        self.step_count += 1

        # Routing: which pre/post populations for each connection
        routing = {
            "ctx_d1": ("cortex",  "d1_msn"),
            "ctx_d2": ("cortex",  "d2_msn"),
            "d1_gpi": ("d1_msn",  "gpi"),
            "d2_gpe": ("d2_msn",  "gpe"),
        }

        dW_out = {}

        for name, (pre_name, post_name) in routing.items():
            pre_spikes  = spike_dict.get(pre_name,
                            np.zeros(self.engines[name].n_pre))
            post_spikes = spike_dict.get(post_name,
                            np.zeros(self.engines[name].n_post))

            pre_spikes  = np.asarray(pre_spikes,  dtype=float)
            post_spikes = np.asarray(post_spikes, dtype=float)

            # Pad/truncate
            n_pre  = self.engines[name].n_pre
            n_post = self.engines[name].n_post
            pspk   = np.zeros(n_pre)
            ospk   = np.zeros(n_post)
            pspk[:min(len(pre_spikes),  n_pre)]  = \
                pre_spikes[:min(len(pre_spikes),  n_pre)]
            ospk[:min(len(post_spikes), n_post)] = \
                post_spikes[:min(len(post_spikes), n_post)]

            # Update STDP traces
            self.kernels[name].update_traces(pspk, ospk)
            stdp_dW = self.kernels[name].compute_stdp(pspk, ospk)

            # Update eligibility traces
            e_total = self.traces[name].step(stdp_dW)

            # Gate: only update weights if |delta| is large enough
            do_update = (abs(delta_prime) > self.min_delta
                         or force_update)

            if do_update:
                lr   = self.alphas[name] * alpha_t
                dW   = self.engines[name].update(
                    e_total     = e_total,
                    delta_total = delta_prime,
                    alpha_t     = lr)

                # Update timescale weights
                self.traces[name].update_weights(
                    delta_prime, delta_prime)

                dW_mag = float(np.abs(dW).mean())
                self.update_count += 1
            else:
                dW_mag = 0.0

            dW_out[name] = dW_mag
            self.weight_change_history[name].append(dW_mag)

        return dW_out

    def get_weights(self) -> dict:
        """Returns current weight matrices for all connections."""
        return {name: eng.W.copy()
                for name, eng in self.engines.items()}

    def plasticity_summary(self) -> dict:
        return {
            "step_count"   : self.step_count,
            "update_count" : self.update_count,
            "connections"  : {
                name: eng.stde_summary()
                for name, eng in self.engines.items()
            },
            "trace_summary": {
                name: tr.timescale_summary()
                for name, tr in self.traces.items()
            },
        }