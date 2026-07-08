"""
BGNetwork: the full BG-cortex-thalamus spiking network.
Wraps all populations and synaptic connections into a
single simulate_step() call.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from neurons.population_factory        import build_all_populations
from connectivity.network_connectivity import build_connectivity


class BGNetwork:

    def __init__(self, dt: float = 0.1e-3,
                 param_noise: float = 0.05):
        self.dt = dt
        self.t  = 0.0

        print("\n  Building BGNetwork populations...")
        self.pops = build_all_populations(dt=dt, param_noise=param_noise)

        print("\n  Building synaptic connections...")
        self.cons = build_connectivity(self.pops, dt=dt)

        # Current buffers — use .items() so 'pop' is always defined
        self._I = {}
        for name, pop in self.pops.items():
            self._I[name] = np.zeros(pop.N)

        # Spike buffers from last step
        self.spikes = {}
        for name, pop in self.pops.items():
            self.spikes[name] = np.zeros(pop.N, dtype=bool)

        # Tonic drive for each population (baseline firing)
        self._setup_tonic_drives()

    def _setup_tonic_drives(self):
        """
        Computes the DC bias current that keeps each population
        near its biological baseline firing rate.
        """
        from neurons.population_params import POPULATION_PARAMS
        self.tonic = {}
        for name, pop in self.pops.items():
            params   = POPULATION_PARAMS[name]
            gL       = params.get("gL", 10e-9)
            EL       = params.get("EL", -70e-3)
            VT       = params.get("VT", -50e-3)
            hz       = params["target_rate_hz"]
            rheobase = gL * (VT - EL)
            self.tonic[name] = np.full(pop.N, rheobase * (1.0 + hz / 80.0))

    def step(self, cortex_input: np.ndarray = None,
             dopamine_signal: float = 0.0) -> dict:
        """
        One network timestep:
        1. Accumulate tonic + synaptic currents
        2. Step each population
        3. Return spike dict
        """
        # Reset current buffers to tonic baseline
        for name in self._I:
            self._I[name][:] = self.tonic[name].copy()

        # Spikes from previous timestep
        sp = self.spikes

        # Route synaptic currents through all connections
        for cname, con in self.cons.items():
            pre_name  = self._resolve_pre(cname)
            post_name = self._resolve_post(cname)
            if pre_name is None or post_name is None:
                continue

            I_syn = con.step(sp[pre_name],
                             self.pops[post_name].V)
            self._I[post_name] += I_syn

            # Update eligibility traces for plastic synapses
            if con.plastic:
                con.update_eligibility(sp[pre_name],
                                       sp[post_name])

        # External cortex input (task state encoding)
        if cortex_input is not None:
            n = min(len(cortex_input), self._I["cortex"].shape[0])
            self._I["cortex"][:n] += cortex_input[:n]

        # Small independent noise on every neuron
        for name, pop in self.pops.items():
            self._I[name] += np.random.normal(0, 0.05e-9, pop.N)

        # Step all populations
        for name, pop in self.pops.items():
            self.spikes[name] = pop.step(self._I[name], self.t)

        # Reward-modulated weight update
        if abs(dopamine_signal) > 1e-12:
            for cname, con in self.cons.items():
                if con.plastic:
                    con.apply_reward_modulated_update(
                        dopamine_signal, alpha=1e-12)

        self.t += self.dt
        return dict(self.spikes)

    # Connection routing tables (class-level constants)
    _PRE_MAP = {
        "ctx_d1"        : "cortex",
        "ctx_d2"        : "cortex",
        "ctx_stn"       : "cortex",
        "d1_gpi"        : "d1_msn",
        "d2_gpe"        : "d2_msn",
        "gpe_stn"       : "gpe",
        "gpe_gpi"       : "gpe",
        "stn_gpi"       : "stn",
        "gpi_thal"      : "gpi",
        "thal_ctx"      : "thalamus",
        "snc_d1"        : "snc",
        "snc_d2"        : "snc",
        "sero_gpi"      : "serotonin",
        "ne_stn"        : "norepinephrine",
        "bayes_striatum": "bayesian_layer",
        "bayes_gpi"     : "bayesian_layer",
    }

    _POST_MAP = {
        "ctx_d1"        : "d1_msn",
        "ctx_d2"        : "d2_msn",
        "ctx_stn"       : "stn",
        "d1_gpi"        : "gpi",
        "d2_gpe"        : "gpe",
        "gpe_stn"       : "stn",
        "gpe_gpi"       : "gpi",
        "stn_gpi"       : "gpi",
        "gpi_thal"      : "thalamus",
        "thal_ctx"      : "cortex",
        "snc_d1"        : "d1_msn",
        "snc_d2"        : "d2_msn",
        "sero_gpi"      : "gpi",
        "ne_stn"        : "stn",
        "bayes_striatum": "d1_msn",
        "bayes_gpi"     : "gpi",
    }

    def _resolve_pre(self, cname: str):
        return self._PRE_MAP.get(cname)

    def _resolve_post(self, cname: str):
        return self._POST_MAP.get(cname)

    def reset(self):
        """Full network reset between episodes."""
        for pop in self.pops.values():
            pop.reset_state()
        self.t = 0.0
        for name, pop in self.pops.items():
            self._I[name]     = np.zeros(pop.N)
            self.spikes[name] = np.zeros(pop.N, dtype=bool)