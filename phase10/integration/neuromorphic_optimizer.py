"""
NeuromorphicOptimizer — Phase 10 integration.
Steps 29 and 30 combined into one pipeline call.

Wraps around the full Phases 2-9 pipeline and applies
energy optimization at every timestep.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sparse.sparse_coder          import SparseCoder
from sparse.event_driven_filter   import EventDrivenFilter
from sparse.firing_rate_limiter   import FiringRateLimiter
from pruning.synapse_pruner       import SynapsePruner
from gates.conditional_gates      import ConditionalGates
from budget.energy_budget         import EnergyBudget


class NeuromorphicOptimizer:

    def __init__(self,
                 pop_sizes        : dict,
                 n_actions        : int,
                 target_sparsity  : float = 0.08,
                 stn_U_threshold  : float = 0.55,
                 action_C_threshold: float = 0.55,
                 episode_budget_nJ: float = 2000.0,
                 prune_every      : int   = 1000,
                 dt               : float = 0.1e-3):
        """
        pop_sizes         : dict {name: N_neurons} for each population
        n_actions         : number of action channels
        target_sparsity   : target fraction active per timestep
        episode_budget_nJ : maximum energy per episode
        prune_every       : steps between pruning operations
        """
        self.dt          = dt
        self.prune_every = prune_every
        self.step_count  = 0

        # Step 29: sparse coders per population
        self.coders = {
            name: SparseCoder(
                n_neurons       = N,
                target_sparsity = target_sparsity,
                dt              = dt,
                name            = name)
            for name, N in pop_sizes.items()
        }

        # Step 29: shared event-driven filter
        self.ev_filter = EventDrivenFilter(dt=dt)

        # Step 29: firing rate limiters per population
        rate_targets = {
            "cortex"        : 5.0,
            "d1_msn"        : 2.0,
            "d2_msn"        : 2.0,
            "stn"           : 20.0,
            "gpe"           : 50.0,
            "gpi"           : 60.0,
            "thalamus"      : 10.0,
            "snc"           : 4.0,
            "serotonin"     : 2.0,
            "norepinephrine": 2.0,
            "bayesian_layer": 10.0,
            "reasoning_layer": 8.0,
        }
        self.limiters = {
            name: FiringRateLimiter(
                n_neurons      = N,
                target_rate_hz = rate_targets.get(name, 10.0),
                dt             = dt,
                name           = name)
            for name, N in pop_sizes.items()
        }

        # Step 29: synapse pruners for plastic connections
        self.pruners = {}
        plastic_connections = {
            "ctx_d1" : ("cortex",  "d1_msn"),
            "ctx_d2" : ("cortex",  "d2_msn"),
            "d1_gpi" : ("d1_msn",  "gpi"),
            "d2_gpe" : ("d2_msn",  "gpe"),
        }
        for cname, (pre, post) in plastic_connections.items():
            n_pre  = pop_sizes.get(pre,  40)
            n_post = pop_sizes.get(post, 40)
            self.pruners[cname] = SynapsePruner(
                n_pre  = n_pre,
                n_post = n_post,
                name   = cname,
                dt     = dt)

        # Step 30: conditional gates
        self.gates = ConditionalGates(
            stn_U_threshold     = stn_U_threshold,
            action_C_threshold  = action_C_threshold,
            dt                  = dt)

        # Energy budget monitor
        self.budget = EnergyBudget(
            episode_budget_nJ = episode_budget_nJ,
            dt                = dt)

        # AHP currents accumulator (added back to populations)
        self.ahp_currents = {name: np.zeros(N)
                              for name, N in pop_sizes.items()}

    def reset(self) -> None:
        for c in self.coders.values():   c.reset()
        for l in self.limiters.values(): l.reset()
        for p in self.pruners.values():  p.reset()
        self.ev_filter.reset()
        self.gates.reset()
        self.budget.reset()
        self.step_count = 0
        for name in self.ahp_currents:
            self.ahp_currents[name][:] = 0.0

    def optimize_spikes(self,
                         spike_dict   : dict,
                         membrane_V   : dict) -> dict:
        """
        Step 29: applies sparse coding + rate limiting to all
        population spike trains.

        spike_dict  : {pop_name: bool_array}
        membrane_V  : {pop_name: float_array}

        Returns optimized spike_dict.
        """
        optimized = {}
        for name, spikes in spike_dict.items():
            sp = np.asarray(spikes, dtype=bool)
            V  = np.asarray(membrane_V.get(name,
                             np.zeros(len(sp))), dtype=float)

            # k-WTA sparse coding
            if name in self.coders:
                sp = self.coders[name].apply(V, sp)

            # Rate limiting
            if name in self.limiters:
                sp, I_ahp, _ = self.limiters[name].step(sp)
                # Store AHP for injection back into network
                if name in self.ahp_currents:
                    self.ahp_currents[name] = I_ahp

            optimized[name] = sp

        return optimized

    def apply_pruning(self, weight_dict: dict) -> dict:
        """
        Step 29 (technique 3): applies synapse pruning when due.
        weight_dict: {connection_name: weight_matrix}
        Returns pruned weight_dict.
        """
        pruned = {}
        for cname, W in weight_dict.items():
            if cname in self.pruners:
                W_pruned, _ = self.pruners[cname].prune(W)
                pruned[cname] = W_pruned
            else:
                pruned[cname] = W
        return pruned

    def evaluate_gates(self,
                        U              : float,
                        C              : float,
                        delta_prime    : float,
                        conflict_score : float) -> dict:
        """
        Step 30: evaluates all four conditional gates.
        Returns gate state dict consumed by the pipeline.
        """
        return self.gates.evaluate(U, C, delta_prime, conflict_score)

    def record_energy(self,
                       optimized_spikes: dict,
                       gate_state      : dict,
                       n_plasticity    : int,
                       n_reasoning     : int) -> float:
        """
        Records step energy consumption.
        Returns energy for this step in pJ.
        """
        self.step_count += 1

        # Count spikes across all optimized populations
        n_spikes = sum(
            int(np.asarray(sp, dtype=bool).sum())
            for sp in optimized_spikes.values())

        # Synapse events: spikes * active synapses
        # Approximate: spikes * mean_connections
        n_syn = int(n_spikes * 15)   # ~15 synapses per active neuron

        return self.budget.record_step(
            n_spikes           = n_spikes,
            n_synapse_events   = n_syn,
            n_weight_updates   = n_plasticity,
            n_reasoning_calls  = n_reasoning)

    def full_step(self,
                   spike_dict      : dict,
                   membrane_V      : dict,
                   U               : float,
                   C               : float,
                   delta_prime     : float,
                   conflict_score  : float,
                   weight_dict     : dict = None,
                   n_reasoning     : int  = 0) -> dict:
        """
        Combined Phase 10 step: Steps 29 + 30 + energy budget.

        Returns output dict with optimized spikes, gate states,
        AHP currents, and energy estimate.
        """
        # Step 29a: sparse coding + rate limiting
        opt_spikes = self.optimize_spikes(spike_dict, membrane_V)

        # Step 29b: pruning (periodic)
        if (weight_dict and self.step_count > 0
                and self.step_count % self.prune_every == 0):
            weight_dict = self.apply_pruning(weight_dict)

        # Step 30: conditional gates
        gate_state = self.evaluate_gates(
            U, C, delta_prime, conflict_score)

        # Event-driven check
        is_spike_event = self.ev_filter.check_spike_event(
            opt_spikes.get("cortex",
                           np.zeros(1, dtype=bool)))
        is_reward_event = self.ev_filter.check_reward_event(
            delta_prime, delta_prime)

        # Gate-modulated plasticity count
        n_plasticity = (20 if is_reward_event
                             and gate_state["neuromod_gate"]
                        else 0)

        # Energy recording
        step_energy = self.record_energy(
            opt_spikes, gate_state, n_plasticity, n_reasoning)

        return {
            # Step 29 outputs
            "optimized_spikes"    : opt_spikes,
            "ahp_currents"        : dict(self.ahp_currents),
            "is_spike_event"      : bool(is_spike_event),
            "is_reward_event"     : bool(is_reward_event),

            # Step 30 outputs
            "stn_active"          : gate_state["stn_gate"],
            "action_release_ok"   : gate_state["action_gate"],
            "neuromod_active"     : gate_state["neuromod_gate"],
            "indirect_active"     : gate_state["indirect_gate"],
            "hyperdirect_active"  : gate_state["hyperdirect_gate"],

            # Energy
            "step_energy_pJ"      : float(step_energy),
            "total_energy_nJ"     : self.budget.total_energy_nJ(),
            "within_budget"       : self.budget.within_budget(),

            # Pruned weights (if pruning was applied)
            "weight_dict"         : weight_dict,
        }

    def optimizer_summary(self) -> dict:
        return {
            "budget"   : self.budget.budget_summary(),
            "gates"    : self.gates.gate_summary(),
            "sparsity" : {
                name: c.sparsity_summary()
                for name, c in self.coders.items()
            },
            "pruning"  : {
                name: p.pruner_summary()
                for name, p in self.pruners.items()
            },
        }