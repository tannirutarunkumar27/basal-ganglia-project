"""
Constraint Validator: ensures the SNN agent never violates
the seven design principles at runtime.
"""

class ConstraintValidator:

    @staticmethod
    def check_no_backprop(model) -> bool:
        """
        Confirms no gradient-based weight updates are applied.
        Returns True if clean (no autograd in use).
        """
        for param in getattr(model, 'parameters', []):
            if getattr(param, 'requires_grad', False):
                return False
        return True

    @staticmethod
    def check_spike_based(neuron_output) -> bool:
        """
        Verifies outputs are binary spike trains {0, 1}.
        """
        import numpy as np
        unique_vals = set(np.unique(neuron_output))
        return unique_vals.issubset({0, 1})

    @staticmethod
    def check_local_plasticity(update_rule: str) -> bool:
        """
        Confirms the update rule uses only locally available signals:
        pre-synaptic, post-synaptic, and neuromodulatory (dopamine).
        """
        allowed_rules = {"stdp", "stde", "reward_modulated_stdp",
                         "hebbian", "meta_dopamine"}
        return update_rule.lower() in allowed_rules

    @staticmethod
    def check_energy_budget(spike_count: int, budget: int = 50000) -> bool:
        """
        Checks that total spikes per episode stay within energy budget.
        """
        return spike_count <= budget

    @staticmethod
    def full_report(model, neuron_output, update_rule,
                    spike_count, budget=50000) -> None:
        print("\n--- Constraint Validation Report ---")
        checks = {
            "No backpropagation"  : ConstraintValidator.check_no_backprop(model),
            "Spike-based output"  : ConstraintValidator.check_spike_based(neuron_output),
            "Local plasticity"    : ConstraintValidator.check_local_plasticity(update_rule),
            "Energy budget"       : ConstraintValidator.check_energy_budget(spike_count, budget),
        }
        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")
        print("------------------------------------\n")