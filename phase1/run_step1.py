from objectives.unified_objective   import UnifiedObjective
from objectives.constraint_validator import ConstraintValidator
import numpy as np

# --- instantiate objective ---
obj = UnifiedObjective(gamma=0.99, T=200)
obj.print_objective_summary()

# --- test discounted return ---
rewards = [1.0, 0.5, 1.0, 0.0, 1.0]
G = obj.compute_discounted_return(rewards)
print(f"\n  Discounted return for sample rewards: G = {G:.4f}")

# --- validate a sample agent config ---
agent_config = {
    "spike_based_computation"          : True,
    "biologically_plausible_plasticity": True,
    "no_backpropagation"               : True,
    "bg_mediated_action_selection"     : True,
    "uncertainty_aware_learning"       : True,
    "explanation_generation"           : True,
    "energy_efficient_operation"       : True,
}
report = obj.validate_constraints(agent_config)
print("\n  Agent Constraint Report:")
for k, v in report.items():
    print(f"    [{v}] {k.replace('_',' ')}")

# --- constraint validator demo ---
spike_output = np.array([0, 1, 0, 0, 1, 1, 0])
ConstraintValidator.full_report(
    model         = object(),       # placeholder
    neuron_output = spike_output,
    update_rule   = "reward_modulated_stdp",
    spike_count   = 12000,
    budget        = 50000
)