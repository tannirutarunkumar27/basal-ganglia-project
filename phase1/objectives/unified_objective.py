"""
Unified Objective: Sequential Decision-Making Under SNN Constraints
max E[sum_{t=0}^{T} gamma^t * r_t]
"""

import numpy as np

class UnifiedObjective:
    """
    Encodes the unified RL objective for the SNN-BG system.
    All seven constraints are enforced here as flags and validators.
    """

    def __init__(self, gamma=0.99, T=200):
        self.gamma   = gamma    # discount factor
        self.T       = T        # max episode horizon

        # Seven hard constraints
        self.constraints = {
            "spike_based_computation"       : True,
            "biologically_plausible_plasticity": True,
            "no_backpropagation"            : True,
            "bg_mediated_action_selection"  : True,
            "uncertainty_aware_learning"    : True,
            "explanation_generation"        : True,
            "energy_efficient_operation"    : True,
        }

    def compute_discounted_return(self, rewards: list) -> float:
        """
        Computes G_t = sum_{k=0}^{T} gamma^k * r_{t+k}
        This is the quantity the system maximises in expectation.
        """
        G = 0.0
        for k, r in enumerate(rewards):
            G += (self.gamma ** k) * r
        return G

    def validate_constraints(self, agent_config: dict) -> dict:
        """
        Checks that the agent configuration satisfies all seven constraints.
        Returns a report dict: {constraint_name: pass/fail}.
        """
        report = {}
        for constraint, required in self.constraints.items():
            provided = agent_config.get(constraint, False)
            report[constraint] = "PASS" if provided == required else "FAIL"
        return report

    def print_objective_summary(self):
        print("=" * 55)
        print("  Unified Objective: SNN-BG Decision System")
        print("=" * 55)
        print(f"  Objective : max E[sum_{{t=0}}^{{{self.T}}} {self.gamma}^t * r_t]")
        print(f"  Gamma     : {self.gamma}")
        print(f"  Horizon T : {self.T}")
        print()
        print("  Active Constraints:")
        for k, v in self.constraints.items():
            status = "[ON]" if v else "[OFF]"
            print(f"    {status}  {k.replace('_',' ')}")
        print("=" * 55)