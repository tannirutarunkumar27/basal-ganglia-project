import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tasks.standard.probabilistic_bandit    import ProbabilisticBandit
from tasks.standard.reversal_learning       import ReversalLearning
from tasks.standard.stop_signal             import StopSignalTask
from tasks.standard.sequential_decision     import SequentialDecisionTask
from tasks.standard.grid_world              import GridWorld
from tasks.advanced.risk_sensitive          import RiskSensitiveChoice
from tasks.advanced.partial_observability   import PartialObservabilityTask
from tasks.advanced.delayed_reward          import DelayedRewardTask
from tasks.advanced.counterfactual_task     import CounterfactualTask
from tasks.advanced.multi_objective         import MultiObjectiveTask
from tasks.advanced.energy_constrained      import EnergyConstrainedTask

STANDARD_TASKS = {
    "probabilistic_bandit"   : ProbabilisticBandit,
    "reversal_learning"      : ReversalLearning,
    "stop_signal"            : StopSignalTask,
    "sequential_decision"    : SequentialDecisionTask,
    "grid_world"             : GridWorld,
}

ADVANCED_TASKS = {
    "risk_sensitive"         : RiskSensitiveChoice,
    "partial_observability"  : PartialObservabilityTask,
    "delayed_reward"         : DelayedRewardTask,
    "counterfactual"         : CounterfactualTask,
    "multi_objective"        : MultiObjectiveTask,
    "energy_constrained"     : EnergyConstrainedTask,
}

ALL_TASKS = {**STANDARD_TASKS, **ADVANCED_TASKS}

def get_task(name: str, **kwargs):
    if name not in ALL_TASKS:
        raise ValueError(f"Unknown task: {name}. "
                         f"Available: {list(ALL_TASKS.keys())}")
    return ALL_TASKS[name](**kwargs)