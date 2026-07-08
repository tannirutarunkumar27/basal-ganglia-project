"""
Phase 12 — Step 32: Experimental Validation
Runs all 9 benchmark tasks on the full SNN-BG pipeline.
"""

import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ["phase1","phase2","phase3","phase4","phase5",
          "phase6","phase7","phase8","phase9","phase10"]:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), p))
sys.path.insert(0, HERE)

from network.bg_network                      import BGNetwork
from integration.bayesian_reasoning_pipeline import BayesianReasoningPipeline
from integration.bg_pathway_controller       import BGPathwayController
from integration.action_gating_controller    import ActionGatingController
from rl_engine.rl_engine                     import RLEngine
from plasticity.plasticity_manager           import PlasticityManager
from controllers.neuromodulator_controller   import NeuromodulatorController
from integration.reasoning_pipeline          import ReasoningPipeline
from integration.neuromorphic_optimizer      import NeuromorphicOptimizer

from tasks.standard.probabilistic_bandit  import ProbabilisticBandit
from tasks.standard.reversal_learning     import ReversalLearning
from tasks.standard.stop_signal          import StopSignalTask
from tasks.standard.sequential_decision   import SequentialDecisionTask
from tasks.standard.grid_world           import GridWorld
from tasks.advanced.risk_sensitive        import RiskSensitiveChoice
from tasks.advanced.counterfactual_task   import CounterfactualTask

from tasks.volatile_reward_task   import VolatileRewardTask
from tasks.hidden_rule_task       import HiddenRuleTask

from harness.validation_harness   import ValidationHarness
from metrics.capability_scorer    import CapabilityScorer
from reports.validation_report    import ValidationReport

np.random.seed(42)

DT         = 0.1e-3
N_ACTIONS  = 4
STATE_DIM  = 2 * N_ACTIONS + 4
RESULTS    = os.path.join(HERE, "results")


def build_pipeline():
    """Builds all Phases 2-10 components."""
    print("  Building pipeline components...")
    net = BGNetwork(dt=DT)

    pipeline = BayesianReasoningPipeline(
        n_actions=N_ACTIONS,
        n_neurons_total=net.pops["bayesian_layer"].N,
        window_steps=100, lam=0.85, dt=DT)

    p4_ctrl = BGPathwayController(
        n_actions=N_ACTIONS,
        n_d1_per_action=net.pops["d1_msn"].N // N_ACTIONS,
        n_d2_per_action=net.pops["d2_msn"].N // N_ACTIONS,
        conflict_eps=0.3, dt=DT)

    p5_ctrl = ActionGatingController(
        n_actions=N_ACTIONS, theta_0=0.5, beta=0.4,
        kappa=0.3, refractory_ms=150.0,
        log_dir=RESULTS, dt=DT)

    rl = RLEngine(n_actions=N_ACTIONS, state_dim=STATE_DIM,
                   n_d1_per_action=net.pops["d1_msn"].N // N_ACTIONS,
                   dt=DT)

    pop_sizes = {k: net.pops[k].N
                 for k in ["cortex","d1_msn","d2_msn","gpi","gpe"]}
    plast = PlasticityManager(
        pop_sizes=pop_sizes, n_actions=N_ACTIONS,
        base_alpha=0.05, min_delta=0.03, dt=DT)

    nm_ctrl = NeuromodulatorController(
        alpha_0=0.05, eta=0.10, dt=DT)

    reasoning = ReasoningPipeline(
        n_actions=N_ACTIONS,
        action_names=["a0","a1","a2","a3"],
        log_dir=RESULTS, dt=DT)

    pop_sizes_all = {name: pop.N
                     for name, pop in net.pops.items()}
    optimizer = NeuromorphicOptimizer(
        pop_sizes=pop_sizes_all, n_actions=N_ACTIONS,
        target_sparsity=0.08, episode_budget_nJ=10000.0,
        prune_every=5000, dt=DT)

    return (net, pipeline, p4_ctrl, p5_ctrl,
            rl, plast, nm_ctrl, reasoning, optimizer)


def main():

    print("\n" + "="*65)
    print("  Phase 12 — Step 32: Experimental Validation")
    print("="*65)

    os.makedirs(RESULTS, exist_ok=True)

    # ── Build pipeline ──────────────────────────────────────────
    (net, pipeline, p4_ctrl, p5_ctrl,
     rl, plast, nm_ctrl, reasoning,
     optimizer) = build_pipeline()

    harness = ValidationHarness(
        net=net, pipeline=pipeline, p4_ctrl=p4_ctrl,
        p5_ctrl=p5_ctrl, rl_engine=rl, plast=plast,
        nm_ctrl=nm_ctrl, reasoning=reasoning,
        optimizer=optimizer, results_dir=RESULTS, dt=DT)

    scorer = CapabilityScorer(N_ACTIONS)
    report = ValidationReport(RESULTS)

    # ── Define task suite ───────────────────────────────────────
    # Steps and n_steps chosen for meaningful convergence signal
    task_suite = [
        # Standard tasks
        (ProbabilisticBandit(k=N_ACTIONS, max_steps=3000),
         "probabilistic_bandit", 3000),
        (ReversalLearning(max_steps=3000),
         "reversal_learning", 3000),
        (StopSignalTask(max_steps=3000, stop_prob=0.25),
         "stop_signal", 3000),
        (SequentialDecisionTask(n_actions=N_ACTIONS,
                                 max_steps=3000),
         "sequential_decision", 3000),
        (GridWorld(grid_size=5, max_steps=3000),
         "grid_world", 3000),
        # Advanced tasks
        (RiskSensitiveChoice(max_steps=3000, rho=0.5),
         "risk_sensitive", 3000),
        (VolatileRewardTask(k=N_ACTIONS, max_steps=3000),
         "volatile_reward", 3000),
        (HiddenRuleTask(n_actions=N_ACTIONS, max_steps=3000),
         "hidden_rule", 3000),
        (CounterfactualTask(k=N_ACTIONS, max_steps=3000),
         "counterfactual", 3000),
    ]

    print(f"\n  Running {len(task_suite)} benchmark tasks...\n")

    # ── Run all tasks ───────────────────────────────────────────
    for task_obj, task_name, n_steps in task_suite:
        harness.run_task(
            task          = task_obj,
            n_steps       = n_steps,
            explain_every = 500,
            task_name     = task_name)

    # ── Compute capability scores ───────────────────────────────
    scores = scorer.score_all(harness.results)
    scorer.print_scores(scores)

    # ── Attach scores to results ────────────────────────────────
    for name, r in harness.results.items():
        r.capability_scores = scores

    # ── Generate full report ────────────────────────────────────
    report.print_full_report(harness.results, scores)

    json_path = report.save_json(
        harness.results, scores,
        "phase12_validation.json")
    radar_path = report.plot_capability_radar(scores)
    comp_path  = report.plot_task_comparison(harness.results)
    curve_path = report.plot_learning_curves(harness.results)

    print(f"  Saved JSON report      : {json_path}")
    print(f"  Saved capability radar : {radar_path}")
    print(f"  Saved task comparison  : {comp_path}")
    print(f"  Saved learning curves  : {curve_path}")
    print("="*65 + "\n")

    return harness.results, scores


if __name__ == "__main__":
    main()