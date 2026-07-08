"""
Phase 11 — Step 31: Full Online Training Loop
Entry point. Constructs all components from Phases 2-10
and runs the complete multi-episode training procedure.
"""

import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ["phase2","phase3","phase4","phase5",
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

from config.training_config           import TrainingConfig
from loop.training_loop               import TrainingLoop
from metrics.training_metrics         import TrainingMetrics
from checkpointing.checkpoint_manager import CheckpointManager
from utils.plot_training              import plot_training_curves


def build_all_components(cfg: TrainingConfig):
    """
    Instantiates all pipeline components with config values.
    """
    print("\n  Building Phase 2 — BGNetwork...")
    net = BGNetwork(dt=cfg.dt)

    print("  Building Phase 3 — BayesianReasoningPipeline...")
    pipeline = BayesianReasoningPipeline(
        n_actions       = cfg.n_actions,
        n_neurons_total = net.pops["bayesian_layer"].N,
        window_steps    = cfg.belief_window,
        lam             = cfg.belief_lam,
        alpha_prior     = cfg.alpha_prior,
        dt              = cfg.dt)

    print("  Building Phase 4 — BGPathwayController...")
    p4_ctrl = BGPathwayController(
        n_actions       = cfg.n_actions,
        n_d1_per_action = net.pops["d1_msn"].N // cfg.n_actions,
        n_d2_per_action = net.pops["d2_msn"].N // cfg.n_actions,
        conflict_eps    = cfg.conflict_eps,
        dt              = cfg.dt)

    print("  Building Phase 5 — ActionGatingController...")
    results_dir = os.path.join(HERE, cfg.results_dir)
    p5_ctrl = ActionGatingController(
        n_actions     = cfg.n_actions,
        theta_0       = cfg.theta_0,
        beta          = cfg.gate_beta,
        kappa         = cfg.gate_kappa,
        refractory_ms = cfg.refractory_ms,
        log_dir       = results_dir,
        dt            = cfg.dt)

    print("  Building Phase 6 — RLEngine...")
    rl_engine = RLEngine(
        n_actions       = cfg.n_actions,
        state_dim       = cfg.state_dim,
        n_d1_per_action = net.pops["d1_msn"].N // cfg.n_actions,
        dt              = cfg.dt)

    print("  Building Phase 7 — PlasticityManager...")
    pop_sizes = {k: net.pops[k].N
                 for k in ["cortex","d1_msn","d2_msn","gpi","gpe"]}
    # prune_every is a Phase 10 parameter, not Phase 7
    plast = PlasticityManager(
        pop_sizes  = pop_sizes,
        n_actions  = cfg.n_actions,
        base_alpha = cfg.base_alpha,
        min_delta  = cfg.min_delta,
        dt         = cfg.dt)

    print("  Building Phase 8 — NeuromodulatorController...")
    nm_ctrl = NeuromodulatorController(
        alpha_0 = cfg.alpha_0,
        eta     = cfg.eta,
        omega_d = cfg.omega_d,
        omega_s = cfg.omega_s,
        omega_n = cfg.omega_n,
        dt      = cfg.dt)

    print("  Building Phase 9 — ReasoningPipeline...")
    reasoning = ReasoningPipeline(
        n_actions    = cfg.n_actions,
        action_names = cfg.action_names,
        log_dir      = results_dir,
        dt           = cfg.dt)

    print("  Building Phase 10 — NeuromorphicOptimizer...")
    pop_sizes_all = {name: pop.N
                     for name, pop in net.pops.items()}
    optimizer = NeuromorphicOptimizer(
        pop_sizes          = pop_sizes_all,
        n_actions          = cfg.n_actions,
        target_sparsity    = cfg.target_sparsity,
        stn_U_threshold    = cfg.stn_U_threshold,
        action_C_threshold = cfg.action_C_threshold,
        episode_budget_nJ  = cfg.episode_budget_nJ,
        prune_every        = cfg.prune_every,
        dt                 = cfg.dt)

    return (net, pipeline, p4_ctrl, p5_ctrl,
            rl_engine, plast, nm_ctrl, reasoning, optimizer)


def main():

    # ── Configuration ──────────────────────────────────────────
    cfg = TrainingConfig(
        dt             = 0.1e-3,
        episode_steps  = 5000,      # 500 ms per episode
        n_episodes     = 5,
        seed           = 42,
        n_actions      = 4,
        state_dim      = 12,        # 2*4 + 4
        correct_action = 2,
        log_every      = 1000,
        save_every     = 5000,
        explain_every  = 1000,
        results_dir    = "results",
        checkpoint_dir = "checkpoints",
    )
    cfg.validate()

    np.random.seed(cfg.seed)

    os.makedirs(os.path.join(HERE, cfg.results_dir),    exist_ok=True)
    os.makedirs(os.path.join(HERE, cfg.checkpoint_dir), exist_ok=True)

    # ── Build all components ───────────────────────────────────
    (net, pipeline, p4_ctrl, p5_ctrl,
     rl_engine, plast, nm_ctrl,
     reasoning, optimizer) = build_all_components(cfg)

    # ── Support objects ────────────────────────────────────────
    metrics  = TrainingMetrics(
        n_actions   = cfg.n_actions,
        window      = 500,
        results_dir = os.path.join(HERE, cfg.results_dir))

    ckpt_mgr = CheckpointManager(
        checkpoint_dir = os.path.join(HERE, cfg.checkpoint_dir))

    # ── Training loop ──────────────────────────────────────────
    loop = TrainingLoop(
        net       = net,
        pipeline  = pipeline,
        p4_ctrl   = p4_ctrl,
        p5_ctrl   = p5_ctrl,
        rl_engine = rl_engine,
        plast     = plast,
        nm_ctrl   = nm_ctrl,
        reasoning = reasoning,
        optimizer = optimizer,
        metrics   = metrics,
        ckpt_mgr  = ckpt_mgr,
        config    = cfg)

    # ── Step 31: execute the complete algorithm ────────────────
    all_ep_metrics = loop.run_training()

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  Training complete.")

    final_acc = all_ep_metrics[-1]["accuracy"] * 100
    best_acc  = max(m["accuracy"] for m in all_ep_metrics) * 100
    mean_eff  = float(np.mean(
        [m["efficiency_score"] for m in all_ep_metrics]))

    print(f"  Final episode accuracy : {final_acc:.1f}%")
    print(f"  Best episode accuracy  : {best_acc:.1f}%")
    print(f"  Mean efficiency score  : {mean_eff:.4f}")
    print(f"  Convergence step       : {metrics.convergence_step()}")
    print(f"  Total regret           : {metrics.regret():.1f}")
    print(f"  Total steps            : {metrics.total_steps:,}")

    print("\n  Neuromodulator fusion weights (final):")
    nm_s = nm_ctrl.controller_summary()
    print(f"    omega_d = {nm_s['fusion']['omega_d']:.3f}  "
          f"omega_s = {nm_s['fusion']['omega_s']:.3f}  "
          f"omega_n = {nm_s['fusion']['omega_n']:.3f}")
    print(f"    dominant NM : {nm_s['fusion']['dominant']}")
    print(f"    NE arousal  : {nm_s['NE_arousal']}")

    print("\n  Plasticity (final weight means):")
    plast_s = plast.plasticity_summary()
    # Extract into local variables to avoid f-string line-break SyntaxError
    for cname, cs in plast_s["connections"].items():
        trace_mean = plast_s["trace_summary"][cname]["e_total_mean"]
        print(f"    {cname:<12s}: mean_W={cs['mean_W']:.4f}  "
              f"trace_mean={trace_mean:.6f}")

    # ── Save training plots ────────────────────────────────────
    print("\n  Running one final episode for step-level plots...")
    _, last_step_log = loop.run_episode(cfg.n_episodes + 1)

    plot_path = plot_training_curves(
        all_ep_metrics = all_ep_metrics,
        step_log       = last_step_log,
        results_dir    = os.path.join(HERE, cfg.results_dir),
        dt             = cfg.dt)
    print(f"\n  Training curves saved: {plot_path}")

    reasoning.save_explanations("phase11_explanations.json")
    metrics_path = metrics.save("phase11_final_metrics.json")
    print(f"  Final metrics saved  : {metrics_path}")

    optimizer.budget.print_report()
    print("="*60 + "\n")

    return all_ep_metrics


if __name__ == "__main__":
    main()