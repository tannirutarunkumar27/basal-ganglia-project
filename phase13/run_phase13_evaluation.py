"""
Phase 13 — Step 33: Full Metric Evaluation
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

from aggregator.metric_aggregator import MetricAggregator
from reporter.metric_reporter     import MetricReporter

np.random.seed(42)

DT        = 0.1e-3
N_ACTIONS = 4
STATE_DIM = 12
CORRECT   = 2
RESULTS   = os.path.join(HERE, "results")


def build_pipeline():
    net = BGNetwork(dt=DT)
    pipeline = BayesianReasoningPipeline(
        n_actions       = N_ACTIONS,
        n_neurons_total = net.pops["bayesian_layer"].N,
        window_steps    = 100, lam=0.85, dt=DT)
    p4_ctrl = BGPathwayController(
        n_actions       = N_ACTIONS,
        n_d1_per_action = net.pops["d1_msn"].N // N_ACTIONS,
        n_d2_per_action = net.pops["d2_msn"].N // N_ACTIONS,
        conflict_eps    = 0.3, dt=DT)
    p5_ctrl = ActionGatingController(
        n_actions     = N_ACTIONS, theta_0=0.5, beta=0.4,
        kappa         = 0.3, refractory_ms=150.0,
        log_dir       = RESULTS, dt=DT)
    rl = RLEngine(
        n_actions       = N_ACTIONS, state_dim=STATE_DIM,
        n_d1_per_action = net.pops["d1_msn"].N // N_ACTIONS,
        dt              = DT)
    pop_sizes = {k: net.pops[k].N
                 for k in ["cortex","d1_msn","d2_msn","gpi","gpe"]}
    plast = PlasticityManager(
        pop_sizes  = pop_sizes,
        n_actions  = N_ACTIONS,
        base_alpha = 0.05,
        min_delta  = 0.03,
        dt         = DT)
    nm_ctrl = NeuromodulatorController(
        alpha_0=0.05, eta=0.10, dt=DT)
    reasoning = ReasoningPipeline(
        n_actions    = N_ACTIONS,
        action_names = ["a0","a1","a2","a3"],
        log_dir      = RESULTS, dt=DT)
    pop_sizes_all = {name: pop.N
                     for name, pop in net.pops.items()}
    optimizer = NeuromorphicOptimizer(
        pop_sizes          = pop_sizes_all,
        n_actions          = N_ACTIONS,
        target_sparsity    = 0.08,
        episode_budget_nJ  = 10000.0,
        prune_every        = 5000,
        dt                 = DT)
    return (net, pipeline, p4_ctrl, p5_ctrl,
            rl, plast, nm_ctrl, reasoning, optimizer)


def run_evaluation(task, task_name: str,
                    n_steps: int, components: tuple) -> tuple:
    """Runs one task evaluation and returns (metrics, step_log)."""
    (net, pipeline, p4_ctrl, p5_ctrl,
     rl, plast, nm_ctrl, reasoning, optimizer) = components

    # Reset all components
    for obj in components:
        if hasattr(obj, "reset"):
            obj.reset()

    agg          = MetricAggregator(N_ACTIONS, CORRECT, dt=DT)
    state        = task.reset()
    prev_action  = None
    prev_reward  = None
    step_log_out = []

    print(f"\n  [{task_name}] running {n_steps} steps...")

    for step in range(n_steps):
        t    = step * DT
        t_ms = t * 1000

        # ── Encode state as cortical drive ───────────────────
        st = np.asarray(state, dtype=float).flatten()
        st_range = float(st.max() - st.min())    # replaces ptp()
        if st_range > 1e-8:
            st = (st - st.min()) / (st_range + 1e-8)
        ctx_amp = 0.3e-9 + 0.4e-9 * float(st.mean())
        ctx_in  = np.full(net.pops["cortex"].N, ctx_amp)

        # ── Phase 2 + sparse optimization ────────────────────
        raw_spks   = net.step(ctx_in, dopamine_signal=0.0)
        membrane_V = {n: p.V.copy()
                      for n, p in net.pops.items()}
        n_neurons_t = sum(p.N for p in net.pops.values())

        p10 = optimizer.full_step(
            spike_dict     = raw_spks,
            membrane_V     = membrane_V,
            U              = 0.5,
            C              = 0.5,
            delta_prime    = float(prev_reward or 0.0),
            conflict_score = 0.1,
            n_reasoning    = 0)
        opt_spks = p10["optimized_spikes"]
        n_opt    = sum(int(np.asarray(sp).sum())
                       for sp in opt_spks.values())

        # ── Phase 3 ───────────────────────────────────────────
        p3 = pipeline.step(
            opt_spks.get("bayesian_layer",
                          np.zeros(net.pops["bayesian_layer"].N)),
            reward      = prev_reward,
            prev_action = prev_action)

        snc_rate = net.pops["snc"].population_rate(50)
        da_level = float(np.clip(
            1.0 + (snc_rate - 4.0) / 20.0, 0.1, 3.0))

        # ── Phase 4 ───────────────────────────────────────────
        p4 = p4_ctrl.step(
            cortex_spikes = opt_spks.get(
                "cortex", np.zeros(net.pops["cortex"].N)),
            d1_spikes     = opt_spks.get(
                "d1_msn", np.zeros(net.pops["d1_msn"].N)),
            d2_spikes     = opt_spks.get(
                "d2_msn", np.zeros(net.pops["d2_msn"].N)),
            belief_scores = p3["V_combined"],
            U             = p3["U"],
            C             = p3["C"],
            dopamine_level= da_level)

        # ── Phase 5 ───────────────────────────────────────────
        p5 = p5_ctrl.step(
            direct_inh    = p4["direct_inh"],
            indirect_exc  = p4["indirect_exc"],
            stn_global    = p4["stn_global"],
            w_go          = p4["w_go"],
            w_nogo        = p4["w_nogo"],
            w_stn         = p4["w_stn"],
            U             = p3["U"],
            C             = p3["C"],
            action_probs  = p3["prob"],
            belief_scores = p3["V_combined"],
            conflict_score= p4["conflict_score"],
            t_ms          = t_ms)

        action = (p5["released_action"]
                  if p5["action_released"] else p3["action"])

        # ── Phase 6 ───────────────────────────────────────────
        rl_out = rl.step(
            d1_spikes     = opt_spks.get(
                "d1_msn", np.zeros(net.pops["d1_msn"].N)),
            belief_scores = p3["V_combined"],
            raw_reward    = float(prev_reward or 0.0),
            action        = action,
            U             = p3["U"],
            C             = p3["C"],
            conflict_score= p4["conflict_score"],
            stn_burst     = p4["stn_burst"],
            dopamine_level= da_level,
            done          = False)

        # ── Phase 8 ───────────────────────────────────────────
        nm_out = nm_ctrl.step(
            delta_prime    = rl_out["delta_prime"],
            U              = p3["U"],
            C              = p3["C"],
            reward         = float(prev_reward or 0.0),
            rho            = rl_out["rho"],
            conflict_score = p4["conflict_score"],
            snc_rate_hz    = snc_rate)

        # ── Phase 7 ───────────────────────────────────────────
        if p10.get("is_reward_event", False):
            plast.step(
                opt_spks,
                rl_out["delta_prime"] * nm_out["Mt"],
                nm_out["alpha_t"])

        # ── Phase 9 ───────────────────────────────────────────
        gm = p5.get("gate_margins", np.zeros(N_ACTIONS))
        if not isinstance(gm, np.ndarray):
            gm = np.zeros(N_ACTIONS)
        do_explain = (step % 500 == 0)
        p9 = reasoning.step(
            V_combined     = p3["V_combined"],
            U              = p3["U"],
            C              = p3["C"],
            Q_risk         = rl_out.get(
                "Q_risk", np.zeros(N_ACTIONS)),
            conflict_score = p4["conflict_score"],
            stn_burst      = p4["stn_burst"],
            direct_inh     = p4["direct_inh"],
            indirect_exc   = p4["indirect_exc"],
            stn_global     = p4["stn_global"],
            gate_margins   = gm,
            DA             = nm_out["DA"],
            ht5            = nm_out["5HT"],
            NE             = nm_out["NE"],
            alpha_t        = nm_out["alpha_t"],
            Mt             = nm_out["Mt"],
            rho            = rl_out["rho"],
            reward         = prev_reward,
            gate_action    = action,
            t_ms           = t_ms,
            explain        = do_explain)

        # ── Task step + feedback ──────────────────────────────
        if not task.done:
            next_state, reward, done, info = task.step(action)
            state = next_state
        else:
            reward = 0.0
            state  = task.reset()

        net.step(ctx_in,
                 rl_out["delta_prime"] * nm_out["DA"])
        pipeline.inject_dopamine_signal(
            rl_out["delta_prime"] * nm_out["Mt"])
        p4_ctrl.apply_reward(reward, action)

        # ── Derived signals for metrics ───────────────────────
        mean_w = float(np.mean([
            float(np.abs(eng.W).mean())
            for eng in plast.engines.values()
            if eng.W is not None]))

        trace_mag = float(np.mean([
            float(np.abs(tr.e_total).mean())
            for tr in plast.traces.values()]))

        ahp_mean = float(np.mean([
            float(np.abs(lim.ahp).mean())
            for lim in optimizer.limiters.values()]))

        energy_pJ = optimizer.budget.record_step(
            n_opt,
            int(n_opt * 15),
            20 if p10.get("is_reward_event") else 0,
            int(do_explain))

        # ── Record all metrics ────────────────────────────────
        agg.record(
            action           = action,
            reward           = reward,
            V_combined       = p3["V_combined"],
            U                = p3["U"],
            C                = p3["C"],
            delta_total      = rl_out["delta_total"],
            alpha_t          = nm_out["alpha_t"],
            DA               = nm_out["DA"],
            ht5              = nm_out["5HT"],
            NE               = nm_out["NE"],
            rho              = rl_out["rho"],
            expl_conf        = p9["explanation_conf"],
            n_rules          = p9["n_rules_fired"],
            n_spikes         = n_opt,
            n_neurons_total  = n_neurons_t,
            direct_inh       = p4["direct_inh"],
            indirect_exc     = p4["indirect_exc"],
            trace_mag        = trace_mag,
            ahp_mag          = ahp_mean,
            mean_weight_mag  = mean_w,
            n_syn_events     = int(n_opt * 15),
            n_weight_updates = 20 if p10.get(
                "is_reward_event") else 0,
            gate_open        = bool(p5["action_released"]),
            cf_delta         = 0.1,
            is_reversal      = False,
            new_stimulus     = (step % 50 == 0))

        step_log_out.append({
            "reward"     : float(reward),
            "U"          : float(p3["U"]),
            "delta_total": float(rl_out["delta_total"]),
            "alpha_t"    : float(nm_out["alpha_t"]),
            "energy_pJ"  : float(energy_pJ),
            "expl_conf"  : float(p9["explanation_conf"]),
            "DA"         : float(nm_out["DA"]),
            "5HT"        : float(nm_out["5HT"]),
            "NE"         : float(nm_out["NE"]),
        })

        prev_action = action
        prev_reward = reward

    agg.record_episode(agg.behavioral.accuracy())
    agg.record_weights(
        {n: e.W for n, e in plast.engines.items()})
    metrics = agg.compute_all()
    return metrics, step_log_out


def main():

    print("\n" + "="*60)
    print("  Phase 13 — Step 33: Evaluation Metrics")
    print("="*60)

    os.makedirs(RESULTS, exist_ok=True)
    reporter = MetricReporter(RESULTS)

    tasks = [
        (ProbabilisticBandit(k=N_ACTIONS, max_steps=2000),
         "probabilistic_bandit", 2000),
        (ReversalLearning(max_steps=2000),
         "reversal_learning", 2000),
        (StopSignalTask(max_steps=2000, stop_prob=0.25),
         "stop_signal", 2000),
        (SequentialDecisionTask(n_actions=N_ACTIONS,
                                 max_steps=2000),
         "sequential_decision", 2000),
        (GridWorld(grid_size=5, max_steps=2000),
         "grid_world", 2000),
    ]

    all_metrics = {}

    for task_obj, task_name, n_steps in tasks:
        print(f"\n  Building pipeline for: {task_name}")
        components = build_pipeline()

        metrics, step_log = run_evaluation(
            task_obj, task_name, n_steps, components)

        all_metrics[task_name] = metrics

        reporter.print_report(metrics, task_name)
        json_path = reporter.save_json(metrics, task_name)
        plot_path = reporter.plot_summary(metrics, task_name)
        ts_path   = reporter.plot_time_series(
            step_log, task_name, DT)

        print(f"  Saved: {json_path}")
        print(f"  Saved: {plot_path}")
        if ts_path:
            print(f"  Saved: {ts_path}")

    # Cross-task summary
    print("\n" + "="*60)
    print("  Cross-task metric averages:")
    cats = ["behavioral","learning","reasoning","neural",
            "meta_learning","neuromodulation","energy"]
    for cat in cats:
        vals = []
        for m in all_metrics.values():
            for v in m.get(cat, {}).values():
                if isinstance(v, float) and 0 <= v <= 1:
                    vals.append(v)
        mean = float(np.mean(vals)) if vals else 0.0
        bar  = "#" * int(mean * 25)
        print(f"  {cat:<22s}: {mean:.3f}  {bar}")

    # Save cross-task JSON
    import json
    summary_path = os.path.join(RESULTS,
                                 "phase13_all_metrics.json")

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, float) and (
                np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    with open(summary_path, "w") as f:
        json.dump(_clean(all_metrics), f, indent=2)

    print(f"\n  Cross-task JSON: {summary_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()