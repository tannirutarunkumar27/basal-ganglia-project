"""
Phase 10 full integration test — Phases 2-10.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ["phase2","phase3","phase4","phase5",
          "phase6","phase7","phase8","phase9"]:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), p))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network.bg_network                      import BGNetwork
from integration.bayesian_reasoning_pipeline import BayesianReasoningPipeline
from integration.bg_pathway_controller       import BGPathwayController
from integration.action_gating_controller    import ActionGatingController
from rl_engine.rl_engine                     import RLEngine
from plasticity.plasticity_manager           import PlasticityManager
from controllers.neuromodulator_controller   import NeuromodulatorController
from integration.reasoning_pipeline          import ReasoningPipeline
from integration.neuromorphic_optimizer      import NeuromorphicOptimizer

np.random.seed(42)

DT        = 0.1e-3
SIM_TIME  = 1.0
N_STEPS   = int(SIM_TIME / DT)
N_ACTIONS = 4
CORRECT   = 2
STATE_DIM = 2 * N_ACTIONS + 4
NAMES     = ["reach_left","reach_right","press_button","wait"]

print("\n" + "="*65)
print("  Phase 10 Integration: Full Pipeline Phases 2-10")
print("="*65)

net      = BGNetwork(dt=DT)
pipeline = BayesianReasoningPipeline(
    n_actions=N_ACTIONS,
    n_neurons_total=net.pops["bayesian_layer"].N,
    window_steps=100, lam=0.85, dt=DT)
p4_ctrl  = BGPathwayController(
    n_actions=N_ACTIONS,
    n_d1_per_action=net.pops["d1_msn"].N // N_ACTIONS,
    n_d2_per_action=net.pops["d2_msn"].N // N_ACTIONS,
    conflict_eps=0.3, dt=DT)
p5_ctrl  = ActionGatingController(
    n_actions=N_ACTIONS, theta_0=0.5, beta=0.4,
    kappa=0.3, refractory_ms=150.0,
    log_dir=os.path.join(HERE, "results"), dt=DT)
rl       = RLEngine(n_actions=N_ACTIONS, state_dim=STATE_DIM,
    n_d1_per_action=net.pops["d1_msn"].N // N_ACTIONS, dt=DT)
pop_sizes_p7 = {k: net.pops[k].N for k in
                ["cortex","d1_msn","d2_msn","gpi","gpe"]}
plast    = PlasticityManager(pop_sizes=pop_sizes_p7,
    n_actions=N_ACTIONS, base_alpha=0.05, dt=DT)
nm_ctrl  = NeuromodulatorController(
    alpha_0=0.05, eta=0.10, dt=DT)
reasoning = ReasoningPipeline(
    n_actions=N_ACTIONS, action_names=NAMES,
    log_dir=os.path.join(HERE, "results"), dt=DT)

# Phase 10: neuromorphic optimizer
pop_sizes_all = {name: pop.N
                 for name, pop in net.pops.items()}
optimizer = NeuromorphicOptimizer(
    pop_sizes         = pop_sizes_all,
    n_actions         = N_ACTIONS,
    target_sparsity   = 0.08,
    stn_U_threshold   = 0.55,
    action_C_threshold= 0.55,
    episode_budget_nJ = 5000.0,
    prune_every       = 2000,
    dt                = DT)

logs = {k: [] for k in [
    "action","reward","energy_pJ","total_nJ",
    "n_spikes_raw","n_spikes_opt","stn_gate","acc_running"]}

prev_action   = None
prev_reward   = None
cum_reward    = 0.0
correct_count = 0

print(f"  Running {SIM_TIME*1000:.0f} ms...")

for step in range(N_STEPS):
    t    = step * DT
    t_ms = t * 1000

    ctx_amp = 0.5e-9 * (np.sin(2*np.pi*5*t) + 1.0)
    ctx_in  = np.full(net.pops["cortex"].N, ctx_amp)

    # Phase 2
    spks = net.step(cortex_input=ctx_in, dopamine_signal=0.0)

    # Phase 10 FIRST: optimize spikes before downstream processing
    membrane_V = {name: pop.V.copy()
                  for name, pop in net.pops.items()}
    n_raw = sum(int(np.asarray(sp).sum()) for sp in spks.values())

    p10_out = optimizer.full_step(
        spike_dict     = spks,
        membrane_V     = membrane_V,
        U              = 0.5,     # placeholder; updated after Phase 3
        C              = 0.5,
        delta_prime    = float(prev_reward or 0.0),
        conflict_score = 0.1,
        n_reasoning    = 1)

    opt_spks = p10_out["optimized_spikes"]
    n_opt    = sum(int(np.asarray(sp).sum())
                   for sp in opt_spks.values())

    # Phase 3 (uses optimized spikes)
    p3 = pipeline.step(
        opt_spks.get("bayesian_layer",
                      np.zeros(net.pops["bayesian_layer"].N)),
        reward=prev_reward, prev_action=prev_action)

    # Re-evaluate Phase 10 gates with real U, C
    gate_state = optimizer.evaluate_gates(
        p3["U"], p3["C"],
        float(prev_reward or 0.0),
        0.1)

    da = float(np.clip(
        1.0 + (net.pops["snc"].population_rate(50)-4.0)/20.0,
        0.1, 3.0))

    # Phase 4 — skip indirect/hyperdirect if gates closed
    p4 = p4_ctrl.step(
        cortex_spikes  = opt_spks.get("cortex",
                          np.zeros(net.pops["cortex"].N)),
        d1_spikes      = opt_spks.get("d1_msn",
                          np.zeros(net.pops["d1_msn"].N)),
        d2_spikes      = opt_spks.get("d2_msn",
                          np.zeros(net.pops["d2_msn"].N)),
        belief_scores  = p3["V_combined"],
        U=p3["U"], C=p3["C"], dopamine_level=da)

    p5 = p5_ctrl.step(
        direct_inh=p4["direct_inh"],
        indirect_exc=p4["indirect_exc"],
        stn_global=p4["stn_global"],
        w_go=p4["w_go"], w_nogo=p4["w_nogo"], w_stn=p4["w_stn"],
        U=p3["U"], C=p3["C"], action_probs=p3["prob"],
        belief_scores=p3["V_combined"],
        conflict_score=p4["conflict_score"], t_ms=t_ms)

    action = (p5["released_action"]
              if p5["action_released"] else p3["action"])

    reward = 1.0 if action == CORRECT else -0.1
    cum_reward += reward
    if action == CORRECT:
        correct_count += 1

    rl_out = rl.step(
        d1_spikes=opt_spks.get("d1_msn",
                   np.zeros(net.pops["d1_msn"].N)),
        belief_scores=p3["V_combined"],
        raw_reward=reward, action=action,
        U=p3["U"], C=p3["C"],
        conflict_score=p4["conflict_score"],
        stn_burst=p4["stn_burst"],
        dopamine_level=da, done=False)

    nm_out = nm_ctrl.step(
        delta_prime=rl_out["delta_prime"],
        U=p3["U"], C=p3["C"], reward=reward,
        rho=rl_out["rho"],
        conflict_score=p4["conflict_score"], snc_rate_hz=da*4)

    # Only run plasticity if reward event gate is open
    if p10_out["is_reward_event"]:
        plast.step(opt_spks, rl_out["delta_prime"]*nm_out["Mt"],
                   nm_out["alpha_t"])

    # Phase 9 reasoning
    gm = p5.get("gate_margins", np.zeros(N_ACTIONS))
    if not isinstance(gm, np.ndarray):
        gm = np.zeros(N_ACTIONS)
    p9 = reasoning.step(
        V_combined=p3["V_combined"], U=p3["U"], C=p3["C"],
        Q_risk=rl_out.get("Q_risk", np.zeros(N_ACTIONS)),
        conflict_score=p4["conflict_score"],
        stn_burst=p4["stn_burst"],
        direct_inh=p4["direct_inh"], indirect_exc=p4["indirect_exc"],
        stn_global=p4["stn_global"], gate_margins=gm,
        DA=nm_out["DA"], ht5=nm_out["5HT"], NE=nm_out["NE"],
        alpha_t=nm_out["alpha_t"], Mt=nm_out["Mt"],
        rho=rl_out["rho"], reward=reward,
        gate_action=action, t_ms=t_ms)

    net.step(cortex_input=ctx_in,
             dopamine_signal=rl_out["delta_prime"] * nm_out["DA"])
    pipeline.inject_dopamine_signal(
        rl_out["delta_prime"] * nm_out["Mt"])
    p4_ctrl.apply_reward(reward, action)

    prev_action = action
    prev_reward = reward

    logs["action"].append(action)
    logs["reward"].append(reward)
    logs["energy_pJ"].append(p10_out["step_energy_pJ"])
    logs["total_nJ"].append(optimizer.budget.total_energy_nJ())
    logs["n_spikes_raw"].append(n_raw)
    logs["n_spikes_opt"].append(n_opt)
    logs["stn_gate"].append(int(gate_state["stn_gate"]))
    logs["acc_running"].append(correct_count / (step + 1))

acc = correct_count / N_STEPS * 100
summary = optimizer.optimizer_summary()
budget  = summary["budget"]
gates_s = summary["gates"]

print(f"\n  Accuracy           : {acc:.1f}%")
print(f"  Cumulative reward  : {cum_reward:.1f}")
print(f"\n  Energy summary:")
print(f"    Total energy      : {budget['total_energy_nJ']:.2f} nJ")
print(f"    Budget (5000 nJ)  : "
      f"{'WITHIN' if budget['within_budget'] else 'EXCEEDED'}")
print(f"    Mean energy/step  : {budget['mean_energy_pJ_step']:.2f} pJ")

eff = optimizer.budget.compute_efficiency_score(acc/100.0)
print(f"    Efficiency score  : {eff:.4f}")

mean_raw = float(np.mean(logs["n_spikes_raw"]))
mean_opt = float(np.mean(logs["n_spikes_opt"]))
spike_save = (1 - mean_opt / max(mean_raw, 1e-9)) * 100
print(f"\n  Spike optimization : {spike_save:.1f}% reduction")

print(f"\n  Gate activation rates:")
for g, r in gates_s["activation_rates"].items():
    print(f"    {g:<20s}: {r*100:.1f}%")

print(f"\n  Synapse pruning:")
for cname, ps in summary["pruning"].items():
    print(f"    {cname:<12s}: {ps['sparsity']*100:.1f}% pruned")

optimizer.budget.print_report()

fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
t_ms_arr = np.arange(N_STEPS) * DT * 1000
smooth   = lambda v, w=100: np.convolve(
    v, np.ones(w)/w, mode="same")

axes[0].plot(t_ms_arr, logs["acc_running"],
             color="forestgreen", lw=1.2)
axes[0].set_title(f"Running accuracy  (final={acc:.1f}%)")
axes[0].set_ylabel("accuracy")
axes[0].set_ylim(0, 1)

axes[1].plot(t_ms_arr, smooth(logs["n_spikes_raw"]),
             color="coral", lw=1, label="raw spikes")
axes[1].plot(t_ms_arr, smooth(logs["n_spikes_opt"]),
             color="steelblue", lw=1.2, label="optimized spikes")
axes[1].set_title(f"Spike count  (Step 29 — {spike_save:.0f}% reduction)")
axes[1].set_ylabel("spikes/step")
axes[1].legend(fontsize=8)

axes[2].plot(t_ms_arr, smooth(logs["energy_pJ"]),
             color="darkorange", lw=1)
axes[2].set_title("Energy per step (pJ) — Step 29+30 optimization")
axes[2].set_ylabel("energy pJ")

axes[3].plot(t_ms_arr, logs["total_nJ"],
             color="slateblue", lw=1)
axes[3].axhline(y=5000, color="red", ls="--",
                lw=0.8, label="budget limit")
axes[3].set_title("Cumulative energy (nJ)")
axes[3].set_ylabel("nJ")
axes[3].legend(fontsize=8)

axes[4].plot(t_ms_arr, smooth(logs["stn_gate"]),
             color="crimson", lw=1, label="STN gate")
axes[4].set_title("STN gate activation (Step 30 — conditional)")
axes[4].set_xlabel("Time (ms)")
axes[4].set_ylabel("active fraction")
axes[4].legend(fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "phase10_integration.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("\n  Plot saved: results/phase10_integration.png")
print("="*65 + "\n")