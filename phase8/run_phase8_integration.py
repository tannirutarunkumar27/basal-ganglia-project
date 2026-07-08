"""
Phase 8 full integration test — Phases 2-8.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ["phase2","phase3","phase4","phase5","phase6","phase7"]:
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

np.random.seed(42)

DT        = 0.1e-3
SIM_TIME  = 1.0
N_STEPS   = int(SIM_TIME / DT)
N_ACTIONS = 4
CORRECT   = 2
STATE_DIM = 2 * N_ACTIONS + 4

print("\n" + "="*65)
print("  Phase 8 Integration: Full Pipeline Phases 2-8")
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
rl       = RLEngine(
    n_actions=N_ACTIONS, state_dim=STATE_DIM,
    n_d1_per_action=net.pops["d1_msn"].N // N_ACTIONS,
    dt=DT)
pop_sizes = {
    "cortex": net.pops["cortex"].N,
    "d1_msn": net.pops["d1_msn"].N,
    "d2_msn": net.pops["d2_msn"].N,
    "gpi"   : net.pops["gpi"].N,
    "gpe"   : net.pops["gpe"].N,
}
plast    = PlasticityManager(
    pop_sizes=pop_sizes, n_actions=N_ACTIONS,
    base_alpha=0.05, min_delta=0.03, dt=DT)

# Phase 8: neuromodulator controller
nm_ctrl  = NeuromodulatorController(
    alpha_0=0.05, eta=0.10,
    omega_d=0.5, omega_s=0.3, omega_n=0.2, dt=DT)

logs = {k: [] for k in [
    "action","reward","alpha_t","Mt",
    "DA","5HT","NE","learn_rate","explore_temp",
    "w_go_scale","w_nogo_scale","acc_running"]}

prev_action   = None
prev_reward   = None
cum_reward    = 0.0
correct_count = 0

print(f"  Running {SIM_TIME*1000:.0f} ms...")

for step in range(N_STEPS):
    t = step * DT
    ctx_amp = 0.5e-9 * (np.sin(2*np.pi*5*t) + 1.0)
    ctx_in  = np.full(net.pops["cortex"].N, ctx_amp)

    spks = net.step(cortex_input=ctx_in, dopamine_signal=0.0)
    p3   = pipeline.step(spks["bayesian_layer"],
                          reward=prev_reward,
                          prev_action=prev_action)
    da   = float(np.clip(
        1.0 + (net.pops["snc"].population_rate(50)-4.0)/20.0,
        0.1, 3.0))
    snc_rate = net.pops["snc"].population_rate(50)

    p4 = p4_ctrl.step(
        cortex_spikes=spks["cortex"],
        d1_spikes=spks["d1_msn"],
        d2_spikes=spks["d2_msn"],
        belief_scores=p3["V_combined"],
        U=p3["U"], C=p3["C"],
        dopamine_level=da)

    p5 = p5_ctrl.step(
        direct_inh=p4["direct_inh"],
        indirect_exc=p4["indirect_exc"],
        stn_global=p4["stn_global"],
        w_go=p4["w_go"], w_nogo=p4["w_nogo"], w_stn=p4["w_stn"],
        U=p3["U"], C=p3["C"],
        action_probs=p3["prob"],
        belief_scores=p3["V_combined"],
        conflict_score=p4["conflict_score"],
        t_ms=t*1000)

    action = (p5["released_action"]
              if p5["action_released"] else p3["action"])

    reward = 1.0 if action == CORRECT else -0.1
    cum_reward += reward
    if action == CORRECT:
        correct_count += 1

    rl_out = rl.step(
        d1_spikes=spks["d1_msn"],
        belief_scores=p3["V_combined"],
        raw_reward=reward, action=action,
        U=p3["U"], C=p3["C"],
        conflict_score=p4["conflict_score"],
        stn_burst=p4["stn_burst"],
        dopamine_level=da, done=False)

    # Phase 8: compute meta-dopamine and neuromodulator fusion
    nm_out = nm_ctrl.step(
        delta_prime    = rl_out["delta_prime"],
        U              = p3["U"],
        C              = p3["C"],
        reward         = reward,
        rho            = rl_out["rho"],
        conflict_score = p4["conflict_score"],
        snc_rate_hz    = snc_rate)

    # Feed Phase 8 alpha_t back into Phase 7
    dW_dict = plast.step(
        spike_dict  = spks,
        delta_prime = rl_out["delta_prime"] * nm_out["Mt"],
        alpha_t     = nm_out["alpha_t"])

    # Update Phase 2 SNc with enriched DA
    net.step(cortex_input=ctx_in,
             dopamine_signal=rl_out["delta_prime"] * nm_out["DA"])
    pipeline.inject_dopamine_signal(
        rl_out["delta_prime"] * nm_out["Mt"])
    p4_ctrl.apply_reward(reward, action)

    prev_action = action
    prev_reward = reward

    logs["action"].append(action)
    logs["reward"].append(reward)
    logs["alpha_t"].append(nm_out["alpha_t"])
    logs["Mt"].append(nm_out["Mt"])
    logs["DA"].append(nm_out["DA"])
    logs["5HT"].append(nm_out["5HT"])
    logs["NE"].append(nm_out["NE"])
    logs["learn_rate"].append(nm_out["learning_rate"])
    logs["explore_temp"].append(nm_out["explore_temp"])
    logs["w_go_scale"].append(nm_out["w_go_scale"])
    logs["w_nogo_scale"].append(nm_out["w_nogo_scale"])
    logs["acc_running"].append(correct_count / (step + 1))

acc = correct_count / N_STEPS * 100
print(f"\n  Accuracy   : {acc:.1f}%")
print(f"  Cum reward : {cum_reward:.1f}")

nm_summary = nm_ctrl.controller_summary()
print(f"\n  Phase 8 summary:")
print(f"    NE arousal  : {nm_summary['NE_arousal']}")
print(f"    5HT patience: {nm_summary['5HT_patience']:.3f}")
print(f"    NE gain     : {nm_summary['NE_gain']:.3f}")
print(f"    Meta regime : {nm_summary['meta_da']['regime']}")
print(f"    Dominant NM : {nm_summary['fusion']['dominant']}")
print(f"    Mean Mt     : {nm_summary['fusion']['mean_Mt']:.3f}")
print(f"    omega_d/s/n : "
      f"{nm_summary['fusion']['omega_d']:.3f} / "
      f"{nm_summary['fusion']['omega_s']:.3f} / "
      f"{nm_summary['fusion']['omega_n']:.3f}")

fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
t_ms_arr = np.arange(N_STEPS) * DT * 1000
smooth   = lambda v, w=100: np.convolve(v, np.ones(w)/w, mode="same")

axes[0].plot(t_ms_arr, logs["acc_running"],
             color="forestgreen", lw=1.2)
axes[0].set_title(f"Running accuracy  (final={acc:.1f}%)")
axes[0].set_ylabel("accuracy")
axes[0].set_ylim(0, 1)

axes[1].plot(t_ms_arr, smooth(logs["alpha_t"]),
             color="darkorange", lw=1.2, label="alpha_t (Step 23)")
axes[1].plot(t_ms_arr, smooth(logs["learn_rate"]),
             color="purple", lw=1, label="learning rate (Mt-modulated)")
axes[1].set_title("Step 23: meta-dopamine alpha_t and Mt-modulated learning rate")
axes[1].set_ylabel("rate")
axes[1].legend(fontsize=8)

axes[2].plot(t_ms_arr, smooth(logs["DA"]),  color="steelblue",
             lw=1, label="DA")
axes[2].plot(t_ms_arr, smooth(logs["5HT"]), color="coral",
             lw=1, label="5-HT")
axes[2].plot(t_ms_arr, smooth(logs["NE"]),  color="goldenrod",
             lw=1, label="NE")
axes[2].plot(t_ms_arr, smooth(logs["Mt"]),  color="slateblue",
             lw=1.5, ls="--", label="Mt (fused)")
axes[2].set_title("Step 24: neuromodulator levels and fused signal Mt")
axes[2].set_ylabel("level [0,1]")
axes[2].legend(fontsize=8)
axes[2].set_ylim(0, 1.1)

axes[3].plot(t_ms_arr, smooth(logs["explore_temp"]),
             color="goldenrod", lw=1, label="explore temp (NE-driven)")
axes[3].plot(t_ms_arr, smooth(logs["w_go_scale"]),
             color="steelblue", lw=1, label="wGo scale (DA-driven)")
axes[3].plot(t_ms_arr, smooth(logs["w_nogo_scale"]),
             color="coral", lw=1, label="wNoGo scale (5HT-driven)")
axes[3].set_title("Mt-regulated pathway balance and exploration temperature")
axes[3].set_ylabel("scale")
axes[3].legend(fontsize=8)

axes[4].plot(t_ms_arr, np.cumsum(logs["reward"]),
             color="forestgreen", lw=1)
axes[4].set_title("Cumulative reward")
axes[4].set_xlabel("Time (ms)")
axes[4].set_ylabel("reward")

plt.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
plt.savefig(os.path.join(HERE, "results", "phase8_integration.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print("\n  Plot saved: results/phase8_integration.png")
print("="*65 + "\n")